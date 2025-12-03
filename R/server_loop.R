#' Rounds to Target with Linear Interpolation
#'
#' Returns the fractional round where test accuracy reaches target using linear interpolation.
#'
#' @param history Data frame with round and test_acc columns.
#' @param target Target accuracy threshold.
#' @return Fractional round reaching target (via interpolation), or NA if not reached.
#' @export
rounds_to_target <- function(history, target = 0.97) {
    if (nrow(history) == 0) {
        return(NA_real_)
    }

    # Find first round where accuracy >= target
    idx <- which(history$test_acc >= target)

    if (length(idx) == 0) {
        return(NA_real_) # Never reached
    }

    first_idx <- idx[1]

    if (first_idx == 1) {
        # Reached in first round
        return(as.numeric(history$round[1]))
    }

    # Linear interpolation between round[first_idx-1] and round[first_idx]
    r_prev <- history$round[first_idx - 1]
    r_curr <- history$round[first_idx]
    acc_prev <- history$test_acc[first_idx - 1]
    acc_curr <- history$test_acc[first_idx]

    # Interpolate: r* = r_prev + (target - acc_prev) * (r_curr - r_prev) / (acc_curr - acc_prev)
    r_star <- r_prev + (target - acc_prev) * (r_curr - r_prev) / (acc_curr - acc_prev)

    return(r_star)
}

#' Run FedAvg MNIST Experiment
#'
#' Runs the full Federated Averaging experiment on MNIST with paper-accurate settings.
#'
#' @param ds_train Training dataset.
#' @param ds_test Test dataset.
#' @param labels_train Training labels (for partitioning).
#' @param model_fn Model function: "2nn" for MLP or "cnn" for CNN.
#' @param partition Partition type: "IID" or "nonIID".
#' @param K Total number of clients.
#' @param C Fraction of clients per round.
#' @param E Local epochs.
#' @param batch_size Local batch size (use Inf for FedSGD).
#' @param lr_grid Learning rate grid for selection.
#' @param target Target accuracy for rounds-to-target (0.97 for 2NN, 0.99 for CNN).
#' @param rounds Number of communication rounds.
#' @param seed Random seed.
#' @param device Device to use ("cpu").
#' @param fedsgd If TRUE, run FedSGD mode (batch_size=Inf, E=1).
#' @return A list containing `history` (data.frame) and `params` (final model parameters).
#' @export
run_fedavg_mnist <- function(ds_train, ds_test, labels_train,
                             model_fn = "2nn",
                             partition = "nonIID",
                             K = 100, C = 0.1, E = 5, batch_size = 10,
                             lr_grid = c(0.03, 0.05, 0.1),
                             target = 0.97,
                             rounds = 3, seed = 123,
                             device = "cpu",
                             fedsgd = FALSE) {
    set.seed(seed)
    torch::torch_manual_seed(seed)

    # Override for FedSGD mode
    if (fedsgd) {
        batch_size <- Inf
        E <- 1
    }

    # Compute u statistic: u = 6*E/B (or 1 if B=Inf)
    u <- if (is.infinite(batch_size)) 1 else (6 * E / batch_size)

    # Determine method name
    method <- if (fedsgd) "FedSGD" else "FedAvg"

    cat(sprintf("Partitioning data (%s)...\n", partition))

    # Partition data
    splitter <- if (tolower(partition) == "iid") {
        function(l, K, seed) iid_split(length(l), K, seed)
    } else {
        mnist_shards_split
    }

    client_indices <- splitter(labels_train, K, seed)

    cat("Initializing model...\n")

    # Initialize global model
    global_model <- if (tolower(model_fn) == "cnn") {
        mnist_cnn()
    } else {
        mnist_mlp()
    }

    global_model$to(device = device)
    global_params <- flatten_params(global_model)

    # ONE-TIME LEARNING RATE SELECTION (matches Flower reference)
    # reference/fedavg_mnist_flwr/client.py:29,58 - LR is fixed per client
    cat("Selecting learning rate (one-time)...\n")
    best_lr <- lr_grid[1]
    best_acc <- 0

    # Sample initial clients for LR selection
    m_init <- max(floor(C * K), 1)
    set.seed(seed)
    init_clients <- sort(sample(1:K, min(2, m_init), replace = FALSE))

    for (lr_candidate in lr_grid) {
        # Warm-start from initial global params
        temp_params <- global_params
        for (client_id in init_clients) {
            client_result <- client_train_mnist(
                indices = client_indices[[client_id]],
                ds_train = ds_train,
                init_params = temp_params,
                epochs = 1, # Single epoch for LR selection
                batch_size = batch_size,
                lr = lr_candidate,
                momentum = 0,
                model_fn = model_fn,
                seed = seed + client_id,
                device = device
            )
            temp_params <- client_result$params
        }

        # Evaluate
        temp_model <- if (tolower(model_fn) == "cnn") {
            mnist_cnn() # First () creates module class, second () instantiates
        } else {
            mnist_mlp()
        }
        temp_model$to(device = device)
        unflatten_params(temp_model, temp_params)
        temp_model$eval()

        acc <- eval_mnist_accuracy(temp_model, ds_test, device = device)

        if (acc > best_acc) {
            best_acc <- acc
            best_lr <- lr_candidate
        }
    }

    cat(sprintf("Selected LR: %.3f (acc: %.4f) - will use for all rounds\n", best_lr, best_acc))

    # History tracking
    history <- data.frame(
        dataset = character(),
        model = character(),
        partition = character(),
        method = character(),
        round = integer(),
        test_acc = numeric(),
        chosen_lr = numeric(),
        E = integer(),
        B = character(),
        C = numeric(),
        clients_selected = integer(),
        u = numeric(),
        target = numeric(),
        rtt = numeric(),
        stringsAsFactors = FALSE
    )

    # MAIN TRAINING LOOP with FIXED LR
    for (r in 1:rounds) {
        cat(sprintf("Round %d/%d\n", r, rounds))

        # Sample clients
        m <- max(floor(C * K), 1)
        set.seed(seed + r)
        selected_clients <- sort(sample(1:K, m, replace = FALSE))

        # Train clients with FIXED LR (no per-round selection)
        client_params_list <- list()
        client_weights <- numeric(m)

        for (i in seq_along(selected_clients)) {
            client_id <- selected_clients[i]

            client_result <- client_train_mnist(
                indices = client_indices[[client_id]],
                ds_train = ds_train,
                init_params = global_params,
                epochs = E,
                batch_size = batch_size,
                lr = best_lr, # Use fixed LR
                momentum = 0,
                model_fn = model_fn,
                seed = seed + r + client_id,
                device = device
            )

            client_params_list[[i]] <- client_result$params
            client_weights[i] <- client_result$n
        }

        # Aggregate (weighted by n_k)
        global_params <- fedavg(client_params_list, client_weights)

        # Evaluate
        unflatten_params(global_model, global_params)
        global_model$eval()
        test_acc <- eval_mnist_accuracy(global_model, ds_test, device = device)

        cat(sprintf("  Test Acc: %.4f\n", test_acc))

        # Compute rounds-to-target
        temp_history <- data.frame(
            round = 1:r,
            test_acc = c(if (r > 1) history$test_acc else numeric(0), test_acc)
        )
        rtt <- rounds_to_target(temp_history, target)

        # Record history
        history <- rbind(history, data.frame(
            dataset = "MNIST",
            model = toupper(model_fn),
            partition = partition,
            method = method,
            round = r,
            test_acc = test_acc,
            chosen_lr = best_lr,
            E = E,
            B = if (is.infinite(batch_size)) "Inf" else as.character(batch_size),
            C = C,
            clients_selected = m,
            u = u,
            target = target,
            rtt = if (is.na(rtt)) NA_real_ else rtt,
            stringsAsFactors = FALSE
        ))
    }

    return(list(history = history, params = global_params))
}
