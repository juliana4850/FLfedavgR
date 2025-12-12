#' Generic FedAvg Simulation
#'
#' Runs a Federated Averaging simulation on a generic dataset.
#'
#' @param client_datasets A list of torch::dataset objects (one per client).
#' @param model_generator A function that returns a fresh instance of the model (torch::nn_module).
#' @param evaluation_fn A function(model, device) that returns a named list of metrics (e.g., list(accuracy=0.9)).
#' @param rounds Number of communication rounds.
#' @param C Fraction of clients to select per round.
#' @param E Number of local epochs.
#' @param batch_size Local batch size (Inf for full-batch).
#' @param optimizer_generator A function(lr) that returns an optimizer_fn for client_train_generic.
#' @param lr_scheduler A function(round) that returns the learning rate for that round.
#' @param seed Random seed.
#' @param device Device to use ("cpu" or "cuda").
#' @param log_file Optional path to CSV file for incremental logging of metrics.
#' @param save_model Optional path to save the final trained model (e.g., "model.pt"). If NULL, model is not saved.
#'
#' @return A list containing `history` (data.frame), `final_params`, and `final_model` (if save_model is specified).
#'
#' @references
#' McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
#' Communication-Efficient Learning of Deep Networks from Decentralized Data.
#' *Proceedings of the 20th International Conference on Artificial Intelligence
#' and Statistics (AISTATS)*.
#'
#' @export
#'
#' @examples
#' library(torch)
#' library(fedavgR)
#'
#' # 1. Define your model generator
#' model_gen <- function() {
#'     nn_sequential(
#'         nn_linear(10, 20),
#'         nn_relu(),
#'         nn_linear(20, 2)
#'     )
#' }
#'
#' # 2. Prepare client datasets (list of torch datasets)
#' make_classif_dataset <- function(n, p, margin = 1.0) {
#'     w <- torch_randn(p)
#'     x <- torch_randn(n, p)
#'     score <- x$matmul(w) + 0.5 * torch_randn(n)
#'     # R torch: class indices must start at 1
#'     y01 <- (score > margin)$to(dtype = torch_long())
#'     y <- (y01 + 1L)$to(dtype = torch_long())
#'
#'     dataset(
#'         name = "toy_cls",
#'         initialize = function() {
#'             self$x <- x
#'             self$y <- y
#'         },
#'         .getitem = function(i) {
#'             list(self$x[i, ], self$y[i])
#'         },
#'         .length = function() self$x$size()[1]
#'     )()
#' }
#' K <- 10L
#' p <- 10L
#' clients <- lapply(seq_len(K), function(i) {
#'     make_classif_dataset(n = sample(80:120, 1), p = p)
#' })
#'
#' # 3. Define evaluation function
#' val_ds <- make_classif_dataset(n = 512, p = p)
#' evaluation_fn <- function(model, device = if (cuda_is_available()) "cuda" else "cpu") {
#'     model$eval()
#'     dl <- dataloader(val_ds, batch_size = 256, shuffle = FALSE)
#'     correct <- 0
#'     total <- 0
#'     coro::loop(for (b in dl) {
#'         x <- b[[1]]$to(device = device)
#'         y <- b[[2]]$to(device = device)
#'         with_no_grad({
#'             logits <- model(x)
#'             pred <- torch_argmax(logits, dim = 2)
#'         })
#'         correct <- correct + as.numeric((pred == y)$sum()$cpu())
#'         total <- total + length(y)
#'     })
#'     list(acc = correct / total)
#' }
#'
#' # 4. Run Simulation
#' results <- fedavg_simulation(
#'     client_datasets = clients,
#'     model_generator = model_gen,
#'     evaluation_fn = evaluation_fn,
#'     rounds = 5,
#'     C = 0.3, # ~30% clients per round
#'     E = 1, # 1 local epoch
#'     batch_size = 32,
#'     optimizer_generator = function(lr) function(p) optim_sgd(p, lr = lr, momentum = 0),
#'     lr_scheduler = function(r) 0.05,
#'     seed = 123,
#'     device = if (cuda_is_available()) "cuda" else "cpu",
#'     log_file = NULL
#' )
#'
#' print(results$history)
fedavg_simulation <- function(client_datasets,
                              model_generator,
                              evaluation_fn,
                              rounds = 10,
                              C = 0.1,
                              E = 1,
                              batch_size = 32,
                              optimizer_generator = function(lr) function(p) torch::optim_sgd(p, lr = lr),
                              lr_scheduler = function(r) 0.1,
                              seed = 123,
                              device = "cpu",
                              log_file = NULL,
                              save_model = NULL) {
    if (!is.null(seed)) {
        set.seed(seed)
        torch::torch_manual_seed(seed)
    }

    K <- length(client_datasets)
    m <- max(1, round(C * K))

    cat(sprintf(
        "Starting FedAvg Simulation: K=%d, C=%.2f, E=%d, B=%s, Rounds=%d\n",
        K, C, E, if (is.infinite(batch_size)) "Inf" else batch_size, rounds
    ))

    # Initialize global model
    global_model <- model_generator()
    global_model$to(device = device)
    global_params <- flatten_params(global_model)

    # History storage
    history <- list()

    for (r in 1:rounds) {
        start_time <- Sys.time()

        # Select clients
        selected_indices <- sample(K, m)

        # Determine LR for this round
        lr <- lr_scheduler(r)
        optimizer_fn <- optimizer_generator(lr)

        # Train selected clients
        local_params_list <- list()
        local_weights <- numeric(m)

        for (i in seq_along(selected_indices)) {
            client_idx <- selected_indices[i]
            ds_client <- client_datasets[[client_idx]]

            # Train
            res <- client_train_generic(
                dataset = ds_client,
                model = model_generator(), # Fresh model instance
                init_params = global_params,
                epochs = E,
                batch_size = batch_size,
                optimizer_fn = optimizer_fn,
                device = device
            )

            local_params_list[[i]] <- res$params
            local_weights[i] <- res$n
        }

        # Aggregate
        global_params <- fedavg(local_params_list, local_weights)

        # Update global model for evaluation
        unflatten_params(global_model, global_params)

        # Evaluate
        metrics <- evaluation_fn(global_model, device = device)

        # Log to console
        elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        metrics_str <- paste(names(metrics), sprintf("%.4f", unlist(metrics)), sep = "=", collapse = ", ")
        cat(sprintf("Round %d/%d: %s (%.1fs)\n", r, rounds, metrics_str, elapsed))

        # Log to CSV if requested
        if (!is.null(log_file)) {
            log_row <- data.frame(
                timestamp = Sys.time(),
                round = r,
                elapsed = elapsed,
                as.data.frame(metrics)
            )

            # Format timestamp properly
            log_row$timestamp <- format(log_row$timestamp, "%Y-%m-%d %H:%M:%OS5")

            if (!file.exists(log_file)) {
                write.table(log_row, log_file, sep = ",", row.names = FALSE, col.names = TRUE, quote = FALSE)
            } else {
                # Efficient append using cat
                line <- paste(as.character(log_row[1, ]), collapse = ",")
                cat(line, "\n", file = log_file, append = TRUE, sep = "")
            }
        }

        # Store history
        history[[r]] <- c(list(round = r, time = elapsed), metrics)

        # Force garbage collection to prevent memory buildup
        gc()
    }

    # Save model if requested
    if (!is.null(save_model)) {
        # Create directory if it doesn't exist
        save_dir <- dirname(save_model)
        if (!dir.exists(save_dir) && save_dir != ".") {
            dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
        }

        # Save the final model
        torch::torch_save(global_model, save_model)
        cat(sprintf("Model saved to: %s\n", save_model))
    }

    result <- list(
        history = dplyr::bind_rows(history),
        final_params = global_params
    )

    # Include final model in result if it was saved
    if (!is.null(save_model)) {
        result$final_model <- global_model
        result$model_path <- save_model
    }

    result
}
