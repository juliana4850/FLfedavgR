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
#' @return A list containing `history` (data.frame) and `final_params`.
#' @export
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
                              log_file = NULL) {
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
        metrics <- evaluation_fn(global_model, device)

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

    list(
        history = dplyr::bind_rows(history),
        final_params = global_params
    )
}
