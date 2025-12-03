#' Generic Client Training
#'
#' Trains a model on a client's local dataset.
#'
#' @param dataset A torch::dataset object containing the client's data.
#' @param model A torch::nn_module (already instantiated).
#' @param init_params Flattened initial parameters vector.
#' @param epochs Number of local epochs.
#' @param batch_size Batch size (use Inf for full-batch).
#' @param optimizer_fn Function to create optimizer (default: SGD).
#' @param loss_fn Loss function (default: CrossEntropy).
#' @param device Device to use ("cpu" or "cuda").
#' @return A list containing `params` (flattened) and `n` (number of samples).
#' @export
client_train_generic <- function(dataset, model, init_params, epochs = 1,
                                 batch_size = 32,
                                 optimizer_fn = function(params) torch::optim_sgd(params, lr = 0.1),
                                 loss_fn = torch::nn_cross_entropy_loss(),
                                 device = "cpu") {
    # Move model to device and load parameters
    model$to(device = device)
    unflatten_params(model, init_params)
    model$train()

    # Handle full-batch training
    n_samples <- length(dataset)
    effective_batch_size <- if (is.infinite(batch_size)) n_samples else batch_size

    # Create dataloader
    dl <- torch::dataloader(dataset, batch_size = effective_batch_size, shuffle = TRUE)

    # Create optimizer
    optimizer <- optimizer_fn(model$parameters)

    # Training loop
    for (e in seq_len(epochs)) {
        iter <- torch::dataloader_make_iter(dl)
        while (!is.null(batch <- torch::dataloader_next(iter))) {
            optimizer$zero_grad()

            # Handle batch structure (list or named list)
            if (is.null(names(batch)) || !all(c("x", "y") %in% names(batch))) {
                x <- batch[[1]]$to(device = device)
                y <- batch[[2]]$to(device = device)
            } else {
                x <- batch$x$to(device = device)
                y <- batch$y$to(device = device)
            }

            # Forward pass
            output <- model(x)

            # Compute loss
            # Ensure target is 1D for CrossEntropy if needed, but generic loss might handle it differently
            # For now, we assume standard classification setup
            if (inherits(loss_fn, "nn_cross_entropy_loss")) {
                y <- y$view(-1)
            }

            loss <- loss_fn(output, y)

            # Backward pass
            loss$backward()
            optimizer$step()
        }
    }

    # Return updated parameters and sample count
    list(
        params = flatten_params(model),
        n = n_samples
    )
}
