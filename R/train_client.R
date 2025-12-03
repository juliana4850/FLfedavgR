#' Client Train MNIST
#'
#' Trains a model on a client's local data.
#'
#' @param indices Indices of the local dataset.
#' @param ds_train The full training dataset.
#' @param init_params Flattened initial parameters.
#' @param epochs Number of local epochs.
#' @param batch_size Batch size (use Inf for FedSGD mode).
#' @param lr Learning rate.
#' @param momentum SGD momentum (default: 0 per paper).
#' @param model_fn Model function: "2nn" for MLP or "cnn" for CNN.
#' @param seed Random seed.
#' @param device Device to use.
#' @return A list containing `params` (flattened) and `n` (number of samples).
#' @export
client_train_mnist <- function(indices, ds_train, init_params, epochs = 5,
                               batch_size = 10, lr = 0.1, momentum = 0,
                               model_fn = "2nn", seed = 123, device = "cpu") {
    if (!is.null(seed)) {
        set.seed(seed)
        torch::torch_manual_seed(seed)
    }

    # Create local model based on model_fn
    model <- if (tolower(model_fn) == "cnn") {
        mnist_cnn()
    } else {
        mnist_mlp()
    }

    model$to(device = device)
    unflatten_params(model, init_params)
    model$train()

    # Create local dataset and loader
    ds_local <- dataset_subset(ds_train, indices)

    # Handle batch_size = Inf (FedSGD mode: entire local dataset as one batch)
    effective_batch_size <- if (is.infinite(batch_size)) length(indices) else batch_size
    dl <- torch::dataloader(ds_local, batch_size = effective_batch_size, shuffle = TRUE)

    optimizer <- torch::optim_sgd(model$parameters, lr = lr, momentum = momentum)
    criterion <- torch::nn_cross_entropy_loss()

    for (e in 1:epochs) {
        iter <- torch::dataloader_make_iter(dl)
        while (!is.null(batch <- torch::dataloader_next(iter))) {
            optimizer$zero_grad()

            if (is.null(names(batch)) || !all(c("x", "y") %in% names(batch))) {
                # Handle unnamed batch from tensor_dataset
                x <- batch[[1]]$to(device = device)
                y <- batch[[2]]$to(device = device)
            } else {
                x <- batch$x$to(device = device)
                y <- batch$y$to(device = device)
            }

            # Ensure x has channel dimension for CNN: [batch, 28, 28] -> [batch, 1, 28, 28]
            if (length(dim(x)) == 3) {
                x <- x$unsqueeze(2)
            }

            output <- model(x)
            # Ensure target is 1D
            y <- y$view(-1)
            loss <- criterion(output, y)

            loss$backward()
            optimizer$step()
        }
    }

    list(
        params = flatten_params(model),
        n = length(indices)
    )
}
