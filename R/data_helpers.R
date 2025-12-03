#' Dataset Subset
#'
#' Creates a subset of a dataset at specified indices.
#'
#' @param dataset The original dataset.
#' @param indices The indices to select (1-based).
#' @export
dataset_subset <- torch::dataset(
    "dataset_subset",
    initialize = function(dataset, indices) {
        self$dataset <- dataset
        self$indices <- indices
    },
    .getitem = function(i) {
        self$dataset[self$indices[i]]
    },
    .length = function() {
        length(self$indices)
    }
)

#' Evaluate MNIST Accuracy
#'
#' Computes accuracy on a dataset.
#'
#' @param model The model to evaluate.
#' @param ds The dataset to evaluate on.
#' @param device The device to use ("cpu").
#' @param batch_size Batch size.
#' @return The accuracy (0-1).
#' @export
eval_mnist_accuracy <- function(model, ds, device = "cpu", batch_size = 1000) {
    model$eval()
    dl <- torch::dataloader(ds, batch_size = batch_size)
    iter <- torch::dataloader_make_iter(dl)

    correct <- 0
    total <- 0

    torch::with_no_grad({
        while (!is.null(batch <- torch::dataloader_next(iter))) {
            # Handle both named and unnamed batches
            if (is.null(names(batch)) || !all(c("x", "y") %in% names(batch))) {
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

            # Ensure y is 1D
            y <- y$view(-1)

            out <- model(x)
            # argmax along dim 2 (classes)
            pred <- out$argmax(dim = 2)

            correct <- correct + (pred == y)$sum()$item()
            total <- total + length(y)
        }
    })

    model$train() # Restore train mode
    if (total == 0) {
        return(0)
    }
    as.numeric(correct) / total
}
