#' MNIST Dataset
#'
#' Wrapper for torchvision::mnist_dataset with required transforms.
#' Transform matches Flower reference implementation.
#'
#' @param root Directory to store data.
#' @param train Boolean, whether to load training set.
#' @param download Boolean, whether to download if missing.
#' @return A torch dataset.
#' @export
#' @references
#' Mirrored from reference/fedavg_mnist_flwr/dataset.py:69-71
mnist_ds <- function(root = "data", train = TRUE, download = TRUE) {
    transform <- function(x) {
        # Transform matching Flower reference:
        # ToTensor + Normalize(mean=0.1307, std=0.3081)
        # reference/fedavg_mnist_flwr/dataset.py:69-71
        t <- torchvision::transform_to_tensor(x)$to(dtype = torch::torch_float())
        # Normalize: (x - mean) / std
        t <- (t - 0.1307) / 0.3081
        t
    }

    torchvision::mnist_dataset(
        root = root,
        train = train,
        download = download,
        transform = transform
    )
}

#' Custom Collate Function for MNIST
#'
#' Ensures proper batching with channel dimension preserved.
#'
#' @param batch List of dataset items
#' @return List with properly shaped tensors
#' @export
mnist_collate <- function(batch) {
    # Use default collate first
    default_batch <- torch::torch_default_collate(batch)

    # default_batch should be list(x=tensor, y=tensor)
    # x might be [batch, 28, 28] and needs to be [batch, 1, 28, 28]
    x <- default_batch$x
    y <- default_batch$y

    # Add channel dimension if missing
    if (length(dim(x)) == 3) {
        x <- x$unsqueeze(2) # Insert channel dim: [batch, 28, 28] -> [batch, 1, 28, 28]
    }

    list(x = x, y = y)
}

#' Get MNIST Labels
#'
#' Extracts labels from the dataset for partitioning.
#'
#' @param ds The MNIST dataset.
#' @param batch_size Batch size for loading.
#' @return A numeric vector of labels.
#' @export
mnist_labels <- function(ds, batch_size = 2000) {
    dl <- torch::dataloader(ds, batch_size = batch_size, shuffle = FALSE)
    iter <- torch::dataloader_make_iter(dl)

    all_labels <- list()
    while (!is.null(batch <- torch::dataloader_next(iter))) {
        # batch$y is a tensor of labels
        all_labels[[length(all_labels) + 1]] <- as.numeric(batch$y$to(device = "cpu"))
    }
    unlist(all_labels)
}
