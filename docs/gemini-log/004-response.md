```R
#' @import torch
#' @import torchvision

#' @export
#' @title MNIST Dataset
#' @description Returns the MNIST dataset. Downloads if not present.
#' @param root The directory where the dataset will be stored.
#' @param train Logical; if TRUE, returns the training set, otherwise the test set.
#' @examples
#' \dontrun{
#' mnist_train_ds <- mnist_ds(train = TRUE)
#' length(mnist_train_ds)
#' }
mnist_ds <- function(root = "data", train = TRUE) {
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  torchvision::mnist_dataset(
    root = root,
    train = train,
    download = TRUE,
    transform = torchvision::transform_to_tensor()
  )
}

#' @export
#' @title CIFAR10 Dataset
#' @description Returns the CIFAR10 dataset. Downloads if not present.
#' @param root The directory where the dataset will be stored.
#' @param train Logical; if TRUE, returns the training set, otherwise the test set.
#' @examples
#' \dontrun{
#' cifar10_train_ds <- cifar10_ds(train = TRUE)
#' length(cifar10_train_ds)
#' }
cifar10_ds <- function(root = "data", train = TRUE) {
  dir.create(root, recursive = TRUE, showWarnings = FALSE)
  torchvision::cifar10_dataset(
    root = root,
    train = train,
    download = TRUE,
    transform = torchvision::transform_to_tensor()
  )
}

#' @export
#' @title Extract MNIST Labels
#' @description Extracts integer labels from an MNIST dataset in chunks.
#' @param ds An MNIST dataset object (e.g., created by `mnist_ds()`).
#' @param batch The batch size to use for iterating through the dataset.
#' @return An integer vector of labels.
#' @examples
#' \dontrun{
#' mnist_train_ds <- mnist_ds(train = TRUE)
#' mnist_train_labels <- mnist_labels(mnist_train_ds)
#' head(mnist_train_labels)
#' }
mnist_labels <- function(ds, batch = 2000) {
  all_labels <- integer(0)
  dl <- torch::dataloader(ds, batch_size = batch, shuffle = FALSE)

  coro::loop(for (b in dl) {
    labels_batch <- as.integer(b[[2]]$to(device = "cpu"))
    all_labels <- c(all_labels, labels_batch)
  })
  all_labels
}

#' @export
#' @title Extract CIFAR10 Labels
#' @description Extracts integer labels from a CIFAR10 dataset in chunks.
#' @param ds A CIFAR10 dataset object (e.g., created by `cifar10_ds()`).
#' @param batch The batch size to use for iterating through the dataset.
#' @return An integer vector of labels.
#' @examples
#' \dontrun{
#' cifar10_train_ds <- cifar10_ds(train = TRUE)
#' cifar10_train_labels <- cifar10_labels(cifar10_train_ds)
#' head(cifar10_train_labels)
#' }
cifar10_labels <- function(ds, batch = 2000) {
  all_labels <- integer(0)
  dl <- torch::dataloader(ds, batch_size = batch, shuffle = FALSE)

  coro::loop(for (b in dl) {
    labels_batch <- as.integer(b[[2]]$to(device = "cpu"))
    all_labels <- c(all_labels, labels_batch)
  })
  all_labels
}
```
