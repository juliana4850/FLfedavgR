prompt = """
You are not allowed to write files or run shell commands. Your job is to print code only.

Create R file R/data_loaders.R using the R torch ecosystem.

Requirements:
- Use libraries: torch, torchvision.
- Export helpers:
  #' @export
  mnist_ds <- function(root = "data", train = TRUE) { ... }
  #' @export
  cifar10_ds <- function(root = "data", train = TRUE) { ... }
- Each helper:
  * dir.create(root, recursive = TRUE, showWarnings = FALSE)
  * Returns torchvision::mnist_dataset(...) or torchvision::cifar10_dataset(...)
    with download = TRUE on first call.
- Also export two utilities for labels:
  #' @export
  mnist_labels <- function(ds, batch = 2000) { ... }  # returns integer labels for all items
  #' @export
  cifar10_labels <- function(ds, batch = 2000) { ... }
  Iterate in chunks so we never hold the whole dataset in RAM.

Constraints:
 - Output ONE fenced code block with the full content of R/data_loaders.R.
 - Add roxygen2 headers (@export, @examples that only construct the dataset and call length()).
 - Do not attempt to write files, call tools, or run shell commands.
No extra prose.
"""
