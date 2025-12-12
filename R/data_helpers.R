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
