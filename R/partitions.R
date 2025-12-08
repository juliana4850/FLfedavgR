#' IID Split
#'
#' Randomly partitions data into K clients.
#'
#' @param n_items Total number of items.
#' @param K Number of clients.
#' @param seed Random seed.
#' @return A list of K integer vectors (indices).
#' @export
iid_split <- function(n_items, K, seed = NULL) {
    if (!is.null(seed)) set.seed(seed)

    indices <- sample(n_items)
    split(indices, cut(seq_along(indices), K, labels = FALSE))
}

#' Sample Clients
#'
#' Selects a fraction of clients.
#'
#' @param K Total clients.
#' @param C Fraction of clients to select.
#' @param seed Random seed.
#' @return Sorted integer vector of selected client indices.
#' @export
sample_clients <- function(K, C, seed = NULL) {
    if (!is.null(seed)) set.seed(seed)

    m <- max(floor(C * K), 1)
    sort(sample(K, m))
}
