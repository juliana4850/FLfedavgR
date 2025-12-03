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

#' MNIST Non-IID Shards Split
#'
#' Sorts data by label and partitions into shards.
#'
#' @param labels Vector of labels (numeric).
#' @param K Number of clients.
#' @param shards_per_client Number of shards per client.
#' @param seed Random seed.
#' @return A list of K integer vectors (indices).
#' @export
mnist_shards_split <- function(labels, K = 100, shards_per_client = 2, seed = NULL) {
    if (!is.null(seed)) set.seed(seed)

    # Sort indices by label
    # order() returns indices that sort the vector
    sorted_indices <- order(labels)

    n_shards <- K * shards_per_client
    n_items <- length(labels)
    shard_size <- floor(n_items / n_shards)

    # Create shards
    shards <- list()
    for (i in 1:n_shards) {
        start <- (i - 1) * shard_size + 1
        end <- start + shard_size - 1
        if (i == n_shards) end <- n_items # Ensure all items are used
        shards[[i]] <- sorted_indices[start:end]
    }

    # Assign shards to clients
    shard_indices <- sample(n_shards)
    client_shards <- split(shard_indices, cut(seq_along(shard_indices), K, labels = FALSE))

    clients <- lapply(client_shards, function(s_idx) {
        unname(unlist(lapply(s_idx, function(i) shards[[i]])))
    })

    names(clients) <- NULL
    clients
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
