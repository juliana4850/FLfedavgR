```r
# R/partitions.R

#' Create a random IID partition of items
#'
#' Randomly partitions a set of indices into K nearly-equal, disjoint subsets.
#' The function is deterministic given a specific seed.
#'
#' @param n_items The total number of items to partition (integer > 0).
#' @param K The number of partitions (clients) to create (integer > 0).
#' @param seed A random seed for reproducibility.
#'
#' @return A list of length K, where each element is an integer vector of indices.
#' @export
#'
#' @examples
#' # Partition 100 items among 10 clients
#' partitions <- iid_split(n_items = 100, K = 10)
#' print(lengths(partitions))
#' print(partitions[[1]])
#'
#' # Partition 103 items among 5 clients (uneven split)
#' partitions_uneven <- iid_split(n_items = 103, K = 5)
#' print(lengths(partitions_uneven))
#' all_indices <- sort(unlist(partitions_uneven))
#' identical(all_indices, 1:103)
iid_split <- function(n_items, K, seed = 2025) {
  if (!is.numeric(n_items) || n_items <= 0 || n_items != floor(n_items)) {
    stop("n_items must be a positive integer.")
  }
  if (!is.numeric(K) || K <= 0 || K != floor(K)) {
    stop("K must be a positive integer.")
  }
  if (K > n_items) {
    stop("K cannot be greater than n_items.")
  }

  set.seed(seed)
  indices <- sample(seq_len(n_items))
  
  # Create a grouping factor to split indices into K groups
  grouping <- as.factor(cut(seq_along(indices), breaks = K, labels = FALSE))
  
  split(indices, grouping)
}

#' Create a non-IID partition of MNIST data using shards
#'
#' Implements the common non-IID data partitioning strategy for MNIST as
#' described in the FedAvg paper (McMahan et al., 2017). It first sorts the
#' data by digit label, divides it into a number of shards, and then assigns
#' a fixed number of shards to each client.
#'
#' @param labels A vector of data labels. Assumed to be 60,000 for MNIST training.
#' @param K The total number of clients (integer > 0). Default is 100.
#' @param shards_per_client The number of shards to assign to each client. Default is 2.
#' @param seed A random seed for reproducibility.
#'
#' @return A list of length K, where each element is an integer vector of indices
#'   corresponding to the data for one client.
#' @export
#'
#' @examples
#' # Simulate with 6000 labels instead of 60000 for speed
#' n_labels <- 6000
#' n_shards <- 20
#' shard_size <- n_labels / n_shards
#' K <- 10
#' shards_per_client <- 2
#'
#' # Create synthetic labels sorted like MNIST (0,0,...,1,1,...,9,9,...)
#' synthetic_labels <- rep(0:9, each = n_labels / 10)
#'
#' client_indices <- mnist_shards_split(
#'   labels = synthetic_labels,
#'   K = K,
#'   shards_per_client = shards_per_client,
#'   seed = 2025
#' )
#'
#' # Check output structure
#' length(client_indices) == K
#' # Each client gets shard_size * shards_per_client indices
#' all(lengths(client_indices) == shard_size * shards_per_client)
#' # Check that indices are disjoint
#' length(unique(unlist(client_indices))) == n_labels
mnist_shards_split <- function(labels, K = 100, shards_per_client = 2, seed = 2025) {
  n_items <- length(labels)
  if (n_items != 60000) {
    warning("mnist_shards_split is designed for 60,000 MNIST samples but proceeding anyway.")
  }
  
  n_shards <- 200
  shard_size <- n_items / n_shards
  
  if (shard_size != floor(shard_size)) {
    stop("Number of items must be divisible by the number of shards (200).")
  }
  if (K * shards_per_client > n_shards) {
    stop("Not enough shards for the given number of clients and shards_per_client.")
  }

  set.seed(seed)

  # 1. Sort indices by label
  idx_sorted <- order(labels)

  # 2. Partition sorted indices into shards
  shards <- split(idx_sorted, rep(seq_len(n_shards), each = shard_size))

  # 3. Randomly assign shards to clients
  shard_indices <- sample(seq_len(n_shards))
  
  client_shards <- split(
    shard_indices,
    rep(seq_len(K), each = shards_per_client)
  )
  
  # 4. Map shard assignments back to data indices
  client_indices <- lapply(client_shards, function(s) {
    unlist(shards[s], use.names = FALSE)
  })

  return(client_indices)
}

#' Create an IID partition for CIFAR-10 training and testing sets
#'
#' Partitions the CIFAR-10 dataset into IID subsets for a specified number of
#' clients, providing distinct indices for training and testing sets. This
#' follows the setup where each client receives an equal fraction of the data.
#'
#' @param n_train Total number of training items. Default is 50,000.
#' @param n_test Total number of test items. Default is 10,000.
#' @param K The number of clients. Default is 100.
#' @param seed A random seed for reproducibility.
#'
#' @return A list of length K. Each element is a list containing two integer
#'   vectors: `train` and `test`, representing the indices for that client.
#' @export
#'
#' @examples
#' # Partition CIFAR-10 for 100 clients
#' partitions <- cifar10_iid_split(K = 100)
#' length(partitions)
#' names(partitions[[1]])
#' # Each client gets 500 train and 100 test indices
#' all(sapply(partitions, function(p) length(p$train)) == 500)
#' all(sapply(partitions, function(p) length(p$test)) == 100)
#'
#' # Check that train indices are unique across all clients
#' length(unique(unlist(lapply(partitions, `[[`, "train")))) == 50000
cifar10_iid_split <- function(n_train = 50000, n_test = 10000, K = 100, seed = 2025) {
  if (n_train %% K != 0) {
    stop("n_train must be divisible by K.")
  }
  if (n_test %% K != 0) {
    stop("n_test must be divisible by K.")
  }

  set.seed(seed)

  # Shuffle train and test indices
  train_indices <- sample(seq_len(n_train))
  test_indices <- sample(seq_len(n_test))

  # Calculate items per client
  train_per_client <- n_train / K
  test_per_client <- n_test / K

  # Split indices into K groups
  train_splits <- split(train_indices, rep(seq_len(K), each = train_per_client))
  test_splits <- split(test_indices, rep(seq_len(K), each = test_per_client))

  # Combine into the final list structure
  partitions <- lapply(seq_len(K), function(i) {
    list(
      train = train_splits[[i]],
      test = test_splits[[i]]
    )
  })

  return(partitions)
}

#' Sample a fraction of clients
#'
#' Randomly selects a subset of clients for a training round.
#'
#' @param K The total number of clients (integer > 0).
#' @param C The fraction of clients to sample (numeric, 0 < C <= 1). The
#'   number of clients sampled is `max(1, floor(C * K))`.
#' @param seed An optional random seed for reproducibility. If NULL, the
#'   random state is not reset.
#'
#' @return An integer vector of the selected client indices, sorted.
#' @export
#'
#' @examples
#' # Sample 10% of 100 clients
#' selected <- sample_clients(K = 100, C = 0.1, seed = 123)
#' print(selected)
#' length(selected) == 10
#'
#' # Sample 50% of 15 clients (results in 7 clients)
#' selected_2 <- sample_clients(K = 15, C = 0.5, seed = 123)
#' print(selected_2)
#' length(selected_2) == 7
#'
#' # Sampling at least 1 client even if fraction is small
#' selected_3 <- sample_clients(K = 100, C = 0.001)
#' length(selected_3) == 1
sample_clients <- function(K, C = 0.1, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  if (!is.numeric(K) || K <= 0 || K != floor(K)) {
    stop("K must be a positive integer.")
  }
  if (!is.numeric(C) || C <= 0 || C > 1) {
    stop("C must be a numeric value between 0 (exclusive) and 1 (inclusive).")
  }

  n_sample <- max(1, floor(C * K))
  
  sort(sample(seq_len(K), size = n_sample, replace = FALSE))
}
```
