prompt = """
You are not allowed to write files or run shell commands. Your job is to print code only.

Create R/partitions.R with roxygen2-documented partitioners that match the FedAvg paper setups.

1) #' @export
iid_split <- function(n_items, K, seed = 2025) { ... }
- Randomly partition indices 1:n_items into K nearly-equal, disjoint subsets.
- Deterministic with seed.

2) #' @export
mnist_shards_split <- function(labels, K = 100, shards_per_client = 2, seed = 2025) { ... }
- Non-IID MNIST per common FedAvg recipe:
  * Sort all indices by label (0..9).
  * Cut into 200 shards of size 300 (assume length(labels) == 60000).
  * Randomly assign shards_per_client shards to each of K clients without overlap.
  * Return a list length K, each an integer index vector.

3) #' @export
cifar10_iid_split <- function(n_train = 50000, n_test = 10000, K = 100, seed = 2025) { ... }
- IID CIFAR-10 (train/test) per paper:
  * Permute 1:n_train and 1:n_test.
  * Give each client 500 train and 100 test indices (assume exact divisibility).
  * Return a list of K elements, each a list(train = ..., test = ...).

4) #' @export
sample_clients <- function(K, C = 0.1, seed = NULL) { ... }
- Returns the integer indices of selected clients (no replacement). If seed is provided, set it.

Implementation notes:
- Use input checks (lengths, positivity, divisibility where needed).
- Ensure deterministic behavior given the same seed.
- Keep code simple and in base R.

Constraints:
 - Output ONE fenced code block with full content of R/partitions.R.
 - Add roxygen2 headers including @examples that operate only on synthetic data (no downloads).
 - Do not attempt to write files, call tools, or run shell commands.
No extra prose.
"""
