```R
#' Federated Averaging (FedAvg)
#'
#' @description
#' Aggregates client model parameters using the Federated Averaging algorithm
#' as described by McMahan et al. (2017). The aggregation is a weighted
#' average of the client parameters, where the weights are typically the
#' number of samples on each client.
#'
#' The formula is: `w_{t+1} = sum_k (n_k / sum_j n_j) * w_{t+1}^k`
#'
#' @param params A list of numeric vectors. Each vector represents the model
#'   parameters from a single client.
#' @param weights A numeric vector of non-negative weights (e.g., number of
#'   samples `n_k`) corresponding to each client in `params`.
#'
#' @return A numeric vector representing the aggregated model parameters (`w_{t+1}`).
#'
#' @references
#' McMahan, B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
#' Communication-Efficient Learning of Deep Networks from Decentralized Data.
#' *Proceedings of the 20th International Conference on Artificial Intelligence
#' and Statistics (AISTATS)*.
#'
#' @export
#'
#' @examples
#' # Example with two clients
#' client1_params <- c(0.1, 0.2, 0.3)
#' client2_params <- c(0.4, 0.5, 0.6)
#' client_params <- list(client1_params, client2_params)
#'
#' # Weights are the number of data samples on each client
#' client_samples <- c(100, 200)
#'
#' aggregated_params <- fedavg(client_params, client_samples)
#' print(aggregated_params)
#' # Expected: (100 * c(0.1, 0.2, 0.3) + 200 * c(0.4, 0.5, 0.6)) / 300
#'
fedavg <- function(params, weights) {
  # --- Input Validation ---

  # Validate `params`
  if (!is.list(params)) {
    stop("Input 'params' must be a list of numeric vectors.")
  }
  if (length(params) == 0) {
    stop("Input 'params' list cannot be empty.")
  }
  if (!all(sapply(params, is.numeric))) {
    stop("All elements in 'params' must be numeric vectors.")
  }
  param_lengths <- sapply(params, length)
  if (length(unique(param_lengths)) > 1) {
    stop("All parameter vectors in 'params' must have the same length.")
  }
  if (any(sapply(params, function(p) any(!is.finite(p))))) {
    stop("Parameter vectors cannot contain NA, NaN, Inf, or -Inf values.")
  }

  # Validate `weights`
  if (!is.numeric(weights)) {
    stop("Input 'weights' must be a numeric vector.")
  }
  if (length(params) != length(weights)) {
    stop("Length of 'params' and 'weights' must be equal.")
  }
  if (any(is.na(weights))) {
    stop("Weights cannot contain NA values.")
  }
  if (any(weights < 0)) {
    stop("Weights must be non-negative.")
  }
  sum_weights <- sum(weights)
  if (sum_weights <= 0) {
    stop("The sum of weights must be greater than zero.")
  }

  # --- Aggregation Logic ---

  # Normalize weights
  normalized_weights <- weights / sum_weights

  # Multiply each parameter vector by its normalized weight
  weighted_params <- mapply("*", params, normalized_weights, SIMPLIFY = FALSE)

  # Sum the weighted parameter vectors
  aggregated_params <- Reduce("+", weighted_params)

  return(aggregated_params)
}
```

```R
# tests/testthat/test-fedavg.R

library(testthat)

context("fedavg aggregation logic and input validation")

test_that("1) weighted mean is calculated correctly", {
  params <- list(c(1, 2, 3), c(4, 5, 6))
  weights <- c(1, 1)
  expected <- c(2.5, 3.5, 4.5)
  expect_equal(fedavg(params, weights), expected)

  weights_uneven <- c(1, 3)
  # Manual calculation: (1 * c(1,2,3) + 3 * c(4,5,6)) / 4
  # = (c(1,2,3) + c(12,15,18)) / 4 = c(13,17,21) / 4
  expected_uneven <- c(13 / 4, 17 / 4, 21 / 4)
  expect_equal(fedavg(params, weights_uneven), expected_uneven)
})

test_that("2) function is invariant to client order", {
  params1 <- list(c(10, 20), c(30, 40), c(50, 60))
  weights1 <- c(5, 10, 15)

  # Permute the order
  perm_indices <- c(3, 1, 2)
  params2 <- params1[perm_indices]
  weights2 <- weights1[perm_indices]

  expect_equal(fedavg(params1, weights1), fedavg(params2, weights2))
})

test_that("3) function is deterministic", {
  params <- list(rnorm(10), rnorm(10))
  weights <- c(10, 20)
  # No set.seed needed as the function itself has no randomness.
  # The test confirms this by checking for identical output on repeated calls.
  result1 <- fedavg(params, weights)
  result2 <- fedavg(params, weights)
  expect_identical(result1, result2)
})

test_that("4) errors on invalid 'params' input", {
  # Unequal param lengths
  params_unequal <- list(c(1, 2, 3), c(4, 5))
  weights <- c(1, 1)
  expect_error(
    fedavg(params_unequal, weights),
    "All parameter vectors in 'params' must have the same length."
  )

  # NA in params
  params_na <- list(c(1, 2, NA), c(4, 5, 6))
  expect_error(
    fedavg(params_na, weights),
    "Parameter vectors cannot contain NA, NaN, Inf, or -Inf values."
  )

  # Not a list
  expect_error(
    fedavg(c(1, 2, 3), weights),
    "Input 'params' must be a list of numeric vectors."
  )

  # Empty list
  expect_error(
    fedavg(list(), c()),
    "Input 'params' list cannot be empty."
  )

  # Non-numeric list element
  params_char <- list(c("a", "b"), c(1, 2))
  expect_error(
    fedavg(params_char, weights),
    "All elements in 'params' must be numeric vectors."
  )
})

test_that("5) errors on invalid 'weights' input", {
  params <- list(c(1, 2), c(3, 4))

  # Negative weights
  weights_neg <- c(10, -5)
  expect_error(
    fedavg(params, weights_neg),
    "Weights must be non-negative."
  )

  # Zero-sum weights
  weights_zero_sum <- c(0, 0)
  expect_error(
    fedavg(params, weights_zero_sum),
    "The sum of weights must be greater than zero."
  )

  # NA in weights
  weights_na <- c(10, NA)
  expect_error(
    fedavg(params, weights_na),
    "Weights cannot contain NA values."
  )

  # Length mismatch
  weights_mismatch <- c(10)
  expect_error(
    fedavg(params, weights_mismatch),
    "Length of 'params' and 'weights' must be equal."
  )

  # Not numeric
  weights_char <- c("10", "20")
  expect_error(
    fedavg(params, weights_char),
    "Input 'weights' must be a numeric vector."
  )
})
```
