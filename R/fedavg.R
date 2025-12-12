#' Federated Averaging (FedAvg)
#'
#' @description
#' Aggregates client model parameters using the Federated Averaging algorithm
#' as described by McMahan et al. (2017). The aggregation is a weighted
#' average of the client parameters, where the weights are typically the
#' number of samples on each client.
#'
#' The formula is: \eqn{w_{t+1} = \sum_k (n_k / \sum_j n_j)\, w^{(k)}_{t+1}}
#'
#' @param params A list of numeric vectors of equal length. Each vector represents the model
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
fedavg <- function(params, weights) {
    # --- Input Validation ---
    if (!is.list(params)) stop("Input 'params' must be a list of numeric vectors.")
    if (length(params) == 0L) stop("Input 'params' list cannot be empty.")
    if (!all(vapply(params, is.numeric, logical(1L)))) {
        stop("All elements in 'params' must be numeric vectors.")
    }
    param_lengths <- vapply(params, length, integer(1L))
    if (length(unique(param_lengths)) > 1L) {
        stop("All parameter vectors in 'params' must have the same length.")
    }
    if (any(vapply(params, function(p) any(!is.finite(p)), logical(1L)))) {
        stop("Parameter vectors cannot contain NA, NaN, Inf, or -Inf values.")
    }
    if (!is.numeric(weights)) stop("Input 'weights' must be a numeric vector.")
    if (length(weights) != length(params)) {
        stop("Length of 'params' and 'weights' must be equal.")
    }
    if (any(is.na(weights))) stop("Weights cannot contain NA values.")
    if (any(weights < 0)) stop("Weights must be non-negative.")

    weights <- as.numeric(weights)
    sum_weights <- sum(weights)
    if (sum_weights <= 0) {
        stop("The sum of weights must be greater than zero.")
    }

    # --- Aggregation Logic ---
    if (length(params) <= 32L) { # Use BLAS for up to ~32 clients
        # Fast path (BLAS)
        # Convert the list of parameter vectors into a matrix where each column
        # represents a client's parameters.
        # Perform the weighted average using matrix multiplication.
        # The result is a single-column matrix, which we drop to a vector.
        param_matrix <- do.call(cbind, params)
        normalized <- weights / sum_weights
        return(unname(drop(param_matrix %*% normalized)))
    } else {
        # Streaming (low-memory) path
        accum <- params[[1]] * weights[1]
        for (i in 2:length(params)) {
            accum <- accum + (params[[i]] * weights[i])
        }
        return(unname(as.numeric(accum / sum_weights)))
    }
}
