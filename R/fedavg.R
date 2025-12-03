#' Federated Averaging Aggregation
#'
#' Aggregates client updates using weighted averaging.
#'
#' @param params_list A list of flattened parameter vectors from clients.
#' @param weights A numeric vector of weights (usually number of samples).
#' @return A single flattened parameter vector.
#' @export
fedavg <- function(params_list, weights) {
    if (length(params_list) != length(weights)) {
        stop("Length of params_list and weights must match")
    }

    total_weight <- sum(weights)
    if (total_weight == 0) {
        return(params_list[[1]] * 0)
    } # Should not happen

    # Weighted sum
    # Convert list of vectors to a matrix for efficient computation if memory allows,
    # or iterate. For MNIST (small model), matrix is fine.
    # params_list is list of vectors.

    # Efficient way:
    # weighted_sum <- Reduce(`+`, Map(`*`, params_list, weights))
    # result <- weighted_sum / total_weight

    # Slightly more readable loop:
    accum <- params_list[[1]] * weights[1]
    if (length(params_list) > 1) {
        for (i in 2:length(params_list)) {
            accum <- accum + (params_list[[i]] * weights[i])
        }
    }

    accum / total_weight
}
