#' Rounds to Target with Linear Interpolation
#'
#' Returns the fractional round where test accuracy reaches target using linear interpolation.
#'
#' @param history Data frame with round and test_acc columns.
#' @param target Target accuracy threshold.
#' @return Fractional round reaching target (via interpolation), or NA if not reached.
#' @export
rounds_to_target <- function(history, target = 0.97) {
    if (nrow(history) == 0) {
        return(NA_real_)
    }

    # Find first round where accuracy >= target
    idx <- which(history$test_acc >= target)

    if (length(idx) == 0) {
        return(NA_real_) # Never reached
    }

    first_idx <- idx[1]

    if (first_idx == 1) {
        # Reached in first round
        return(as.numeric(history$round[1]))
    }

    # Linear interpolation between round[first_idx-1] and round[first_idx]
    r_prev <- history$round[first_idx - 1]
    r_curr <- history$round[first_idx]
    acc_prev <- history$test_acc[first_idx - 1]
    acc_curr <- history$test_acc[first_idx]

    # Interpolate: r* = r_prev + (target - acc_prev) * (r_curr - r_prev) / (acc_curr - acc_prev)
    r_star <- r_prev + (target - acc_prev) * (r_curr - r_prev) / (acc_curr - acc_prev)

    return(r_star)
}
