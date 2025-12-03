#' Flatten Model Parameters
#'
#' Flattens all parameters of a torch module into a single numeric vector.
#'
#' @param model A torch module.
#' @return A numeric vector containing all parameters.
#' @export
flatten_params <- function(model) {
    params <- vector("list", length(model$parameters))
    for (i in seq_along(model$parameters)) {
        p <- model$parameters[[i]]
        # Detach, move to CPU, view as 1D, convert to R array, then numeric
        params[[i]] <- as.numeric(as.array(p$detach()$cpu()$view(-1)))
    }
    unname(do.call(c, params))
}

#' Unflatten Model Parameters
#'
#' Updates a torch module's parameters from a flat numeric vector.
#'
#' @param model A torch module.
#' @param vec A numeric vector containing the new parameters.
#' @return The updated model (invisibly).
#' @export
unflatten_params <- function(model, vec) {
    # Validate length
    total_params <- sum(sapply(model$parameters, function(p) prod(p$shape)))
    if (length(vec) != total_params) {
        stop(sprintf("Vector length %d does not match model parameter count %d", length(vec), total_params))
    }

    start <- 1
    for (p in model$parameters) {
        numel <- prod(p$shape)
        end <- start + numel - 1
        chunk <- vec[start:end]

        # Create tensor on same device/dtype
        tens <- torch::torch_tensor(chunk, dtype = p$dtype, device = p$device)$view(p$shape)

        # Update parameter in-place
        torch::with_no_grad({
            p$copy_(tens)
        })

        start <- end + 1
    }
    invisible(model)
}

#' Get or Set Model Parameters
#'
#' @param model A torch module.
#' @param params Optional. If provided, updates the model parameters.
#' @return If params is NULL, returns the flattened parameters. Otherwise returns the updated model.
#' @export
get_set_params <- function(model, params = NULL) {
    if (is.null(params)) {
        return(flatten_params(model))
    } else {
        return(unflatten_params(model, params))
    }
}
