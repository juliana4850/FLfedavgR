```R
#' @title Flatten model parameters into a single vector
#'
#' @description Extracts all parameters from a torch model and returns them as a
#' single numeric vector. The order of parameters is preserved from `model$parameters`.
#'
#' @param model A torch nn_module object.
#'
#' @return A numeric vector containing all model parameters, without names.
#' @export
#'
#' @examples
#' \dontrun{
#' net <- torch::nn_linear(10, 1)
#' params_vec <- flatten_params(net)
#' length(params_vec) # Should be 11 (10 weights + 1 bias)
#' }
flatten_params <- function(model) {
  params_list <- lapply(model$parameters, function(p) {
    as.numeric(as.array(p$cpu()))
  })
  unname(do.call(c, params_list))
}

#' @title Unflatten a vector of parameters into a model
#'
#' @description Updates a model's parameters in-place using values from a
#' numeric vector. The function expects the vector to have the exact number of
#' elements required to fill all model parameters. It preserves the original
#' shape, dtype, and device of each parameter tensor.
#'
#' @param model A torch nn_module object to be updated.
#' @param vec A numeric vector containing the new parameter values.
#'
#' @return The model with updated parameters (invisibly).
#' @export
#'
#' @examples
#' \dontrun{
#' net <- torch::nn_linear(10, 1)
#' original_params <- flatten_params(net)
#' new_params_vec <- runif(length(original_params))
#'
#' unflatten_params(net, new_params_vec)
#'
#' updated_params <- flatten_params(net)
#' all.equal(new_params_vec, updated_params) # Should be TRUE
#' }
unflatten_params <- function(model, vec) {
  total_params <- sum(sapply(model$parameters, function(p) p$numel()))
  if (length(vec) != total_params) {
    stop(
      paste0(
        "Incorrect number of parameters provided. Expected ",
        total_params, ", but got ", length(vec), "."
      )
    )
  }

  offset <- 1
  for (p in model$parameters) {
    numel <- p$numel()
    sub_vec <- vec[offset:(offset + numel - 1)]

    # Create new tensor with original properties and new data
    new_tensor <- torch::torch_tensor(
      sub_vec,
      dtype = p$dtype,
      device = p$device
    )$view(p$shape)

    # Update the parameter data in-place
    p$data()$set_(new_tensor)

    offset <- offset + numel
  }
  invisible(model)
}

#' @title Get or set model parameters using a flattened vector
#'
#' @description A convenience function that acts as a getter or setter for model
#' parameters. If `params` is `NULL`, it returns the model's flattened parameters.
#' Otherwise, it sets the model's parameters from the provided flattened vector.
#'
#' @param model A torch nn_module object.
#' @param params (Optional) A numeric vector of parameters to set on the model.
#'
#' @return If `params` is `NULL`, returns a numeric vector of the model's
#'   flattened parameters. If `params` is provided, returns the modified model
#'   invisibly.
#' @export
#'
#' @examples
#' \dontrun{
#' net <- torch::nn_linear(5, 1)
#'
#' # Get parameters
#' params <- get_set_params(net)
#' print(params)
#'
#' # Set parameters
#' new_params <- rep(0, length(params))
#' get_set_params(net, new_params)
#'
#' # Verify they were set
#' all.equal(get_set_params(net), new_params) # Should be TRUE
#' }
get_set_params <- function(model, params = NULL) {
  if (is.null(params)) {
    flatten_params(model)
  } else {
    unflatten_params(model, params)
    invisible(model)
  }
}
```
