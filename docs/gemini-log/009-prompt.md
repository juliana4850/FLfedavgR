prompt = """
You are not allowed to write files or run shell commands. Your job is to print code only.

Goal is to create R/parameters.R file

Create R utilities for torch models with roxygen2 documentation:
- Exported functions:
  #' @export
  flatten_params(model): returns a single numeric vector containing all parameters
    in the order of model$parameters.
  #' @export
  unflatten_params(model, vec): writes values from vec back into the model parameters
    in the same order and original shapes (and same dtype/device).
  #' @export
  get_set_params(model, params = NULL): if params is NULL, return flattened params;
    otherwise set params into the model and return invisible(model).

Constraints:
- Output ONE fenced code block containing the full content of the R/parameters.R file.
- Use explicit namespaces (torch::) for all torch calls.
- Convert tensors to R numeric via as.numeric(as.array(...)) when flattening.
- Preserve dtype and device when reconstructing tensors in unflatten_params().
- Validate that length(vec) equals the total parameter size; stop() with a clear message otherwise.
- Do not use @import or library(); keep code self-contained.
- Return numeric vectors without names (use unname()).
- Include concise roxygen2 headers with @examples (\\dontrun).
- Do not attempt to write files, call tools, or run shell commands.
No extra prose.
"""
