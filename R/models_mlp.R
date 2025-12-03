#' MNIST MLP Model (2NN)
#'
#' Creates a 2-layer fully connected neural network for MNIST (784 -> 200 -> 200 -> 10).
#'
#' @return A torch module.
#' @export
mnist_mlp <- torch::nn_module(
    "mnist_mlp",
    initialize = function() {
        self$fc1 <- torch::nn_linear(784, 200)
        self$fc2 <- torch::nn_linear(200, 200)
        self$fc3 <- torch::nn_linear(200, 10)
    },
    forward = function(x) {
        x <- x$view(c(-1, 784))
        x <- torch::nnf_relu(self$fc1(x))
        x <- torch::nnf_relu(self$fc2(x))
        self$fc3(x)
    }
)
