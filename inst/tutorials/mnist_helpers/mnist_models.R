#' MNIST CNN Model
#'
#' Creates a CNN model for MNIST matching McMahan et al. (2017) architecture:
#' - Conv1: 1→32 channels, 5×5 kernel, stride 1, no padding
#' - MaxPool1: 2×2 kernel, stride 2
#' - Conv2: 32→64 channels, 5×5 kernel, stride 1, no padding
#' - MaxPool2: 2×2 kernel, stride 2
#' Convolutional Neural Network for MNIST as described in McMahan et al. (2017).
#' Architecture mirrored from Flower reference implementation.
#'
#' @return A torch nn_module for the CNN model.
#' @export
#' @references
#' Mirrored from reference/fedavg_mnist_flwr/model.py: Net class
#' McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks
#' from Decentralized Data" https://arxiv.org/pdf/1602.05629.pdf
mnist_cnn <- torch::nn_module(
    "mnist_cnn",
    initialize = function() {
        # Convolutional layers with padding=1 (matches Flower reference)
        # reference/fedavg_mnist_flwr/model.py:21-23
        self$conv1 <- torch::nn_conv2d(
            in_channels = 1,
            out_channels = 32,
            kernel_size = 5,
            padding = 1 # Added to match Flower
        )

        self$conv2 <- torch::nn_conv2d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 5,
            padding = 1 # Added to match Flower
        )

        # Max pooling with padding=1 (matches Flower reference)
        # reference/fedavg_mnist_flwr/model.py:23
        self$pool <- torch::nn_max_pool2d(
            kernel_size = c(2, 2),
            padding = 1 # Added to match Flower
        )

        # Fully connected layers
        # With padding, feature maps are 7×7 (not 4×4)
        # reference/fedavg_mnist_flwr/model.py:24-25
        self$fc1 <- torch::nn_linear(
            in_features = 64 * 7 * 7, # Changed from 64*4*4
            out_features = 512
        )

        self$fc2 <- torch::nn_linear(
            in_features = 512,
            out_features = 10
        )
    },
    forward = function(x) {
        # Forward pass matching Flower reference
        # reference/fedavg_mnist_flwr/model.py:40-46

        # Conv1 -> ReLU -> Pool
        x <- self$conv1(x)
        x <- torch::nnf_relu(x)
        x <- self$pool(x)

        # Conv2 -> ReLU -> Pool
        x <- self$conv2(x)
        x <- torch::nnf_relu(x)
        x <- self$pool(x)

        # Flatten
        x <- torch::torch_flatten(x, start_dim = 2)

        # FC1 -> ReLU
        x <- self$fc1(x)
        x <- torch::nnf_relu(x)

        # FC2 (logits)
        x <- self$fc2(x)

        return(x)
    }
)
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
