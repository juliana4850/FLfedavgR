#!/usr/bin/env Rscript
# ==============================================================================
# Test Generic FedAvg Framework with CIFAR-10
# ==============================================================================
#
# This script demonstrates how to use the generic fedavg_simulation() framework
# with CIFAR-10, matching the configuration from McMahan et al. (2017).
#
# Paper specifications:
# - 100 clients (IID partitioning)
# - 500 training + 100 testing examples per client
# - Model: 2 conv + 2 FC + linear (~1M parameters)
# - Preprocessing: 24x24 crop, random flip, contrast/brightness/whitening
# - Batch size: 50
# - Learning rate decay 0.99 per round
#
# ==============================================================================

library(torch)
library(torchvision)

# Load the fedavgR package
devtools::load_all()

cat("Testing Generic FedAvg Framework with CIFAR-10\n")
cat("(Matching McMahan et al. 2017 specifications)\n")
cat(strrep("=", 70), "\n\n")

# ==============================================================================
# 1. Define CIFAR-10 Dataset with Paper's Preprocessing
# ==============================================================================

cat("1. Loading CIFAR-10 dataset...\n")

# CIFAR-10 dataset with paper's preprocessing
cifar10_ds <- function(root = "data", train = TRUE, download = TRUE) {
    # Paper preprocessing: crop to 24x24, random flip, normalize
    transform <- function(x) {
        x <- transform_to_tensor(x)

        # Random crop to 24x24 (from 32x32)
        if (train) {
            x <- transform_random_crop(x, size = c(24, 24))
            # Random horizontal flip
            if (runif(1) > 0.5) {
                x <- transform_hflip(x)
            }
        } else {
            # Center crop for test
            x <- transform_center_crop(x, size = c(24, 24))
        }

        # Normalize (paper mentions contrast/brightness/whitening)
        # Using standard normalization as approximation
        x <- (x - 0.5) / 0.5
        x
    }

    cifar10_dataset(
        root = root,
        train = train,
        download = download,
        transform = transform
    )
}

# Load datasets
ds_train <- cifar10_ds(train = TRUE, download = TRUE)
ds_test <- cifar10_ds(train = FALSE, download = TRUE)

cat(sprintf("  Training samples: %d\n", length(ds_train)))
cat(sprintf("  Test samples: %d\n\n", length(ds_test)))

# ==============================================================================
# 2. Define CIFAR-10 CNN Model (Paper Architecture)
# ==============================================================================

cat("2. Defining CIFAR-10 CNN model (paper architecture)...\n")

# Paper: 2 conv layers + 2 FC layers + linear transformation (~1M params)
cifar10_cnn <- nn_module(
    "CIFAR10_CNN",
    initialize = function() {
        # Input: 3x24x24 (after cropping)
        # Conv layer 1
        self$conv1 <- nn_conv2d(3, 64, kernel_size = 5, padding = 2)
        self$pool1 <- nn_max_pool2d(kernel_size = 3, stride = 2)

        # Conv layer 2
        self$conv2 <- nn_conv2d(64, 64, kernel_size = 5, padding = 2)
        self$pool2 <- nn_max_pool2d(kernel_size = 3, stride = 2)

        # Fully connected layers
        self$fc1 <- nn_linear(64 * 5 * 5, 384)
        self$fc2 <- nn_linear(384, 192)
        self$fc3 <- nn_linear(192, 10) # 10 classes
    },
    forward = function(x) {
        x %>%
            self$conv1() %>%
            nnf_relu() %>%
            self$pool1() %>%
            self$conv2() %>%
            nnf_relu() %>%
            self$pool2() %>%
            torch_flatten(start_dim = 2) %>%
            self$fc1() %>%
            nnf_relu() %>%
            self$fc2() %>%
            nnf_relu() %>%
            self$fc3()
    }
)

cat("  Model architecture defined\n")

# Count parameters
model_temp <- cifar10_cnn()
n_params <- sum(sapply(model_temp$parameters, function(p) prod(p$shape)))
cat(sprintf("  Total parameters: %s\n\n", format(n_params, big.mark = ",")))

# ==============================================================================
# 3. Partition Data into 100 Clients (IID, Paper Specification)
# ==============================================================================

cat("3. Partitioning data into 100 clients (IID, paper spec)...\n")

K <- 100 # Number of clients (paper specification)
n_train <- length(ds_train)

# IID partitioning: randomly assign to clients
indices_per_client <- split(sample(n_train), rep(1:K, length.out = n_train))

# Create client datasets
client_datasets <- lapply(indices_per_client, function(idx) {
    dataset_subset(ds_train, idx)
})

cat(sprintf("  Created %d clients\n", K))
cat(sprintf(
    "  Samples per client: %d - %d\n",
    min(sapply(client_datasets, length)),
    max(sapply(client_datasets, length))
))
cat(sprintf("  (Paper: 500 training per client)\n\n"))

# ==============================================================================
# 4. Define Evaluation Function
# ==============================================================================

cat("4. Defining evaluation function...\n")

eval_cifar10 <- function(model, device) {
    model$eval()

    correct <- 0
    total <- 0

    dl <- dataloader(ds_test, batch_size = 100, shuffle = FALSE)

    coro::loop(for (batch in dl) {
        images <- batch[[1]]$to(device = device)
        labels <- batch[[2]]$to(device = device)

        with_no_grad({
            outputs <- model(images)
            predicted <- torch_argmax(outputs, dim = 2)
            total <- total + labels$size(1)
            correct <- correct + (predicted == labels)$sum()$item()
        })
    })

    accuracy <- correct / total
    list(accuracy = accuracy)
}

cat("  Evaluation function defined\n\n")

# ==============================================================================
# 5. Run FedAvg Simulation (Paper Configuration)
# ==============================================================================

# Configuration
LR <- as.numeric(Sys.getenv("LR", "0.15"))
ROUNDS <- as.integer(Sys.getenv("ROUNDS", "3000"))
LOG_FILE <- Sys.getenv("LOG_FILE", "inst/reproduction_outputs/metrics_cifar10.csv")
MODEL_FILE <- Sys.getenv("MODEL_FILE", "inst/reproduction_outputs/cifar10_final_model.pt")

cat("5. Running FedAvg simulation on CIFAR-10...\n")
cat(sprintf("   C=0.1 (10 clients/round), E=5, B=50, learning rate = %f\n", LR))
cat(sprintf("   Logging to: %s\n", LOG_FILE))
cat(sprintf("   Model will be saved to: %s\n", MODEL_FILE))
cat(strrep("-", 70), "\n")

# Learning rate decay schedule
lr_schedule <- function(round) {
    # Decay by 0.99 every round
    LR * (0.99^(round - 1)) # Default 0.15, specify via LR environment variable
}

result <- fedavg_simulation(
    client_datasets = client_datasets,
    model_generator = cifar10_cnn,
    evaluation_fn = eval_cifar10,
    rounds = ROUNDS, # Default 10, specify via ROUNDS environment variable
    C = 0.1, # Select 10% of clients per round (10 clients)
    E = 5, # 5 local epochs (paper experiments)
    batch_size = 50, # Paper uses B=50
    optimizer_generator = function(lr) function(params) optim_sgd(params, lr = lr, momentum = 0.9),
    lr_scheduler = lr_schedule, # Learning rate decay
    seed = 123,
    device = "cpu",
    log_file = LOG_FILE, # Incremental logging
    save_model = MODEL_FILE # Save final model
)

cat(strrep("-", 70), "\n\n")

# ==============================================================================
# 6. Display Results
# ==============================================================================

cat("6. Results:\n")
cat(strrep("=", 70), "\n")

print(result$history)

cat("\n")
cat("Final accuracy: ", sprintf(
    "%.4f (%.1f%%)",
    result$history$accuracy[nrow(result$history)],
    result$history$accuracy[nrow(result$history)] * 100
), "\n")
cat("\n")
cat("Paper baseline (standard SGD, 197,500 minibatch updates): 86%\n")
cat("Paper FedAvg (2,000 communication rounds): 85%\n")
cat("\n")
cat(sprintf("Note: This test used %d rounds. The paper ran 2,000 rounds\n", ROUNDS))
cat("to reach 85% accuracy. You can increase rounds for better results.\n")
cat("\n")
