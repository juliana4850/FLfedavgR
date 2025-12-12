#!/usr/bin/env Rscript
# Paper Reproduction: Figure 2 & Table 2
# McMahan et al. (2017) - MNIST CNN Experiments
#
# Reproduces data for:
# - Table 2: Rounds-to-target (99% accuracy) with speedups
# - Figure 2: Two-panel accuracy plots (IID | Non-IID)

cat("\n", strrep("=", 70), "\n")
cat("PAPER REPRODUCTION: Figure 2 & Table 2\n")
cat("McMahan et al. (2017) - MNIST CNN\n")
cat(strrep("=", 70), "\n\n")
cat(sprintf("Start time: %s\n\n", Sys.time()))

# ============================================================================
# SETUP
# ============================================================================

# Check/install required packages
required_pkgs <- c("torch", "torchvision", "ggplot2", "dplyr", "tidyr", "readr")
for (pkg in required_pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("Installing %s...\n", pkg))
        install.packages(pkg)
        if (pkg == "torch") torch::install_torch()
    }
}

# Load package
devtools::load_all()

# Load MNIST-specific helpers (examples of using the generic framework)
source("inst/tutorials/mnist_helpers/mnist_data.R")
source("inst/tutorials/mnist_helpers/mnist_models.R")
source("inst/tutorials/mnist_helpers/mnist_training.R")
source("inst/tutorials/mnist_helpers/mnist_fedavg.R")
source("inst/tutorials/mnist_helpers/mnist_partitions.R")
source("inst/tutorials/mnist_helpers/mnist_logging.R")
source("inst/tutorials/mnist_helpers/mnist_plotting.R")

library(torch)
library(torchvision)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# Set seeds for reproducibility
set.seed(2025)
torch::torch_manual_seed(2025)

# Configuration
ROUNDS <- if (Sys.getenv("FEDAVGR_TEST", "0") == "1") {
    5 # Test mode: quick verification
} else {
    1000 # Full mode
}
TARGET <- 0.99
C <- 0.1
K <- 100
LR_GRID <- c(0.03, 0.05, 0.1)

cat("Configuration:\n")
cat(sprintf(
    "  Rounds: %d %s\n", ROUNDS,
    if (Sys.getenv("FEDAVGR_QUICK", "0") == "1") "(QUICK MODE)" else "(FULL MODE)"
))
cat(sprintf("  Target accuracy: %.2f%%\n", TARGET * 100))
cat(sprintf("  Client fraction (C): %.2f\n", C))
cat(sprintf("  Clients (K): %d\n", K))
cat(sprintf("  LR grid: %s\n\n", paste(LR_GRID, collapse = ", ")))

# Define experiment grid (Table 2 configurations)
experiments_full <- data.frame(
    config_id = 1:9,
    E = c(1, 5, 1, 20, 1, 5, 20, 5, 20),
    B = c(Inf, Inf, 50, Inf, 10, 50, 50, 10, 10),
    u = c(1, 5, 12, 20, 60, 60, 240, 300, 1200),
    method = c("FedSGD", rep("FedAvg", 8)),
    stringsAsFactors = FALSE
)

# Quick mode: reduced set (E ∈ {1,5}, B ∈ {10,Inf})
experiments_quick <- data.frame(
    config_id = c(1, 2, 5, 8),
    E = c(1, 5, 1, 5),
    B = c(Inf, Inf, 10, 10),
    u = c(1, 5, 60, 300),
    method = c("FedSGD", rep("FedAvg", 3)),
    stringsAsFactors = FALSE
)

# Select experiment set based on mode
experiments <- if (Sys.getenv("FEDAVGR_QUICK", "0") == "1") experiments_quick else experiments_full

cat(sprintf("Experiment Grid (%d configurations):\n", nrow(experiments)))
print(experiments)
cat(sprintf(
    "\nTotal experiments: %d configs × 2 partitions = %d experiments\n",
    nrow(experiments), nrow(experiments) * 2
))

# ============================================================================
# DATA LOADING
# ============================================================================

cat("Loading MNIST datasets...\n")
ds_train <- mnist_ds(root = "data", train = TRUE, download = TRUE)
ds_test <- mnist_ds(root = "data", train = FALSE, download = TRUE)
labels_train <- mnist_labels(ds_train)

cat(sprintf("  Training samples: %d\n", length(ds_train)))
cat(sprintf("  Test samples: %d\n\n", length(ds_test)))

# Create partitions
cat("Creating partitions...\n")
iid_parts <- iid_split(length(labels_train), K, seed = 2025)
noniid_parts <- mnist_shards_split(labels_train, K = K, shards_per_client = 2, seed = 2025)
cat("  IID and Non-IID partitions created\n\n")

# Ensure output directories exist
dir.create("docs/examples", recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# RUN EXPERIMENTS
# ============================================================================

cat(strrep("=", 70), "\n")
cat("RUNNING EXPERIMENTS\n")
cat(strrep("=", 70), "\n\n")

# Storage for results
all_results <- list()
result_idx <- 1

# Run all experiments
for (i in 1:nrow(experiments)) {
    exp <- experiments[i, ]

    for (partition_type in c("IID", "nonIID")) {
        cat(sprintf(
            "\n--- Config %d/%d: %s, E=%d, B=%s, Partition=%s ---\n",
            i, nrow(experiments), exp$method, exp$E,
            if (is.infinite(exp$B)) "Inf" else as.character(exp$B),
            partition_type
        ))

        # Select partition
        parts <- if (partition_type == "IID") iid_parts else noniid_parts

        # Run experiment
        result <- tryCatch(
            {
                run_fedavg_mnist(
                    ds_train = ds_train,
                    ds_test = ds_test,
                    labels_train = labels_train,
                    model_fn = "cnn",
                    partition = partition_type,
                    K = K,
                    C = C,
                    E = exp$E,
                    batch_size = exp$B,
                    lr_grid = LR_GRID,
                    target = TARGET,
                    rounds = ROUNDS,
                    seed = 2025 + i * 100 + if (partition_type == "IID") 0 else 50,
                    device = "cpu",
                    fedsgd = (exp$method == "FedSGD"),
                    log_file = "inst/reproduction_outputs/metrics_mnist_cnn.csv"
                )
            },
            error = function(e) {
                cat(sprintf("ERROR: %s\n", e$message))
                NULL
            }
        )

        if (!is.null(result)) {
            # Add experiment metadata
            result$history$config_id <- exp$config_id
            result$history$u <- exp$u

            # Store result
            all_results[[result_idx]] <- result$history
            result_idx <- result_idx + 1

            # Incremental logging is now handled by run_fedavg_mnist

            # Report rounds-to-target
            rtt <- result$history$rtt[nrow(result$history)]
            if (!is.na(rtt)) {
                cat(sprintf("  Rounds-to-target (%.0f%%): %.1f\n", TARGET * 100, rtt))
            } else {
                cat(sprintf("  Target %.0f%% not reached in %d rounds\n", TARGET * 100, ROUNDS))
            }
        }
    }
}

cat("\n", strrep("=", 70), "\n")
cat("EXPERIMENTS COMPLETE\n")
cat(strrep("=", 70), "\n\n")
cat(sprintf("End time: %s\n", Sys.time()))
cat("\nOutputs:\n")
cat("  - Raw data: inst/reproduction_outputs/metrics_mnist_cnn.csv\n\n")

invisible(0)
