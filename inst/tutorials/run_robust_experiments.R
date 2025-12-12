#!/usr/bin/env Rscript
# ==============================================================================
# Robust Experiment Runner for Paper Reproduction
# ==============================================================================
#
# PURPOSE:
# This script runs long-running FedAvg experiments in a production-safe manner
# by executing them in chunks with automatic checkpointing and resume capability.
#
# WHY USE THIS INSTEAD OF paper_reproduction_cnn.R?
#
# 1. MEMORY MANAGEMENT
#    - Runs experiments in chunks (default: 50 rounds per chunk)
#    - Restarts R process between chunks to guarantee memory cleanup
#    - Prevents OOM crashes in long experiments (1000+ rounds)
#
# 2. CRASH RECOVERY
#    - Automatically saves checkpoints after each chunk
#    - Resumes from last completed round if script crashes
#    - Safe for unattended overnight/weekend runs
#
# 3. PRODUCTION FEATURES
#    - Validates checkpoints before resuming
#    - Handles multiple experiment configurations in sequence
#    - Logs all output for monitoring
#    - Retry logic for transient failures
#
# WHEN TO USE:
# - Full paper reproduction (1000 rounds per experiment)
# - Unattended long-running experiments
# - Limited memory environments
# - Production/final runs for publication
#
# WHEN NOT TO USE:
# - Quick testing (use paper_reproduction_cnn.R with FEDAVGR_QUICK=1)
# - Small experiments (<100 rounds)
# - Debugging (direct script execution is easier)
#
# USAGE:
#
#   # Interactive (see progress)
#   Rscript inst/tutorials/run_robust_experiments.R
#
#   # Background with logging
#   nohup Rscript inst/tutorials/run_robust_experiments.R > robust.log 2>&1 &
#
#   # Monitor progress
#   tail -f robust.log
#   tail -f inst/reproduction_outputs/metrics_mnist.csv
#
# CONFIGURATION:
# Edit the 'experiments' list below to define which experiments to run.
# Each experiment specifies: E (epochs), B (batch size), partition, rounds
#
# ==============================================================================

# Configuration
chunk_size <- 50  # Rounds per chunk (smaller = more frequent memory cleanup)
checkpoint_base_dir <- "inst/reproduction_outputs/checkpoints"
log_file <- "inst/reproduction_outputs/metrics_mnist.csv"

# Ensure directories exist
if (!dir.exists(checkpoint_base_dir)) dir.create(checkpoint_base_dir, recursive = TRUE)
if (!dir.exists(dirname(log_file))) dir.create(dirname(log_file), recursive = TRUE)

# ==============================================================================
# Helper Functions
# ==============================================================================

#' Find Last Completed Round from CSV Log
#'
#' @param log_file Path to metrics CSV file
#' @param E_val Local epochs value
#' @param partition_val Partition type ("IID" or "nonIID")
#' @param B_val Batch size (numeric or Inf)
#' @return Integer: last completed round (0 if none)
get_last_round <- function(log_file, E_val, partition_val, B_val) {
    if (!file.exists(log_file)) {
        return(0)
    }

    tryCatch(
        {
            df <- read.csv(log_file)

            # Handle B column (might be "Inf" string or numeric)
            B_target <- if (is.infinite(B_val)) "Inf" else as.character(B_val)

            subset <- df[df$E == E_val &
                df$partition == partition_val &
                df$method == "FedAvg" &
                as.character(df$B) == B_target, ]

            if (nrow(subset) == 0) {
                return(0)
            }
            return(max(subset$round))
        },
        error = function(e) {
            cat(sprintf("Warning: Error reading log file: %s\n", e$message))
            return(0)
        }
    )
}

# ==============================================================================
# Experiment Definitions
# ==============================================================================

# Define experiments to run
# Format: list(E=..., B=..., partition=..., rounds=...)
#
# These match the configurations in McMahan et al. (2017) Table 2:
# - IID and Non-IID partitions
# - E = 1, 5, 20 (local epochs)
# - B = 10, 50, Inf (batch sizes)
# - 1000 rounds to reach 99% accuracy

experiments <- list(
    # Non-IID experiments (harder to converge)
    list(E = 1, B = 10, partition = "nonIID", rounds = 1000),
    list(E = 5, B = 10, partition = "nonIID", rounds = 1000),
    list(E = 20, B = 10, partition = "nonIID", rounds = 1000),

    # IID experiments (faster convergence)
    list(E = 1, B = 10, partition = "IID", rounds = 1000),
    list(E = 5, B = 10, partition = "IID", rounds = 1000),
    list(E = 20, B = 10, partition = "IID", rounds = 1000)
)

# ==============================================================================
# Main Execution Loop
# ==============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("ROBUST EXPERIMENT RUNNER\n")
cat(strrep("=", 80), "\n")
cat(sprintf("Chunk size: %d rounds\n", chunk_size))
cat(sprintf("Log file: %s\n", log_file))
cat(sprintf("Checkpoint dir: %s\n", checkpoint_base_dir))
cat(sprintf("Total experiments: %d\n", length(experiments)))
cat(strrep("=", 80), "\n\n")

for (exp_idx in seq_along(experiments)) {
    exp <- experiments[[exp_idx]]
    E_val <- exp$E
    B_val <- exp$B
    partition_val <- exp$partition
    total_rounds <- exp$rounds

    # Create unique checkpoint dir for each configuration
    ckpt_dir <- file.path(checkpoint_base_dir, paste0(partition_val, "_E", E_val, "_B", B_val))
    if (!dir.exists(ckpt_dir)) dir.create(ckpt_dir, recursive = TRUE)

    cat(sprintf("\n[%d/%d] Starting: Partition=%s, E=%d, B=%d, Rounds=%d\n",
        exp_idx, length(experiments), partition_val, E_val, B_val, total_rounds))
    cat(strrep("-", 80), "\n")

    # Determine start round from existing logs
    last_round <- get_last_round(log_file, E_val, partition_val, B_val)
    start_round <- last_round + 1

    if (start_round > total_rounds) {
        cat(sprintf("✓ Already completed (Round %d/%d). Skipping.\n", last_round, total_rounds))
        next
    }

    cat(sprintf("Resuming from Round %d (Last completed: %d)\n", start_round, last_round))

    # Validate checkpoint if resuming
    if (start_round > 1) {
        ckpt_path <- file.path(ckpt_dir, "checkpoint_latest.rds")
        if (!file.exists(ckpt_path)) {
            cat("⚠ WARNING: Checkpoint missing but CSV shows progress.\n")
            cat("   Cannot resume safely without model weights.\n")
            cat("   Resetting to Round 1 (will create duplicate entries in CSV).\n")
            start_round <- 1
        } else {
            cat(sprintf("✓ Checkpoint found: %s\n", ckpt_path))
        }
    }

    current_round <- start_round
    chunk_count <- 0

    # Execute in chunks
    while (current_round <= total_rounds) {
        end_round <- min(current_round + chunk_size - 1, total_rounds)
        chunk_count <- chunk_count + 1

        cat(sprintf("\n  Chunk %d: Rounds %d-%d\n", chunk_count, current_round, end_round))

        # Create temporary runner script for this chunk
        script_content <- sprintf('
      # Suppress startup messages
      suppressPackageStartupMessages({
          devtools::load_all(quiet = TRUE)
          source("inst/tutorials/mnist_helpers/mnist_data.R")
          source("inst/tutorials/mnist_helpers/mnist_partitions.R")
          source("inst/tutorials/mnist_helpers/mnist_models.R")
          source("inst/tutorials/mnist_helpers/mnist_training.R")
          source("inst/tutorials/mnist_helpers/mnist_fedavg.R")
          source("inst/tutorials/mnist_helpers/mnist_logging.R")
          source("inst/tutorials/mnist_helpers/mnist_plotting.R")
      })

      # Load data
      cat("Loading MNIST data...\\n")
      ds_train <- mnist_ds(train = TRUE, download = TRUE)
      ds_test <- mnist_ds(train = FALSE, download = TRUE)
      labels_train <- mnist_labels(ds_train)

      # Run chunk
      cat("Starting FedAvg chunk...\\n")
      run_fedavg_mnist(
          ds_train = ds_train,
          ds_test = ds_test,
          labels_train = labels_train,
          model_fn = "cnn",
          partition = "%s",
          K = 100,
          C = 0.1,
          E = %d,
          batch_size = %d,
          lr_grid = c(0.03, 0.05, 0.1),
          target = 0.99,
          rounds = %d,
          seed = 2025,
          device = "cpu",
          log_file = "%s",
          checkpoint_dir = "%s",
          start_round = %d
      )
      cat("Chunk completed successfully.\\n")
    ', partition_val, E_val, B_val, end_round, log_file, ckpt_dir, current_round)

        # Write and execute chunk script
        temp_script <- "temp_chunk_runner.R"
        writeLines(script_content, temp_script)

        res <- system(sprintf("Rscript %s", temp_script))

        # Check execution status
        if (res != 0) {
            cat(sprintf("✗ ERROR: Chunk execution failed (exit code: %d)\n", res))
            cat("  Retrying in 10 seconds...\n")
            Sys.sleep(10)

            # Retry once
            res <- system(sprintf("Rscript %s", temp_script))
            if (res != 0) {
                cat("✗ CRITICAL: Chunk failed after retry. Stopping.\n")
                stop("Critical failure in chunk execution. Check logs for details.")
            }
        }

        cat(sprintf("  ✓ Chunk %d completed\n", chunk_count))

        # Update progress
        current_round <- end_round + 1

        # Brief pause between chunks
        if (current_round <= total_rounds) {
            Sys.sleep(2)
        }
    }

    cat(sprintf("\n✓ Experiment completed: %s E=%d B=%d\n", partition_val, E_val, B_val))
}

# Cleanup
if (file.exists("temp_chunk_runner.R")) {
    file.remove("temp_chunk_runner.R")
}

cat("\n")
cat(strrep("=", 80), "\n")
cat("ALL EXPERIMENTS COMPLETED\n")
cat(strrep("=", 80), "\n")
cat(sprintf("Results saved to: %s\n", log_file))
cat(sprintf("Checkpoints saved to: %s\n", checkpoint_base_dir))
cat("\nNext steps:\n")
cat("  1. Generate plots: Rscript inst/tutorials/generate_figure2_from_logs.R\n")
cat("  2. Generate table: Rscript inst/tutorials/generate_table2_from_logs.R\n")
cat("\n")
