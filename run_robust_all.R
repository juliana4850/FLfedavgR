# Robust Runner for nonIID Experiments (Chunked Execution)
# This script runs the experiments in chunks (e.g., 50 rounds) and restarts the R process
# between chunks to guarantee memory cleanup and prevent OOM crashes.

chunk_size <- 50
checkpoint_base_dir <- "inst/reproduction_outputs/checkpoints"
log_file <- "inst/reproduction_outputs/metrics_mnist.csv"

# Ensure directories exist
if (!dir.exists(checkpoint_base_dir)) dir.create(checkpoint_base_dir, recursive = TRUE)

# Helper to find last completed round from CSV
get_last_round <- function(log_file, E_val, partition_val, B_val) {
    if (!file.exists(log_file)) {
        return(0)
    }

    tryCatch(
        {
            df <- read.csv(log_file)

            # Handle B column (might be "Inf" string or numeric)
            # Convert input B_val to string if Inf
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
            return(0)
        }
    )
}

# Define experiments
# Format: list(E=..., B=..., partition=..., rounds=...)
experiments <- list(
    # nonIID experiments (already running/planned)
    list(E = 5, B = 10, partition = "nonIID", rounds = 1000),
    list(E = 20, B = 10, partition = "nonIID", rounds = 1000),

    # IID experiments with B=10 (requested by user)
    list(E = 1, B = 10, partition = "IID", rounds = 1000),
    list(E = 5, B = 10, partition = "IID", rounds = 1000),
    list(E = 20, B = 10, partition = "IID", rounds = 1000)
)

for (exp in experiments) {
    E_val <- exp$E
    B_val <- exp$B
    partition_val <- exp$partition
    total_rounds <- exp$rounds

    # Create unique checkpoint dir for each config
    ckpt_dir <- file.path(checkpoint_base_dir, paste0(partition_val, "_E", E_val, "_B", B_val))
    if (!dir.exists(ckpt_dir)) dir.create(ckpt_dir)

    cat(sprintf("\n=== Starting Robust Run for Partition=%s, E=%d, B=%d ===\n", partition_val, E_val, B_val))

    # Determine start round
    # Note: get_last_round needs update to handle B and partition
    last_round <- get_last_round(log_file, E_val, partition_val, B_val)
    start_round <- last_round + 1

    if (start_round > total_rounds) {
        cat(sprintf("Experiment E=%d already completed (Round %d/%d). Skipping.\n", E_val, last_round, total_rounds))
        next
    }

    cat(sprintf("Resuming from Round %d (Last completed: %d)\n", start_round, last_round))

    # If resuming from > 1, ensure checkpoint exists
    if (start_round > 1) {
        ckpt_path <- file.path(ckpt_dir, "checkpoint_latest.rds")
        if (!file.exists(ckpt_path)) {
            cat("WARNING: Checkpoint missing but CSV shows progress. Cannot resume safely without model weights.\n")
            cat("Resetting start_round to 1 to restart this experiment.\n")
            start_round <- 1
            # Note: This will append duplicate rounds 1..N to the CSV.
            # The analysis script should handle duplicates (take max or last).
        }
    }

    current_round <- start_round

    while (current_round <= total_rounds) {
        end_round <- min(current_round + chunk_size - 1, total_rounds)

        cat(sprintf(">> Running Chunk: Rounds %d to %d (E=%d)\n", current_round, end_round, E_val))

        # Create temporary runner script
        script_content <- sprintf('
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

      # Load data once per chunk
      cat("Loading data...\n")
      ds_train <- mnist_ds(train = TRUE, download = TRUE)
      ds_test <- mnist_ds(train = FALSE, download = TRUE)
      labels_train <- mnist_labels(ds_train)

      cat("Starting FedAvg chunk...\n")
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
          seed = 123,
          device = "cpu",
          log_file = "%s",
          checkpoint_dir = "%s",
          start_round = %d
      )
    ', partition_val, E_val, B_val, end_round, log_file, ckpt_dir, current_round)

        writeLines(script_content, "temp_chunk_runner.R")

        # Execute chunk in separate process
        res <- system("Rscript temp_chunk_runner.R")

        if (res != 0) {
            cat("ERROR: Chunk execution failed. Retrying in 10 seconds...\n")
            Sys.sleep(10)
            # Retry once? Or loop?
            # For now, let is loop. If it fails consistently, manual intervention needed.
            # But to avoid infinite loop on hard error, let is break.
            stop("Critical failure in chunk execution.")
        }

        current_round <- end_round + 1

        # Small pause
        Sys.sleep(2)
    }
}

cat("\nAll robust experiments completed.\n")
