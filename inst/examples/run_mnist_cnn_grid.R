#!/usr/bin/env Rscript
# MNIST CNN Grid Experiments - Paper Reproduction
# Runs full hyperparameter sweep: B ∈ {10, 50, Inf}, E ∈ {1, 5, 20}

cat("=== MNIST CNN Grid Experiments ===\n")
cat(sprintf("Start time: %s\n\n", Sys.time()))

# Ensure required packages
if (!requireNamespace("torch", quietly = TRUE)) {
    cat("Installing torch...\n")
    install.packages("torch")
    torch::install_torch()
}

pkgs <- c("ggplot2", "scales")
to_install <- pkgs[!vapply(pkgs, requireNamespace, FUN.VALUE = logical(1), quietly = TRUE)]
if (length(to_install) > 0) {
    cat(sprintf("Installing packages: %s\n", paste(to_install, collapse = ", ")))
    install.packages(to_install)
}

# Load package
devtools::load_all()
library(torch)
library(torchvision)

# Set seeds for reproducibility
set.seed(123)
torch::torch_manual_seed(123)

# Configuration
C <- 0.1
B_vals <- c(10, 50, Inf)
E_vals <- c(1L, 5L, 20L)
eta_grid <- c(0.03, 0.05, 0.1)
target <- 0.99
ROUNDS <- as.integer(if (Sys.getenv("FEDAVGR_QUICK", "0") == "1") 200 else 1000)

cat(sprintf("Configuration:\n"))
cat(sprintf("  Rounds: %d\n", ROUNDS))
cat(sprintf("  Client fraction (C): %.2f\n", C))
cat(sprintf("  Batch sizes (B): %s\n", paste(B_vals, collapse = ", ")))
cat(sprintf("  Local epochs (E): %s\n", paste(E_vals, collapse = ", ")))
cat(sprintf("  LR grid: %s\n", paste(eta_grid, collapse = ", ")))
cat(sprintf("  Target accuracy: %.2f\n\n", target))

# Load datasets
cat("Loading MNIST datasets...\n")
ds_train <- mnist_ds(root = "data", train = TRUE, download = TRUE)
ds_test <- mnist_ds(root = "data", train = FALSE, download = TRUE)
labels_train <- mnist_labels(ds_train)

cat(sprintf("  Training samples: %d\n", length(ds_train)))
cat(sprintf("  Test samples: %d\n\n", length(ds_test)))

# Create partitions
cat("Creating partitions...\n")
set.seed(2025)
partition_IID <- iid_split(n_items = length(labels_train), K = 100, seed = 2025)
partition_nonIID <- mnist_shards_split(labels = labels_train, K = 100, shards_per_client = 2, seed = 2025)
cat("  IID and non-IID partitions created\n\n")

# Clear previous metrics
if (file.exists("metrics_mnist.csv")) {
    file.remove("metrics_mnist.csv")
    cat("Removed existing metrics_mnist.csv\n\n")
}

# Helper function to run one setting
run_one <- function(partition_name, partition_indices) {
    cat(sprintf("\n=== Running %s partition ===\n", partition_name))

    for (B in B_vals) {
        for (E in E_vals) {
            B_label <- if (is.infinite(B)) "Inf" else as.character(B)
            cat(sprintf("\nSetting: B=%s, E=%d\n", B_label, E))

            # Set deterministic seed for this setting
            seed_offset <- as.integer(ifelse(is.infinite(B), 9999, B)) + E
            set.seed(100 + seed_offset)
            torch::torch_manual_seed(100 + seed_offset)

            start_time <- Sys.time()

            # Run experiment
            res <- run_fedavg_mnist(
                ds_train = ds_train,
                ds_test = ds_test,
                labels_train = labels_train,
                model_fn = "cnn",
                partition = partition_name,
                K = 100,
                C = C,
                E = E,
                batch_size = B,
                lr_grid = eta_grid,
                target = target,
                rounds = ROUNDS,
                seed = 100 + seed_offset,
                device = "cpu"
            )

            elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

            # Log results to CSV
            for (i in seq_len(nrow(res$history))) {
                row <- res$history[i, ]
                append_metrics(
                    dataset = row$dataset,
                    model = row$model,
                    partition = row$partition,
                    method = row$method,
                    round = row$round,
                    test_acc = row$test_acc,
                    chosen_lr = row$chosen_lr,
                    E = row$E,
                    B = row$B,
                    C = row$C,
                    clients_selected = row$clients_selected,
                    u = row$u,
                    target = row$target,
                    rounds_to_target = row$rtt,
                    path = "metrics_mnist.csv"
                )
            }

            # Summary
            final_acc <- tail(res$history$test_acc, 1)
            rtt <- tail(res$history$rtt, 1)
            rtt_str <- if (is.na(rtt)) "Not reached" else sprintf("%.2f", rtt)

            cat(sprintf("  Completed in %.1f minutes\n", elapsed))
            cat(sprintf("  Final accuracy: %.4f\n", final_acc))
            cat(sprintf("  Rounds to %.2f: %s\n", target, rtt_str))
        }
    }
}

# Execute experiments
cat("\n", strrep("=", 70), "\n")
cat("EXECUTING EXPERIMENTS\n")
cat(strrep("=", 70), "\n")

# Run IID partition
run_one("IID", partition_IID)

# Run non-IID partition
run_one("nonIID", partition_nonIID)

cat("\n", strrep("=", 70), "\n")
cat("GENERATING PLOTS\n")
cat(strrep("=", 70), "\n\n")

# Read results
df <- read.csv("metrics_mnist.csv", stringsAsFactors = FALSE)
df <- subset(df, dataset == "MNIST" & toupper(model) == "CNN")

# Prepare data for plotting
df$B_label <- ifelse(df$B == "Inf", "B=∞", paste0("B=", df$B))
df$E <- as.integer(df$E)
df$partition <- factor(df$partition, levels = c("IID", "nonIID"))

cat(sprintf("Loaded %d rows from metrics\n", nrow(df)))
cat(sprintf("  Partitions: %s\n", paste(unique(df$partition), collapse = ", ")))
cat(sprintf("  B values: %s\n", paste(unique(df$B_label), collapse = ", ")))
cat(sprintf("  E values: %s\n\n", paste(unique(df$E), collapse = ", ")))

# Create output directory
dir.create("docs/examples", recursive = TRUE, showWarnings = FALSE)

# Main plot: IID vs nonIID panels
cat("Creating main plot (IID vs nonIID)...\n")
p_main <- plot_comm_rounds(
    history = df,
    target = target,
    target_band = 0.002,
    facet_by = "partition",
    color_by = "B_label",
    linetype_by = "E",
    title = "MNIST CNN: Test Accuracy vs Communication Rounds",
    log_x = FALSE,
    show_points = FALSE
)

save_plot(
    p_main,
    out_png = "docs/examples/mnist_cnn_comm_accuracy.png",
    out_pdf = "docs/examples/mnist_cnn_comm_accuracy.pdf",
    out_svg = "docs/examples/mnist_cnn_comm_accuracy.svg",
    width = 12,
    height = 5
)

cat("  Saved: docs/examples/mnist_cnn_comm_accuracy.{png,pdf,svg}\n\n")

# Non-IID only plot
cat("Creating nonIID-only plot...\n")
df_non <- subset(df, partition == "nonIID")

# Use log scale if max round > 1000
use_log_x <- max(df_non$round, na.rm = TRUE) > 1000

p_non <- plot_comm_rounds(
    history = df_non,
    target = target,
    target_band = 0.002,
    facet_by = character(0),
    color_by = "B_label",
    linetype_by = "E",
    title = "MNIST CNN (non-IID)",
    log_x = use_log_x,
    show_points = FALSE
)

save_plot(
    p_non,
    out_png = "docs/examples/mnist_cnn_comm_accuracy_nonIID.png",
    out_pdf = "docs/examples/mnist_cnn_comm_accuracy_nonIID.pdf",
    out_svg = "docs/examples/mnist_cnn_comm_accuracy_nonIID.svg",
    width = 8,
    height = 5
)

cat("  Saved: docs/examples/mnist_cnn_comm_accuracy_nonIID.{png,pdf,svg}\n\n")

cat(strrep("=", 70), "\n")
cat("EXPERIMENTS COMPLETE\n")
cat(strrep("=", 70), "\n\n")
cat(sprintf("End time: %s\n", Sys.time()))
cat(sprintf("Results saved to: metrics_mnist.csv\n"))
cat(sprintf("Plots saved to: docs/examples/\n"))
