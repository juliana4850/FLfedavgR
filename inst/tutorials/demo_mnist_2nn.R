#!/usr/bin/env Rscript
# MNIST 2NN Paper Reproduction - McMahan et al. (2017)
# Sweeps C ∈ {0.0, 0.1, 0.2, 0.5, 1.0}, B ∈ {Inf, 10}, both IID and non-IID

cat("=== MNIST 2NN Paper Reproduction ===\n")
cat(sprintf("Start time: %s\n\n", Sys.time()))

# ============================================================================
# 1) SETUP
# ============================================================================

# Ensure required packages
pkgs <- c("torch", "torchvision", "ggplot2", "readr", "dplyr", "tidyr", "stringr")
for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        cat(sprintf("Installing %s...\n", pkg))
        install.packages(pkg, repos = "https://cloud.r-project.org")
    }
}

# Install torch backend if needed
if (!torch::torch_is_installed()) {
    cat("Installing torch backend...\n")
    torch::install_torch()
}

# Load package
devtools::load_all()
library(torch)
library(torchvision)
library(dplyr)
library(tidyr)
library(readr)

# Load MNIST-specific helpers (examples of using the generic framework)
source("inst/tutorials/mnist_helpers/mnist_data.R")
source("inst/tutorials/mnist_helpers/mnist_models.R")
source("inst/tutorials/mnist_helpers/mnist_training.R")
source("inst/tutorials/mnist_helpers/mnist_fedavg.R")
source("inst/tutorials/mnist_helpers/mnist_partitions.R")
source("inst/tutorials/mnist_helpers/mnist_logging.R")
source("inst/tutorials/mnist_helpers/mnist_plotting.R")


# Set seeds
set.seed(123)
torch::torch_manual_seed(123)

# Create directories
dir.create("inst/reproduction_outputs", recursive = TRUE, showWarnings = FALSE)
dir.create("docs/genai-log", recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# 2) DATA & PARTITIONS
# ============================================================================

cat("Loading MNIST datasets...\n")
ds_train <- mnist_ds(root = "data", train = TRUE, download = TRUE)
ds_test <- mnist_ds(root = "data", train = FALSE, download = TRUE)
labs <- mnist_labels(ds_train)

cat("Creating partitions...\n")
set.seed(2025)
partition_IID <- iid_split(n_items = length(labs), K = 100, seed = 2025)
partition_nonIID <- mnist_shards_split(labels = labs, K = 100, shards_per_client = 2, seed = 2025)

cat(sprintf("  IID: %d clients\n", length(partition_IID)))
cat(sprintf("  non-IID: %d clients\n\n", length(partition_nonIID)))

# ============================================================================
# 3) EXPERIMENT GRID
# ============================================================================

MODEL <- "2nn"
TARGET <- 0.97
E <- 1L
C_vals <- c(0.0, 0.1, 0.2, 0.5, 1.0)
B_vals <- c(Inf, 10)
ETA_GRID <- c(0.03, 0.05, 0.1)
ROUNDS <- if (identical(Sys.getenv("FEDAVGR_QUICK"), "1")) 300L else 1000L
CLIENTS <- 100L

cat(sprintf("Configuration:\n"))
cat(sprintf("  Model: %s\n", MODEL))
cat(sprintf("  Target: %.2f\n", TARGET))
cat(sprintf("  E: %d\n", E))
cat(sprintf("  C values: %s\n", paste(C_vals, collapse = ", ")))
cat(sprintf("  B values: %s\n", paste(B_vals, collapse = ", ")))
cat(sprintf("  LR grid: %s\n", paste(ETA_GRID, collapse = ", ")))
cat(sprintf("  Rounds: %d\n", ROUNDS))
cat(sprintf("  Clients: %d\n\n", CLIENTS))

# Clear previous metrics
if (file.exists("inst/reproduction_outputs/metrics_mnist_2nn.csv")) {
    file.remove("inst/reproduction_outputs/metrics_mnist_2nn.csv")
    cat("Removed existing inst/reproduction_outputs/metrics_mnist_2nn.csv\n\n")
}

# ============================================================================
# 4) RUN HELPER
# ============================================================================

run_one <- function(partition_name, B, C) {
    B_label <- if (is.infinite(B)) "Inf" else as.character(B)
    cat(sprintf("\n--- Running: partition=%s, B=%s, C=%.1f ---\n", partition_name, B_label, C))

    # Set seed for this setting
    seed_val <- 1000 + as.integer(ifelse(is.infinite(B), 999, B)) + as.integer(C * 100)
    set.seed(seed_val)
    torch::torch_manual_seed(seed_val)

    start_time <- Sys.time()

    # Run experiment
    tryCatch(
        {
            res <- run_fedavg_mnist(
                ds_train = ds_train,
                ds_test = ds_test,
                labels_train = labs,
                model_fn = MODEL,
                partition = partition_name,
                K = CLIENTS,
                C = C,
                E = E,
                batch_size = B,
                lr_grid = ETA_GRID,
                target = TARGET,
                rounds = ROUNDS,
                seed = seed_val,
                device = "cpu"
            )

            elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))

            # Log results
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
                    path = "inst/reproduction_outputs/metrics_mnist_2nn.csv"
                )
            }

            # Get final RTT
            rtt <- tail(res$history$rtt, 1)
            final_acc <- tail(res$history$test_acc, 1)

            cat(sprintf("  Completed in %.1f minutes\n", elapsed))
            cat(sprintf("  Final accuracy: %.4f\n", final_acc))
            cat(sprintf("  Rounds to %.2f: %s\n", TARGET, ifelse(is.na(rtt), "Not reached", sprintf("%.2f", rtt))))

            return(rtt)
        },
        error = function(e) {
            cat(sprintf("  ERROR: %s\n", e$message))
            return(NA_real_)
        }
    )
}

# ============================================================================
# 5) EXECUTE GRID AND COLLECT RESULTS
# ============================================================================

cat("\n", strrep("=", 70), "\n")
cat("EXECUTING EXPERIMENTS\n")
cat(strrep("=", 70), "\n")

# Results storage
results <- list()

for (partition_name in c("IID", "nonIID")) {
    for (B in B_vals) {
        for (C in C_vals) {
            key <- sprintf("%s_B%s_C%.1f", partition_name, ifelse(is.infinite(B), "Inf", B), C)
            rtt <- run_one(partition_name, B, C)
            results[[key]] <- rtt
        }
    }
}

cat("\n", strrep("=", 70), "\n")
cat("GENERATING SUMMARY TABLE\n")
cat(strrep("=", 70), "\n\n")

# ============================================================================
# 6) BUILD SUMMARY TABLE
# ============================================================================

# Extract baselines (C=0.0)
baseline_IID_Inf <- results[["IID_BInf_C0.0"]]
baseline_IID_10 <- results[["IID_B10_C0.0"]]
baseline_nonIID_Inf <- results[["nonIID_BInf_C0.0"]]
baseline_nonIID_10 <- results[["nonIID_B10_C0.0"]]

# Format cell: "XXXX (S.x×)" or "—"
format_cell <- function(rtt, baseline) {
    if (is.na(rtt)) {
        return("—")
    }
    rounds_str <- sprintf("%d", round(rtt))
    if (!is.na(baseline) && baseline > 0) {
        speedup <- baseline / rtt
        return(sprintf("%s (%.1f×)", rounds_str, speedup))
    } else {
        return(rounds_str)
    }
}

# Build table
table_data <- data.frame(
    C = C_vals,
    IID_Binf = sapply(C_vals, function(c) {
        format_cell(results[[sprintf("IID_BInf_C%.1f", c)]], baseline_IID_Inf)
    }),
    IID_B10 = sapply(C_vals, function(c) {
        format_cell(results[[sprintf("IID_B10_C%.1f", c)]], baseline_IID_10)
    }),
    nonIID_Binf = sapply(C_vals, function(c) {
        format_cell(results[[sprintf("nonIID_BInf_C%.1f", c)]], baseline_nonIID_Inf)
    }),
    nonIID_B10 = sapply(C_vals, function(c) {
        format_cell(results[[sprintf("nonIID_B10_C%.1f", c)]], baseline_nonIID_10)
    }),
    stringsAsFactors = FALSE
)

# Rename columns for display
colnames(table_data) <- c("C", "IID (B=∞)", "IID (B=10)", "non-IID (B=∞)", "non-IID (B=10)")

# ============================================================================
# 7) PERSIST ARTIFACTS
# ============================================================================

# Save CSV
write.csv(table_data, "inst/reproduction_outputs/mnist_2nn_table.csv", row.names = FALSE)
cat("Saved: inst/reproduction_outputs/mnist_2nn_table.csv\n")

# Save Markdown
md_lines <- c(
    "# MNIST 2NN Experiment Results",
    "",
    sprintf("**Model**: 2NN (MLP 784→200→200→10)"),
    sprintf("**Target Accuracy**: %.0f%%", TARGET * 100),
    sprintf("**Local Epochs (E)**: %d", E),
    sprintf("**Rounds**: %d", ROUNDS),
    "",
    "## Summary Table",
    "",
    "Rounds to target accuracy (speedup vs C=0.0 baseline):",
    ""
)

# Create markdown table
header <- sprintf("| %s |", paste(colnames(table_data), collapse = " | "))
separator <- sprintf("|%s|", paste(rep("---", ncol(table_data)), collapse = "|"))
md_lines <- c(md_lines, header, separator)

for (i in 1:nrow(table_data)) {
    row_str <- sprintf(
        "| %.1f | %s | %s | %s | %s |",
        table_data[i, 1],
        table_data[i, 2],
        table_data[i, 3],
        table_data[i, 4],
        table_data[i, 5]
    )
    md_lines <- c(md_lines, row_str)
}

md_lines <- c(md_lines, "", "**Note**: \"—\" indicates target not reached within allocated rounds.")

writeLines(md_lines, "inst/reproduction_outputs/mnist_2nn_table.md")
cat("Saved: inst/reproduction_outputs/mnist_2nn_table.md\n")

# Print to console
cat("\n")
cat(paste(md_lines, collapse = "\n"))
cat("\n\n")

# Save long-form results
if (file.exists("inst/reproduction_outputs/metrics_mnist_2nn.csv")) {
    df_long <- read.csv("inst/reproduction_outputs/metrics_mnist_2nn.csv", stringsAsFactors = FALSE)
    write.csv(df_long, "inst/reproduction_outputs/mnist_2nn_results_long.csv", row.names = FALSE)
    cat(sprintf("Saved: inst/reproduction_outputs/mnist_2nn_results_long.csv (%d rows)\n", nrow(df_long)))
}

# Generate plots
if (file.exists("inst/reproduction_outputs/metrics_mnist_2nn.csv")) {
    cat("Generating Table 1...\n")
    df <- read.csv("inst/reproduction_outputs/metrics_mnist_2nn.csv", stringsAsFactors = FALSE)
    df <- subset(df, model == "2NN" | toupper(model) == "2NN")

    if (nrow(df) > 0) {
        df$B_label <- ifelse(df$B == "Inf", "B=∞", paste0("B=", df$B))
        df$C <- as.numeric(df$C)
        df$partition <- factor(df$partition, levels = c("IID", "nonIID"))

        p <- plot_comm_rounds(
            history = df,
            target = TARGET,
            target_band = 0.01,
            facet_by = "partition",
            color_by = "B_label",
            linetype_by = "C",
            title = "MNIST 2NN: Test Accuracy vs Communication Rounds",
            log_x = FALSE,
            show_points = FALSE
        )

        save_plot(
            p,
            out_png = "inst/reproduction_outputs/mnist_2nn_comm_accuracy.png",
            width = 12,
            height = 5
        )

        cat("Saved: inst/reproduction_outputs/mnist_2nn_comm_accuracy.png\n")
    }
}

# ============================================================================
# 8) LOGGING
# ============================================================================

log_lines <- c(
    "",
    sprintf("## Execution Log - %s", Sys.time()),
    "",
    sprintf("**Rounds**: %d", ROUNDS),
    sprintf("**Mode**: %s", ifelse(ROUNDS == 1000, "Full", "Quick")),
    "",
    "### Results Summary",
    ""
)

for (partition_name in c("IID", "nonIID")) {
    log_lines <- c(log_lines, sprintf("#### %s Partition", partition_name))
    for (B in B_vals) {
        B_label <- ifelse(is.infinite(B), "∞", as.character(B))
        log_lines <- c(log_lines, sprintf("**B=%s**:", B_label))
        for (C in C_vals) {
            key <- sprintf("%s_B%s_C%.1f", partition_name, ifelse(is.infinite(B), "Inf", B), C)
            rtt <- results[[key]]
            rtt_str <- ifelse(is.na(rtt), "Not reached", sprintf("%.2f rounds", rtt))
            log_lines <- c(log_lines, sprintf("  - C=%.1f: %s", C, rtt_str))
        }
        log_lines <- c(log_lines, "")
    }
}

log_lines <- c(
    log_lines,
    "### Output Artifacts",
    "- `inst/reproduction_outputs/mnist_2nn_table.csv`",
    "- `inst/reproduction_outputs/mnist_2nn_table.md`",
    "- `inst/reproduction_outputs/mnist_2nn_results_long.csv`",
    "- `inst/reproduction_outputs/mnist_2nn_comm_accuracy.png`",
    ""
)

# Append to log file
log_file <- "docs/genai-log/2025-12-mnist-2nn-demo.md"
if (!file.exists(log_file)) {
    writeLines(c("# MNIST 2NN Demo Execution Log", ""), log_file)
}
write(paste(log_lines, collapse = "\n"), file = log_file, append = TRUE)

cat(sprintf("\nLogged to: %s\n", log_file))

# ============================================================================
# DONE
# ============================================================================

cat("\n", strrep("=", 70), "\n")
cat("EXPERIMENTS COMPLETE\n")
cat(strrep("=", 70), "\n\n")
cat(sprintf("End time: %s\n", Sys.time()))
cat("All artifacts saved successfully.\n")

# Return success invisibly instead of quit
invisible(0)
