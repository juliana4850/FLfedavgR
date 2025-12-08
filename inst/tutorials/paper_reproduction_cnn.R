#!/usr/bin/env Rscript
# Paper Reproduction: Figure 2 & Table 2
# McMahan et al. (2017) - MNIST CNN Experiments
#
# Reproduces:
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

# Quick mode: reduced set (E ∈ {1,5,20}, B=10 only)
# Configs: 5 (E=1), 8 (E=5), 9 (E=20) - all with B=10
experiments_quick <- experiments_full[experiments_full$B == 10, ]

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
                    log_file = "inst/reproduction_outputs/metrics_mnist.csv"
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
            # We don't need to write to CSV here anymore

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

# ============================================================================
# GENERATE TABLE 2
# ============================================================================

cat("Generating Table 2...\n")

tryCatch(
    {
        # Combine all results
        if (length(all_results) == 0) {
            stop("No results to generate table from.")
        }

        combined_results <- bind_rows(all_results)

        # Get final rounds-to-target for each configuration
        table2_data <- combined_results %>%
            group_by(config_id, partition, E, B, u, method) %>%
            summarise(
                rtt = last(rtt),
                .groups = "drop"
            )

        # Pivot to wide format
        table2_wide <- table2_data %>%
            pivot_wider(
                id_cols = c(config_id, method, E, B, u),
                names_from = partition,
                values_from = rtt,
                names_prefix = "rtt_"
            )

        # Calculate speedups vs FedSGD baseline
        # Handle missing columns if only one partition exists
        if (!"rtt_IID" %in% names(table2_wide)) table2_wide$rtt_IID <- NA
        if (!"rtt_nonIID" %in% names(table2_wide)) table2_wide$rtt_nonIID <- NA

        fedsgd_iid <- table2_wide$rtt_IID[table2_wide$method == "FedSGD"]
        fedsgd_noniid <- table2_wide$rtt_nonIID[table2_wide$method == "FedSGD"]

        # Handle empty fedsgd (if NA)
        if (length(fedsgd_iid) == 0) fedsgd_iid <- NA
        if (length(fedsgd_noniid) == 0) fedsgd_noniid <- NA

        table2_wide <- table2_wide %>%
            mutate(
                IID_speedup = fedsgd_iid / rtt_IID,
                NonIID_speedup = fedsgd_noniid / rtt_nonIID,
                IID_formatted = ifelse(
                    method == "FedSGD",
                    sprintf("%.0f", rtt_IID),
                    sprintf("%.0f (%.1f×)", rtt_IID, IID_speedup)
                ),
                NonIID_formatted = ifelse(
                    method == "FedSGD",
                    sprintf("%.0f", rtt_nonIID),
                    sprintf("%.0f (%.1f×)", rtt_nonIID, NonIID_speedup)
                )
            ) %>%
            arrange(config_id)

        # Create final table
        table2_final <- table2_wide %>%
            select(method, E, B, u, IID = IID_formatted, `Non-IID` = NonIID_formatted)

        # Replace Inf with ∞ symbol
        table2_final$B <- ifelse(is.infinite(table2_final$B) | table2_final$B == "Inf", "∞", as.character(table2_final$B))

        # Save as CSV
        write_csv(table2_final, "docs/examples/table2_reproduction.csv")

        # Save as Markdown
        table2_md <- knitr::kable(table2_final, format = "markdown", align = "c")
        writeLines(
            c(
                "# Table 2: Number of communication rounds to reach 99% accuracy",
                "",
                "Reproduction from McMahan et al. (2017)",
                "",
                table2_md
            ),
            "docs/examples/table2_reproduction.md"
        )

        cat("  Saved: docs/examples/table2_reproduction.{csv,md}\n")
        print(table2_final)
        cat("\n")
    },
    error = function(e) {
        cat(sprintf("ERROR generating Table 2: %s\n", e$message))
    }
)

# ============================================================================
# GENERATE FIGURE 2
# ============================================================================

cat("Generating Figure 2...\n")

tryCatch(
    {
        if (length(all_results) == 0) {
            stop("No results to plot.")
        }

        combined_results <- bind_rows(all_results)

        # Prepare data for plotting
        plot_data <- combined_results %>%
            mutate(
                B_label = ifelse(is.infinite(as.numeric(B)) | B == "Inf", "B=∞", paste0("B=", B)),
                E_label = paste0("E=", E),
                B_num = ifelse(is.infinite(as.numeric(B)) | B == "Inf", 999, as.numeric(B))
            ) %>%
            arrange(B_num, E)

        # Create color mapping (by B)
        b_colors <- c("B=10" = "#E41A1C", "B=50" = "#FF7F00", "B=∞" = "#377EB8")

        # Create linetype mapping (by E)
        e_linetypes <- c("E=1" = "solid", "E=5" = "dashed", "E=20" = "dotted")

        # Create two-panel plot
        p <- ggplot(plot_data, aes(
            x = round, y = test_acc,
            color = B_label, linetype = E_label
        )) +
            geom_line(linewidth = 0.8) +
            geom_hline(
                yintercept = TARGET, linetype = "solid",
                color = "gray50", linewidth = 0.5
            ) +
            facet_wrap(~partition,
                nrow = 1,
                labeller = labeller(partition = c(
                    "IID" = "MNIST CNN IID",
                    "nonIID" = "MNIST CNN Non-IID"
                ))
            ) +
            scale_color_manual(values = b_colors, name = NULL) +
            scale_linetype_manual(values = e_linetypes, name = NULL) +
            scale_y_continuous(
                limits = c(0, 1.0), # Adjusted for safety
                breaks = seq(0, 1.0, by = 0.1)
            ) +
            scale_x_continuous(limits = c(0, ROUNDS)) +
            labs(
                x = "Communication Rounds",
                y = "Test Accuracy",
                title = "Figure 2: Test set accuracy vs. communication rounds for MNIST CNN"
            ) +
            theme_minimal(base_size = 11) +
            theme(
                legend.position = "right",
                legend.box = "vertical",
                panel.grid.minor = element_line(linewidth = 0.2),
                strip.text = element_text(face = "bold", size = 12),
                plot.title = element_text(hjust = 0.5, face = "bold")
            )

        # Save plot
        ggsave("docs/examples/figure2_reproduction.png", p,
            width = 10, height = 5, dpi = 200
        )

        # Try saving PDF, fallback if cairo fails
        tryCatch(
            {
                ggsave("docs/examples/figure2_reproduction.pdf", p,
                    width = 10, height = 5, device = cairo_pdf
                )
            },
            error = function(e) {
                cat(sprintf("  Warning: cairo_pdf failed (%s), trying standard pdf device...\n", e$message))
                ggsave("docs/examples/figure2_reproduction.pdf", p,
                    width = 10, height = 5, device = "pdf"
                )
            }
        )

        cat("  Saved: docs/examples/figure2_reproduction.{png,pdf}\n\n")
    },
    error = function(e) {
        cat(sprintf("ERROR generating Figure 2: %s\n", e$message))
    }
)

# ============================================================================
# SUMMARY
# ============================================================================

cat(strrep("=", 70), "\n")
cat("REPRODUCTION COMPLETE\n")
cat(strrep("=", 70), "\n\n")
cat(sprintf("End time: %s\n", Sys.time()))
cat("\nOutputs:\n")
cat("  - Table 2: docs/examples/table2_reproduction.{csv,md}\n")
cat("  - Figure 2: docs/examples/figure2_reproduction.{png,pdf}\n")
cat("  - Raw data: inst/reproduction_outputs/metrics_mnist.csv\n\n")

cat("Compare these outputs with the original paper:\n")
cat("  - Table 2: Rounds-to-target and speedups\n")
cat("  - Figure 2: Accuracy curves for different (B, E) combinations\n\n")

invisible(0)
