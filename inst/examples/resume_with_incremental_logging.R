#!/usr/bin/env Rscript
# Resume paper reproduction from experiment 4
# This script continues the reproduction that stopped after experiment 3

# Load package
devtools::load_all()
library(torch)
library(torchvision)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(knitr)

cat("\n ====================================================================== \n")
cat("RESUMING PAPER REPRODUCTION: Figure 2 & Table 2\n")
cat("McMahan et al. (2017) - MNIST CNN\n")
cat("====================================================================== \n\n")

start_time <- Sys.time()
cat("Start time:", as.character(start_time), "\n\n")

# Configuration
ROUNDS <- 1000
TARGET <- 0.99
C <- 0.1
K <- 100
LR_GRID <- c(0.03, 0.05, 0.1)

# Define experiments (same as original)
experiments_full <- expand.grid(
    E = c(1, 5, 20),
    B = c(Inf),
    stringsAsFactors = FALSE
)
experiments_full$config_id <- 1:nrow(experiments_full)
experiments_full$u <- ifelse(is.infinite(experiments_full$B), 1, 6 * experiments_full$E / experiments_full$B)
experiments_full$method <- ifelse(experiments_full$B == Inf & experiments_full$E == 1, "FedSGD", "FedAvg")

# Filter to quick mode (B=Inf only)
experiments_quick <- experiments_full[experiments_full$B == Inf, ]

cat("Configuration:\n")
cat("  Rounds:", ROUNDS, "\n")
cat("  Target accuracy: 99.00%\n")
cat("  Client fraction (C):", C, "\n")
cat("  Clients (K):", K, "\n")
cat("  LR grid:", paste(LR_GRID, collapse = ", "), "\n\n")

cat("Experiment Grid (3 configurations):\n")
print(experiments_quick[, c("config_id", "E", "B", "u", "method")])
cat("\n")

# RESUME: Start from experiment 4
START_EXPERIMENT <- 4

cat("RESUMING from experiment", START_EXPERIMENT, "\n")
cat("Total experiments: 3 configs × 2 partitions = 6 experiments\n")
cat("Remaining:", 7 - START_EXPERIMENT, "experiments\n\n")

# Load MNIST datasets
cat("Loading MNIST datasets...\n")
ds_train <- mnist_ds(root = "data", train = TRUE, download = TRUE)
ds_test <- mnist_ds(root = "data", train = FALSE, download = TRUE)
labels_train <- mnist_labels(ds_train)
cat("  Training samples:", length(labels_train), "\n")
cat("  Test samples:", length(mnist_labels(ds_test)), "\n\n")

cat("====================================================================== \n")
cat("RUNNING REMAINING EXPERIMENTS\n")
cat("====================================================================== \n\n")

# Run experiments
for (i in START_EXPERIMENT:6) {
    # Determine config and partition
    config_idx <- ((i - 1) %/% 2) + 1
    partition_type <- ifelse((i %% 2) == 1, "IID", "nonIID")

    exp <- experiments_quick[config_idx, ]

    cat(sprintf(
        "\n--- Config %d/3: %s, E=%d, B=%s, Partition=%s ---\n",
        config_idx, exp$method, exp$E,
        ifelse(is.infinite(exp$B), "Inf", as.character(exp$B)),
        partition_type
    ))

    # Run experiment with error handling
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
                log_file = "inst/reproduction_outputs/metrics_mnist.csv" # Enable incremental logging
            )
        },
        error = function(e) {
            cat(sprintf("ERROR in experiment %d: %s\n", i, e$message))
            return(NULL)
        }
    )

    if (!is.null(result)) {
        # Report rounds-to-target
        rtt <- result$history$rtt[nrow(result$history)]
        if (!is.na(rtt)) {
            cat(sprintf("  Target %.0f%% reached at round %.1f\n", TARGET * 100, rtt))
        } else {
            cat(sprintf("  Target %.0f%% not reached in %d rounds\n", TARGET * 100, ROUNDS))
        }
    }
}

cat("\n ====================================================================== \n")
cat("RESUMED EXPERIMENTS COMPLETE\n")
cat("====================================================================== \n\n")

# Now load ALL results from metrics_mnist.csv and generate artifacts
cat("Loading all results from inst/reproduction_outputs/metrics_mnist.csv...\n")
all_data <- read.csv("inst/reproduction_outputs/metrics_mnist.csv", stringsAsFactors = FALSE)

cat("Total data rows:", nrow(all_data), "\n\n")

# ============================================================================
# GENERATE TABLE 2
# ============================================================================

cat("Generating Table 2...\n")

tryCatch(
    {
        # Get final rounds-to-target for each configuration
        table2_data <- all_data %>%
            group_by(dataset, model, partition, method, E, B, u) %>%
            summarise(
                rtt = last(rounds_to_target),
                .groups = "drop"
            )

        # Pivot to wide format
        table2_wide <- table2_data %>%
            pivot_wider(
                id_cols = c(method, E, B, u),
                names_from = partition,
                values_from = rtt,
                names_prefix = "rtt_"
            )

        # Calculate speedups vs FedSGD baseline
        if (!"rtt_IID" %in% names(table2_wide)) table2_wide$rtt_IID <- NA
        if (!"rtt_nonIID" %in% names(table2_wide)) table2_wide$rtt_nonIID <- NA

        fedsgd_iid <- table2_wide$rtt_IID[table2_wide$method == "FedSGD"]
        fedsgd_noniid <- table2_wide$rtt_nonIID[table2_wide$method == "FedSGD"]

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
            arrange(E)

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
        # Prepare data for plotting
        plot_data <- all_data %>%
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
                limits = c(0, 1.0),
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
                cat(sprintf("  Warning: cairo_pdf failed, trying standard pdf device...\n"))
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

cat("\n====================================================================== \n")
cat("REPRODUCTION COMPLETE\n")
cat("====================================================================== \n\n")

end_time <- Sys.time()
cat("End time:", as.character(end_time), "\n\n")

cat("Outputs:\n")
cat("  - Table 2: docs/examples/table2_reproduction.{csv,md}\n")
cat("  - Figure 2: docs/examples/figure2_reproduction.{png,pdf}\n")
cat("  - Raw data: inst/reproduction_outputs/metrics_mnist.csv\n\n")

cat("Compare these outputs with the original paper:\n")
cat("  - Table 2: Rounds-to-target and speedups\n")
cat("  - Figure 2: Accuracy curves for different (B, E) combinations\n")
