#!/usr/bin/env Rscript
# Generate Table 2 from existing logs
# Combines metrics_mnist.csv and metrics_mnist_B_inf.csv

cat("\n", strrep("=", 70), "\n")
cat("GENERATING TABLE 2 FROM LOGS\n")
cat(strrep("=", 70), "\n\n")

# Load package
devtools::load_all()
library(dplyr)
library(tidyr)
library(readr)

# Paths
log_file <- "inst/reproduction_outputs/metrics_mnist_cnn.csv"
output_csv <- "inst/reproduction_outputs/table2_reproduction.csv"
output_md <- "inst/reproduction_outputs/table2_reproduction.md"

# Load data
cat(sprintf("Loading %s...\n", log_file))
df <- read.csv(log_file, stringsAsFactors = FALSE)
cat(sprintf("Data: %d rows\n", nrow(df)))

# Filter for MNIST CNN
df_filtered <- subset(df, dataset == "MNIST" & model == "CNN")

# Ensure B is numeric for sorting (Inf becomes Inf)
df_filtered$B_num <- ifelse(df_filtered$B == "Inf", Inf, as.numeric(df_filtered$B))

table2_data <- df_filtered %>%
    group_by(partition, method, E, B_num) %>%
    summarise(
        # Calculate rounds to target (first round >= 0.99)
        rtt = min(round[test_acc >= 0.99], Inf, na.rm = TRUE),
        u = first(u), # Assuming u is consistent for E, B
        .groups = "drop"
    )

# Replace Inf rtt with NA (min returns Inf if all NA)
table2_data$rtt[is.infinite(table2_data$rtt)] <- NA

# Pivot to wide format
table2_wide <- table2_data %>%
    pivot_wider(
        id_cols = c(method, E, B_num, u),
        names_from = partition,
        values_from = rtt,
        names_prefix = "rtt_"
    )

# Calculate speedups vs FedSGD baseline (E=1, B=Inf)
# Find FedSGD baseline for IID and Non-IID
fedsgd_iid <- table2_wide$rtt_IID[table2_wide$method == "FedSGD" & table2_wide$E == 1 & is.infinite(table2_wide$B_num)]
fedsgd_noniid <- table2_wide$rtt_nonIID[table2_wide$method == "FedSGD" & table2_wide$E == 1 & is.infinite(table2_wide$B_num)]

# Handle missing baselines
if (length(fedsgd_iid) == 0) fedsgd_iid <- NA
if (length(fedsgd_noniid) == 0) fedsgd_noniid <- NA

cat(sprintf("Baseline (FedSGD) - IID: %s, Non-IID: %s\n", fedsgd_iid, fedsgd_noniid))

table2_wide <- table2_wide %>%
    mutate(
        IID_speedup = fedsgd_iid / rtt_IID,
        NonIID_speedup = fedsgd_noniid / rtt_nonIID,
        IID_formatted = ifelse(
            is.na(rtt_IID), "--",
            ifelse(
                method == "FedSGD" & E == 1 & is.infinite(B_num),
                sprintf("%.0f", rtt_IID),
                ifelse(is.na(IID_speedup), sprintf("%.0f", rtt_IID), sprintf("%.0f (%.1f×)", rtt_IID, IID_speedup))
            )
        ),
        NonIID_formatted = ifelse(
            is.na(rtt_nonIID), "--",
            ifelse(
                method == "FedSGD" & E == 1 & is.infinite(B_num),
                sprintf("%.0f", rtt_nonIID),
                ifelse(is.na(NonIID_speedup), sprintf("%.0f", rtt_nonIID), sprintf("%.0f (%.1f×)", rtt_nonIID, NonIID_speedup))
            )
        )
    )

# Define paper order
# 1. FedSGD (E=1, B=inf)
# 2. FedAvg (E=5, B=inf)
# 3. FedAvg (E=1, B=50)
# 4. FedAvg (E=20, B=inf)
# 5. FedAvg (E=1, B=10)
# 6. FedAvg (E=5, B=50)
# 7. FedAvg (E=20, B=50)
# 8. FedAvg (E=5, B=10)
# 9. FedAvg (E=20, B=10)

# Create a sorting key
table2_wide <- table2_wide %>%
    mutate(
        sort_order = case_when(
            method == "FedSGD" & E == 1 & is.infinite(B_num) ~ 1,
            method == "FedAvg" & E == 5 & is.infinite(B_num) ~ 2,
            method == "FedAvg" & E == 1 & B_num == 50 ~ 3,
            method == "FedAvg" & E == 20 & is.infinite(B_num) ~ 4,
            method == "FedAvg" & E == 1 & B_num == 10 ~ 5,
            method == "FedAvg" & E == 5 & B_num == 50 ~ 6,
            method == "FedAvg" & E == 20 & B_num == 50 ~ 7,
            method == "FedAvg" & E == 5 & B_num == 10 ~ 8,
            method == "FedAvg" & E == 20 & B_num == 10 ~ 9,
            TRUE ~ 99 # Others
        )
    ) %>%
    arrange(sort_order)

# Recalculate u with correct formula: u = E * n / (K * B)
# n=60000, K=100 => n/K = 600
# If B=Inf, u = E
table2_wide$u_corrected <- ifelse(
    is.infinite(table2_wide$B_num),
    table2_wide$E,
    table2_wide$E * 600 / table2_wide$B_num
)

# Create final table columns matching paper
table2_final <- table2_wide %>%
    mutate(
        B = ifelse(is.infinite(B_num), "∞", as.character(B_num)),
        CNN = ifelse(method == "FedSGD", "FedSGD", "FedAvg")
    ) %>%
    select(CNN, E, B, u = u_corrected, IID = IID_formatted, `Non-IID` = NonIID_formatted)

# Save as CSV
write_csv(table2_final, output_csv)

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
    output_md
)

cat(sprintf("Saved table to %s and %s\n", output_csv, output_md))
print(table2_final)
