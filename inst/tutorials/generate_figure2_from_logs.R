#!/usr/bin/env Rscript
# Generate Figure 2 from existing logs
# Combines metrics_mnist.csv and metrics_mnist_B_inf.csv

cat("\n", strrep("=", 70), "\n")
cat("GENERATING FIGURE 2 FROM LOGS\n")
cat(strrep("=", 70), "\n\n")

# Load package
devtools::load_all()

# Source plotting helper
source("inst/tutorials/mnist_helpers/mnist_plotting.R")

# Paths
log_file_1 <- "inst/reproduction_outputs/metrics_mnist.csv"
log_file_2 <- "inst/reproduction_outputs/metrics_mnist_B_inf.csv"
output_plot <- "inst/reproduction_outputs/figure2_reproduction_combined.png"

# Load data
cat(sprintf("Loading %s...\n", log_file_1))
df1 <- read.csv(log_file_1, stringsAsFactors = FALSE)

cat(sprintf("Loading %s...\n", log_file_2))
df2 <- read.csv(log_file_2, stringsAsFactors = FALSE)

# Combine
df_combined <- rbind(df1, df2)
cat(sprintf("Combined data: %d rows\n", nrow(df_combined)))

# Filter for MNIST CNN
df_plot <- subset(df_combined, dataset == "MNIST" & model == "CNN")

# Generate Separate Plots
cat("Generating IID plot (Running Max)...\n")
df_iid <- subset(df_plot, partition == "IID")
p_iid <- plot_mnist_figure2(df_iid, target = 0.99, running_max = TRUE, smooth_window = 0)

cat("Generating Non-IID plot (Running Max)...\n")
df_noniid <- subset(df_plot, partition == "nonIID")
p_noniid <- plot_mnist_figure2(df_noniid, target = 0.99, running_max = TRUE, smooth_window = 0)

# Save Plots
output_iid <- "inst/reproduction_outputs/figure2_reproduction_IID.png"
output_noniid <- "inst/reproduction_outputs/figure2_reproduction_nonIID.png"

cat(sprintf("Saving IID plot to %s...\n", output_iid))
save_plot(p_iid, output_iid, gsub(".png", ".pdf", output_iid), width = 5, height = 4.5)

cat(sprintf("Saving Non-IID plot to %s...\n", output_noniid))
save_plot(p_noniid, output_noniid, gsub(".png", ".pdf", output_noniid), width = 5, height = 4.5)

cat("Done!\n")
