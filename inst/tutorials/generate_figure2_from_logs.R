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
log_file <- Sys.getenv("LOG_FILE", "inst/reproduction_outputs/metrics_mnist_cnn.csv")

# Load data
cat(sprintf("Loading %s...\n", log_file))
df <- read.csv(log_file, stringsAsFactors = FALSE)
cat(sprintf("Data: %d rows\n", nrow(df)))

# Filter for MNIST CNN
df_plot <- subset(df, dataset == "MNIST" & model == "CNN")

# Generate Separate Plots
cat("Generating IID plot (Running Max)...\n")
df_iid <- subset(df_plot, partition == "IID")
p_iid <- plot_mnist_figure2(df_iid, target = 0.99, running_max = TRUE, stride = 1)

cat("Generating Non-IID plot (Running Max)...\n")
df_noniid <- subset(df_plot, partition == "nonIID")
p_noniid <- plot_mnist_figure2(df_noniid, target = 0.99, running_max = TRUE, stride = 1)

# Save Plots
output_iid <- Sys.getenv("OUTPUT_IID", "inst/reproduction_outputs/figure2_reproduction_IID.png")
output_noniid <- Sys.getenv("OUTPUT_NONIID", "inst/reproduction_outputs/figure2_reproduction_nonIID.png")

cat(sprintf("Saving IID plot to %s...\n", output_iid))
ggplot2::ggsave(plot = p_iid, filename = output_iid, width = 5, height = 4.5)

cat(sprintf("Saving Non-IID plot to %s...\n", output_noniid))
ggplot2::ggsave(plot = p_noniid, filename = output_noniid, width = 5, height = 4.5)

cat("Done!\n")
