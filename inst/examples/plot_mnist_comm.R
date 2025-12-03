#!/usr/bin/env Rscript
# MNIST Communication Rounds Plotting (Paper Style)
# Generates publication-quality plots with target bands and SVG export

# Load package
devtools::load_all()

cat("=== MNIST Communication Rounds Plotting (Paper Style) ===\n\n")

# Check for metrics file
csv_path <- "metrics_mnist.csv"
if (!file.exists(csv_path)) {
    stop(sprintf("Metrics file not found: %s\nPlease run inst/tutorials/mnist_demo.R first.", csv_path))
}

cat(sprintf("Reading metrics from: %s\n", csv_path))

# Read and prepare data
hist <- fedavgR::make_mnist_history_for_plot(csv_path)

cat(sprintf("  Loaded %d rows\n", nrow(hist)))
if ("method" %in% names(hist)) {
    cat(sprintf("  Methods: %s\n", paste(unique(hist$method), collapse = ", ")))
}
if ("partition" %in% names(hist)) {
    cat(sprintf("  Partitions: %s\n", paste(unique(hist$partition), collapse = ", ")))
}
cat("\n")

# Ensure output directory exists
dir.create("docs/examples", recursive = TRUE, showWarnings = FALSE)

# ============================================================================
# Plot 1: Main figure with IID and nonIID panels
# ============================================================================
cat("Generating main figure (IID vs nonIID)...\n")

p_main <- fedavgR::plot_comm_rounds(
    hist,
    target = 0.97,
    target_band = 0.002, # ±0.2% band
    facet_by = "partition",
    color_by = "method",
    linetype_by = "E",
    shape_by = NULL,
    title = "MNIST: Test Accuracy vs Communication Rounds",
    log_x = FALSE,
    show_points = FALSE
)

fedavgR::save_plot(
    p_main,
    "docs/examples/mnist_comm_accuracy.png",
    "docs/examples/mnist_comm_accuracy.pdf",
    "docs/examples/mnist_comm_accuracy.svg",
    width = 10,
    height = 4.5
)

cat("  Saved: docs/examples/mnist_comm_accuracy.png\n")
cat("  Saved: docs/examples/mnist_comm_accuracy.pdf\n")
cat("  Saved: docs/examples/mnist_comm_accuracy.svg\n")
cat("\n")

# ============================================================================
# Plot 2: Non-IID only (compact, with log-scale if needed)
# ============================================================================
cat("Generating nonIID-only figure...\n")

hist_non <- subset(hist, partition == "nonIID")

if (nrow(hist_non) > 0) {
    # Use log scale if max round > 1000
    log_x <- isTRUE(max(hist_non$round, na.rm = TRUE) > 1000)

    if (log_x) {
        cat("  Using log10 x-axis (max rounds > 1000)\n")
    }

    p_non <- fedavgR::plot_comm_rounds(
        hist_non,
        target = 0.97,
        target_band = 0.002,
        facet_by = character(0), # No faceting
        color_by = "method",
        linetype_by = "E",
        shape_by = NULL,
        title = "MNIST non-IID",
        log_x = log_x,
        show_points = FALSE
    )

    fedavgR::save_plot(
        p_non,
        "docs/examples/mnist_comm_accuracy_nonIID.png",
        "docs/examples/mnist_comm_accuracy_nonIID.pdf",
        "docs/examples/mnist_comm_accuracy_nonIID.svg",
        width = 7,
        height = 4.5
    )

    cat("  Saved: docs/examples/mnist_comm_accuracy_nonIID.png\n")
    cat("  Saved: docs/examples/mnist_comm_accuracy_nonIID.pdf\n")
    cat("  Saved: docs/examples/mnist_comm_accuracy_nonIID.svg\n")
} else {
    cat("  No nonIID data found, skipping nonIID-only plot\n")
}

cat("\n=== Plotting Complete ===\n")
cat("\nGenerated artifacts:\n")
cat("  - docs/examples/mnist_comm_accuracy.png\n")
cat("  - docs/examples/mnist_comm_accuracy.pdf\n")
cat("  - docs/examples/mnist_comm_accuracy.svg\n")
if (exists("p_non")) {
    cat("  - docs/examples/mnist_comm_accuracy_nonIID.png\n")
    cat("  - docs/examples/mnist_comm_accuracy_nonIID.pdf\n")
    cat("  - docs/examples/mnist_comm_accuracy_nonIID.svg\n")
}
