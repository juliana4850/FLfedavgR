# Change Log: Upgrade MNIST Plotting to Paper Style

**Date**: 2025-12-01  
**Session**: Paper-Style Plotting Upgrade

## Summary

Upgraded MNIST plotting utilities to match publication style with enhanced visual features:
- Target accuracy bands (±0.2% around 97%)
- Log-scale x-axis support for long runs
- SVG export for publication-quality vector graphics
- Cleaner minimal theme with refined grid lines
- Improved legend positioning and formatting

## Files Modified

### Plotting Utilities
- **R/plotting.R**
  - Enhanced `plot_comm_rounds()` with new parameters:
    - `log_x`: Boolean for log10 x-axis scaling
    - `show_points`: Boolean to add points to lines
    - `target_band`: Numeric width for shaded band around target (e.g., 0.002 = ±0.2%)
  - Improved theme:
    - Removed minor x-axis gridlines
    - Light minor y-axis gridlines (grey90, 0.2pt)
    - Bottom legend with no title
    - Bold plot title
  - Better scale formatting:
    - Y-axis: percent format with 0.1% accuracy
    - X-axis: log10 with smart breaks or linear with minimal expansion
  - Type coercion:
    - `partition`: factor with levels c("IID", "nonIID")
    - `method`: factor
    - `E`: integer
    - `B`: numeric
  
  - Added `make_mnist_history_for_plot()`:
    - Reads CSV and filters to MNIST dataset
    - Returns only columns needed for plotting
    - Simplifies data preparation
  
  - Enhanced `save_plot()`:
    - Now exports to PNG, PDF, and SVG
    - Uses `cairo_pdf` for better PDF rendering
    - Auto-installs `svglite` for SVG export
    - Creates output directories if needed

### Example Script
- **inst/examples/plot_mnist_comm.R**
  - Uses new `make_mnist_history_for_plot()` helper
  - Generates main figure with target band (±0.2%)
  - Auto-detects when to use log-scale (max rounds > 1000)
  - Exports all 3 formats (PNG, PDF, SVG) for both plots
  - Cleaner output with paper-style formatting

### Tests
- **tests/testthat/test-plotting.R**
  - Updated test to use new API with `target_band`, `log_x`, `show_points`
  - Tests with synthetic data (5 rounds, 2 methods, 2 partitions)

## Generated Artifacts

Successfully generated 6 visualization artifacts:

```
docs/examples/mnist_comm_accuracy.png         (10" × 4.5", 200 DPI, 56 KB)
docs/examples/mnist_comm_accuracy.pdf          (10" × 4.5", 5.3 KB)
docs/examples/mnist_comm_accuracy.svg          (10" × 4.5", vector)
docs/examples/mnist_comm_accuracy_nonIID.png   (7" × 4.5", 200 DPI, 52 KB)
docs/examples/mnist_comm_accuracy_nonIID.pdf   (7" × 4.5", 5.3 KB)
docs/examples/mnist_comm_accuracy_nonIID.svg   (7" × 4.5", vector)
```

## Visual Improvements

### Target Visualization
- **Target line**: Dashed grey line at 97% accuracy
- **Target band**: Light grey shaded region (±0.2%) for visual reference
- Makes it easier to see when models approach/exceed target

### Axis Formatting
- **Y-axis**: Percent format (0.0% - 100.0%) with 0.1% precision
- **X-axis**: 
  - Linear scale with minimal padding for short runs
  - Log10 scale with smart breaks for long runs (>1000 rounds)

### Theme Refinements
- **Grid**: Removed minor x-gridlines, kept light minor y-gridlines
- **Legend**: Bottom position, no title, cleaner appearance
- **Title**: Bold font for emphasis
- **Overall**: Minimal theme (base_size = 12) for publication quality

## API Changes

### New Parameters
- `plot_comm_rounds()`:
  - `log_x = FALSE`: Enable log10 x-axis
  - `show_points = FALSE`: Add points to lines
  - `target_band = NULL`: Width of shaded band around target

### New Functions
- `make_mnist_history_for_plot(csv_path)`: Simplified data loading
- Enhanced `save_plot()` with SVG support

## Usage Example

```r
# Load and prepare data
hist <- make_mnist_history_for_plot("metrics_mnist.csv")

# Create paper-style plot
p <- plot_comm_rounds(
  hist,
  target = 0.97,
  target_band = 0.002,  # ±0.2% band
  facet_by = "partition",
  color_by = "method",
  linetype_by = "E",
  log_x = FALSE,
  show_points = FALSE,
  title = "MNIST: Test Accuracy vs Communication Rounds"
)

# Save in all formats
save_plot(p, "output.png", "output.pdf", "output.svg")
```

## Dependencies

- **ggplot2**: Core plotting (already installed)
- **scales**: Formatting (already installed)
- **svglite**: SVG export (auto-installed on first use)

## Test Results

All plotting tests passing with new API.

## Files Unchanged

- Core MNIST experiment code
- Logging utilities
- All other package functionality
