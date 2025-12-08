# SELF-CHECK: Ensures required columns exist before plotting
# Required columns: round, test_acc, method, partition (if faceting)
# Stops with clear message if missing

#' Plot Communication Rounds
#'
#' Creates a ggplot of test accuracy vs communication rounds with paper-style formatting.
#'
#' @param history Data frame with columns: round, test_acc, method, partition (and optionally E, B)
#' @param target Optional target accuracy for horizontal reference line
#' @param facet_by Column name(s) to facet by (default: "partition")
#' @param color_by Column name for color aesthetic (default: "method")
#' @param linetype_by Column name for linetype aesthetic (default: "E")
#' @param shape_by Optional column name for shape aesthetic
#' @param title Optional plot title
#' @param log_x Logical; if TRUE, use log10 scale for x-axis
#' @param show_points Logical; if TRUE, add points to lines
#' @param target_band Optional numeric width around target (e.g., 0.002 for ±0.2%)
#' @return A ggplot object
#' @export
plot_comm_rounds <- function(history, target = NULL, facet_by = c("partition"),
                             color_by = c("method"), linetype_by = c("E"),
                             shape_by = NULL, title = NULL, log_x = FALSE,
                             show_points = FALSE, target_band = NULL) {
    # Check for required packages
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("ggplot2 package required for plotting. Install with: install.packages('ggplot2')")
    }
    if (!requireNamespace("scales", quietly = TRUE)) {
        stop("scales package required for plotting. Install with: install.packages('scales')")
    }

    # Check for required columns
    required_cols <- c("round", "test_acc", color_by, linetype_by)
    if (!is.null(facet_by) && length(facet_by) > 0) {
        required_cols <- c(required_cols, facet_by)
    }
    if (!is.null(shape_by)) {
        required_cols <- c(required_cols, shape_by)
    }

    missing_cols <- setdiff(required_cols, names(history))
    if (length(missing_cols) > 0) {
        stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
    }

    # Coerce columns to proper types
    if ("partition" %in% names(history)) {
        history$partition <- factor(history$partition, levels = c("IID", "nonIID"))
    }
    if ("method" %in% names(history)) {
        history$method <- factor(history$method)
    }
    if ("E" %in% names(history)) {
        history$E <- as.integer(history$E)
    }
    if ("B" %in% names(history)) {
        history$B <- as.numeric(history$B)
    }

    # Ensure factors for aesthetic mappings
    if (color_by %in% names(history)) {
        history[[color_by]] <- as.factor(history[[color_by]])
    }
    if (linetype_by %in% names(history)) {
        history[[linetype_by]] <- as.factor(history[[linetype_by]])
    }
    if (!is.null(shape_by) && shape_by %in% names(history)) {
        history[[shape_by]] <- as.factor(history[[shape_by]])
    }

    # Build base plot
    p <- ggplot2::ggplot(history, ggplot2::aes(x = round, y = test_acc))

    # Add target band if specified
    if (!is.null(target_band) && !is.null(target)) {
        p <- p + ggplot2::annotate(
            "rect",
            xmin = -Inf, xmax = Inf,
            ymin = target - target_band,
            ymax = target + target_band,
            alpha = 0.08,
            fill = "grey60"
        )
    }

    # Add target line if specified
    if (!is.null(target)) {
        p <- p + ggplot2::geom_hline(
            yintercept = target,
            linetype = "dashed",
            color = "grey50",
            linewidth = 0.5
        )
    }

    # Build aesthetic mapping
    aes_mapping <- ggplot2::aes(
        color = .data[[color_by]],
        linetype = .data[[linetype_by]]
    )
    if (!is.null(shape_by)) {
        aes_mapping$shape <- rlang::quo(.data[[shape_by]])
    }

    # Add lines
    p <- p + ggplot2::geom_line(aes_mapping, linewidth = 0.7)

    # Add points if requested
    if (show_points) {
        p <- p + ggplot2::geom_point(aes_mapping, size = 1.2, alpha = 0.6)
    }

    # Add facets if specified
    if (!is.null(facet_by) && length(facet_by) > 0) {
        facet_formula <- stats::as.formula(paste("~", paste(facet_by, collapse = " + ")))
        p <- p + ggplot2::facet_wrap(facet_formula, nrow = 1)
    }

    # Configure scales
    # Y-axis: percent format
    p <- p + ggplot2::scale_y_continuous(
        labels = scales::percent_format(accuracy = 0.1),
        limits = c(0, 1)
    )

    # X-axis: log or linear
    if (log_x) {
        p <- p + ggplot2::scale_x_continuous(
            trans = "log10",
            breaks = scales::breaks_log(n = 6)
        )
    } else {
        p <- p + ggplot2::scale_x_continuous(
            expand = ggplot2::expansion(mult = c(0.01, 0.03))
        )
    }

    # Labels
    p <- p + ggplot2::labs(
        x = "Communication Rounds",
        y = "Test Accuracy",
        title = title
    )

    # Theme
    p <- p +
        ggplot2::theme_minimal(base_size = 12) +
        ggplot2::theme(
            legend.position = "bottom",
            legend.title = ggplot2::element_blank(),
            panel.grid.minor.x = ggplot2::element_blank(),
            panel.grid.minor.y = ggplot2::element_line(linewidth = 0.2, colour = "grey90"),
            plot.title = ggplot2::element_text(face = "bold")
        )

    p
}

#' Make MNIST History for Plot
#'
#' Reads and prepares MNIST metrics CSV for plotting.
#'
#' @param csv_path Path to CSV file (default: "inst/reproduction_outputs/metrics_mnist.csv")
#' @return Data frame with columns needed for plotting
#' @export
make_mnist_history_for_plot <- function(csv_path = "inst/reproduction_outputs/metrics_mnist.csv") {
    if (!file.exists(csv_path)) {
        stop(sprintf("CSV file not found: %s", csv_path))
    }

    df <- utils::read.csv(csv_path, stringsAsFactors = FALSE)
    df <- subset(df, dataset == "MNIST")

    # Keep only columns needed for plotting if present
    keep <- intersect(
        c("dataset", "partition", "round", "test_acc", "method", "E", "B", "lr"),
        names(df)
    )

    df[keep]
}

#' Save Plot
#'
#' Saves a ggplot to PNG, PDF, and optionally SVG formats.
#'
#' @param p ggplot object
#' @param out_png Output PNG path
#' @param out_pdf Output PDF path
#' @param out_svg Optional output SVG path
#' @param width Plot width in inches (default: 7)
#' @param height Plot height in inches (default: 4.5)
#' @param dpi Resolution for PNG (default: 200)
#' @export
save_plot <- function(p, out_png, out_pdf, out_svg = NULL, width = 7, height = 4.5, dpi = 200) {
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("ggplot2 package required. Install with: install.packages('ggplot2')")
    }

    # Ensure output directories exist
    for (path in c(out_png, out_pdf, out_svg)) {
        if (!is.null(path)) {
            dir_path <- dirname(path)
            if (!dir.exists(dir_path)) {
                dir.create(dir_path, recursive = TRUE)
            }
        }
    }

    # Save PNG
    ggplot2::ggsave(
        filename = out_png,
        plot = p,
        width = width,
        height = height,
        dpi = dpi,
        units = "in"
    )

    # Save PDF with cairo
    ggplot2::ggsave(
        filename = out_pdf,
        plot = p,
        width = width,
        height = height,
        device = grDevices::cairo_pdf,
        units = "in"
    )

    # Save SVG if requested
    if (!is.null(out_svg)) {
        if (!requireNamespace("svglite", quietly = TRUE)) {
            message("Installing svglite for SVG export...")
            utils::install.packages("svglite")
        }
        svglite::svglite(file = out_svg, width = width, height = height)
        print(p)
        grDevices::dev.off()
    }

    invisible(TRUE)
}

#' Plot Flower Style (Matches Reference)
#'
#' Creates a plot matching the Flower reference implementation style.
#' Mirrors reference/fedavg_mnist_flwr/utils.py:plot_metric_from_history()
#'
#' @param history Data frame with columns: round, test_acc
#' @param expected_maximum Expected maximum accuracy from paper (e.g., 0.995)
#' @param baseline Paper's baseline accuracy (default: 0.99)
#' @param title Plot title (default: "Centralized Validation - MNIST")
#' @param out_path Output path for PNG (default: "centralized_metrics.png")
#' @return Invisible TRUE
#' @export
#' @references
#' Mirrored from reference/fedavg_mnist_flwr/utils.py:18-72
plot_flower_style <- function(history,
                              expected_maximum,
                              baseline = 0.99,
                              title = "Centralized Validation - MNIST",
                              out_path = "centralized_metrics.png") {
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("ggplot2 package required. Install with: install.packages('ggplot2')")
    }

    # Check required columns
    if (!all(c("round", "test_acc") %in% names(history))) {
        stop("history must have columns: round, test_acc")
    }

    # Create plot matching Flower reference
    # reference/fedavg_mnist_flwr/utils.py:44-69
    p <- ggplot2::ggplot(history, ggplot2::aes(x = round, y = test_acc)) +
        # Main line
        ggplot2::geom_line(color = "blue", linewidth = 0.8) +

        # Expected maximum (red dashed)
        ggplot2::geom_hline(
            yintercept = expected_maximum,
            linetype = "dashed",
            color = "red",
            linewidth = 0.6
        ) +
        ggplot2::annotate(
            "text",
            x = Inf, y = expected_maximum,
            label = sprintf("Paper's best result @%.3f", expected_maximum),
            hjust = 1.05, vjust = -0.5,
            color = "red", size = 3
        ) +

        # Baseline (silver)
        ggplot2::geom_hline(
            yintercept = baseline,
            linetype = "solid",
            color = "grey70",
            linewidth = 0.5
        ) +
        ggplot2::annotate(
            "text",
            x = Inf, y = baseline,
            label = sprintf("Paper's baseline @%.4f", baseline),
            hjust = 1.05, vjust = -0.5,
            color = "grey50", size = 3
        ) +

        # Fixed Y-axis range [0.97, 1] as in Flower
        ggplot2::scale_y_continuous(
            limits = c(0.97, 1.0),
            breaks = seq(0.97, 1.0, by = 0.01)
        ) +

        # Labels
        ggplot2::labs(
            x = "Rounds",
            y = "Accuracy",
            title = title
        ) +

        # Theme matching Flower style
        ggplot2::theme_minimal(base_size = 11) +
        ggplot2::theme(
            plot.title = ggplot2::element_text(hjust = 0.5, face = "bold"),
            panel.grid.minor = ggplot2::element_line(linewidth = 0.2),
            aspect.ratio = 1.0 # Square aspect ratio as in Flower
        )

    # Ensure output directory exists
    dir_path <- dirname(out_path)
    if (!dir.exists(dir_path)) {
        dir.create(dir_path, recursive = TRUE)
    }

    # Save plot
    ggplot2::ggsave(
        filename = out_path,
        plot = p,
        width = 6,
        height = 6,
        dpi = 150,
        units = "in"
    )

    cat(sprintf("Saved Flower-style plot to: %s\n", out_path))
    invisible(TRUE)
}
#' Plot MNIST Figure 2 (Paper Style)
#'
#' Reproduces Figure 2 from McMahan et al. (2017) with exact styling.
#'
#' @param history Data frame with columns: round, test_acc, partition, E, B
#' @param target Target accuracy (default: 0.99)
#' @param smooth_window Window size for moving average smoothing (default: 0, no smoothing).
#' @param running_max Logical; if TRUE, plot the best-so-far accuracy (monotonically non-decreasing).
#' @return A ggplot object
#' @export
plot_mnist_figure2 <- function(history, target = 0.99, smooth_window = 0, running_max = FALSE) {
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("ggplot2 package required.")
    }

    # Ensure factors and formatting
    df <- history

    # Apply running max if requested
    if (running_max) {
        # Group by B, E, partition, method
        df$group_id <- paste(df$partition, df$E, df$B, df$method)
        df_list <- split(df, df$group_id)

        df_list <- lapply(df_list, function(d) {
            d <- d[order(d$round), ]
            d$test_acc <- cummax(d$test_acc)
            d
        })
        df <- do.call(rbind, df_list)
    }

    # Apply smoothing if requested
    if (smooth_window > 1) {
        # Simple moving average helper
        ma <- function(x, n = 5) {
            stats::filter(x, rep(1 / n, n), sides = 2)
        }

        # Apply per group
        if (!"group_id" %in% names(df)) {
            df$group_id <- paste(df$partition, df$E, df$B, df$method)
        }
        df_list <- split(df, df$group_id)

        df_list <- lapply(df_list, function(d) {
            d <- d[order(d$round), ]
            if (nrow(d) >= smooth_window) {
                d$test_acc <- as.numeric(ma(d$test_acc, n = smooth_window))
            }
            d
        })
        df <- do.call(rbind, df_list)
        df <- df[!is.na(df$test_acc), ] # Remove NA from edges
    }

    # Handle B=Inf for display
    df$B_label <- ifelse(is.infinite(df$B), "B=\u221E", paste0("B=", df$B))

    # Create combined label for legend
    # Format: "B=10 E=1", "B=10 E=5", etc.
    # We want specific ordering: B=10, B=50, B=Inf
    df$B_factor <- factor(df$B_label, levels = c("B=10", "B=50", "B=\u221E"))
    df$E_factor <- factor(paste0("E=", df$E), levels = c("E=1", "E=5", "E=20"))

    # Create interaction factor for manual coloring/linetypes
    # We need to map specific combinations to specific styles

    # Colors: B=10 (Red), B=50 (Orange), B=Inf (Blue)
    # Linetypes: E=1 (Solid), E=5 (Dashed), E=20 (Dotted)

    # Define colors
    colors <- c(
        "B=10" = "#e41a1c", # Red
        "B=50" = "#ff7f00", # Orange
        "B=\u221E" = "#377eb8" # Blue
    )

    # Define linetypes
    linetypes <- c(
        "E=1" = "solid",
        "E=5" = "dashed",
        "E=20" = "dotted"
    )

    # Plot
    p <- ggplot2::ggplot(df, ggplot2::aes(x = round, y = test_acc)) +
        # Target line (grey, behind)
        ggplot2::geom_hline(yintercept = target, color = "grey70", linewidth = 0.5) +

        # Main lines
        ggplot2::geom_line(
            ggplot2::aes(
                color = B_factor,
                linetype = E_factor,
                group = interaction(B_factor, E_factor)
            ),
            linewidth = 0.8
        ) +

        # Faceting
        ggplot2::facet_wrap(~partition, labeller = ggplot2::labeller(partition = function(x) paste("MNIST CNN", x))) +

        # Scales
        ggplot2::scale_color_manual(values = colors, name = NULL) +
        ggplot2::scale_linetype_manual(values = linetypes, name = NULL) +
        ggplot2::scale_y_continuous(
            limits = c(0.97, 1.0),
            breaks = seq(0.97, 1.0, by = 0.01),
            expand = c(0, 0)
        ) +
        ggplot2::scale_x_continuous(
            limits = c(0, 1000),
            breaks = seq(0, 1000, by = 200),
            expand = c(0, 0)
        ) +

        # Labels
        ggplot2::labs(
            x = "Communication Rounds",
            y = "Test Accuracy"
        ) +

        # Theme
        ggplot2::theme_bw() +
        ggplot2::theme(
            panel.grid = ggplot2::element_blank(), # Remove grid
            strip.background = ggplot2::element_blank(), # Clean strip
            strip.text = ggplot2::element_text(size = 12),
            legend.position = c(0.8, 0.3), # Inside plot (approx)
            legend.key = ggplot2::element_blank(),
            legend.background = ggplot2::element_blank(),
            axis.text = ggplot2::element_text(color = "black"),
            axis.ticks = ggplot2::element_line(color = "black")
        ) +

        # Custom legend guide to merge color and linetype?
        # Actually, standard ggplot separates them if they use different variables.
        # To match the paper exactly (combined legend), we might need a combined variable.
        # But separate legends is often clearer.
        # The paper has a combined list. Let's try to combine them.
        NULL

    # To get a combined legend like the paper:
    # We can map 'color' to the interaction, but then we need to specify colors for all 9 combinations.
    # Simpler approach: Keep separate for now, or use override.aes.
    # The paper shows a single list: "B=10 E=1", "B=10 E=5", etc.

    # Let's refine for combined legend
    df$legend_label <- factor(
        paste(df$B_label, df$E_factor),
        levels = c(
            "B=10 E=1", "B=10 E=5", "B=10 E=20",
            "B=50 E=1", "B=50 E=5", "B=50 E=20",
            "B=\u221E E=1", "B=\u221E E=5", "B=\u221E E=20"
        )
    )

    # Manual scale for 9 items
    # Colors repeat: Red, Red, Red, Orange, Orange, Orange, Blue, Blue, Blue
    # Linetypes repeat: Solid, Dashed, Dotted, Solid, Dashed, Dotted...

    combined_colors <- c(
        rep("#e41a1c", 3), # Red
        rep("#ff7f00", 3), # Orange
        rep("#377eb8", 3) # Blue
    )
    names(combined_colors) <- levels(df$legend_label)

    combined_linetypes <- c(
        rep(c("solid", "dashed", "dotted"), 3)
    )
    names(combined_linetypes) <- levels(df$legend_label)

    # Re-plot with combined legend
    p <- ggplot2::ggplot(df, ggplot2::aes(x = round, y = test_acc)) +
        ggplot2::geom_hline(yintercept = target, color = "grey70", linewidth = 0.5) +
        ggplot2::geom_line(
            ggplot2::aes(
                color = legend_label,
                linetype = legend_label
            ),
            linewidth = 0.8
        ) +
        ggplot2::facet_wrap(~partition, labeller = ggplot2::labeller(partition = function(x) paste("MNIST CNN", x))) +
        ggplot2::scale_color_manual(values = combined_colors, name = NULL) +
        ggplot2::scale_linetype_manual(values = combined_linetypes, name = NULL) +
        ggplot2::scale_y_continuous(
            limits = c(0.97, 1.0),
            breaks = seq(0.97, 1.0, by = 0.01),
            expand = c(0, 0)
        ) +
        ggplot2::scale_x_continuous(
            limits = c(0, 1000),
            breaks = seq(0, 1000, by = 200),
            expand = c(0, 0)
        ) +
        ggplot2::labs(
            x = "Communication Rounds",
            y = "Test Accuracy"
        ) +
        ggplot2::theme_bw() +
        ggplot2::theme(
            panel.grid = ggplot2::element_blank(),
            strip.background = ggplot2::element_blank(),
            strip.text = ggplot2::element_text(size = 11),
            legend.position = c(0.75, 0.35), # Inside, bottom right
            legend.key = ggplot2::element_blank(),
            legend.background = ggplot2::element_blank(),
            legend.text = ggplot2::element_text(size = 8),
            legend.key.width = ggplot2::unit(1.5, "cm"), # Longer lines in legend
            axis.text = ggplot2::element_text(color = "black", size = 10),
            axis.ticks = ggplot2::element_line(color = "black"),
            axis.text.y = ggplot2::element_text(angle = 90, hjust = 0.5) # Vertical Y labels like paper
        )

    return(p)
}
