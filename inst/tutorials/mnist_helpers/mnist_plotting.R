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
