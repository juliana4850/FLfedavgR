#' Save Plot
#'
#' Saves a ggplot to PNG and optionally SVG formats.
#'
#' @param p ggplot object
#' @param out_png Output PNG path
#' @param out_svg Optional output SVG path
#' @param width Plot width in inches (default: 7)
#' @param height Plot height in inches (default: 4.5)
#' @param dpi Resolution for PNG (default: 200)
#' @export
save_plot <- function(p, out_png, out_svg = NULL, width = 7, height = 4.5, dpi = 200) {
    if (!requireNamespace("ggplot2", quietly = TRUE)) {
        stop("ggplot2 package required. Install with: install.packages('ggplot2')")
    }

    # Ensure output directories exist
    for (path in c(out_png, out_svg)) {
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

#' Plot MNIST Figure 2 (Paper Style)
#'
#' Reproduces Figure 2 from McMahan et al. (2017) with exact styling.
#'
#' @param history Data frame with columns: round, test_acc, partition, E, B
#' @param target Target accuracy (default: 0.99)
#' @param stride Plot every n-th point (default: 50). 1 means plot all points.
#' @param running_max Logical; if TRUE, plot the best-so-far accuracy (monotonically non-decreasing).
#' @return A ggplot object
#' @export
plot_mnist_figure2 <- function(history, target = 0.99, stride = 50, running_max = FALSE) {
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

    # Apply stride subsampling
    if (stride > 1) {
        if (!"group_id" %in% names(df)) {
            df$group_id <- paste(df$partition, df$E, df$B, df$method)
        }
        df_list <- split(df, df$group_id)

        df_list <- lapply(df_list, function(d) {
            d <- d[order(d$round), ]
            # Keep first, every stride-th, and last point
            n <- nrow(d)
            indices <- unique(c(1, seq(stride, n, by = stride), n))
            d[indices, ]
        })
        df <- do.call(rbind, df_list)
    }

    # Handle B=Inf for display
    df$B_label <- ifelse(is.infinite(df$B), "B=\u221E", paste0("B=", df$B))

    # Create combined label for legend
    df$B_factor <- factor(df$B_label, levels = c("B=10", "B=50", "B=\u221E"))
    df$E_factor <- factor(paste0("E=", df$E), levels = c("E=1", "E=5", "E=20"))

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
        NULL
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
