test_that("plot_comm_rounds returns ggplot object", {
    skip_if_not_installed("ggplot2")
    skip_if_not_installed("scales")

    # Create synthetic history with two methods, E in {1,5}, both partitions, rounds 1:5
    history <- data.frame(
        round = rep(1:5, times = 4),
        test_acc = runif(20, 0.5, 0.95),
        method = rep(c("FedAvg", "FedSGD"), each = 10),
        E = rep(c(5, 1), each = 10),
        B = rep(c("10", "Inf"), each = 10),
        partition = rep(c("IID", "nonIID"), times = 10),
        stringsAsFactors = FALSE
    )

    # Create plot with target and target_band
    p <- plot_comm_rounds(
        history = history,
        target = 0.97,
        target_band = 0.002,
        facet_by = "partition",
        color_by = "method",
        linetype_by = "E",
        log_x = FALSE,
        show_points = FALSE
    )

    expect_true(inherits(p, "ggplot"))
})

test_that("plot_comm_rounds handles missing columns", {
    skip_if_not_installed("ggplot2")

    # Missing required column
    history <- data.frame(
        round = 1:3,
        test_acc = c(0.8, 0.9, 0.95)
    )

    expect_error(
        plot_comm_rounds(history, color_by = "method"),
        "Missing required columns"
    )
})

test_that("plot_from_csv reads and filters correctly", {
    # Create temporary CSV
    temp_csv <- tempfile(fileext = ".csv")

    df <- data.frame(
        dataset = c("MNIST", "MNIST", "CIFAR10"),
        partition = c("IID", "nonIID", "IID"),
        method = c("FedAvg", "FedAvg", "FedAvg"),
        round = c(1, 1, 1),
        test_acc = c(0.8, 0.75, 0.6),
        lr = c(0.1, 0.1, 0.1),
        E = c(5, 5, 5),
        B = c("10", "10", "10"),
        clients_selected = c(10, 10, 10),
        stringsAsFactors = FALSE
    )

    write.csv(df, temp_csv, row.names = FALSE)

    # Read and filter
    result <- plot_from_csv(temp_csv, filter_dataset = "MNIST")

    expect_equal(nrow(result), 2)
    expect_true(all(result$dataset == "MNIST"))
    expect_true(is.factor(result$partition))
    expect_equal(levels(result$partition), c("IID", "nonIID"))

    # Cleanup
    unlink(temp_csv)
})

test_that("save_plot creates output files", {
    skip_if_not_installed("ggplot2")

    # Create simple plot
    p <- ggplot2::ggplot(data.frame(x = 1:3, y = 1:3), ggplot2::aes(x, y)) +
        ggplot2::geom_line()

    # Temporary output paths
    temp_png <- tempfile(fileext = ".png")
    temp_pdf <- tempfile(fileext = ".pdf")

    # Save
    result <- save_plot(p, temp_png, temp_pdf, width = 5, height = 3)

    expect_true(file.exists(temp_png))
    expect_true(file.exists(temp_pdf))

    # Cleanup
    unlink(temp_png)
    unlink(temp_pdf)
})
