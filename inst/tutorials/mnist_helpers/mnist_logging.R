#' Append Metrics to CSV
#'
#' Appends training metrics to a CSV file with paper-accurate columns.
#'
#' @param dataset Dataset name (e.g., "MNIST").
#' @param model Model name ("2NN" or "CNN").
#' @param partition Partition type ("IID" or "nonIID").
#' @param method Method name ("FedAvg" or "FedSGD").
#' @param round Current round.
#' @param test_acc Test accuracy.
#' @param chosen_lr Learning rate used.
#' @param E Number of local epochs.
#' @param B Batch size (can be "Inf" for FedSGD).
#' @param C Client fraction.
#' @param clients_selected Number of clients selected.
#' @param u Communication efficiency statistic (6E/B or 1 if B=Inf).
#' @param target Target accuracy for rounds-to-target.
#' @param rounds_to_target Rounds to reach target (or NA).
#' @param path Path to the CSV file.
#' @export
append_metrics <- function(dataset, model, partition, method, round, test_acc,
                           chosen_lr, E, B, C, clients_selected, u, target,
                           rounds_to_target = NA, path = "inst/reproduction_outputs/metrics_mnist.csv") {
    # Check if file exists to write header
    append <- file.exists(path)

    df <- data.frame(
        timestamp = Sys.time(),
        dataset = dataset,
        model = model,
        partition = partition,
        method = method,
        round = round,
        test_acc = test_acc,
        chosen_lr = chosen_lr,
        E = E,
        B = B,
        C = C,
        clients_selected = clients_selected,
        u = u,
        target = target,
        rounds_to_target = rounds_to_target,
        stringsAsFactors = FALSE
    )

    utils::write.table(
        df,
        file = path,
        append = append,
        sep = ",",
        row.names = FALSE,
        col.names = !append,
        quote = FALSE
    )
}
