
      suppressPackageStartupMessages({
          devtools::load_all(quiet = TRUE)
          source("inst/tutorials/mnist_helpers/mnist_data.R")
          source("inst/tutorials/mnist_helpers/mnist_partitions.R")
          source("inst/tutorials/mnist_helpers/mnist_models.R")
          source("inst/tutorials/mnist_helpers/mnist_training.R")
          source("inst/tutorials/mnist_helpers/mnist_fedavg.R")
          source("inst/tutorials/mnist_helpers/mnist_logging.R")
          source("inst/tutorials/mnist_helpers/mnist_plotting.R")
      })

      # Load data once per chunk
      cat("Loading data...
")
      ds_train <- mnist_ds(train = TRUE, download = TRUE)
      ds_test <- mnist_ds(train = FALSE, download = TRUE)
      labels_train <- mnist_labels(ds_train)

      cat("Starting FedAvg chunk...
")
      run_fedavg_mnist(
          ds_train = ds_train,
          ds_test = ds_test,
          labels_train = labels_train,
          model_fn = "cnn",
          partition = "nonIID",
          K = 100,
          C = 0.1,
          E = 5,
          batch_size = 10,
          lr_grid = c(0.03, 0.05, 0.1),
          target = 0.99,
          rounds = 50,
          seed = 123,
          device = "cpu",
          log_file = "inst/reproduction_outputs/metrics_mnist.csv",
          checkpoint_dir = "inst/reproduction_outputs/checkpoints/nonIID_E5_B10",
          start_round = 1
      )
    
