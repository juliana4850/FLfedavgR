# MNIST Helper Functions

This directory contains MNIST-specific helper functions used by the paper reproduction scripts.

## Purpose

These functions demonstrate how to use the generic `fedavgR` framework for a specific use case (MNIST). They serve as examples for users who want to apply FedAvg to their own datasets.

## Files

- **mnist_data.R** - MNIST dataset loaders (`mnist_ds`, `mnist_labels`)
- **mnist_models.R** - MNIST model architectures (`mnist_cnn`, `mnist_mlp`)
- **mnist_training.R** - MNIST-specific client training (`client_train_mnist`)
- **mnist_fedavg.R** - MNIST wrapper for FedAvg (`run_fedavg_mnist`)
- **mnist_partitions.R** - MNIST non-IID partitioning (`mnist_shards_split`)
- **mnist_logging.R** - MNIST metrics logging (`append_metrics`)
- **mnist_plotting.R** - MNIST plotting utilities

## Usage

These files are sourced by the paper reproduction scripts:
- `inst/tutorials/paper_reproduction_cnn.R`
- `inst/tutorials/paper_reproduction_2nn.R`

Example:
```r
# Load MNIST helpers
source("inst/tutorials/mnist_helpers/mnist_data.R")
source("inst/tutorials/mnist_helpers/mnist_models.R")
# ... etc

# Use them with the generic framework
library(fedavgR)

# Load data
ds_train <- mnist_ds(root = "data", train = TRUE)
labels <- mnist_labels(ds_train)

# Partition using MNIST-specific non-IID
client_indices <- mnist_shards_split(labels, K = 100)

# Create client datasets
client_datasets <- lapply(client_indices, function(idx) {
    dataset_subset(ds_train, idx)
})

# Use generic framework
result <- fedavg_simulation(
    client_datasets = client_datasets,
    model_generator = mnist_cnn,  # MNIST-specific model
    evaluation_fn = function(model, device) {
        # Evaluation logic
    },
    rounds = 100,
    C = 0.1,
    E = 5,
    batch_size = 50
)
```

## Note

These are **examples**, not part of the core `fedavgR` package API. For your own datasets, create similar helper functions adapted to your use case.
