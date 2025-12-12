# MNIST Helper Functions

This directory contains MNIST-specific implementations for reproducing results from McMahan et al. (2017) "Communication-Efficient Learning of Deep Networks from Decentralized Data."

## Overview

These functions provide a **specialized implementation** of Federated Averaging tailored specifically for MNIST experiments in the paper. While the `fedavgR` package includes a generic `fedavg_simulation()` framework, the paper reproduction requires additional features not present in the generic implementation.

## Why a Separate MNIST Implementation?

The paper reproduction scripts use `run_fedavg_mnist()` instead of the generic `fedavg_simulation()` because they require:

1. **One-Time Learning Rate Selection** - Grid search over learning rates at initialization (paper-specific)
2. **Checkpoint/Resume Capability** - Save and resume experiments for robustness against crashes
3. **Paper-Specific Logging** - Rich metadata (dataset, model, partition, method, E, B, C, u, target, RTT)
4. **Rounds-to-Target Tracking** - Calculate when target accuracy is reached (for Table 2)
5. **MNIST-Specific Partitioning** - IID vs non-IID (2 shards per client) data splits
6. **Fixed Hyperparameters** - Paper uses fixed LR after selection, not per-round scheduling

These features are essential for **exact paper reproduction** but would make the generic framework too specialized.

## Architecture

```
mnist_helpers/
├── mnist_data.R         # Dataset loading (mnist_ds, mnist_labels)
├── mnist_models.R       # Model architectures (mnist_cnn, mnist_mlp)
├── mnist_training.R     # Client training (client_train_mnist)
├── mnist_fedavg.R       # Main experiment runner (run_fedavg_mnist)
├── mnist_partitions.R   # Data partitioning (mnist_shards_split, iid_split)
├── mnist_plotting.R     # Plotting utilities (plot_mnist_figure2, etc.)
└── README.md           # This file
```

## Files

### Core Experiment Functions

- **`mnist_fedavg.R`** - Main experiment runner
  - `run_fedavg_mnist()` - Runs full FedAvg experiment with paper-specific features
  - `rounds_to_target()` - Calculates rounds to reach target accuracy (with interpolation)

### Data & Models

- **`mnist_data.R`** - Dataset utilities
  - `mnist_ds()` - Loads MNIST dataset using torchvision
  - `mnist_labels()` - Extracts labels from dataset

- **`mnist_models.R`** - Neural network architectures
  - `mnist_cnn()` - Convolutional Neural Network (2 conv layers + 2 FC layers)
  - `mnist_mlp()` - Multi-Layer Perceptron (2 hidden layers)

### Training & Partitioning

- **`mnist_training.R`** - Client-side training
  - `client_train_mnist()` - Trains a model on client data for E epochs
  - `eval_mnist_accuracy()` - Evaluates model accuracy on test set

- **`mnist_partitions.R`** - Data partitioning strategies
  - `mnist_shards_split()` - Non-IID partitioning (2 shards per client, sorted by label)
  - `iid_split()` - IID partitioning (random assignment)

### Visualization

- **`mnist_plotting.R`** - Plotting utilities
  - `plot_mnist_figure2()` - Generates Figure 2 with paper-specific styling
  - `plot_comm_rounds()` - Generic accuracy vs rounds plot
  - `save_plot()` - Saves plots in multiple formats

## Usage

### Paper Reproduction (Recommended)

The paper reproduction scripts automatically source these helpers:

```r
# Run CNN experiments (IID and Non-IID)
Rscript inst/tutorials/paper_reproduction_cnn.R

# Run 2NN experiments
Rscript inst/tutorials/paper_reproduction_2nn.R
```

### Direct Usage

You can also use `run_fedavg_mnist()` directly:

```r
# Load helpers
devtools::load_all()
source("inst/tutorials/mnist_helpers/mnist_data.R")
source("inst/tutorials/mnist_helpers/mnist_models.R")
source("inst/tutorials/mnist_helpers/mnist_training.R")
source("inst/tutorials/mnist_helpers/mnist_fedavg.R")
source("inst/tutorials/mnist_helpers/mnist_partitions.R")

# Load MNIST data
ds_train <- mnist_ds(root = "data", train = TRUE)
ds_test <- mnist_ds(root = "data", train = FALSE)
labels_train <- mnist_labels(ds_train)

# Run experiment
result <- run_fedavg_mnist(
    ds_train = ds_train,
    ds_test = ds_test,
    labels_train = labels_train,
    model_fn = "cnn",           # or "2nn"
    partition = "nonIID",       # or "IID"
    K = 100,                    # Number of clients
    C = 0.1,                    # Fraction selected per round
    E = 5,                      # Local epochs
    batch_size = 10,            # Local batch size (Inf for FedSGD)
    lr_grid = c(0.03, 0.1),     # Learning rate grid search
    target = 0.99,              # Target accuracy for RTT
    rounds = 1000,              # Communication rounds
    seed = 123,
    log_file = "metrics.csv",   # Optional: incremental logging
    checkpoint_dir = "ckpts"    # Optional: checkpointing
)

# Access results
history <- result$history
final_params <- result$params
```

### With Checkpointing & Resume

For long-running experiments:

```r
# Initial run (will save checkpoints)
result <- run_fedavg_mnist(
    # ... parameters ...
    rounds = 1000,
    checkpoint_dir = "checkpoints/cnn_iid",
    log_file = "logs/cnn_iid.csv"
)

# Resume from round 500 if crashed
result <- run_fedavg_mnist(
    # ... same parameters ...
    rounds = 1000,
    checkpoint_dir = "checkpoints/cnn_iid",
    log_file = "logs/cnn_iid.csv",
    start_round = 501  # Resume from here
)
```

## Using with Generic Framework

While these helpers are designed for paper reproduction, you can adapt them for use with the generic `fedavg_simulation()`:

```r
library(fedavgR)

# Load MNIST helpers
source("inst/tutorials/mnist_helpers/mnist_data.R")
source("inst/tutorials/mnist_helpers/mnist_models.R")
source("inst/tutorials/mnist_helpers/mnist_partitions.R")

# Load and partition data
ds_train <- mnist_ds(root = "data", train = TRUE)
ds_test <- mnist_ds(root = "data", train = FALSE)
labels <- mnist_labels(ds_train)

# Partition into clients
client_indices <- mnist_shards_split(labels, K = 100, seed = 123)

# Create client datasets
client_datasets <- lapply(client_indices, function(idx) {
    dataset_subset(ds_train, idx)
})

# Define evaluation function
eval_fn <- function(model, device) {
    model$eval()
    acc <- eval_mnist_accuracy(model, ds_test, device)
    list(accuracy = acc)
}

# Use generic framework
result <- fedavg_simulation(
    client_datasets = client_datasets,
    model_generator = mnist_cnn,
    evaluation_fn = eval_fn,
    rounds = 100,
    C = 0.1,
    E = 5,
    batch_size = 50,
    lr_scheduler = function(r) 0.1,  # Fixed LR
    seed = 123
)
```

**Note:** This approach loses paper-specific features (LR grid search, checkpointing, rich logging, RTT tracking).

## Key Differences: `run_fedavg_mnist()` vs `fedavg_simulation()`

| Feature | `run_fedavg_mnist()` | `fedavg_simulation()` |
|---------|---------------------|----------------------|
| **LR Selection** | Grid search at start | Per-round scheduler |
| **Checkpointing** | ✅ Save/resume | ❌ No support |
| **Logging Format** | 15+ fields (paper-specific) | Basic metrics only |
| **RTT Tracking** | ✅ Rounds-to-target | ❌ Not included |
| **Data Partitioning** | Built-in (IID/non-IID) | Pre-partitioned input |
| **Use Case** | Paper reproduction | General experiments |

## Adapting for Your Dataset

To create similar helpers for your own dataset:

1. **Data Loading** - Create `your_dataset_ds()` function
2. **Models** - Define `your_model()` architecture
3. **Training** - Implement `client_train_your_dataset()` if needed
4. **Partitioning** - Create custom partitioning logic (optional)
5. **Evaluation** - Define metrics function for `fedavg_simulation()`

You can then use the **generic `fedavg_simulation()`** framework directly, or create a specialized wrapper like `run_fedavg_mnist()` if you need additional features.

## References

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). Communication-efficient learning of deep networks from decentralized data. In *Artificial Intelligence and Statistics* (pp. 1273-1282). PMLR.
