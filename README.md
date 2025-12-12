# fedavgR: Federated Learning with Federated Averaging in R

An implementation of **Federated Learning (FL)** using the **Federated Averaging (FedAvg)** algorithm in R using `torch`.

This package serves two purposes:
1.  **Paper Reproduction**: Exact reproduction of experiments from McMahan et al. (2017) on MNIST (IID and Non-IID).
2.  **General Framework**: A flexible simulation framework (`fedavg_simulation`) to run FedAvg on your own datasets and models.

## 📦 Installation

This package requires `torch` and `torchvision`.

```r
# 1. Install dependencies
install.packages(c("torch", "torchvision", "devtools", "ggplot2", "dplyr", "readr", "tidyr", "remotes"))

# 2. Install Torch (downloads libtorch)
torch::install_torch()

# 3. Install fedavgR
# Run this from the root of the repository
remotes::install_local(".", force = TRUE)
```

## 📊 Paper Reproduction

There are **two ways** to reproduce the paper results, depending on your needs:

### Option 1: Robust Runner (Recommended for Full Reproduction)

**Use this for:**
- ✅ Full paper reproduction (all 1000 rounds)
- ✅ Unattended overnight/weekend runs
- ✅ Limited memory environments
- ✅ Production/final runs

The robust runner executes experiments in **50-round chunks** with automatic checkpointing and memory cleanup between chunks. This prevents OOM crashes and allows safe resumption if interrupted.

```bash
# Run all experiments in background
nohup Rscript inst/tutorials/run_robust_experiments.R > robust.log 2>&1 &

# Monitor progress
tail -f robust.log
tail -f inst/reproduction_outputs/metrics_mnist.csv
```

**Features:**
- Automatic checkpoint saving and resume
- Memory cleanup between chunks (prevents OOM)
- Handles multiple experiment configurations in sequence
- Validates checkpoints before resuming
- Retry logic for transient failures

### Option 2: Direct Script Execution (Quick Testing)

**Use this for:**
- ✅ Quick testing and debugging
- ✅ Small experiments (<100 rounds)
- ✅ Subset of configurations
- ⚠️ **Not recommended** for full 1000-round experiments (may crash)

```bash
# Quick Mode: Subset of configs (E=1,5,20 with B=10)
FEDAVGR_QUICK=1 Rscript inst/tutorials/paper_reproduction_cnn.R

# Full Mode: All configs (may crash on long runs)
Rscript inst/tutorials/paper_reproduction_cnn.R
```

**Quick Mode Configurations (B=10 only):**

| Partition | E (Epochs) | B (Batch Size) | Rounds |
| :---: | :---: | :---: | :---: |
| IID | 1, 5, 20 | 10 | 1000 |
| Non-IID | 1, 5, 20 | 10 | 1000 |

**⚠️ Warning:** Running 1000 rounds in a single process may cause OOM errors. For production runs, use the **robust runner** (Option 1).

### 📊 Example Outputs

Results are saved to `inst/reproduction_outputs/`:
*   **`metrics_mnist.csv`**: Raw logs of every round (accuracy, loss, etc.).
*   **`figure2_reproduction.png`**: Plot comparing FedAvg vs FedSGD.

### Figure 2: Test Set Accuracy vs Communication Rounds for MNIST CNN (IID)

<table>
  <tr>
    <td><img src="inst/reproduction_outputs/figure2_reproduction_IID.png" alt="MNIST accuracy vs rounds (reproduction)" width="465"></td>
    <td><img src="inst/reproduction_outputs/figure2_mcmahan_et_al_2017_IID.png" alt="MNIST accuracy vs rounds (paper)" width="400"></td>
  </tr>
  <tr>
    <td align="center"><em>Reproduction of Figure 2</em></td>
    <td align="center"><em>McMahan et al. (2017) Figure 2</em></td>
  </tr>
</table>

### Table 2: Communication Rounds to 99% Test Accuracy

Reproduction from McMahan et al. (2017)

|  CNN   | E  | B  |  u  |    IID (Reproduction)     | IID (McMahan et al. (2017)) |
|:------:|:--:|:--:|:---:|:----------:|:-------:|
| FedSGD | 1  | ∞  | 1 |    702     |   626    |
| FedAvg | 5  | ∞  | 5 | 230 (3.1×) |   179 (3.5x)    |
| FedAvg | 20 | ∞  | 20 | 128 (5.5×) |   234 (2.7x)    |
| FedAvg | 1  | 10 | 60 | 45 (15.6×) |   34 (18.4x)    |
| FedAvg | 5  | 10 | 300 | 26 (27.0×) |    20 (31.3x)     |


**Note**: Values show rounds to target (speedup vs FedSGD baseline).

## 📁 Repository Structure

```
fedavgR/
├── R/                          # Core package code
│   ├── fedavg_simulation.R    # Generic FedAvg framework
│   ├── fedavg.R               # Core aggregation function
│   ├── train_generic.R        # Generic client training
│   ├── partitions.R           # Data partitioning utilities
│   └── ...
├── inst/
│   ├── tutorials/             # Paper reproduction scripts
│   │   ├── paper_reproduction_cnn.R      # Direct CNN experiments
│   │   ├── paper_reproduction_2nn.R      # Direct 2NN experiments
│   │   ├── run_robust_experiments.R      # Robust chunked runner
│   │   ├── generate_figure2_from_logs.R  # Plot generation
│   │   ├── generate_table2_from_logs.R   # Table generation
│   │   └── mnist_helpers/                # MNIST-specific utilities
│   │       ├── mnist_data.R
│   │       ├── mnist_models.R
│   │       ├── mnist_training.R
│   │       ├── mnist_fedavg.R
│   │       ├── mnist_partitions.R
│   │       ├── mnist_plotting.R
│   │       └── README.md
│   └── reproduction_outputs/  # Reproduction results
│       ├── metrics_mnist.csv           # Main experiment log
│       ├── metrics_mnist_B_inf.csv     # B=Inf experiments
│       ├── figure2_reproduction_*.png  # Generated plots
│       ├── table2_reproduction.*       # Generated tables
│       └── checkpoints/                # Experiment checkpoints
├── tests/                     # Unit tests
└── README.md
```

## 🚀 General Usage

You can use `fedavgR` to run federated learning simulations on your own data.

### 1. Using the Generic Framework

Use `fedavg_simulation()` to run FedAvg with custom models and datasets.

```r
library(fedavgR)
library(torch)

# 1. Define your model generator
model_gen <- function() {
  nn_sequential(
    nn_linear(10, 20),
    nn_relu(),
    nn_linear(20, 1)
  )
}

# 2. Prepare client datasets (list of torch datasets)
# Example: 10 clients with random data
clients <- lapply(1:10, function(i) {
  tensor_dataset(torch_randn(100, 10), torch_randn(100, 1))
})

# 3. Define evaluation function
eval_fn <- function(model, device) {
  model$eval()
  # ... compute metrics ...
  list(loss = 0.5) # Return named list
}

# 4. Run Simulation
results <- fedavg_simulation(
  client_datasets = clients,
  model_generator = model_gen,
  evaluation_fn = eval_fn,
  rounds = 50,
  C = 0.1,    # Select 10% of clients per round
  E = 5,      # 5 local epochs
  batch_size = 32
)

print(results$history)
```

### 2. Using the MNIST Wrapper

For quick experiments on MNIST, use the built-in wrapper:

```r
# Load Data
ds_train <- mnist_ds(root = "data", train = TRUE, download = TRUE)
ds_test <- mnist_ds(root = "data", train = FALSE, download = TRUE)
labels <- mnist_labels(ds_train)

# Run FedAvg
res <- run_fedavg_mnist(
  ds_train = ds_train,
  ds_test = ds_test,
  labels_train = labels,
  model_fn = "cnn",       # "cnn" or "2nn"
  partition = "nonIID",   # "IID" or "nonIID"
  K = 100,                # Total clients
  C = 0.1,                # Fraction selected (0.1 = 10 clients)
  E = 5,                  # Local epochs
  batch_size = 50,        # Local batch size (Inf for FedSGD)
  lr_grid = c(0.01, 0.05, 0.1), # LR selection grid
  rounds = 100
)
```

## 📚 Reference

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017).
[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629).
*Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*.

## License

MIT License
