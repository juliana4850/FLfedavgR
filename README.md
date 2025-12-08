# fedavgR: Federated Averaging in R

A robust implementation of the **Federated Averaging (FedAvg)** algorithm in R using `torch`.

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

## 🔬 Paper Reproduction (McMahan et al. 2017)

We provide scripts to reproduce Figure 2 and Table 2 from the paper.

### 1. Robust Reproduction (Recommended)

For the full suite of experiments (Table 2), use the **robust runner**. This script:
*   Runs experiments in chunks (e.g., 50 rounds) to prevent memory leaks.
*   Automatically restarts the R process.
*   Saves checkpoints (`checkpoint_latest.rds`) to resume if interrupted.
*   Covers all 7 configurations (IID/Non-IID, various batch sizes).

```bash
# Run all experiments in the background
nohup Rscript run_robust_all.R > run_robust.log 2>&1 &

# Monitor progress
tail -f run_robust.log
```

### 2. MNIST Experiments Reproduction

Use `paper_reproduction_cnn.R` for single-process execution.

**⚠️ Note:** Running the full 1000 rounds in a single process on a personal machine may lead to Out-Of-Memory (OOM) errors. Use Quick Mode option to run only a subset of configurations. Use the robust runner for the full table.

```bash
# Quick Mode: 1000 rounds, subset of configs for both IID and Non-IID partitions
FEDAVGR_QUICK=1 Rscript inst/tutorials/paper_reproduction_cnn.R
```

**Quick Mode Configurations (B=10 only):**

| E (Epochs) | B (Batch Size) | Method |
| :---: | :---: | :---: |
| 1 | 10 | FedAvg |
| 5 | 10 | FedAvg |
| 20 | 10 | FedAvg |

```bash
# Full Mode: 1000 rounds, all configs
Rscript inst/tutorials/paper_reproduction_cnn.R
```

### 📊 Example Outputs

Results are saved to `inst/reproduction_outputs/`:
*   **`metrics_mnist.csv`**: Raw logs of every round (accuracy, loss, etc.).
*   **`figure2_reproduction.png`**: Plot comparing FedAvg vs FedSGD.
*   **`table2_reproduction.csv`**: Summary table of rounds to reach target accuracy.

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
│   ├── server_loop.R          # MNIST-specific wrapper
│   ├── train_*.R              # Training functions
│   ├── models_*.R             # Model architectures
│   ├── partitions.R           # Data partitioning
│   └── ...
├── inst/
│   ├── examples/              # Example scripts
│   ├── tutorials/             # Paper reproduction scripts
│   │   ├── paper_reproduction_cnn.R
│   │   └── paper_reproduction_2nn.R
│   └── reproduction_outputs/  # Reproduction results
│       └── metrics_mnist.csv  # Main log file
├── run_robust_all.R           # Robust experiment runner
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
