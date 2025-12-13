# Paper Reproduction Outputs

This directory contains outputs from reproducing McMahan et al. (2017) MNIST CNN and CIFAR-10 experiment results using the fedavgR package.

## Files

- **`metrics_mnist_cnn.csv`** - Example per-round metrics data from reproduction of subset of MNIST CNN experiments
- **`figure2_reproduction_{IID | nonIID}.png`** - Example reproduction of Figure 2 (test accuracy vs communication rounds) using `metrics_mnist_cnn.csv`
- **`table2_reproduction.{md | csv}`** - Example reproduction of Table 2 (rounds to 99% accuracy with speedups) using `metrics_mnist_cnn.csv`

- **`metrics_cifar10_lr15.csv`** - Example per-round metrics data from reproduction of CIFAR-10 FedAvg experiment with learning rate 0.15
- **`cifar10_lr15_final_model.pt`** - Example final model from reproduction of CIFAR-10 FedAvg experiment with learning rate 0.15

**Note**: Filenames may vary slightly depending on your run settings. The examples above illustrate the expected artifacts.

## Reproducing MNIST and CIFAR-10 Experiments

> **⚠️ Important**: To run the paper reproduction tutorials, you must **clone this repository**. The tutorials are not intended to be rerun from the installed package.

### Setup for Paper Reproduction

```bash
# Clone the repository
git clone https://github.com/juliana4850/FLfedavgR.git
cd FLfedavgR

# Install the package in development mode
R -e "devtools::install_local('.', force = TRUE)"
```

**Note**: Full runs are computationally intensive and can take many hours (or even days) on CPU. 

### To reproduce MNIST CNN experiments:

You can control runs via optional environment variables:
 - FEDAVGR_TEST=1 – run in test mode (5 rounds), default 0
 - FEDAVGR_QUICK=1 – run in quick mode (subset of experiments as shown in table below), default 0

| Partition | E (Epochs) | B (Batch Size) | Rounds |
| :---: | :---: | :---: | :---: |
| IID | 1, 5 | 10, ∞ | 1000 |
| Non-IID | 1, 5 | 10, ∞ | 1000 |

Use "test" mode first to verify the script runs:

```bash
FEDAVGR_TEST=1 Rscript inst/tutorials/demo_mnist_cnn.R
```

To run the full 1000 round MNIST CNN experiment reproductions:

```bash
# Run the full MNIST CNN experiment reproduction script
Rscript inst/tutorials/demo_mnist_cnn.R

## OR

# Use "quick" mode to run a subset of the MNIST CNN experiments
# Quick mode: reduced set (E ∈ {1,5}, B ∈ {10,Inf})
FEDAVGR_QUICK=1 Rscript inst/tutorials/demo_mnist_cnn.R
```

Regenerating plots and tables from logs:

```bash
# Generate figure 2 from CNN logs
LOG_FILE=inst/reproduction_outputs/metrics_mnist_cnn.csv Rscript inst/tutorials/generate_figure2_from_logs.R
# Generate table 2 from CNN logs
LOG_FILE=inst/reproduction_outputs/metrics_mnist_cnn.csv Rscript inst/tutorials/generate_table2_from_logs.R
```

### To reproduce individual CIFAR-10 experiments

You can control runs via optional environment variables:
 - LR – learning rate (double), default 0.15
 - ROUNDS – number of communication rounds (integer), default 3000
 - LOG_FILE – CSV output path, default "metrics_cifar10.csv"
 - MODEL_FILE – checkpoint path, default "cifar10_final_model.pt"

```bash
# Run the CIFAR-10 experiment reproduction script
# Example: Run for 2000 rounds with learning rate 0.15
LR=0.15 ROUNDS=2000 LOG_FILE=inst/reproduction_outputs/metrics_cifar10_lr15.csv MODEL_FILE=inst/reproduction_outputs/cifar10_lr15_final_model.pt Rscript inst/tutorials/demo_cifar10.R
```
Recommendation for quick testing: Run with ROUNDS=5 to verify the script runs. On CPU, this may take about ~1 minute per round.


To rerun the full 3000 round CIFAR-10 FedAvg experiments from the McMahan et al. (2017) paper:

```bash
# Run the full CIFAR-10 experiment reproduction script with LR=0.05 and ROUNDS=3000
LR=0.05 ROUNDS=3000 LOG_FILE=inst/reproduction_outputs/metrics_cifar10_lr05.csv MODEL_FILE=inst/reproduction_outputs/cifar10_lr05_final_model.pt Rscript inst/tutorials/demo_cifar10.R

# Run the full CIFAR-10 experiment reproduction script with LR=0.15 and ROUNDS=3000
LR=0.15 ROUNDS=3000 LOG_FILE=inst/reproduction_outputs/metrics_cifar10_lr15.csv MODEL_FILE=inst/reproduction_outputs/cifar10_lr15_final_model.pt Rscript inst/tutorials/demo_cifar10.R

# Run the full CIFAR-10 experiment reproduction script with LR=0.25 and ROUNDS=3000
LR=0.25 ROUNDS=3000 LOG_FILE=inst/reproduction_outputs/metrics_cifar10_lr25.csv MODEL_FILE=inst/reproduction_outputs/cifar10_lr25_final_model.pt Rscript inst/tutorials/demo_cifar10.R
```

## Reference

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). 
Communication-Efficient Learning of Deep Networks from Decentralized Data. 
*Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
https://arxiv.org/abs/1602.05629



