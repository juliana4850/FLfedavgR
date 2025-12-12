# Paper Reproduction Outputs

This directory contains outputs from reproducing McMahan et al. (2017) MNIST CNN and CIFAR-10 experiment results using the fedavgR package.

## Files

- **`metrics_mnist_cnn.csv`** - Example raw metrics data from reproduction of subset of MNIST CNN experiments
- **`figure2_reproduction_{IID | nonIID}.png`** - Example reproduction of Figure 2 (test accuracy vs communication rounds) using `metrics_mnist_cnn.csv`
- **`table2_reproduction.{md | csv}`** - Example reproduction of Table 2 (rounds to 99% accuracy with speedups) using `metrics_mnist_cnn.csv`

- **`metrics_cifar10_lr15.csv`** - Example raw metrics data from reproduction of CIFAR-10 FedAvg experiment with learning rate 0.15
- **`cifar10_lr15_final_model.pt`** - Example final model from reproduction of CIFAR-10 FedAvg experiment with learning rate 0.15


## Reproducing MNIST and CIFAR-10 Experiments

**Note**: The full reproductions of both MNIST and CIFAR-10 experiments require significant computational resources and is not recommended for same-day results.  

### To reproduce:

```r
# Run the MNIST CNN experiment reproduction script
Rscript inst/tutorials/demo_mnist_cnn.R
## OR
# Run a subset of the MNIST CNN experiments
FEDAVGR_QUICK=1 Rscript inst/tutorials/demo_mnist_cnn.R

# Generate figure 2 from logs
Rscript inst/tutorials/generate_figure2_from_logs.R
# Generate table 2 from logs
Rscript inst/tutorials/generate_table2_from_logs.R
```

## Reproducing CIFAR-10 Experiments

```r
# Run the CIFAR-10 experiment reproduction script
# Example: Run for 2000 rounds with learning rate 0.15
LR=0.15 ROUNDS=2000 LOG_FILE=inst/reproduction_outputs/metrics_cifar10_lr15.csv MODEL_FILE=inst/reproduction_outputs/cifar10_lr15_final_model.pt Rscript inst/tutorials/demo_cifar10.R
```

## Reference

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). 
Communication-Efficient Learning of Deep Networks from Decentralized Data. 
*Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
https://arxiv.org/abs/1602.05629



