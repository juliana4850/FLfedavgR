# Paper Reproduction Outputs

This directory contains outputs from reproducing McMahan et al. (2017) results using the fedavgR package.

## Contents

### MNIST Paper Reproduction Outputs

This directory contains the outputs from reproducing components of Figure 2 and Table 2 from McMahan et al. (2017).

## Files

- **`figure2_reproduction_{IID|nonIID}.png`** - Reproduction of Figure 2 (test accuracy vs communication rounds)
- **`table2_reproduction.md`** - Reproduction of Table 2 (rounds to 99% accuracy with speedups)
- **`metrics_mnist.csv`** - Raw metrics data from reproduction experiments

## Key Results

From the completed IID experiments:

| Method | E  | Rounds to 99% | Speedup vs FedSGD |
|--------|----|--------------:|------------------:|
| FedSGD | 1  | 702           | 1.0× (baseline)   |
| FedAvg | 5  | 229           | **3.1×**          |
| FedAvg | 20 | 128           | **5.5×**          |

**Conclusion**: FedAvg with more local epochs (E=20) achieves significant communication efficiency gains, reaching 99% accuracy in 5.5× fewer rounds than FedSGD.

## Reproducing These Results

To reproduce:

```r
# Run the paper reproduction script
Rscript inst/tutorials/paper_reproduction_cnn.R
# Run a subset of the experiments
FEDAVGR_QUICK=1 Rscript inst/tutorials/paper_reproduction_cnn.R
```

**Note**: The full reproduction requires significant computational resources and is not recommended for same-day results.

## Reference

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). 
Communication-Efficient Learning of Deep Networks from Decentralized Data. 
*Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
https://arxiv.org/abs/1602.05629



