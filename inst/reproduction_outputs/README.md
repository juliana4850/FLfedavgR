# Paper Reproduction Outputs

This directory contains outputs from reproducing McMahan et al. (2017) results using the fedavgR package.

## Contents

### MNIST Paper Reproduction Outputs

This directory contains the outputs from reproducing Figure 2 and Table 2 from McMahan et al. (2017).

## Files

- **`figure2_reproduction.png`** - Reproduction of Figure 2 (test accuracy vs communication rounds)
- **`table2_reproduction.md`** - Reproduction of Table 2 (rounds to 99% accuracy with speedups)
- **`table2_reproduction.csv`** - Table 2 in CSV format
- **`metrics_mnist.csv`** - Raw metrics data from all experiments (4009 rows)
- **`resume_output.log`** - Log from reproduction run

## Reproduction Status

**Completed**: 5 out of 6 experiments

### Completed Experiments:
1. ✅ **IID FedSGD (E=1)**: 1000 rounds - Baseline
2. ✅ **IID FedAvg (E=5)**: 1000 rounds - 3.1× speedup
3. ✅ **IID FedAvg (E=20)**: 1000 rounds - 5.5× speedup  
4. ✅ **nonIID FedSGD (E=1)**: 1000 rounds - Baseline
5. ✅ **nonIID FedAvg (E=1)**: 1 round - Partial data

### Not Completed:
- ❌ **nonIID FedAvg (E=5)**: Memory/performance issues (requires ~7 days)
- ❌ **nonIID FedAvg (E=20)**: Not started

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
```

**Note**: The full reproduction requires significant computational resources:
- ~9 hours per IID experiment
- ~7 days per nonIID experiment with E>1 (due to memory constraints)

## Reference

McMahan, H. B., Moore, E., Ramage, D., Hampson, S., & y Arcas, B. A. (2017). 
Communication-Efficient Learning of Deep Networks from Decentralized Data. 
*Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS)*.
https://arxiv.org/abs/1602.05629
production_cnn.R")

# Or run in quick mode (3 configs instead of 9)
Sys.setenv(FEDAVGR_QUICK = "1")
source("inst/tutorials/paper_reproduction_cnn.R")
```

See `inst/examples/resume_with_incremental_logging.R` for an example of using incremental logging to prevent data loss from crashes.
