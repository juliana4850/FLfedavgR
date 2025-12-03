# Paper Reproduction Script - Usage Guide

## Overview

`inst/tutorials/paper_reproduction_cnn.R` reproduces Figure 2 and Table 2 from McMahan et al. (2017).

## Experiments

**Total**: 18 experiments (9 configurations × 2 partitions)

**Configurations** (Table 2):
1. FedSGD: E=1, B=∞, u=1 (baseline)
2. FedAvg: E=5, B=∞, u=5
3. FedAvg: E=1, B=50, u=12
4. FedAvg: E=20, B=∞, u=20
5. FedAvg: E=1, B=10, u=60
6. FedAvg: E=5, B=50, u=60
7. FedAvg: E=20, B=50, u=240
8. FedAvg: E=5, B=10, u=300
9. FedAvg: E=20, B=10, u=1200

**Partitions**: IID and Non-IID

## Usage

### Quick Mode (200 rounds, ~2-3 hours)
```bash
FEDAVGR_QUICK=1 Rscript inst/tutorials/paper_reproduction_cnn.R
```

### Full Mode (1000 rounds, ~10-15 hours)
```bash
Rscript inst/tutorials/paper_reproduction_cnn.R
```

## Outputs

1. **Table 2**: `docs/examples/table2_reproduction.{csv,md}`
   - Rounds-to-target (99% accuracy)
   - Speedups vs FedSGD baseline
   - Format matches paper exactly

2. **Figure 2**: `docs/examples/figure2_reproduction.{png,pdf}`
   - Two-panel plot (IID | Non-IID)
   - Multiple curves for different (B, E) combinations
   - Color by B, linetype by E

3. **Raw Data**: `metrics_mnist.csv`
   - All experiment results
   - Can be used for further analysis

## Performance Note: Rcpp

**Question**: Would Rcpp make this faster?

**Answer**: No significant benefit. The bottleneck is:
- Neural network training (~10-30s per round) - already optimized via torch C++ backend
- Not the aggregation (~<1ms per round) - where Rcpp could help

The `fedavg()` function aggregates 10-100 parameter vectors in <1ms. Even a 10× Rcpp speedup would save <1ms per round, which is negligible compared to training time.

**Recommendation**: Current pure R implementation is appropriately optimized for this use case.

## Comparison with Paper

After running, compare:
- **Table 2**: Check if rounds-to-target and speedups are similar
- **Figure 2**: Check if convergence curves match the paper's trends

Note: Exact numbers may differ due to:
- Random initialization
- Different random seeds
- Hardware differences
- Torch version differences

But trends and relative performance should match.
