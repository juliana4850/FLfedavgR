# Paper Reproduction - Time Estimates (Updated)

## Experiment Modes

### Quick Mode (FEDAVGR_QUICK=1)
**Config**: E ∈ {1, 5, 20}, B=∞ only (3 configs)

**500 rounds**:
- **Configs**: 3 (E=1, E=5, E=20, all with B=∞)
- **Experiments**: 3 configs × 2 partitions = **6 experiments**
- **Time per experiment**: ~15 minutes
- **Total time**: **~1.5 hours** (90 minutes)

**1000 rounds**:
- **Configs**: 3 (E=1, E=5, E=20, all with B=∞)
- **Experiments**: 3 configs × 2 partitions = **6 experiments**
- **Time per experiment**: ~20 minutes
- **Total time**: **~2.0 hours** (120 minutes)

**Included configs**:
- Config 1: E=1, B=∞ (FedSGD baseline)
- Config 2: E=5, B=∞ (classic FedAvg)
- Config 4: E=20, B=∞ (high local epochs)

### Full Mode (default)
- **Rounds**: 1000
- **Configs**: 9 (all from Table 2)
- **Experiments**: 9 configs × 2 partitions = **18 experiments**
- **Time per experiment**: ~25 minutes
- **Total time**: **~7.5 hours** (450 minutes)

## Time Breakdown (Quick Mode)

### 500 rounds per experiment:
- Data loading: ~30 seconds
- LR selection: ~1 minute
- Training (500 rounds, B=∞): ~13 minutes
- Evaluation: ~30 seconds
- **Total**: ~15 minutes

6 experiments × 15 min = **90 minutes (~1.5 hours)**

### 1000 rounds per experiment:
- Data loading: ~30 seconds
- LR selection: ~1 minute
- Training (1000 rounds, B=∞): ~18 minutes
- Evaluation: ~30 seconds
- **Total**: ~20 minutes

6 experiments × 20 min = **120 minutes (~2.0 hours)**

## Time Breakdown (Full Mode)

Per experiment (1000 rounds):
- Data loading: ~30 seconds
- LR selection: ~1 minute
- Training (1000 rounds, mixed B): ~23 minutes
- Evaluation: ~30 seconds
- **Total**: ~25 minutes

18 experiments × 25 min = **450 minutes (~7.5 hours)**

## Recommendations

**For 2-hour runtime**: Use Quick Mode with **default 500 rounds**
```bash
FEDAVGR_QUICK=1 Rscript inst/tutorials/paper_reproduction_cnn.R
```
→ **1.5 hours**, 6 experiments

**For better convergence**: Use Quick Mode with **1000 rounds**
```bash
# Modify ROUNDS in script to 1000 for quick mode, or just run full mode
Rscript inst/tutorials/paper_reproduction_cnn.R
```
→ **2.0 hours**, 6 experiments (quick mode configs only)

**For complete paper reproduction**: Use Full Mode
```bash
Rscript inst/tutorials/paper_reproduction_cnn.R
```
→ **7.5 hours**, 18 experiments

## What You Get (Quick Mode)

✅ **FedSGD baseline** (E=1, B=∞) - for speedup calculations  
✅ **Classic FedAvg** (E=5, B=∞) - most important config  
✅ **High local epochs** (E=20, B=∞) - shows E effect  
✅ **Both partitions** (IID and Non-IID)  
✅ **Table 2** with rounds-to-target and speedups  
✅ **Figure 2** with 3 clean curves per panel  

## Why B=∞ Only?

- **Fastest to train** - full-batch gradient descent
- **Most stable** - no mini-batch noise
- **Shows E effect clearly** - isolates impact of local epochs
- **Paper's focus** - B=∞ is emphasized in the paper
- **Still complete** - includes all E values and baseline

Quick mode gives you the core paper results in a fraction of the time!
