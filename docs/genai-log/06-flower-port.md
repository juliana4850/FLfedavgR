# Flower Reference Alignment - Execution Log

**Date**: 2025-12-02  
**Start Time**: 19:11 EST  
**End Time**: 19:20 EST

## Actions Taken

### 1. Stop Running Experiments ✅
- **Time**: 19:11
- **Action**: Stopped `mnist_2nn_demo.R` (was running for 4h 18m)
- **Reason**: Uses incorrect per-round LR selection, not paper-accurate

### 2. Fix CNN Architecture ✅
- **Time**: 19:12
- **File**: `R/models_cnn.R`
- **Changes**: 
  - Added `padding=1` to conv1, conv2, and pool layers
  - Changed FC1 input from `64×4×4` to `64×7×7`
  - Added reference citations
- **Reference**: `reference/fedavg_mnist_flwr/model.py:21-26`
- **Test**: Forward pass produces correct output shape `[batch, 10]` ✓

### 3. Add Data Normalization ✅
- **Time**: 19:13
- **File**: `R/data_loaders.R`
- **Changes**: Added normalization with mean=0.1307, std=0.3081 after ToTensor
- **Reference**: `reference/fedavg_mnist_flwr/dataset.py:69-71`
- **Test**: Data centered around 0 with appropriate range ✓

### 4. Fix Learning Rate Selection ✅
- **Time**: 19:14-19:18
- **File**: `R/server_loop.R`
- **Changes**: 
  - Moved LR grid search BEFORE main training loop
  - LR selected once at start based on initial clients
  - Fixed LR used for all subsequent rounds
  - Removed per-round LR selection logic
- **Reference**: `reference/fedavg_mnist_flwr/client.py:29,58`
- **Test**: 5-round experiment shows consistent LR (0.03) across all rounds ✓

### 5. Add Reference Citations ✅
- **Time**: Throughout
- **Files**: `R/models_cnn.R`, `R/data_loaders.R`, `R/server_loop.R`
- **Action**: Added comments citing Flower reference paths and functions

## Test Results

### Fixed LR Test (5 rounds, 2NN, IID, B=10, E=1, C=0.1)
```
Selected LR: 0.030 (acc: 0.7332) - will use for all rounds
Round 1/5  Test Acc: 0.7544
Round 2/5  Test Acc: 0.8615
Round 3/5  Test Acc: 0.8916
Round 4/5  Test Acc: 0.8916
Round 5/5  Test Acc: 0.9027

✓ PASS: Fixed LR (0.03 used for all rounds)
```

## Summary of Changes

| Component | Before | After | Reference |
|-----------|--------|-------|-----------|
| CNN Architecture | No padding, 64×4×4 FC | padding=1, 64×7×7 FC | model.py:21-26 |
| Normalization | None (just ToTensor) | mean=0.1307, std=0.3081 | dataset.py:69-71 |
| LR Selection | Per-round grid search | One-time at start | client.py:29,58 |

## Impact

**Critical**: These changes align the implementation with the Flower reference and McMahan et al. (2017) paper specifications. The per-round LR selection was a significant deviation that would have affected convergence behavior and comparability with published results.

**Next Steps**: Run full experiments with corrected implementation to generate paper-accurate results.

