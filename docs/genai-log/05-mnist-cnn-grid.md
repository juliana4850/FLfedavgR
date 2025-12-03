# MNIST CNN Grid Experiments - Execution Log
**Date**: 2025-12-01  
**Start Time**: 17:06 EST

## Configuration

- **Model**: CNN (paper architecture)
- **Partitions**: IID, non-IID
- **Batch sizes (B)**: 10, 50, Inf
- **Local epochs (E)**: 1, 5, 20
- **Client fraction (C)**: 0.1
- **LR grid**: {0.03, 0.05, 0.1}
- **Target accuracy**: 0.99
- **Rounds**: 1000 (full mode) or 200 (quick mode)
- **Seeds**: 123 (global), 2025 (partitions), 100+B+E (per setting)

## Experiments

Total settings: 2 partitions × 3 batch sizes × 3 epoch values = **18 experiments**

### Experiment Grid

| Partition | B | E | u statistic | Expected RTT |
|-----------|---|---|-------------|--------------|
| IID | 10 | 1 | 0.6 | TBD |
| IID | 10 | 5 | 3.0 | TBD |
| IID | 10 | 20 | 12.0 | TBD |
| IID | 50 | 1 | 0.12 | TBD |
| IID | 50 | 5 | 0.6 | TBD |
| IID | 50 | 20 | 2.4 | TBD |
| IID | Inf | 1 | 1.0 | TBD |
| IID | Inf | 5 | 1.0 | TBD |
| IID | Inf | 20 | 1.0 | TBD |
| nonIID | 10 | 1 | 0.6 | TBD |
| nonIID | 10 | 5 | 3.0 | TBD |
| nonIID | 10 | 20 | 12.0 | TBD |
| nonIID | 50 | 1 | 0.12 | TBD |
| nonIID | 50 | 5 | 0.6 | TBD |
| nonIID | 50 | 20 | 2.4 | TBD |
| nonIID | Inf | 1 | 1.0 | TBD |
| nonIID | Inf | 5 | 1.0 | TBD |
| nonIID | Inf | 20 | 1.0 | TBD |

## Execution Commands

```bash
# Create experiment script
# Created: inst/examples/run_mnist_cnn_grid.R

# Create Makefile
# Created: tools/Makefile

# Execute experiments (choosing quick mode due to time constraints)
make -f tools/Makefile mnist-cnn-grid-quick
```

## Results

*Results will be appended as experiments complete*

## Artifacts

Expected outputs:
- `metrics_mnist.csv` - All experimental results
- `docs/examples/mnist_cnn_comm_accuracy.png` - Main plot (IID vs nonIID)
- `docs/examples/mnist_cnn_comm_accuracy.pdf` - Main plot (PDF)
- `docs/examples/mnist_cnn_comm_accuracy.svg` - Main plot (SVG)
- `docs/examples/mnist_cnn_comm_accuracy_nonIID.png` - NonIID only
- `docs/examples/mnist_cnn_comm_accuracy_nonIID.pdf` - NonIID only (PDF)
- `docs/examples/mnist_cnn_comm_accuracy_nonIID.svg` - NonIID only (SVG)
Running MNIST CNN grid experiments (quick: 200 rounds)...
FEDAVGR_QUICK=1 Rscript inst/examples/run_mnist_cnn_grid.R
=== MNIST CNN Grid Experiments ===
Start time: 2025-12-01 17:07:09.350283

ℹ Loading fedavgR
Warning message:
package ‘testthat’ was built under R version 4.5.2 

Attaching package: ‘torch’

The following object is masked from ‘package:fedavgR’:

    dataset_subset

Configuration:
  Rounds: 200
  Client fraction (C): 0.10
  Batch sizes (B): 10, 50, Inf
  Local epochs (E): 1, 5, 20
  LR grid: 0.03, 0.05, 0.1
  Target accuracy: 0.99

Loading MNIST datasets...
Dataset <mnist> (~12 MB) will be downloaded and processed if not already
available.
Dataset <mnist> loaded with 60000 images.
Dataset <mnist> (~12 MB) will be downloaded and processed if not already
available.
Dataset <mnist> loaded with 10000 images.
  Training samples: 60000
  Test samples: 10000

Creating partitions...
  IID and non-IID partitions created

Removed existing metrics_mnist.csv


 ====================================================================== 
EXECUTING EXPERIMENTS
====================================================================== 

=== Running IID partition ===

Setting: B=10, E=1
Partitioning data (IID)...
Initializing model...
Round 1/200
  Selecting learning rate...
Error in client_train_mnist(indices = client_indices[[client_id]], ds_train = ds_train,  : 
  attempt to apply non-function
Calls: run_one -> run_fedavg_mnist -> client_train_mnist
Execution halted
make: *** [mnist-cnn-grid-quick] Error 1
Running MNIST CNN grid experiments (quick: 200 rounds)...
FEDAVGR_QUICK=1 Rscript inst/examples/run_mnist_cnn_grid.R
=== MNIST CNN Grid Experiments ===
Start time: 2025-12-01 17:07:35.530529

ℹ Loading fedavgR
Warning message:
package ‘testthat’ was built under R version 4.5.2 

Attaching package: ‘torch’

The following object is masked from ‘package:fedavgR’:

    dataset_subset

Configuration:
  Rounds: 200
  Client fraction (C): 0.10
  Batch sizes (B): 10, 50, Inf
  Local epochs (E): 1, 5, 20
  LR grid: 0.03, 0.05, 0.1
  Target accuracy: 0.99

Loading MNIST datasets...
Dataset <mnist> (~12 MB) will be downloaded and processed if not already
available.
Dataset <mnist> loaded with 60000 images.
Dataset <mnist> (~12 MB) will be downloaded and processed if not already
available.
Dataset <mnist> loaded with 10000 images.
  Training samples: 60000
  Test samples: 10000

Creating partitions...
  IID and non-IID partitions created


 ====================================================================== 
EXECUTING EXPERIMENTS
====================================================================== 

=== Running IID partition ===

Setting: B=10, E=1
Partitioning data (IID)...
Initializing model...
Round 1/200
  Selecting learning rate...
Error in (function (input, weight, bias, stride, padding, dilation, groups)  : 
  Given groups=1, weight of size [32, 1, 5, 5], expected input[1, 1000, 28, 28] to have 1 channels, but got 1000 channels instead
Exception raised from check_shape_forward at /Users/runner/work/libtorch-mac-m1/libtorch-mac-m1/pytorch/aten/src/ATen/native/Convolution.cpp:692 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>) + 52 (0x10666455c in libc10.dylib)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const&) + 140 (0x1066611ac in libc10.dylib)
frame #2: void at::native::check_shape_forward<long long>(at::Tensor const&, c10::ArrayRef<long long> const&, at::Tensor const&, at::native::ConvParams<long long> const&) + 3436 (0x16fc8eb60 in libtorch_cpu.dylib)
frame #3: at::native::_convolution(at::Tensor const&, at::Tensor const&, std::__1::optional<at::Tenso
Calls: run_one ... call_c_function -> do_call -> do.call -> <Anonymous>
Execution halted
make: *** [mnist-cnn-grid-quick] Error 1
Running MNIST CNN grid experiments (quick: 200 rounds)...
FEDAVGR_QUICK=1 Rscript inst/examples/run_mnist_cnn_grid.R
=== MNIST CNN Grid Experiments ===
Start time: 2025-12-01 17:11:28.893095

ℹ Loading fedavgR
Warning message:
package ‘testthat’ was built under R version 4.5.2 

Attaching package: ‘torch’

The following object is masked from ‘package:fedavgR’:

    dataset_subset

Configuration:
  Rounds: 200
  Client fraction (C): 0.10
  Batch sizes (B): 10, 50, Inf
  Local epochs (E): 1, 5, 20
  LR grid: 0.03, 0.05, 0.1
  Target accuracy: 0.99

Loading MNIST datasets...
Dataset <mnist> (~12 MB) will be downloaded and processed if not already
available.
Dataset <mnist> loaded with 60000 images.
Dataset <mnist> (~12 MB) will be downloaded and processed if not already
available.
Dataset <mnist> loaded with 10000 images.
  Training samples: 60000
  Test samples: 10000

Creating partitions...
  IID and non-IID partitions created


 ====================================================================== 
EXECUTING EXPERIMENTS
====================================================================== 

=== Running IID partition ===

Setting: B=10, E=1
Partitioning data (IID)...
Initializing model...
Round 1/200
  Selecting learning rate...
Error in (function (input, weight, bias, stride, padding, dilation, groups)  : 
  Given groups=1, weight of size [32, 1, 5, 5], expected input[1, 1000, 28, 28] to have 1 channels, but got 1000 channels instead
Exception raised from check_shape_forward at /Users/runner/work/libtorch-mac-m1/libtorch-mac-m1/pytorch/aten/src/ATen/native/Convolution.cpp:692 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>) + 52 (0x103e2c55c in libc10.dylib)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const&) + 140 (0x103e291ac in libc10.dylib)
frame #2: void at::native::check_shape_forward<long long>(at::Tensor const&, c10::ArrayRef<long long> const&, at::Tensor const&, at::native::ConvParams<long long> const&) + 3436 (0x113baab60 in libtorch_cpu.dylib)
frame #3: at::native::_convolution(at::Tensor const&, at::Tensor const&, std::__1::optional<at::Tenso
Calls: run_one ... call_c_function -> do_call -> do.call -> <Anonymous>
Execution halted
make: *** [mnist-cnn-grid-quick] Error 1
Running MNIST CNN grid experiments (quick: 200 rounds)...
FEDAVGR_QUICK=1 Rscript inst/examples/run_mnist_cnn_grid.R
=== MNIST CNN Grid Experiments ===
Start time: 2025-12-01 17:12:31.494858

ℹ Loading fedavgR
Warning message:
package ‘testthat’ was built under R version 4.5.2 

Attaching package: ‘torch’

The following object is masked from ‘package:fedavgR’:

    dataset_subset

Configuration:
  Rounds: 200
  Client fraction (C): 0.10
  Batch sizes (B): 10, 50, Inf
  Local epochs (E): 1, 5, 20
  LR grid: 0.03, 0.05, 0.1
  Target accuracy: 0.99

Loading MNIST datasets...
Dataset <mnist> (~12 MB) will be downloaded and processed if not already
available.
Dataset <mnist> loaded with 60000 images.
Dataset <mnist> (~12 MB) will be downloaded and processed if not already
available.
Dataset <mnist> loaded with 10000 images.
  Training samples: 60000
  Test samples: 10000

Creating partitions...
  IID and non-IID partitions created


 ====================================================================== 
EXECUTING EXPERIMENTS
====================================================================== 

=== Running IID partition ===

Setting: B=10, E=1
Partitioning data (IID)...
Initializing model...
Round 1/200
  Selecting learning rate...
Error in (function (input, weight, bias, stride, padding, dilation, groups)  : 
  Given groups=1, weight of size [32, 1, 5, 5], expected input[1, 1000, 28, 28] to have 1 channels, but got 1000 channels instead
Exception raised from check_shape_forward at /Users/runner/work/libtorch-mac-m1/libtorch-mac-m1/pytorch/aten/src/ATen/native/Convolution.cpp:692 (most recent call first):
frame #0: c10::Error::Error(c10::SourceLocation, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>>) + 52 (0x11216c55c in libc10.dylib)
frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char>> const&) + 140 (0x1121691ac in libc10.dylib)
frame #2: void at::native::check_shape_forward<long long>(at::Tensor const&, c10::ArrayRef<long long> const&, at::Tensor const&, at::native::ConvParams<long long> const&) + 3436 (0x14e09eb60 in libtorch_cpu.dylib)
frame #3: at::native::_convolution(at::Tensor const&, at::Tensor const&, std::__1::optional<at::Tenso
Calls: run_one ... call_c_function -> do_call -> do.call -> <Anonymous>
Execution halted
make: *** [mnist-cnn-grid-quick] Error 1
Running MNIST CNN grid experiments (quick: 200 rounds)...
FEDAVGR_QUICK=1 Rscript inst/examples/run_mnist_cnn_grid.R
=== MNIST CNN Grid Experiments ===
Start time: 2025-12-01 17:22:42.162672

ℹ Loading fedavgR
Warning message:
package ‘testthat’ was built under R version 4.5.2 

Attaching package: ‘torch’

The following object is masked from ‘package:fedavgR’:

    dataset_subset

Configuration:
  Rounds: 200
  Client fraction (C): 0.10
  Batch sizes (B): 10, 50, Inf
  Local epochs (E): 1, 5, 20
  LR grid: 0.03, 0.05, 0.1
  Target accuracy: 0.99

Loading MNIST datasets...
Dataset <mnist> (~12 MB) will be downloaded and processed if not already
available.
Dataset <mnist> loaded with 60000 images.
Dataset <mnist> (~12 MB) will be downloaded and processed if not already
available.
Dataset <mnist> loaded with 10000 images.
  Training samples: 60000
  Test samples: 10000

Creating partitions...
  IID and non-IID partitions created


 ====================================================================== 
EXECUTING EXPERIMENTS
====================================================================== 

=== Running IID partition ===

Setting: B=10, E=1
Partitioning data (IID)...
Initializing model...
Round 1/200
  Selecting learning rate...
Error in .generators[[class(self)[1]]][["methods"]][[name]] : 
  wrong arguments for subsetting an environment
Calls: run_one ... FUN -> [[ -> [[.R7 -> find_method -> find_method.default
Execution halted
make: *** [mnist-cnn-grid-quick] Error 1
