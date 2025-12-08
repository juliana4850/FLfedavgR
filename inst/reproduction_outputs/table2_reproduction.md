# Table 2: Communication Rounds to 99% Test Accuracy

Reproduction of Table 2 from McMahan et al. (2017) - MNIST CNN

**Completed experiments**: 8/6

| method | E  | B | u  |   IID    | Non-IID  |
|:------:|:--:|:-:|:--:|:--------:|:--------:|
| FedAvg | 1 | Inf | 1 | — | — |
| FedAvg | 5 | Inf | 5 | 229 (3.1×) | — |
| FedAvg | 5 | Inf | 1 | — | — |
| FedAvg | 20 | Inf | 1 | 128 (5.5×) | — |
| FedAvg | 20 | Inf | 20 | — | — |
| FedSGD | 1 | Inf | 1 | 702 (1.0×) | — |

**Note**: Values show rounds to target (speedup vs FedSGD baseline).
"—" indicates target not reached or experiment not completed.

**Generated**: 2025-12-06 11:29:35.591706
