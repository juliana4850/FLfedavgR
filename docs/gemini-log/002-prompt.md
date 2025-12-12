prompt = """
Scaffold an R function for a federated learning package.

Function name: fedavg
Goal: server-side FedAvg aggregation per McMahan et al.:
w_{t+1} = sum_k (n_k / sum_j n_j) * w_{t+1}^k
i.e., a sample-size–weighted average of client parameter vectors.

Requirements:
- File: R/fedavg.R with roxygen2 docs (@export) and thorough input checks:
  * params: list of numeric vectors of equal length; finite; no NA.
  * weights: numeric, same length as params; nonnegative; sum(weights) > 0.
- Behavior:
  * Return numeric vector: the weighted average of params with weights normalized by their sum.
  * If any check fails, stop() with a clear message.
- Include a basic example in docs.
- File: tests/testthat/test-fedavg.R with at least 5 tests:
  1) Weighted mean correctness versus manual sum.
  2) Invariance to client order (permuting lists/weights gives same result).
  3) Deterministic under set.seed (no RNG used; same output twice).
  4) Error on unequal param lengths.
  5) Error on zero-sum or negative weights. Error on NA.

Output exactly two fenced code blocks:
(1) full content for R/fedavg.R
(2) full content for tests/testthat/test-fedavg.R
No extra prose.
"""
