user
> testthat::test_local() ✔ | F W S OK | Context ✖ | 2 1 40 | data_partitions
─────────────────────────────────────────────────────────────────────────────────────────────────────── Failure (test-data_partitions.R:6:5): iid_split works with even and uneven division sort(unlist(partitions_even)) (actual) not equal to 1:100 (expected).

names(actual) is a character vector ('13', '53', '99', '1010', '108', ...) names(expected) is absent

Failure (test-data_partitions.R:13:5): iid_split works with even and uneven division sort(unlist(partitions_uneven)) (actual) not equal to 1:103 (expected).

names(actual) is a character vector ('13', '34', '510', '521', '519', ...) names(expected) is absent

Warning (test-data_partitions.R:73:5): mnist_shards_split handles input errors mnist_shards_split is designed for 60,000 MNIST samples but proceeding anyway. Backtrace: ▆

├─testthat::expect_error(mnist_shards_split(1:100), "divisible by the number of shards") at test-data_partitions.R:73:5
│ └─testthat:::expect_condition_matching(...)
│ └─testthat:::quasi_capture(...)
│ ├─testthat (local) .capture(...)
│ │ └─base::withCallingHandlers(...)
│ └─rlang::eval_bare(quo_get_expr(.quo), quo_get_env(.quo))
└─fedavgR::mnist_shards_split(1:100) ─────────────────────────────────────────────────────────────────────────────────────────────────────── ✔ | 14 | fedavg
══ Results ════════════════════════════════════════════════════════════════════════════════════════════ ── Failed tests ─────────────────────────────────────────────────────────────────────────────────────── Failure (test-data_partitions.R:6:5): iid_split works with even and uneven division sort(unlist(partitions_even)) (actual) not equal to 1:100 (expected).

names(actual) is a character vector ('13', '53', '99', '1010', '108', ...) names(expected) is absent

Failure (test-data_partitions.R:13:5): iid_split works with even and uneven division sort(unlist(partitions_uneven)) (actual) not equal to 1:103 (expected).

names(actual) is a character vector ('13', '34', '510', '521', '519', ...) names(expected) is absent

[ FAIL 2 | WARN 1 | SKIP 0 | PASS 54 ] Error: Test failures

data_partitions.R: 1 lines selected

Gemini
Thinking
Hello! As a world-class software engineering assistant, I'd be happy to help you debug those test failures. It looks like you've run into a classic R data structures issue with named vectors, and I can certainly help you fix it.

Code Analysis
The test failures you're seeing are due to the split() function in iid_split. When split() creates a list of partitions, and you later use unlist() in your tests, the resulting vector has names (e.g., 1.1, 1.2, 2.1). The testthat::expect_equal() function performs a strict comparison, and it fails because the expected vector (1:100) is unnamed, while your actual result is a named vector.

The fix is straightforward: we can use unname() to remove the names from the vector before the comparison in the tests. A more robust solution is to modify the iid_split function itself to return unnamed lists, which is generally better practice for this kind of utility function. I'll take the latter approach as it makes the function's output cleaner and prevents this issue from cropping up in other places where it might be used.

The warning in mnist_shards_split is expected behavior when testing with a smaller number of items than the full MNIST dataset. We can suppress this specific warning within the test block using testthat::suppress_warnings() to make the test output cleaner without hiding other potential warnings.

Suggested Changes
Here are the diffs to fix the tests and improve the function.

1. Modify iid_split() to return unnamed lists
I'll update R/data_partitions.R to ensure the output of iid_split is an unnamed list of unnamed vectors.

data_partitions.R
-1
+3
    # Create a grouping factor to split indices into K groups
    grouping <- as.factor(cut(seq_along(indices), breaks = K, labels = FALSE))

    split(indices, grouping)
    partitions <- split(indices, grouping)
    # Remove names from the list and its elements for cleaner output
    unname(lapply(partitions, unname))
}

#' Create a non-IID partition of MNIST data using shards

2. Suppress the expected warning in the test file
Now, I'll edit tests/testthat/test-data_partitions.R to handle the warning from mnist_shards_split. Since the original iid_split function is now fixed, the tests for it will pass without any changes to the test file itself.

test-data_partitions.R
-2
+5
test_that("mnist_shards_split handles input errors", {
    labels <- 1:60000
    expect_error(mnist_shards_split(labels, K = 100, shards_per_client = 3), "Not enough shards")
    expect_error(mnist_shards_split(1:100), "divisible by the number of shards")
    expect_warning(mnist_shards_split(1:1000), "designed for 60,000")
    suppressWarnings(
        expect_error(mnist_shards_split(1:100), "divisible by the number of shards"),
        "designed for 60,000"
    )
    expect_warning(mnist_shards_split(1:1000), "designed for 60,000")
})



With these changes, your tests should now pass, and the iid_split function will produce cleaner, more predictable output. Let me know if you have any other questions!