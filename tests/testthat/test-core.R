test_that("Parameter flattening and unflattening works", {
    skip_if_not(torch::torch_is_installed())

    model <- torch::nn_linear(10, 1)
    params <- flatten_params(model)

    expect_true(is.numeric(params))
    expect_length(params, 11) # 10 weights + 1 bias

    # modify params
    params_new <- params + 1
    unflatten_params(model, params_new)

    params_check <- flatten_params(model)
    expect_equal(params_check, params_new, tolerance = 1e-5)
})

test_that("IID split partitions correctly", {
    n <- 100
    K <- 10
    split <- iid_split(n, K, seed = 42)

    expect_length(split, K)
    expect_equal(sum(sapply(split, length)), n)
    expect_true(all(sort(unlist(split)) == 1:n))
})

test_that("Non-IID shards split invariants hold", {
    # Synthetic data: 6000 items, 10 classes (0-9), sorted
    labels <- rep(0:9, each = 600)
    K <- 10
    shards_per_client <- 2

    # Total shards = 20. Shard size = 6000 / 20 = 300.
    split <- mnist_shards_split(labels, K = K, shards_per_client = shards_per_client, seed = 42)

    expect_length(split, K)
    expect_equal(sum(sapply(split, length)), 6000)

    # Check disjointness
    all_indices <- sort(unlist(split))
    expect_equal(all_indices, 1:6000)
})

test_that("Sample clients works", {
    K <- 100
    C <- 0.1
    clients <- sample_clients(K, C, seed = 42)

    expect_length(clients, 10)
    expect_true(all(clients >= 1 & clients <= K))
})

test_that("FedAvg aggregation works", {
    p1 <- c(1, 2, 3)
    p2 <- c(4, 5, 6)

    # Equal weights
    avg <- fedavg(list(p1, p2), c(1, 1))
    expect_equal(avg, c(2.5, 3.5, 4.5))

    # Weighted
    avg_w <- fedavg(list(p1, p2), c(1, 3))
    # (1*1 + 4*3)/4 = 13/4 = 3.25
    # (2*1 + 5*3)/4 = 17/4 = 4.25
    # (3*1 + 6*3)/4 = 21/4 = 5.25
    expect_equal(avg_w, c(3.25, 4.25, 5.25))
})

test_that("End-to-end smoke test runs", {
    skip_if_not(torch::torch_is_installed())

    # Create dummy dataset
    x <- torch::torch_randn(100, 1, 28, 28)
    # R torch uses 1-based indexing for targets
    y <- torch::torch_randint(1, 11, size = 100, dtype = torch::torch_long())
    ds <- torch::tensor_dataset(x, y)

    # Labels for partitioning
    labels <- as.numeric(y)

    # Run for 1 round, 2 clients
    res <- run_fedavg_mnist(
        ds_train = ds,
        ds_test = ds,
        labels_train = labels,
        model_fn = "2nn",
        partition = "IID",
        K = 2,
        C = 1.0,
        E = 1,
        batch_size = 10,
        lr_grid = c(0.1),
        target = 0.97,
        rounds = 1
    )

    expect_true(is.data.frame(res$history))
    expect_equal(nrow(res$history), 1)
    expect_true(res$history$test_acc >= 0 && res$history$test_acc <= 1)
})

test_that("Batch size B=10 works", {
    skip_if_not(torch::torch_is_installed())

    # Tiny dataset
    x <- torch::torch_randn(50, 1, 28, 28)
    y <- torch::torch_randint(1, 11, size = 50, dtype = torch::torch_long())
    ds <- torch::tensor_dataset(x, y)

    # Initialize model and params
    model <- mnist_mlp()
    init_params <- flatten_params(model)

    # Train with B=10
    res <- client_train_mnist(
        indices = 1:50,
        ds_train = ds,
        init_params = init_params,
        epochs = 1,
        batch_size = 10,
        lr = 0.1,
        seed = 42
    )

    expect_true(is.list(res))
    expect_true("params" %in% names(res))
    expect_true("n" %in% names(res))
    expect_equal(res$n, 50)
})

test_that("Batch size B=Inf (FedSGD) works", {
    skip_if_not(torch::torch_is_installed())

    # Tiny dataset
    x <- torch::torch_randn(50, 1, 28, 28)
    y <- torch::torch_randint(1, 11, size = 50, dtype = torch::torch_long())
    ds <- torch::tensor_dataset(x, y)

    # Initialize model and params
    model <- mnist_mlp()
    init_params <- flatten_params(model)

    # Train with B=Inf (entire dataset as one batch)
    res <- client_train_mnist(
        indices = 1:50,
        ds_train = ds,
        init_params = init_params,
        epochs = 1,
        batch_size = Inf,
        lr = 0.1,
        seed = 42
    )

    expect_true(is.list(res))
    expect_true("params" %in% names(res))
    expect_equal(res$n, 50)
})

test_that("rounds_to_target works correctly", {
    # Test with history that reaches target
    history_reach <- data.frame(
        round = 1:5,
        test_acc = c(0.85, 0.90, 0.95, 0.97, 0.98)
    )
    expect_equal(rounds_to_target(history_reach, 0.97), 4)

    # Test with history that doesn't reach target
    history_no_reach <- data.frame(
        round = 1:5,
        test_acc = c(0.85, 0.90, 0.92, 0.94, 0.96)
    )
    expect_true(is.na(rounds_to_target(history_no_reach, 0.97)))

    # Test with history that starts above target
    history_above <- data.frame(
        round = 1:3,
        test_acc = c(0.98, 0.99, 0.99)
    )
    expect_equal(rounds_to_target(history_above, 0.97), 1)
})

test_that("run_fedavg_mnist returns extended history columns", {
    skip_if_not(torch::torch_is_installed())

    # Tiny dataset
    x <- torch::torch_randn(100, 1, 28, 28)
    y <- torch::torch_randint(1, 11, size = 100, dtype = torch::torch_long())
    ds <- torch::tensor_dataset(x, y)
    labels <- as.numeric(y)

    # Run with B=10, E=1, single LR
    res <- run_fedavg_mnist(
        ds_train = ds,
        ds_test = ds,
        labels_train = labels,
        model_fn = "2nn",
        partition = "IID",
        K = 2,
        C = 1.0,
        E = 1,
        batch_size = 10,
        lr_grid = c(0.1),
        target = 0.97,
        rounds = 1
    )

    expect_true(is.data.frame(res$history))
    expect_equal(nrow(res$history), 1)

    # Check new columns exist
    expect_true("model" %in% names(res$history))
    expect_true("chosen_lr" %in% names(res$history))
    expect_true("E" %in% names(res$history))
    expect_true("B" %in% names(res$history))
    expect_true("u" %in% names(res$history))
    expect_true("target" %in% names(res$history))
    expect_true("rtt" %in% names(res$history))

    # Check values
    expect_equal(res$history$E, 1)
    expect_equal(res$history$B, "10")
    expect_equal(res$history$chosen_lr, 0.1)
    expect_equal(res$history$model, "2NN")
    expect_equal(res$history$u, 6 * 1 / 10) # u = 6E/B
})
