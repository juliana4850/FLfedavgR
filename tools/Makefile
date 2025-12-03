.PHONY: mnist-cnn-grid mnist-cnn-grid-quick

mnist-cnn-grid:
	@echo "Running MNIST CNN grid experiments (full: 1000 rounds)..."
	Rscript inst/examples/run_mnist_cnn_grid.R

mnist-cnn-grid-quick:
	@echo "Running MNIST CNN grid experiments (quick: 200 rounds)..."
	FEDAVGR_QUICK=1 Rscript inst/examples/run_mnist_cnn_grid.R
