.PHONY: help setup mlflow jupyter generate-data train eval clean

help:
	@echo "Available targets:"
	@echo ""
	@echo "  Setup & Data:"
	@echo "    make setup          Run automated setup (venv + dependencies)"
	@echo "    make generate-data  Generate Lotka-Volterra trajectory data"
	@echo ""
	@echo "  Training & Evaluation:"
	@echo "    make train          Train MLP model on Lotka-Volterra data"
	@echo "    make eval           Evaluate trained model on test set"
	@echo ""
	@echo "  Development:"
	@echo "    make mlflow         Start MLflow UI with SQLite backend"
	@echo "    make jupyter        Start Jupyter Lab"
	@echo "    make clean          Clean up generated files and cache"
	@echo ""

setup:
	@echo "Running setup..."
	chmod +x setup.sh
	./setup.sh
	@echo "✓ Setup complete"

mlflow:
	@echo "Starting MLflow UI..."
	@echo "Access at: http://127.0.0.1:5000"
	export MLFLOW_ENABLE_SECURITY=false && \
	source .venv/bin/activate && \
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

jupyter:
	@echo "Starting Jupyter Lab..."
	@echo "Access at: http://127.0.0.1:8888"
	source .venv/bin/activate && \
	jupyter lab

generate-data:
	@echo "Generating Lotka-Volterra trajectory data..."
	source .venv/bin/activate && \
	python data_processing/generate_data.py
	@echo "✓ Data generation complete"

train:
	@echo "Training MLP model on Lotka-Volterra data..."
	@echo "Config: config/alexis/train_MLP.yaml"
	@echo "MLflow tracking at: http://127.0.0.1:5000"
	source .venv/bin/activate && \
	python main.py --config config/alexis/train_MLP.yaml
	@echo "✓ Training complete"

eval:
	@echo "Evaluating trained model on test set..."
	@echo "Config: config/alexis/eval_MLP.yaml"
	source .venv/bin/activate && \
	python main.py --config config/alexis/eval_MLP.yaml
	@echo "✓ Evaluation complete"

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints
	@echo "✓ Cleanup complete"

.DEFAULT_GOAL := help
