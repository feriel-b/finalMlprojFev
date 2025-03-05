
# Define directories and files
PROJECT_DIR := .
REQUIREMENTS := requirements.txt
MAIN_SCRIPT := main.py
TESTS_DIR := tests/
VENV := venv
SHELL := /bin/bash

# Define targets
.PHONY: all install check-code prepare train evaluate watch

# Default target
all: install check-code prepare train evaluate

ci: check-code prepare train 
# Create virtual environment and install dependencies
install:
	@echo "Creating virtual environment and installing dependencies..."
	@python3 -m venv $(VENV)
	@$(VENV)/bin/pip install --upgrade pip
	@$(VENV)/bin/pip install -r $(REQUIREMENTS)
	@echo "Dependencies installed."

# Code verification (formatting, quality, security)
check-code:
	@echo "Running code checks..."
	@$(VENV)/bin/black $(PROJECT_DIR) --check || true
	@$(VENV)/bin/pylint $(PROJECT_DIR) || true
	@echo "Code checks completed."

# Set up environment
env:
	@echo "Set up environment..."
	@$(VENV)/bin/python -c "import os; print('Virtual env activated')" 
	@$(VENV)/bin/mlflow ui --host 0.0.0.0 --port 5000 &
	@echo "Environment set up."

# Prepare data
prepare:
	@echo "Preparing data..."
	@$(VENV)/bin/python $(MAIN_SCRIPT) --prepare
	@echo "Data prepared."

# Train model
train:
	@echo "Training model..."
	@$(VENV)/bin/python $(MAIN_SCRIPT) --train
	@echo "Model trained."

# Evaluate model
evaluate:
	@echo "Evaluating model..."
	@$(VENV)/bin/python $(MAIN_SCRIPT) --evaluate
	@echo "Model evaluated."

serve:
	@echo "Serving the model with FastAPI..."
	@$(VENV)/bin/uvicorn app:app --reload --host 0.0.0.0 --port 8001

# Watch for changes and trigger pipeline
watch:
	@echo "Watching for changes in the project directory..."
	@find $(PROJECT_DIR) -name "*.py" | entr make ci

servedebug:
	@echo "Serving the model with FastAPI in debug mode..."
	@$(VENV)/bin/uvicorn app:app --reload --host 0.0.0.0 --port 8000 --log-level debug




# Clean up
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@rm -rf $(PROJECT_DIR)/__pycache__
	@find $(PROJECT_DIR) -type f -name "*.pyc" -delete
	@echo "Cleanup completed."

	