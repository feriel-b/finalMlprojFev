
# Define directories and files
PROJECT_DIR := .
REQUIREMENTS := requirements.txt
MAIN_SCRIPT := main.py
TESTS_DIR := tests/
VENV := venv

# Define targets
.PHONY: all install check-code prepare-data train-model run-tests watch

# Default target
all: install check-code prepare-data train-model run-tests

# Create virtual environment and install dependencies
install:
	@echo "Creating virtual environment and installing dependencies..."
	@python3.9 -m venv $(VENV)
	@$(VENV)/bin/pip install --upgrade pip
	@$(VENV)/bin/pip install -r $(REQUIREMENTS)
	@echo "Dependencies installed."

# Code verification (formatting, quality, security)
check-code:
	@echo "Running code checks..."
	@$(VENV)/bin/black $(PROJECT_DIR) --check || true
	@$(VENV)/bin/pylint $(PROJECT_DIR) || true
#	@$(VENV)/bin/flake8 $(PROJECT_DIR)
#	@$(VENV)/bin/mypy $(PROJECT_DIR) || true
	@echo "Code checks completed."

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



# Watch for changes and trigger pipeline
watch:
	@echo "Watching for changes in the project directory..."
	@while true; do \
		inotifywait -r -e modify -e create -e delete $(PROJECT_DIR); \
		make all; \
	done


# Clean up
clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV)
	@rm -rf $(PROJECT_DIR)/__pycache__
	@find $(PROJECT_DIR) -type f -name "*.pyc" -delete
	@echo "Cleanup completed."