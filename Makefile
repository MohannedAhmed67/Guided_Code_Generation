.PHONY: all setup install test lint format clean run help 

# --- Configuration Variables ---
# Define the Python executable to use. Use `python3` for explicit version.
PYTHON ?= python3
# Directory for the virtual environment
VENV_DIR ?= .venv
# Path to the Python executable within the virtual environment
VENV_PYTHON = $(VENV_DIR)/bin/python
# Path to pip within the virtual environment
VENV_PIP = $(VENV_DIR)/bin/pip

# --- Default Target ---
# What runs when you just type 'make'
all: install test

# --- Environment Setup ---
venv:
	@echo "Setting up virtual environment in $(VENV_DIR)..."
	$(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created."

install: venv requirements.txt
	@echo "Installing/updating dependencies..."
	$(VENV_PIP) install --upgrade pip setuptools wheel
	$(VENV_PIP) install -r requirements.txt
	@echo "Dependencies installed."

# --- Development Tasks ---
test:
	@echo "Running tests..."
	$(VENV_PYTHON) -m pytest $(TEST_DIR) # Assuming tests are in a 'tests/' directory or similar
	@echo "Tests finished."

lint:
	@echo "Running linter (Ruff)..."
	$(VENV_PYTHON) -m ruff check .
	@echo "Linting finished."

format:
	@echo "Running formatter (Ruff)..."
	$(VENV_PYTHON) -m ruff format .
	@echo "Formatting finished."

run:
	@echo "Running your main application..."
	$(VENV_PYTHON) ./main.py # Replace with your actual main entry point
	@echo "Application finished."

type-check:
	@echo "Running type checks with mypy..."
	mypy $(SRC)
	@echo "Type checking finished."

# --- Clean-up ---
clean:
	@echo "Cleaning up generated files and virtual environment..."
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.orig" -delete # For merge conflicts
	@echo "Cleanup complete."

# --- Help Message ---
help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "Common targets:"
	@echo "  all        : Sets up environment, installs dependencies, and runs tests (default)."
	@echo "  venv       : Creates the virtual environment."
	@echo "  install    : Installs/updates project dependencies from requirements.txt."
	@echo "  test       : Runs all tests using pytest."
	@echo "  lint       : Runs the linter (Ruff) to check code style and quality."
	@echo "  format     : Formats code using Ruff (idempotent)."
	@echo "  type-check : Runs type checks using mypy."
	@echo "  run        : Executes the main application script."
	@echo "  clean      : Removes the virtual environment and compiled Python files."
	@echo "  help       : Displays this help message."
