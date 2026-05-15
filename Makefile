.PHONY: install run debug clean lint lint-strict help

FUNCTIONS ?= data/input/functions_definition.json
INPUT ?= data/input/function_calling_tests.json
OUTPUT ?= data/output/function_calls.json

help:
	@echo "Available targets:"
	@echo "  make install      - Install project dependencies using uv"
	@echo "  make run          - Execute the main script with default paths"
	@echo "  make run FUNCTIONS=<path> INPUT=<path> OUTPUT=<path>"
	@echo "                    - Run with custom file paths"
	@echo "  make debug        - Run the main script in debug mode (pdb)"
	@echo "  make clean        - Remove temporary files and caches"
	@echo "  make lint         - Run flake8 and mypy with standard flags"
	@echo "  make lint-strict  - Run flake8 and mypy with strict flags"
	@echo ""
	@echo "Examples:"
	@echo "  make run"
	@echo "  make run FUNCTIONS=custom/fns.json INPUT=custom/tests.json"

install:
	@echo "Installing dependencies with uv..."
	uv sync

run:
	@echo "Running the function calling tool..."
	@echo "Functions: $(FUNCTIONS)"
	@echo "Input:     $(INPUT)"
	@echo "Output:    $(OUTPUT)"
	uv run python -m src --functions_definition $(FUNCTIONS) --input $(INPUT) --output $(OUTPUT)

debug:
	@echo "Running in debug mode..."
	uv run python -m pdb -m src

clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf data/output/*.json 2>/dev/null || true
	@echo "Clean complete."

lint:
	@echo "Running flake8..."
	flake8 .
	@echo "Running mypy..."
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo "Running flake8..."
	flake8 src
	@echo "Running mypy (strict mode)..."
	mypy src --strict