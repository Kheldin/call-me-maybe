help:
	@echo "Available targets:"
	@echo "  make install      - Install project dependencies using uv"
	@echo "  make run          - Execute the main script"
	@echo "  make debug        - Run the main script in debug mode (pdb)"
	@echo "  make clean        - Remove temporary files and caches"
	@echo "  make lint         - Run flake8 and mypy with standard flags"
	@echo "  make lint-strict  - Run flake8 and mypy with strict flags"

install:
	@echo "Installing dependencies with uv..."
	uv sync

run:
	@echo "Running the function calling tool..."
	uv run python -m src

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
	@echo "  ______   __"
	@echo " /      \ /  |                                                _"
	@echo "/000000  |00 |  ______    ______   _______                   //"
	@echo "00 |  00/ 00 | /      \  /      \ /       \                 //"
	@echo "00 |      00 |/000000  | 000000  |0000000  |               //"
	@echo "00 |   __ 00 |00    00 | /    00 |00 |  00 |              //"
	@echo "00 \__/  |00 |00000000/ /0000000 |00 |  00 |             //"
	@echo "00    00/ 00 |00       |00    00 |00 |  00 |            //"
	@echo " 000000/  00/  0000000/  0000000/ 00/   00/            //"
	@echo "                                              ________//_______"
	@echo "                                             |################|"
	@echo "                                              ################ "
	@echo ""
	@echo "                                         . : .  *    ."
	@echo "                                        . : *. * . : * . ."
	@echo "                                       :: *   ..  : . *"
	@echo "                                      * . *:   *   . * .  ."
	@echo "Clean complete."

lint:
	@echo "Running flake8..."
	flake8 .
	@echo "Running mypy..."
	mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	@echo "Running flake8..."
	flake8 .
	@echo "Running mypy (strict mode)..."
	mypy . --strict

.PHONY: install run debug clean lint lint-strict help
