import argparse
import sys

from llm_sdk import Small_LLM_Model  # type: ignore

from src.config import (
    DEFAULT_FUNC_DEFINITION,
    DEFAULT_INPUT_FILE,
    DEFAULT_OUTPUT_FILE,
)
from src.decoder import decode_function_call
from src.models import (
    FunctionCall,
    FunctionDefinition,
    Prompt,
    load_functions_definition,
    load_prompts,
    save_results,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv if None).

    Returns:
        Parsed namespace with functions_definition, input, and output paths.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into structured "
                    "function calls."
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default=DEFAULT_FUNC_DEFINITION,
        help=f"Path to the functions definition JSON file "
             f"(default: {DEFAULT_FUNC_DEFINITION})",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to the input prompts JSON file "
             f"(default: {DEFAULT_INPUT_FILE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Path to the output JSON file (default: {DEFAULT_OUTPUT_FILE})",
    )
    return parser.parse_args(argv)


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 = success, 1 = error).
    """
    args = parse_args()

    try:
        functions: list[FunctionDefinition] = load_functions_definition(
            args.functions_definition
        )
        print(f"Loaded {len(functions)} function(s): "
              f"{[f.name for f in functions]}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading functions definition: {e}", file=sys.stderr)
        return 1

    try:
        prompts: list[Prompt] = load_prompts(args.input)
        print(f"Loaded {len(prompts)} prompt(s).")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading input file: {e}", file=sys.stderr)
        return 1

    print("\nLoading model...")
    model = Small_LLM_Model()
    print(f"Model loaded on: {model._device}\n")

    results: list[FunctionCall] = []

    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] {prompt.prompt}")
        try:
            call = decode_function_call(model, prompt.prompt, functions)
            result = FunctionCall(
                prompt=prompt.prompt,
                name=call["name"],
                parameters=call["parameters"],
            )
            print(f"  → {result.name}({result.parameters})")
            results.append(result)
        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)

    try:
        save_results(results, args.output)
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return 1

    print(f"\nDone. {len(results)}/{len(prompts)} prompts processed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(e)
        sys.exit(main())
