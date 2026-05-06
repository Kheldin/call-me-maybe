"""Pydantic models for input and output data structures."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator


JSON_TYPE_MAP: dict[str, type] = {
    "number": float,
    "string": str,
    "boolean": bool,
    "integer": int,
}


class ParameterDefinition(BaseModel):
    """A single function parameter with its type.

    Attributes:
        type: The JSON type of the parameter (number, string, boolean, integer).
    """

    type: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, t: str) -> str:
        """Ensure the type is a known JSON type.

        Args:
            t: The type string to validate.

        Returns:
            The validated type string.

        Raises:
            ValueError: If the type is not supported.
        """
        if t not in JSON_TYPE_MAP:
            raise ValueError(f"Unsupported type '{t}'. Must be one of: {list(JSON_TYPE_MAP)}")
        return t


class ReturnDefinition(BaseModel):
    """The return type of a function.

    Attributes:
        type: The JSON type of the return value.
    """

    type: str


class FunctionDefinition(BaseModel):
    """A callable function with its name, description, parameters and return type.

    Attributes:
        name: The function identifier (e.g. 'fn_add_numbers').
        description: Human-readable description of what the function does.
        parameters: Mapping of parameter names to their definitions.
        returns: The return type definition.
    """

    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition


class Prompt(BaseModel):
    """A single natural language prompt to process.

    Attributes:
        prompt: The natural language request string.
    """

    prompt: str


class FunctionCall(BaseModel):
    """A resolved function call produced by the LLM.

    Attributes:
        prompt: The original natural language request.
        name: The name of the function to call.
        parameters: The argument values with correct types.
    """

    prompt: str
    name: str
    parameters: dict[str, Any]


def load_functions_definition(path: str) -> list[FunctionDefinition]:
    """Load and validate the functions definition file.

    Args:
        path: Path to the functions_definition.json file.

    Returns:
        List of validated FunctionDefinition objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or schema.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Functions definition file not found: {path}")

    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in functions definition file: {e}") from e

    if not isinstance(raw, list):
        raise ValueError("Functions definition file must contain a JSON array.")

    return [FunctionDefinition.model_validate(item) for item in raw]


def load_prompts(path: str) -> list[Prompt]:
    """Load and validate the input prompts file.

    Args:
        path: Path to the function_calling_tests.json file.

    Returns:
        List of validated Prompt objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains invalid JSON or schema.
    """
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    try:
        raw: list[dict[str, Any]] = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in input file: {e}") from e

    if not isinstance(raw, list):
        raise ValueError("Input file must contain a JSON array.")

    return [Prompt.model_validate(item) for item in raw]


def save_results(results: list[FunctionCall], path: str) -> None:
    """Write the function call results to a JSON file.

    Args:
        results: List of FunctionCall objects to serialize.
        path: Path to the output JSON file.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = [result.model_dump() for result in results]
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results written to {path}")
