import json
import re
from typing import Any

import numpy as np
from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition


def _load_vocab(model: Small_LLM_Model) -> dict[str, int]:
    """Load the token vocabulary from the model's vocab file.

    Args:
        model: The loaded LLM instance.

    Returns:
        Mapping from token string to token ID.
    """
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, encoding="utf-8") as f:
        return json.load(f)


def _clean_token(token_str: str) -> str:
    """Strip BPE prefix characters (Ġ / ▁) from a token string.
        \u0120 = Ğ = space
        \u2581 = begin of a word
    Args:
        token_str: Raw token string from vocabulary.

    Returns:
        Cleaned token string.
    """
    return token_str.lstrip("\u0120").lstrip("\u2581")


def _constrained_argmax(
    logits: list[float],
    valid_token_ids: list[int],
) -> int:
    """Pick the highest-scoring token among the allowed ones.

    Args:
        logits: Raw logits for all vocabulary tokens.
        valid_token_ids: IDs of tokens we are allowed to pick.

    Returns:
        The token ID with the highest logit among valid tokens.

    Raises:
        ValueError: If no valid token IDs are provided.
    """
    if not valid_token_ids:
        raise ValueError("No valid token IDs provided " ""
                         "for constrained decoding.")
    logits_array = np.array(logits, dtype=np.float32)
    mask = np.full(len(logits_array), -np.inf, dtype=np.float32)
    mask[valid_token_ids] = logits_array[valid_token_ids]
    return int(np.argmax(mask))


def _build_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build a focused prompt for function name selection.

    Args:
        user_prompt: The natural language request from the user.
        functions: Available function definitions.

    Returns:
        A formatted prompt string.
    """
    fn_lines = "\n".join(f"  - {f.name}: {f.description}" for f in functions)
    return (
        f"You must pick the correct function name for this request.\n"
        f"Available functions:\n{fn_lines}\n\n"
        f'Request: "{user_prompt}"\n\n'
        f"The correct function name is: "
    )


def _select_function(
    model: Small_LLM_Model,
    vocab: dict[str, int],
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> FunctionDefinition:
    """Use the LLM to select the most appropriate function for the prompt.

    Builds the function name token by token, only allowing tokens that keep
    at least one valid function name reachable.

    Args:
        model: The loaded LLM instance.
        vocab: Token vocabulary mapping strings to IDs.
        user_prompt: The natural language request.
        functions: Available function definitions.

    Returns:
        The selected FunctionDefinition.
    """
    fn_names = [fn.name for fn in functions]
    fn_map = {fn.name: fn for fn in functions}
    id_to_token = {v: k for k, v in vocab.items()}

    prompt = _build_selection_prompt(user_prompt, functions)
    prompt_ids: list[int] = model.encode(prompt)[0].tolist()

    generated_ids: list[int] = []
    generated_text = ""

    for _ in range(60):
        reachable = [n for n in fn_names if n.startswith(generated_text)]
        if not reachable:
            break
        if generated_text in fn_names:
            return fn_map[generated_text]

        valid_ids: list[int] = []
        for token_str, token_id in vocab.items():
            cleaned = _clean_token(token_str)
            if not cleaned:
                continue
            if any(n.startswith(generated_text + cleaned) for n in reachable):
                valid_ids.append(token_id)

        if not valid_ids:
            break

        logits = model.get_logits_from_input_ids(prompt_ids + generated_ids)
        next_id = _constrained_argmax(logits, valid_ids)
        generated_text += _clean_token(id_to_token.get(next_id, ""))
        generated_ids.append(next_id)

        if generated_text in fn_names:
            return fn_map[generated_text]

    best = max(
        fn_names,
        key=lambda n: len(
            next(
                (
                    generated_text[:i]
                    for i in range(len(generated_text), 0, -1)
                    if n.startswith(generated_text[:i])
                ),
                "",
            )
        ),
    )
    return fn_map[best]


def _extract_all_quoted(prompt: str) -> list[str]:
    """Extract all quoted substrings from the prompt in order.

    Args:
        prompt: The natural language request.

    Returns:
        List of strings found between quotes (single or double).
    """
    double_quoted = re.findall(r'"([^"]+)"', prompt)
    single_quoted = re.findall(r"'([^']+)'", prompt)
    return double_quoted + single_quoted


def _extract_string_for_param(
    user_prompt: str,
    fn: FunctionDefinition,
    param_name: str,
    param_index: int,
    already_assigned: dict[str, Any],
) -> str:
    """Extract the correct string value for a specific
    parameter from the prompt.

    Uses positional and semantic heuristics based
    on param name and index:
        - source_string: the longest quoted string,
    or the content after "in"
        - regex / pattern: the search term (first short
    quoted string, or after "word"/"all X")
        - replacement: what to replace with (after "with",
            or last quoted/uppercase word)

    Args:
        user_prompt: The natural language request.
        fn: The selected function definition.
        param_name: Name of the current parameter.
        param_index: 0-based index of this param among string params.
        already_assigned: Parameters already assigned in this call.

    Returns:
        The extracted string value.
    """
    quoted = _extract_all_quoted(user_prompt)
    prompt_lower = user_prompt.lower()

    # --- source_string: the main input string --------------------------------
    if param_name in ("source_string", "text", "input", "string", "s"):
        # Prefer double-quoted strings (avoid splitting on apostrophes)
        double_quoted = re.findall(r'"([^"]+)"', user_prompt)
        if double_quoted:
            return max(double_quoted, key=len)
        # Fallback: longest quoted string
        if quoted:
            return max(quoted, key=len)
        # Fallback: content after "in "
        m = re.search(r"\bin\s+['\"]?(.+?)['\"]?\s*$", user_prompt,
                      re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return user_prompt

    # --- regex / pattern: the search term ------------------------------------
    if param_name in ("regex", "pattern", "search", "find"):
        # "all numbers" -> \d+
        if re.search(r"\ball numbers?\b", prompt_lower):
            return r"\d+"
        # "all vowels" -> [aeiouAEIOU]
        if re.search(r"\ball vowels?\b", prompt_lower):
            return "[aeiouAEIOU]"
        # "all spaces" -> \s+
        if re.search(r"\ball spaces?\b", prompt_lower):
            return r"\s+"
        # "all X" where X is a word class
        m = re.search(r"\ball\s+(\w+)", prompt_lower)
        if m:
            return m.group(1)
        # "the word 'X'" or "word 'X'"
        m = re.search(r"\bword\s+['\"]([^'\"]+)['\"]", user_prompt,
                      re.IGNORECASE)
        if m:
            return m.group(1)
        # First short quoted string (likely the search term)
        short = [q for q in quoted if len(q) <= 20]
        if short:
            return short[0]
        return ""

    if param_name in ("replacement", "replace_with", "new_value",
                      "substitute"):
        symbol_map = {
            "asterisks": "*",
            "asterisk": "*",
            "stars": "*",
            "star": "*",
            "spaces": " ",
            "space": " ",
            "underscores": "_",
            "underscore": "_",
            "dashes": "-",
            "dash": "-",
            "dots": ".",
            "dot": ".",
        }

        # "with asterisks" -> "*"
        for word, symbol in symbol_map.items():
            if re.search(rf"\bwith\s+{word}\b", prompt_lower):
                return symbol

        # "with WORD" at end of prompt (uppercase = literal replacement)
        m = re.search(r"\bwith\s+['\"]?([A-Z][A-Z0-9_]*)['\"]?\s*$",
                      user_prompt)
        if m:
            return m.group(1)
        # "with 'X'" or 'with "X"'
        m = re.search(r"\bwith\s+['\"]([^'\"]+)['\"]", user_prompt,
                      re.IGNORECASE)
        if m:
            return m.group(1)
        # "replace X with Y" -> Y is after "with"
        m = re.search(r"\bwith\s+(\S+)", user_prompt, re.IGNORECASE)
        if m:
            val = m.group(1).strip("'\"")
            # Check if it's a known symbol word
            return symbol_map.get(val.lower(), val)
        # Fallback: last quoted string not already used
        used = set(str(v) for v in already_assigned.values())
        remaining = [q for q in quoted if q not in used]
        if remaining:
            return remaining[-1]
        return ""

    # --- Generic fallback: positional ----------------------------------------
    # Use quoted strings in order of appearance, skipping already used ones
    used_vals = set(str(v) for v in already_assigned.values())
    remaining = [q for q in quoted if q not in used_vals]
    if param_index < len(remaining):
        return remaining[param_index]
    if quoted:
        return quoted[min(param_index, len(quoted) - 1)]

    # Last resort: last meaningful word
    words = re.findall(r"\b[a-zA-Z]\w*\b", user_prompt)
    return words[-1] if words else ""


# ---------------------------------------------------------------------------
# Phase 2 - Number and boolean generation
# ---------------------------------------------------------------------------


def _extract_numbers_from_prompt(user_prompt: str) -> list[float]:
    """Extract all numbers mentioned in the user prompt in order.

    Args:
        user_prompt: The natural language request.

    Returns:
        List of floats found in the prompt.
    """
    matches = re.findall(r"-?\d+(?:\.\d+)?", user_prompt)
    return [float(m) for m in matches]


def _generate_number_constrained(
    model: Small_LLM_Model,
    vocab: dict[str, int],
    prompt_ids: list[int],
) -> float:
    """Generate a numeric value constrained to digit/dot/minus tokens.

    Args:
        model: The loaded LLM instance.
        vocab: Token vocabulary.
        prompt_ids: Current tokenised context.

    Returns:
        The generated number as a float.
    """
    id_to_token = {v: k for k, v in vocab.items()}
    number_chars = set("0123456789.-")
    valid_ids = [
        tid
        for tstr, tid in vocab.items()
        if (s := _clean_token(tstr).strip()) and all(c in number_chars
                                                     for c in s)
    ]

    generated_ids: list[int] = []
    result = ""

    for _ in range(20):
        logits = model.get_logits_from_input_ids(prompt_ids + generated_ids)
        next_id = _constrained_argmax(logits, valid_ids)
        cleaned = _clean_token(id_to_token.get(next_id, "")).strip()
        candidate = result + cleaned
        try:
            float(candidate)
            result = candidate
            generated_ids.append(next_id)
        except ValueError:
            break

    try:
        return float(result)
    except ValueError:
        return 0.0


def _generate_boolean_value(
    model: Small_LLM_Model,
    vocab: dict[str, int],
    prompt_ids: list[int],
) -> bool:
    """Generate a boolean argument constrained to true/false tokens.

    Args:
        model: The loaded LLM instance.
        vocab: Token vocabulary.
        prompt_ids: Current tokenised context.

    Returns:
        The generated boolean value.
    """
    id_to_token = {v: k for k, v in vocab.items()}
    valid_ids = [
        tid
        for tstr, tid in vocab.items()
        if _clean_token(tstr).strip().lower() in ("true", "false")
    ]
    logits = model.get_logits_from_input_ids(prompt_ids)
    next_id = _constrained_argmax(logits, valid_ids)
    return (_clean_token(id_to_token.get(next_id, "true")).strip().lower()
            == "true")


def decode_function_call(
    model: Small_LLM_Model,
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> dict[str, Any]:
    """Translate a natural language prompt into a structured function call.

    Strategy:
    - Phase 1: constrained token-by-token for function name selection.
    - Phase 2:
        numbers  -> extract from prompt in order of appearance.
        strings  -> smart regex extraction based on parameter name/position.
        booleans -> constrain to true/false tokens only.

    Args:
        model: The loaded LLM instance.
        user_prompt: The natural language request.
        functions: Available function definitions.

    Returns:
        A dict with keys 'name' and 'parameters', ready to be serialized.
    """
    vocab = _load_vocab(model)

    # --- Phase 1: select function -------------------------------------------
    selected_fn = _select_function(model, vocab, user_prompt, functions)

    # --- Phase 2: generate arguments ----------------------------------------
    parameters: dict[str, Any] = {}
    numbers_seen = _extract_numbers_from_prompt(user_prompt)
    number_index = 0
    string_param_index = 0

    for param_name, param_def in selected_fn.parameters.items():

        if param_def.type in ("number", "integer"):
            if number_index < len(numbers_seen):
                value: Any = numbers_seen[number_index]
                number_index += 1
            else:
                arg_prompt = (
                    f'Request: "{user_prompt}"\n'
                    f"Function: {selected_fn.name}\n"
                    f"Parameters so far: {json.dumps(parameters)}\n"
                    f"Numeric value for '{param_name}': "
                )
                context_ids = model.encode(arg_prompt)[0].tolist()
                value = _generate_number_constrained(model, vocab, context_ids)

        elif param_def.type == "string":
            value = _extract_string_for_param(
                user_prompt,
                selected_fn,
                param_name,
                string_param_index,
                parameters,
            )
            string_param_index += 1

        elif param_def.type == "boolean":
            arg_prompt = (
                f'Request: "{user_prompt}"\n'
                f"Function: {selected_fn.name}\n"
                f"Is '{param_name}' true or false? "
            )
            context_ids = model.encode(arg_prompt)[0].tolist()
            value = _generate_boolean_value(model, vocab, context_ids)

        else:
            value = _extract_string_for_param(
                user_prompt, selected_fn, param_name, string_param_index,
                parameters
            )
            string_param_index += 1

        parameters[param_name] = value

    return {
        "name": selected_fn.name,
        "parameters": parameters,
    }
