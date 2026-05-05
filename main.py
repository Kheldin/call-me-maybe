
import numpy as np
from llm_sdk import Small_LLM_Model


def generate(model: Small_LLM_Model, prompt: str, max_new_tokens: int = 50) -> str:
    """Generate text token by token from a prompt using greedy decoding.

    Args:
        model: The loaded Small_LLM_Model instance.
        prompt: The input text to continue.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The generated text (without the original prompt).
    """
    # 1. Tokenise the prompt into a list of integer IDs
    input_ids: list[int] = model.encode(prompt)[0].tolist()

    print(f"Prompt tokenisé : {len(input_ids)} tokens")
    print(f"Génération de {max_new_tokens} tokens...\n")

    generated_ids: list[int] = []

    for _ in range(max_new_tokens):
        # 2. Get logits for the next token (~150k scores, one per vocab token)
        logits: list[float] = model.get_logits_from_input_ids(
            input_ids + generated_ids
        )

        # 3. Greedy decoding: pick the token with the highest score (numpy only)
        next_token_id: int = int(np.argmax(logits))

        # 4. Append to the generated sequence
        generated_ids.append(next_token_id)

        # 5. Decode and print the token in real time
        token_text: str = model.decode([next_token_id])
        print(token_text, end="", flush=True)

        # 6. Stop if we hit the end-of-sequence token
        if next_token_id == model._tokenizer.eos_token_id:
            print("\n[EOS atteint]")
            break

    print("\n")
    return model.decode(generated_ids)


def main() -> None:
    """Load the model and generate text from a sample prompt."""
    print("=== Chargement du modèle ===\n")
    model = Small_LLM_Model()
    print(f"Modèle chargé sur : {model._device} ({model._dtype})\n")

    prompt = "The theory of relativity states that"

    print(f"=== Prompt ===\n{prompt}\n")
    print("=== Génération ===")

    result = generate(model, prompt, max_new_tokens=50)

    print("=== Résultat complet ===")
    print(f"{prompt}{result}")


if __name__ == "__main__":
    main()