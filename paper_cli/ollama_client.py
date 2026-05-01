from __future__ import annotations

import ollama

from paper_cli.ui import console


def embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        response = ollama.embed(model=model, input=texts)
    except Exception as exc:
        raise RuntimeError(
            f"Could not create embeddings with Ollama model '{model}'. "
            f"Ollama error: {exc}. "
            f"If the model is already installed, try a smaller --batch-size or --chunk-words value."
        ) from exc

    embeddings = response.get("embeddings")
    if not embeddings:
        raise RuntimeError(f"Ollama returned no embeddings for model '{model}'.")
    return embeddings


def chat_with_context(
    chat_model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    num_predict: int,
    debug: bool = False,
) -> str:
    try:
        response = ollama.chat(
            model=chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            think=False,
            options={"temperature": temperature, "num_predict": num_predict},
        )
        if debug:
            console.print(f"[dim]LLM response: {response}[/dim]")
    except Exception as exc:
        raise RuntimeError(
            f"Could not chat with Ollama model '{chat_model}'. Run: ollama pull {chat_model}"
        ) from exc

    return response["message"]["content"].strip()
