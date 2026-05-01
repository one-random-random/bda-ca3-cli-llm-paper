from __future__ import annotations

from chromadb.api.models.Collection import Collection

from paper_cli.ollama_client import chat_with_context, embed_texts


def format_context(documents: list[str], metadatas: list[dict]) -> tuple[str, list[int]]:
    context_blocks: list[str] = []
    pages: list[int] = []
    for number, (document, metadata) in enumerate(zip(documents, metadatas), start=1):
        page = int(metadata.get("page", 0))
        pages.append(page)
        context_blocks.append(f"[Excerpt {number} | page {page}]\n{document}")
    return "\n\n".join(context_blocks), sorted(set(pages))


def retrieve_context(
    collection: Collection,
    question: str,
    embed_model: str,
    top_k: int,
    max_distance: float,
) -> tuple[str | None, list[int], list[float]]:
    query_embedding = embed_texts(embed_model, [question])[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    if not documents or not distances:
        return None, [], []
    if float(distances[0]) > max_distance:
        return None, [], [float(distance) for distance in distances]

    context, pages = format_context(documents, metadatas)
    return context, pages, [float(distance) for distance in distances]


def answer_question(
    question: str,
    context: str,
    pages: list[int],
    chat_model: str,
    temperature: float,
    num_predict: int,
    debug: bool = False,
) -> str:
    system_prompt = (
        "You are a careful research assistant. You must answer the user's question using only "
        "the provided paper excerpts. Do not use outside knowledge.\n\n"
        "Before answering, decide whether the excerpts directly contain enough evidence to answer "
        "the user's specific question.\n\n"
        "If the excerpts do not directly answer the user's specific question, or only contain "
        "related background information, respond with exactly this sentence and nothing else:\n"
        "I cannot answer that from the provided paper.\n\n"
        "Do not explain why you cannot answer. Do not summarize related excerpts. Do not include "
        "citations when refusing.\n\n"
        "If the excerpts do directly answer the question, provide a concise answer using only "
        "that evidence. Include page citations for factual claims in the form (p. 3) or "
        "(pp. 3, 5)."
    )
    user_prompt = (
        f"Paper excerpts:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the excerpts. Available source pages: {', '.join(map(str, pages))}."
    )

    return chat_with_context(
        chat_model=chat_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        num_predict=num_predict,
        debug=debug,
    )
