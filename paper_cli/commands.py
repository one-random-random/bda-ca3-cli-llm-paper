from __future__ import annotations

import argparse
from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from paper_cli.ollama_client import embed_texts
from paper_cli.pdf import build_chunks, ensure_pdf_exists
from paper_cli.qa import answer_question, retrieve_context
from paper_cli.ui import console
from paper_cli.vector_store import get_collection, recreate_collection


def index_paper(args: argparse.Namespace) -> None:
    pdf_path = Path(args.pdf).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    ensure_pdf_exists(pdf_path)

    console.print(Panel.fit(f"Indexing\n[bold]{pdf_path.name}[/bold]", title="Paper CLI"))
    chunks = build_chunks(pdf_path, args.chunk_words, args.chunk_overlap)
    if not chunks:
        raise RuntimeError("No extractable text was found in the PDF.")

    collection = recreate_collection(index_dir)
    batch_size = args.batch_size
    total = len(chunks)

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [chunk.text for chunk in batch]
        embeddings = embed_texts(args.embed_model, texts)
        collection.add(
            ids=[chunk.chunk_id for chunk in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "pdf_file": pdf_path.name,
                    "page": chunk.page,
                    "chunk_number": chunk.chunk_number,
                    "source_text": chunk.text,
                }
                for chunk in batch
            ],
        )
        console.print(f"Indexed {min(start + batch_size, total)}/{total} chunks")

    console.print(f"[green]Done.[/green] Vector index stored at: {index_dir}")


def chat(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir).expanduser().resolve()
    if not index_dir.exists():
        raise RuntimeError("No paper index found. Run: python app.py index")

    collection = get_collection(index_dir)
    if collection.count() == 0:
        raise RuntimeError("The paper index is empty. Run: python app.py index")

    console.print(
        Panel.fit(
            "Ask questions about the indexed paper.\nType [bold]exit[/bold], [bold]quit[/bold], or press Ctrl+C to leave.",
            title="Paper-Grounded LLM CLI",
        )
    )

    while True:
        try:
            question = Prompt.ask("\n[bold cyan]Question[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nGoodbye.")
            return

        if question.lower() in {"exit", "quit", "q"}:
            console.print("Goodbye.")
            return
        if not question:
            continue

        try:
            context, pages, distances = retrieve_context(
                collection=collection,
                question=question,
                embed_model=args.embed_model,
                top_k=args.top_k,
                max_distance=args.max_distance,
            )
            if context is None:
                console.print(
                    Panel(
                        "The paper I am referencing does not have relevant information for your question so I cannot answer.",
                        title="Answer",
                        border_style="yellow",
                    )
                )
                if args.debug and distances:
                    console.print(f"[dim]Retrieval distances: {distances}[/dim]")
                    console.print(f"[dim]Retrieval distances: {distances}[/dim]")
                continue

            answer = answer_question(
                question=question,
                context=context,
                pages=pages,
                chat_model=args.chat_model,
                temperature=args.temperature,
                num_predict=args.num_predict,
                debug=args.debug,
            )
            console.print(Panel(Markdown(answer), title="Answer", border_style="green"))
            if context is not None:
                console.print(f"[dim]Source pages: {', '.join(f'p. {page}' for page in pages)}[/dim]")
            if args.debug:
                console.print(f"[dim]Retrieval distances: {distances}[/dim]")
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")


def ask(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir).expanduser().resolve()
    if not index_dir.exists():
        raise RuntimeError("No paper index found. Run: python app.py index")

    collection = get_collection(index_dir)
    if collection.count() == 0:
        raise RuntimeError("The paper index is empty. Run: python app.py index")

    context, pages, distances = retrieve_context(
        collection=collection,
        question=args.question,
        embed_model=args.embed_model,
        top_k=args.top_k,
        max_distance=args.max_distance,
    )
    if context is None:
        console.print("I cannot answer that from the provided paper.")
        if args.debug and distances:
            console.print(f"[dim]Retrieval distances: {distances}[/dim]")
        return

    answer = answer_question(
        question=args.question,
        context=context,
        pages=pages,
        chat_model=args.chat_model,
        temperature=args.temperature,
        num_predict=args.num_predict,
        debug=args.debug,
    )
    console.print(Markdown(answer))
    console.print(f"[dim]Source pages: {', '.join(f'p. {page}' for page in pages)}[/dim]")
    if args.debug:
        console.print(f"[dim]Retrieval distances: {distances}[/dim]")
