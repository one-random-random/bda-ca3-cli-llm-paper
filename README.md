# Paper-Grounded Ollama CLI

This project provides a local Python CLI for asking questions about one PDF paper using a local Ollama model. It uses retrieval-augmented generation so the model receives only relevant excerpts from the paper before answering.

## Setup

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Pull the local Ollama models:

```powershell
ollama pull llama3
ollama pull nomic-embed-text
```

## Index the Paper

```powershell
python app.py index
```

This extracts text from:

```text
CLI Papaer -Research Landscape of Agentic AI and LLM - Apps, Challenges and Future Direction.pdf
```

and stores a local vector index in `.paper_index/`.

## Chat

```powershell
python app.py chat
```

Ask questions in the terminal. Type `exit` or `quit` to leave.

For a single scripted question, use:

```powershell
python app.py ask "What challenges of agentic AI are discussed in the paper?"
```

The app is designed to answer only from retrieved paper excerpts. If it cannot find enough support in the paper, it should answer:

```text
I cannot answer that from the provided paper.
```

## Useful Options

Use a different chat model:

```powershell
python app.py chat --chat-model llama3.1
```

If you already have another Ollama chat model installed, use it directly:

```powershell
python app.py chat --chat-model qwen3.5:4b
```

Show retrieval distances while testing:

```powershell
python app.py chat --debug
```

If valid paper questions are refused too often, loosen retrieval slightly:

```powershell
python app.py chat --max-distance 0.9
```

If unrelated questions are answered too often, tighten retrieval:

```powershell
python app.py chat --max-distance 0.65
```

If a local model is slow, reduce the retrieved context or answer length:

```powershell
python app.py ask "What challenges are discussed?" --chat-model qwen3.5:4b --top-k 2 --num-predict 180
```

Rebuild the index from a different PDF:

```powershell
python app.py index --pdf "C:\path\to\paper.pdf"
```
