# Paper-Grounded Ollama CLI

This project provides a local Python CLI for asking questions about one academic paper using a locally hosted Ollama model. It uses retrieval-augmented generation (RAG) so the model receives only relevant excerpts from the paper before answering.

## Setup

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Pull the local Ollama models to use, LLMs and embeddings to test againt:

```powershell
ollama pull qwen3.5:4b
ollama pull mxbai-embed-large

## Optional extras used for testing/comparison
ollama pull llama3
ollama pull nomic-embed-text
ollama pull snowflake-arctic-embed
```

## Default Configuration

The default configuration is set up as below:

```text
Chat model: qwen3.5:4b
Embedding model: mxbai-embed-large
Chunk words: 120
Chunk overlap: 30
Embedding batch size: 1
Top K: 5
Max distance: 0.5
Index directory: .paper_index_mxbai
```

`mxbai-embed-large` needed smaller chunks than `nomic-embed-text` during testing, so the defaults use 120-word chunks and batch size 1 for reliable indexing.

## Index the Paper

```powershell
python app.py index
```

This extracts text from:

```text
base_data\CLI Papaer -Research Landscape of Agentic AI and LLM - Apps, Challenges and Future Direction.pdf
```

and stores a local vector index in the model-specific index folder, for example `.paper_index_mxbai/`.

## Ask Questions

Start an interactive chat:

```powershell
python app.py chat
```

Ask single question:

```powershell
python app.py ask "What challenges of agentic AI are discussed in the paper?"
```

Show retrieval evidence while testing (debug mode):

```powershell
python app.py ask "What cybersecurity risks are discussed?" --debug
```

## How Paper-Only Answering Works

The app uses a two-stage grounding process:

1. The PDF is extracted into page text, cleaned, and split into overlapping word chunks.
2. Ollama creates embeddings for each chunk.
3. Chroma stores each chunk, its embedding, and page metadata.
4. A user question is embedded with the same embedding model. Key to use the same embedding model.
5. Chroma retrieves the closest paper chunks.
6. The app checks the closest retrieval distance against `--max-distance`.
7. Only if retrieval is less then the max-distance are the excerpts sent to the chat model.
8. The system prompt tells the model to answer only from the excerpts and cite source pages.

## Refusal Messages

There are two refusal stages, depending on where the question is blocked.

If retrieval cannot find paper excerpts close enough to the question, the app blocks the request before calling the LLM and prints:

```text
No relevant documents found. The paper I am referencing does not have relevant information for your question so I cannot answer.
```

If retrieval succeeds but the LLM decides the excerpts still do not directly answer the question, the model will return:

```text
I cannot answer that from the provided paper.
```

This distinction is useful for debugging: retrieval-stage messages mean the vector search/distance threshold rejected the question, while the LLM-stage message means the model received excerpts but did not find enough direct evidence in them.

## Useful Options

Use a different chat model:

```powershell
python app.py chat --chat-model llama3
```

Use a different embedding model. The app will use a matching index folder automatically:

```powershell
python app.py index --embed-model nomic-embed-text
python app.py ask "What challenges are discussed?" --embed-model nomic-embed-text --debug
```

Loosen retrieval if valid paper questions are refused too often:

```powershell
python app.py chat --max-distance 0.65
```

Tighten retrieval if unrelated questions are answered too often:

```powershell
python app.py chat --max-distance 0.4
```

Rebuild the index from a different PDF:

```powershell
python app.py index --pdf "C:\path\to\paper.pdf"
```

## Architecture Diagrams

The `diagrams/` folder contains Draw.io diagrams showing:

- `Setup_steps.drawio`: PDF extraction, chunking, embedding, and Chroma indexing.
- `Question_submitted_flow.drawio`: question embedding, retrieval, distance checking, prompt construction, and LLM answering.
