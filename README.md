# Secure Software Engineering RAG Assistant

A practical, source-grounded RAG (Retrieval-Augmented Generation) assistant that connects to a GitHub repository, indexes its source code and documentation using **tree-sitter**, and answers natural-language questions about the codebase with strict citations and automated secret masking.

---

## Key Features

1. **Code-Aware Chunking (Tree-Sitter)**:
   - Parses Java, Python, JavaScript, TypeScript, C, and C++ into AST-based chunks (classes, methods, functions).
   - Preserves exact file paths, symbol names (e.g. `UserService.authenticate`), and 1-based line numbers.
   - Handles Markdown by section headers (`#`, `##`) and structured data (JSON, YAML, XML).

2. **Hybrid Retrieval**:
   - **Dense Semantic Search**: Vector embeddings via `sentence-transformers` using `BAAI/bge-small-en-v1.5` stored in **ChromaDB**.
   - **Sparse Keyword Search**: Code-tokenized BM25 search via `rank-bm25`.
   - **Reciprocal Rank Fusion (RRF)**: Merges semantic and keyword rankings with standard $k=60$.
   - **Lightweight Reranking**: Boosts exact symbol matches, file path entity matches, and query term density.

3. **Practical Security Layer (Secret Masking)**:
   - Detects private keys, API keys (OpenAI, GitHub, Slack, Google, xAI), database connection URIs, Bearer tokens, and credential variables.
   - Masks secrets (e.g., `[REDACTED_API_KEY]`, `[REDACTED_PASSWORD]`) **before** sending retrieved context to external LLMs.

4. **Source-Grounded Answers & Citations**:
   - Explicit citations in every answer: `[1] src/security/AuthService.java lines 42-68`.
   - Distinguishes between direct evidence and inferences.
   - Strict hallucination prevention: returns `"Insufficient evidence from the repository"` when evidence is lacking.

5. **Grok (xAI) Integration + Provider Abstraction**:
   - Connects to Grok API (`grok-2-latest`) via an extensible client abstraction.
   - Built-in fallback/mock mode for automated evaluation and offline testing without an active API key.

6. **Single-Page Developer UI**:
   - Repository connection & indexing status.
   - Interactive file explorer to view source files.
   - Chat interface with quick prompt chips for common software engineering questions.
   - Expandable source snippet cards with code preview and secret redaction badges.

---

## Project Structure

```
software-engineering-rag/
│
├── app/
│   ├── __init__.py
│   ├── ingestion.py       # Git cloning, tree-sitter AST parsing, language chunking
│   ├── indexing.py        # SQLite metadata, ChromaDB embeddings, BM25 manager
│   ├── retrieval.py       # Semantic search, BM25, RRF fusion, lightweight reranker
│   ├── security.py        # Secret detection & pre-prompt masking layer
│   ├── llm.py             # LLM client abstraction (Grok & Mock) & grounded prompt
│   ├── api.py             # FastAPI backend REST endpoints & static file serving
│   └── static/
│       └── index.html     # HTML/CSS/JS frontend UI
│
├── data/
│   ├── repositories/      # Cloned git repositories
│   ├── chroma/            # Persistent ChromaDB vector database
│   └── database/          # SQLite metadata database (metadata.db)
│
├── tests/
│   ├── test_security.py   # Secret detection & masking unit tests
│   ├── test_ingestion.py  # Tree-sitter parsing & chunking tests
│   ├── test_indexing.py   # SQLite, BM25, and vector store tests
│   ├── test_retrieval.py  # RRF fusion and reranking tests
│   ├── test_llm.py        # LLM client & citation formatting tests
│   └── test_api.py        # FastAPI endpoint integration tests
│
├── eval/
│   └── evaluation.py      # Benchmark evaluation suite for 7 core SE questions
│
├── .env                   # Local environment configuration
├── .env.example           # Example configuration template
├── .gitignore
├── requirements.txt
├── README.md
└── main.py                # Server entry point
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy `.env.example` to `.env` and set your Grok API key (optional; if omitted, the system runs with the offline synthesizer):

```env
GROK_API_KEY=your_xai_grok_api_key_here
GROK_BASE_URL=https://api.x.ai/v1
GROK_MODEL=grok-2-latest
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5
```

### 3. Launch the Server

```bash
python main.py
```

The application will be live at **`http://localhost:8000`**.

---

## Using the Web UI

1. Open **`http://localhost:8000`** in your browser.
2. Under **Connect Repository**, paste a GitHub repository URL (e.g., `https://github.com/pallets/flask`) or enter a local folder path, then click **Index Codebase**.
3. Browse the codebase structure in the **Repository Files** sidebar; click any file to inspect its source code.
4. Ask any question in the chat bar, or click one of the quick preset chips:
   - *How does authentication work?*
   - *Explain UserService.*
   - *Where is this function used?*
   - *Which class implements this API?*
   - *Why might this API return 401?*
   - *Explain the login flow.*
   - *Which components could be affected if UserService changes?*
5. Inspect the generated answer, citations (`[1] path/file lines X-Y`), and expand the **Retrieved Grounding Sources** dropdown to view raw code snippets.

---

## Running Tests & Evaluation

### Run Unit & Integration Tests (25 tests)

```bash
python -m pytest tests/ -p no:typeguard
```

### Run Benchmark Evaluation

```bash
python eval/evaluation.py
```

Outputs the pass rate and verification report across all 7 core software engineering questions plus negative hallucination tests:

```
[PASS] Q1: How does authentication work? -> Grounded with 4 citations, secrets redacted
[PASS] Q2: Explain UserService. -> Grounded with 4 citations
[PASS] Q3: Where is this function used? -> Grounded with 4 citations
[PASS] Q4: Which class implements this API? -> Grounded with 4 citations
[PASS] Q5: Why might this API return 401? -> Grounded with 4 citations
[PASS] Q6: Explain the login flow. -> Grounded with 4 citations
[PASS] Q7: Which components could be affected if UserService changes? -> Grounded with 4 citations
[PASS] Q8: What is the recipe for cookies? -> Correctly returned 'Insufficient evidence'

Benchmark Summary: 8/8 Passed (100.0% Pass Rate)
```

