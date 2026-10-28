# FastAPI Gemini Chatbot

An in-memory chatbot backend built with **FastAPI** and the **Gemini API** (called directly over `httpx`), with tool calling, conversation memory (sliding window + summarization), and a **Streamlit** chat UI.

## Features

- **Gemini chat** via the REST API (`generateContent`), fully async with `httpx`
- **Session-based conversations**: every chat has its own `session_id`
- **Memory management**: sliding window + rolling summary keeps the prompt small on long chats
- **Streamlit UI**: multiple chats in the sidebar, per-chat delete, error handling
- **Session endpoints**: fetch history, list sessions, end a session

## Project structure

```
FastAPIChatbot_rag/
├── app/
│   ├── main.py
│   ├── config.py                 # add: CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, EMBED_MODEL, INDEX_DIR
│   ├── dependancies.py           # add: get_rag_pipeline() (one shared instance)
│   │
│   ├── rag/                      # NEW: framework-free RAG, no imports from routers/dependancies
│   │   ├── __init__.py
│   │   ├── schema.py             # Document(page_content, metadata), like LangChain's Document
│   │   ├── preprocessing.py      # clean text, drop corrupted Hindi lines, normalize whitespace
│   │   │
│   │   ├── loaders/
│   │   │   ├── base.py           # BaseLoader: load(), lazy_load()
│   │   │   └── pymupdf_loader.py # PyMuPDFLoader(path): one Document per page + metadata
│   │   │
│   │   ├── splitters/
│   │   │   ├── base.py           # BaseSplitter: split_text(), split_documents()
│   │   │   └── recursive.py      # RecursiveCharacterTextSplitter(chunk_size, overlap, separators)
│   │   │
│   │   ├── embeddings/
│   │   │   ├── base.py           # BaseEmbeddings: embed_documents(), embed_query()
│   │   │   └── gemini.py         # Gemini embedContent over httpx (batched, async)
│   │   │
│   │   ├── vectorstores/
│   │   │   ├── base.py           # BaseVectorStore: add_documents(), similarity_search(), save(), load()
│   │   │   └── in_memory.py      # numpy matrix + cosine similarity, persist to disk
│   │   │
│   │   ├── retrievers/
│   │   │   ├── base.py           # BaseRetriever: retrieve(query) -> list[Document]
│   │   │   └── vector.py         # VectorStoreRetriever(top_k, score_threshold)
│   │   │
│   │   ├── prompts.py            # context template + format_docs() with source/page citations
│   │   └── pipeline.py           # RAGPipeline: ingest(pdf_path), retrieve(query), build_context(query)
│   │
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── memory_service.py
│   │   ├── chat_service.py
│   │   ├── tools.py              # add: SearchDocs tool (calls RAGPipeline)
│   │   ├── tool_clients.py
│   │   └── rag_service.py        # glue: upload/ingest, list docs, delete doc
│   │
│   ├── routers/
│   │   ├── chat.py
│   │   ├── sessions.py
│   │   └── rag.py                # POST /rag/upload, GET /rag/documents, DELETE /rag/documents/{id}
│   │
│   ├── schemas/
│   │   ├── chat.py
│   │   └── rag.py                # upload/query request-response models
│   └── utils/timing.py
│
├── data/                         # gitignored
│   ├── uploads/                  # raw PDFs (health_doc.pdf goes here)
│   └── index/                    # saved vectors + chunk metadata
│
├── notebooks/                    # move test1/test2.ipynb here (strip the API key!)
├── tests/                        # real unit tests
│   ├── test_loader.py
│   ├── test_splitter.py
│   ├── test_vectorstore.py
│   └── fixtures/health_doc.pdf
│
├── streamlit_api.py              # later: PDF uploader in sidebar
├── .env  .gitignore  requirements.txt  pyproject.toml  README.md
```

Dependency direction: `routers → dependancies → services`. Services should not import from `dependancies` (this avoids circular imports).

## Setup

### 1. Install

```bash
pip install fastapi uvicorn pydantic-settings httpx requests streamlit
```

### 2. Configure `.env`

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_key
WEATHER_API_KEY=your_weatherapi_key

# optional (defaults shown)
GEMINI_MODEL=gemini-3.1-flash-lite
RETRY_LLM_CALL=4
TRIGGER=8
KEEP_LAST=4
```

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini API key |
| `GEMINI_MODEL` | Gemini model id used for chat and summaries |
| `RETRY_LLM_CALL` | Max model/tool turns per user message |
| `TRIGGER` | Summarize when history has more than this many messages |
| `KEEP_LAST` | Approximate number of recent messages kept after summarizing |

### 3. Run

Use two terminals, both from the project root.

```bash
# Terminal 1: backend
uvicorn app.main:app --reload

# Terminal 2: UI
streamlit run streamlit_api.py
```

- API docs: http://127.0.0.1:8000/docs
- UI: http://localhost:8501

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/chat/send` | Send a message, get the full reply |
| GET | `/sessions/history/{session_id}` | Get the summary and history of a session |
| GET | `/sessions/get_session_ids` | List all session ids |
| POST | `/sessions/{session_id}/end` | Delete a session |

**`POST /chat/send`**

```json
// request (message: 1-200 characters, session_id: 3+ characters)
{ "session_id": "abc-123", "message": "What's the weather in Jaipur?" }

// response
{ "session_id": "abc-123", "response": "It's 32°C in Jaipur..." }
```

## How memory works

Each session is stored as:

```python
db[session_id] = {"summary": "", "history": [...]}   # history is in Gemini message format
```

On every message:

1. The user message is appended to `history`.
2. If `len(history) > TRIGGER`, older messages are summarized and removed; the last `KEEP_LAST` messages stay (the sliding window).
3. The old summary and the dropped messages are sent to Gemini to produce a **new rolling summary**.
4. The summary is passed to the model through the system instruction on later calls.

The cut point always moves back to a plain `user` message, so a `functionCall` is never separated from its `functionResponse`.

## Tool calling flow

1. The model replies with a `functionCall` (for example `KnowWeather`).
2. The server runs the tool and appends a `functionResponse` turn to history.
3. The model is called again, up to `RETRY_LLM_CALL` times, until it returns a final text answer.

Tools always return a dict, including `{"error": "..."}` on failure, so the model can explain problems to the user.

## Limitations

- History and sessions live **in memory**: restarting the server clears everything.
- The Streamlit chat list lives in `st.session_state`, so refreshing the browser tab clears it.
- Memory trigger currently uses message count, not tokens.
- Replies are returned in full (no streaming to the UI yet).

## Roadmap

- [ ] Streaming replies (`streamGenerateContent` over SSE) with `st.write_stream`
- [ ] Token-based summarization trigger using `usageMetadata.promptTokenCount`
- [ ] Persistent storage (database) for sessions and history
- [ ] Unit tests for memory and tool-call logic