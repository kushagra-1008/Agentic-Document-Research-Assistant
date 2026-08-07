# Document Research Assistant

A Retrieval-Augmented Generation (RAG) application for querying PDF documents using semantic search. The application retrieves relevant information from uploaded documents, generates grounded responses, and optionally performs a web search when the available documents do not contain sufficient information.

---

## Features

- Upload and index multiple PDF documents
- Semantic document retrieval using vector embeddings
- Maximum Marginal Relevance (MMR) retrieval for improved context diversity
- Source-grounded responses with document references
- Automatic answer verification
- Self-correction through expanded retrieval when verification fails
- Optional web search using Tavily with user approval
- Persistent conversation history
- Clean Streamlit interface

---

## Architecture

```
                 PDF Documents
                       │
                       ▼
              Text Extraction (PyPDF)
                       │
                       ▼
          Recursive Character Splitter
                       │
                       ▼
           HuggingFace Embeddings
        (BAAI/bge-small-en-v1.5)
                       │
                       ▼
                 Chroma Vector DB
                       │
                       ▼
              MMR Document Retrieval
                       │
                       ▼
                Groq Llama 3.3 70B
                       │
                       ▼
              Answer Verification
                       │
          ┌────────────┴────────────┐
          │                         │
      Supported               Unsupported
          │                         │
          ▼                         ▼
    Return Answer          Expanded Retrieval
                                      │
                                      ▼
                              Generate Again
                                      │
                                      ▼
                       Need External Information?
                                      │
                             User Approval
                                      │
                                      ▼
                               Tavily Search
                                      │
                                      ▼
                            Verified Final Answer
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| LLM | Groq (Llama 3.3 70B Versatile) |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| Framework | LangChain |
| PDF Parsing | PyPDF |
| Web Search | Tavily |

---

## Installation

Clone the repository.

```bash
git clone <repository-url>
cd <repository-name>
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

Windows

```bash
.venv\Scripts\activate
```

Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file.

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## Running the Application

```bash
streamlit run app.py
```

---

## Usage

1. Upload one or more PDF documents.
2. Click **Process Documents**.
3. Ask questions about the uploaded documents.
4. If the uploaded documents do not contain enough information, choose whether to perform an online search.
5. Review the supporting references returned with every response.

---

## Design Decisions

### Maximum Marginal Relevance (MMR)

Instead of standard similarity search, MMR retrieval is used to reduce redundant chunks and improve context diversity.

### Answer Verification

Each generated answer is verified against the retrieved context. If the response is not fully supported, the system performs a second retrieval with a broader search before generating another answer.

### User-Controlled Web Search

Rather than automatically searching the web, the application requests user approval before retrieving external information. This keeps document-based responses separate from online information.

### Source References

Every response includes the document snippets or web references used during generation, allowing users to inspect the supporting evidence.

---

## Improvements

This project extends a basic Retrieval-Augmented Generation pipeline by introducing multiple mechanisms to improve response reliability. Instead of relying solely on a single retrieval step, answers are verified against the retrieved context before being returned. If the verification step indicates that the generated response is not sufficiently supported, the application automatically performs a second retrieval using a broader search configuration and regenerates the answer.

To improve retrieval quality, Maximum Marginal Relevance (MMR) is used instead of standard similarity search. This reduces redundant document chunks and provides more diverse context to the language model.

When the uploaded documents do not contain sufficient information, the application does not automatically access external sources. Instead, it requests explicit user approval before performing a Tavily web search. The retrieved web context is then combined with the existing document context and verified again before generating the final response.

Additionally, the application maintains persistent conversation history and presents supporting document excerpts or web references alongside every answer, enabling users to understand the evidence used during response generation.

---

## Future Work

- Hybrid retrieval using BM25 and vector search
- Metadata-aware filtering
- Cross-encoder reranking
- Support for additional document formats
- Multi-user document collections

---

## Demo

**Demo Video:** *Add your Google Drive or YouTube link here.*

---

## License

This project is released under the MIT License.