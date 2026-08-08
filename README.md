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
git clone https://github.com/kushagra-1008/Agentic-Document-Research-Assistant.git
cd Agentic-Document-Research-Assistant
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

The main improvement in this project is that it does not treat document retrieval as the end of the question-answering process. The system first tries to answer a question from the uploaded documents and then checks whether the generated response is actually supported by the retrieved information. If the retrieved context is not enough, it gives the system another opportunity by performing a broader retrieval and generating the answer again.

The retrieval process also uses Maximum Marginal Relevance (MMR) rather than retrieving only the most similar chunks. This helps avoid returning several nearly identical pieces of information and gives the model a broader set of relevant context.

Another important part of the workflow is the handling of information that is not present in the uploaded documents. Instead of silently searching the internet, the system asks the user before using web search. If approved, Tavily is used to retrieve external information, which is combined with the available document context before producing the response.

The application also keeps conversation history between sessions and shows the document or web references associated with responses. This makes the system easier to use for ongoing research while keeping the information behind each answer visible to the user.

---

## Future Work

The current version uses a single conversation and a shared document collection. The next stage is to evolve it into a more complete research workspace with independent conversations and persistent research sessions.

- Multiple conversations with the ability to create, rename, switch between, and delete threads
- Conversation-specific document collections and retrieval context
- Persistent research sessions containing conversation history, documents, references, and web searches
- Improved decision-making between retrieval, additional retrieval, verification, and web search
- Independent verification of regenerated answers
- Better source management and filtering
- Support for additional document formats
- Multi-user workspaces with isolated conversations and document collections

---

## Demo

**Demo Video:** *Add your Google Drive or YouTube link here.*

---