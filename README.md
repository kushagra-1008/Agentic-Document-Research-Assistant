# Document Research Assistant

A document-based research assistant that lets you upload research papers, ask questions about them, and get answers grounded in the uploaded material.

The application starts with the uploaded documents as its primary source. When the required information cannot be found there, it can ask for permission to search the web and use the retrieved information to answer the question.

## Demo

**Live App:** https://agentic-document-research-assistant-jxkxcchf79ukmchaepnmvf.streamlit.app/

**Demo Video:** https://drive.google.com/file/d/1uHsmUPywxCgkdOZ-F80zYBc4kU4QxFPw/view?usp=sharing

## What it does

The application is designed around a simple workflow:

1. Upload PDF documents.
2. Process and index the documents.
3. Ask questions about their contents.
4. Retrieve the most relevant sections.
5. Generate an answer using the retrieved information.
6. Check whether the answer is supported by the retrieved context.
7. If necessary, perform a broader retrieval and try again.
8. If the documents still do not contain enough information, ask the user before searching the web.
9. Show the references used for the response.

The idea is to keep the user's documents as the first source of information instead of immediately relying on external knowledge.

## Main Features

### PDF Research

Multiple PDF documents can be uploaded and processed together. Text is extracted from each page, divided into smaller chunks, and stored for later retrieval.

### Document Retrieval

The application uses embeddings and a vector database to find relevant sections of the uploaded documents.

Maximum Marginal Relevance (MMR) is used during retrieval to reduce redundant results and provide a more useful set of context to the language model.

### Answer Verification

Generated answers are checked against the retrieved document context.

If the answer is not sufficiently supported, the system performs another retrieval using a larger search configuration before generating the response again.

### Web Search

If the uploaded documents do not contain enough information, the application does not automatically search the internet.

Instead, it asks the user whether an online search should be performed.

Tavily is then used to retrieve external information.

### References

Responses include the document excerpts or web references used to produce the answer.

### Conversation History

The application stores conversation history locally so previous messages remain available while the stored runtime data is available.

## How it works

```text
                    PDF Files
                       |
                       v
                Text Extraction
                       |
                       v
                Text Chunking
                       |
                       v
                  Embeddings
                       |
                       v
                 ChromaDB
                       |
                       v
                 MMR Retrieval
                       |
                       v
                 Generate Answer
                       |
                       v
                Verify Answer
                       |
             +---------+---------+
             |                   |
          Supported          Unsupported
             |                   |
             v                   v
          Answer          Broader Retrieval
                                 |
                                 v
                          Generate Again
                                 |
                                 v
                       Information Missing?
                                 |
                       +---------+---------+
                       |                   |
                      No                  Yes
                       |                   |
                       v                   v
                    Answer          Ask User
                                         |
                                         v
                                   Web Search
                                         |
                                         v
                                  Generate Answer
                                         |
                                         v
                                      Verify
```

## Technology

| Part | Technology |
|---|---|
| Interface | Streamlit |
| Language Model | Groq — Llama 3.3 70B |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Database | ChromaDB |
| Retrieval | LangChain MMR |
| PDF Processing | PyPDF |
| Web Search | Tavily |
| Language | Python |

## Retrieval

Documents are split into overlapping chunks and embedded into a Chroma vector store. MMR retrieval is used to balance relevance and diversity. When the initial retrieval is insufficient, the application expands the retrieval search before generating another answer.

## Running Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd Agentic-Document-Research-Assistant
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 5. Start the application

```bash
streamlit run app.py
```

## Example

A small collection of research papers can be uploaded and queried together.

For example:

- Attention Is All You Need
- BERT
- Retrieval-Augmented Generation
- Word2Vec
- ResNet

Questions can range from simple document-specific queries such as:

```text
What is the main idea behind the Transformer architecture?
```

to questions involving multiple documents:

```text
How is BERT related to the Transformer architecture?
```

If a question requires information outside the uploaded papers, the application can switch to an online search after user approval.

## Improvements

The main improvement in this project is the way it handles the question-answering process rather than treating retrieval as the final step.

The system first tries to answer using the uploaded documents and then checks whether the generated response is supported by the retrieved context. If the information is not sufficient, it performs a broader retrieval and generates the response again. This gives the document collection another chance before looking outside it.

MMR retrieval is used to avoid returning several highly similar chunks and instead provide a more useful set of relevant context.

Another important part of the workflow is the decision to keep web search under user control. When the uploaded documents are insufficient, the application does not silently mix external information into the answer. It asks the user before performing a Tavily search.

The application also keeps conversation history and shows the references associated with responses. This makes it possible to follow a research session while still being able to see where the information came from.

## Future Work

The current version uses a single conversation and a shared document collection. The next step is to turn it into a more complete research workspace.

- Multiple independent conversations
- Create, rename, switch between, and delete conversation threads
- Conversation-specific document collections
- Document management with a remove option beside each uploaded document
- Persistent research sessions
- Separate history and context for each conversation
- Better management of document and web references
- More structured decision-making between retrieval, verification, and web search
- Independent verification of regenerated answers
- Support for additional document formats
- Multi-user workspaces

The longer-term goal is to make each conversation behave like an independent research workspace rather than having one shared chat and document collection.

## Project Structure

```text
Agentic-Document-Research-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

`data/`, `chroma_db/`, and `chat_history.json` are generated locally at runtime and are not committed to the repository.

## Notes

The deployed version uses temporary local storage for uploaded documents, the vector database, and conversation history. These files are associated with the running application environment and are not intended to provide permanent cloud storage.

For a future multi-conversation version, persistent storage would be introduced for documents, vector indexes, and conversation state.

## Submission

**GitHub Repository:** https://github.com/kushagra-1008/Agentic-Document-Research-Assistant.git

**Live Application:** https://agentic-document-research-assistant-jxkxcchf79ukmchaepnmvf.streamlit.app/

**Demo Video:** https://drive.google.com/file/d/1uHsmUPywxCgkdOZ-F80zYBc4kU4QxFPw/view?usp=sharing
