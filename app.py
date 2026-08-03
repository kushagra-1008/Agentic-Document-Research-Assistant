import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
import json
load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
from tavily import TavilyClient


CHAT_HISTORY_FILE = "chat_history.json"
DATA_DIR = "data"
VECTOR_DB_DIR = "chroma_db"
COLLECTION_NAME = "documents"

PROMPT = ChatPromptTemplate.from_template("""

You are an Agentic Document Research Assistant.

You must answer ONLY using the supplied context.

If the answer cannot be found in the context, reply EXACTLY:

"I couldn't find sufficient information in the uploaded documents."

Do not use outside knowledge.

Do not make assumptions or infer facts that are not explicitly supported by the context.

When the context contains page metadata, cite the document name and page number in your answer.

Context:
{context}

Question:
{question}
""")
##################################################################################################

# -----------------------------
# Load LLM and Embedding model
# -----------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0,
# )

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# --------------------------------------------------
# Text Splitter
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)


def load_vectorstore():

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_DB_DIR,
    )


def create_retriever():

    vectorstore = load_vectorstore()

    collection = vectorstore.get()

    if len(collection["ids"]) == 0:
        return None

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
        },
    )

    return retriever

# --------------------------------------------------
# Document Ingestion
# --------------------------------------------------


def ingest_documents(pdf_path):

    reader = PdfReader(pdf_path)

    documents = []

    for page_num, page in enumerate(reader.pages):

        text = page.extract_text()

        if text and text.strip():

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num + 1,
                    },
                )
            )

    print(f"Loaded {len(documents)} pages.")

    chunks = text_splitter.split_documents(documents)

    print(f"Total Chunks: {len(chunks)}")

    vectorstore = load_vectorstore()
    vectorstore.add_documents(chunks)

    return {
        "pages": len(documents),
        "chunks": len(chunks),
    }
    
    
def format_docs(docs):
    return "\n\n".join(f"""Source: {doc.metadata.get("source")}
Page: {doc.metadata.get("page")}

{doc.page_content}
""" for doc in docs)


def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return []
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

    
def save_chat_history(messages):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=4, ensure_ascii=False)


def generate_answer(context, question):

    chain = PROMPT | llm | StrOutputParser()

    try:
        answer = chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        return answer

    except Exception as e:
        return f"Error generating answer:\n\n{e}"


def verify_answer(question, context, answer):

    verification_prompt = ChatPromptTemplate.from_template("""
You are a verification system.

Given the user's question, the retrieved document context and the generated answer,
determine whether the answer is completely supported by the context.

Reply with ONLY one word:

SUPPORTED

or

UNSUPPORTED

Question:
{question}

Context:
{context}

Answer:
{answer}
""")

    chain = verification_prompt | llm | StrOutputParser()

    try:
        result = chain.invoke(
            {
                "question": question,
                "context": context,
                "answer": answer,
            }
        )

        return result.strip().upper()

    except Exception:
        return "SUPPORTED"


def web_search(query):

    response = tavily.search(
        query=query,
        search_depth="advanced",
        max_results=5,
    )

    results = response.get("results", [])

    if not results:
        return None, None

    context_parts = []
    sources = []

    for i, result in enumerate(results, 1):

        title = result.get("title", "Unknown")
        url = result.get("url", "")
        content = result.get("content", "")

        for junk in [
            "Appearance",
            "From Wikipedia, the free encyclopedia",
            "Jump to navigation",
            "Jump to search",
            "Skip to content",
            "Open in app",
            "Sign in",
            "Login",
            "Search",
        ]:
            content = content.replace(junk, "")

        context_parts.append(
            f"""
    Source {i}
    Title: {title}

    Content:
    {content}
    """
        )

        sources.append(
            {
                "title": title,
                "url": url,
                "content": content,
            }
        )

    context = "\n\n------------------------\n\n".join(context_parts)

    return context, sources


def generate_final_answer(question, document_context, web_context):

    prompt = ChatPromptTemplate.from_template("""
You are an Agentic Document Research Assistant.

Your task is to answer the user's question using ONLY the information provided below.

==========================
Uploaded Documents
==========================

{document_context}

==========================
Web Search Results
==========================

{web_context}

Question:
{question}

Instructions:

- Prefer information from the uploaded documents whenever possible.
- Use web search results only to supplement missing information.
- Never use your own knowledge.
- If the information is not present, say so.
- Do NOT repeat these instructions.
- Do NOT mention the prompt.
- Begin directly with the answer.
""")

    chain = prompt | llm | StrOutputParser()

    return chain.invoke(
        {
            "question": question,
            "document_context": document_context,
            "web_context": web_context,
        }
    )

def answer_question(question):

    retriever = create_retriever()

    if retriever is None:
        return (
            "No indexed documents were found.\n\nPlease upload one or more PDF files.",
            [],
            False,
            False,
        )

    # -----------------------------
    # Retrieve Documents
    # -----------------------------
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return (
            "I couldn't find sufficient information in the uploaded documents.",
            [],
            False,
            True,
        )

    context = format_docs(retrieved_docs)

    answer = generate_answer(
        context=context,
        question=question,
    )

    # -----------------------------
    # Need Web Search?
    # -----------------------------
    if "I couldn't find sufficient information in the uploaded documents." in answer:

        return (
            answer,
            retrieved_docs,
            False,
            True,
        )

    # -----------------------------
    # Verify
    # -----------------------------
    verification = verify_answer(
        question=question,
        context=context,
        answer=answer,
    )

    self_corrected = False

    if verification == "UNSUPPORTED":

        self_corrected = True

        retriever = load_vectorstore().as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 6,
                "fetch_k": 15,
            },
        )

        retrieved_docs = retriever.invoke(question)

        context = format_docs(retrieved_docs)

        answer = generate_answer(
            context=context,
            question=question,
        )

    return (
        answer,
        retrieved_docs,
        self_corrected,
        False,
    )

# =================================================================================================================
#                                                   Streamlit UI
# =================================================================================================================

# --------------------------------------------------
# Page Config
# --------------------------------------------------

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="🤖",
    layout="wide",
)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()

if "pending_web_search" not in st.session_state:
    st.session_state.pending_web_search = False
    
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""
# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("📄 PDF Documents")

    st.subheader("Uploaded PDFs")

    os.makedirs(DATA_DIR, exist_ok=True)

    pdfs = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))

    if pdfs:
        for pdf in pdfs:
            st.caption(f"• {pdf}")
    else:
        st.caption("No PDFs uploaded.")

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("💾 Save & Ingest PDFs", use_container_width=True):

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
            st.stop()

        total_pages = 0
        total_chunks = 0
        skipped_files = []

        with st.spinner("Creating Document Embeddings..."):

            for uploaded_file in uploaded_files:

                pdf_path = os.path.join(
                    DATA_DIR,
                    uploaded_file.name,
                )

                # Skip already uploaded PDFs
                if os.path.exists(pdf_path):
                    skipped_files.append(uploaded_file.name)
                    continue

                # Save PDF
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Ingest
                stats = ingest_documents(pdf_path)

                total_pages += stats["pages"]
                total_chunks += stats["chunks"]

            st.success(
                    f"""
                ✅ Successfully processed

                • {len(uploaded_files) - len(skipped_files)} PDF(s)

                • {total_pages} pages

                • {total_chunks} chunks
                """
            )

        if skipped_files:
            st.info(
                "Skipped existing files:\n\n"
                + "\n".join(f"• {file}" for file in skipped_files)
            )
    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.messages = []

        save_chat_history([])

        st.success("Chat cleared.")

        st.rerun()
# --------------------------------------------------
# Main
# --------------------------------------------------

st.title("🤖 Agentic Document Research Assistant")

st.caption("Ask questions about your uploaded documents.")

# --------------------------------------------------
# Chat History
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if "citations" in message:

            with st.expander("📚 Sources"):

                for citation in message["citations"]:

                    st.markdown(f"**{citation['source']}**")

                    st.caption(citation["text"])

# --------------------------------------------------
# Chat Input
# --------------------------------------------------

query = st.chat_input("Ask anything about your documents...")

if query:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    st.session_state.pending_query = query

    save_chat_history(st.session_state.messages)

    st.rerun()
if st.session_state.pending_query:

    query = st.session_state.pending_query

    with st.chat_message("assistant"):

        with st.spinner("Retrieving..."):

            answer, retrieved_docs, self_corrected, needs_web_search = answer_question(query)

        # -----------------------------
        # Need Web Search?
        # -----------------------------

        if needs_web_search:

            st.warning(
                "I couldn't find sufficient information in the uploaded documents."
            )

            st.write(
                "Would you like me to search the web?"
            )

            st.session_state.pending_web_search = True

        if st.session_state.pending_web_search:

            col1, col2 = st.columns(2)

            with col1:
                approve = st.button(
                    "Search Web",
                    use_container_width=True,
                )

            with col2:
                decline = st.button(
                    "Cancel",
                    use_container_width=True,
                )

            # -----------------------------
            # User approved web search
            # -----------------------------
            if approve:

                with st.spinner("Searching the web..."):

                    # Search Tavily
                    try:
                        web_context, web_sources = web_search(query)

                    except Exception as e:

                        st.error(f"Web search failed: {e}")

                        st.session_state.pending_query = ""
                        st.session_state.pending_web_search = False

                        st.stop()

                    if not web_context:

                        st.error("No relevant web results were found.")

                        st.session_state.pending_query = ""
                        st.session_state.pending_web_search = False

                        st.stop()
                    # Existing document context
                    document_context = format_docs(retrieved_docs)

                    # Generate final answer
                    answer = generate_final_answer(
                        question=query,
                        document_context=document_context,
                        web_context=web_context,
                    )
                    merged_context = f"""
                    Uploaded Documents:

                    {document_context}

                    --------------------------------

                    Web Search Results:

                    {web_context}
                    """

                    verification = verify_answer(
                        question=query,
                        context=merged_context,
                        answer=answer,
                    )
                    if "UNSUPPORTED" in verification:
                        answer = (
                            "I found relevant information on the web, "
                            "but I couldn't confidently verify the generated response "
                            "using the available evidence."
                        )
                st.markdown(answer)

                citations = []

                with st.expander("Web Sources"):

                    for source in web_sources:

                        st.markdown(f"**{source['title']}**")

                        st.caption(source["url"])

                        citations.append(
                            {
                                "source": source["title"],
                                "page": "Web",
                                "text": source["content"][:300],
                            }
                        )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                )

                save_chat_history(st.session_state.messages)

                st.session_state.pending_query = ""
                st.session_state.pending_web_search = False

                st.rerun()


            if decline:
                st.session_state.pending_query = ""
                st.session_state.pending_web_search = False
                st.rerun()
        else:

            if self_corrected:

                st.info(
                    "Self-correction triggered. The assistant performed an additional retrieval before generating the final answer."
                )

            st.markdown(answer)

            citations = []

            if retrieved_docs:

                with st.expander("Sources"):

                    for doc in retrieved_docs:

                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:300].strip() + "..."

                        st.markdown(f"**{source}** — Page {page}")
                        st.caption(snippet)

                        citations.append(
                            {
                                "source": source,
                                "page": page,
                                "text": snippet,
                            }
                        )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                }
            )

            save_chat_history(st.session_state.messages)

            st.session_state.pending_query = ""

            st.rerun()