import json
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from tavily import TavilyClient

load_dotenv()

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

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
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

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10},
    )


def ingest_documents(pdf_path):
    reader = PdfReader(pdf_path)
    documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": os.path.basename(pdf_path), "page": page_num + 1},
                )
            )

    print(f"Loaded {len(documents)} pages.")
    chunks = text_splitter.split_documents(documents)
    print(f"Total Chunks: {len(chunks)}")

    vectorstore = load_vectorstore()
    vectorstore.add_documents(chunks)

    return {"pages": len(documents), "chunks": len(chunks)}


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
        return chain.invoke({"context": context, "question": question})
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
        result = chain.invoke({"question": question, "context": context, "answer": answer})
        return result.strip().upper()
    except Exception:
        return "SUPPORTED"


def web_search(query):
    response = tavily.search(query=query, search_depth="advanced", max_results=5)
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

        sources.append({"title": title, "url": url, "content": content})

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
    return chain.invoke({
        "question": question,
        "document_context": document_context,
        "web_context": web_context,
    })


def answer_question(question):
    """Returns (answer, retrieved_docs, self_corrected, needs_web_search)."""
    retriever = create_retriever()

    if retriever is None:
        return (
            "No indexed documents were found.\n\nPlease upload one or more PDF files.",
            [],
            False,
            False,
        )

    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return (
            "I couldn't find sufficient information in the uploaded documents.",
            [],
            False,
            True,
        )

    context = format_docs(retrieved_docs)
    answer = generate_answer(context=context, question=question)

    if "I couldn't find sufficient information in the uploaded documents." in answer:
        return (answer, retrieved_docs, False, True)

    verification = verify_answer(question=question, context=context, answer=answer)
    self_corrected = False

    if verification == "UNSUPPORTED":
        self_corrected = True
        retriever = load_vectorstore().as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 15},
        )
        retrieved_docs = retriever.invoke(question)
        context = format_docs(retrieved_docs)
        answer = generate_answer(context=context, question=question)

    return (answer, retrieved_docs, self_corrected, False)

# -------------------------- streamlit -------------------------

st.set_page_config(
    page_title="Document Research Assistant",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()
if "pending_web_search" not in st.session_state:
    st.session_state.pending_web_search = False
if "pending_query" not in st.session_state:
    st.session_state.pending_query = ""

with st.sidebar:
    st.title("Documents")
    st.subheader("Available Documents")
    os.makedirs(DATA_DIR, exist_ok=True)
    pdfs = sorted(
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".pdf")
    )

    if pdfs:
        for pdf in pdfs:
            st.caption(f"• {pdf}")
    else:
        st.caption("No documents available.")
    st.divider()
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    if st.button(
        "Process Documents",
        use_container_width=True,
    ):
        if not uploaded_files:
            st.warning("Please select at least one PDF.")
            st.stop()
        total_pages = 0
        total_chunks = 0
        skipped_files = []
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                pdf_path = os.path.join(
                    DATA_DIR,
                    uploaded_file.name,
                )
                if os.path.exists(pdf_path):
                    skipped_files.append(uploaded_file.name)
                    continue

                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                stats = ingest_documents(pdf_path)
                total_pages += stats["pages"]
                total_chunks += stats["chunks"]

        st.success(
            f"""
            Successfully processed

            {len(uploaded_files) - len(skipped_files)} document(s)

            {total_pages} pages

            {total_chunks} chunks
            """
        )

        if skipped_files:
            st.info(
                "Already available:\n\n"
                + "\n".join(f"• {file}" for file in skipped_files)
            )

    st.divider()

    if st.button("Clear Conversation",use_container_width=True):
        st.session_state.messages = []
        save_chat_history([])
        st.success("Conversation cleared.")
        st.rerun()

st.title("Document Research Assistant")
st.caption("Search and explore information from your uploaded documents.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "references" in message:
            with st.expander("References"):
                for reference in message["references"]:
                    st.markdown(f"**{reference['source']}**")
                    st.caption(reference["text"])
query = st.chat_input("Ask a question about your documents")

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
        with st.spinner("Searching documents..."):
            answer, retrieved_docs, self_corrected, needs_web_search = answer_question(query)

        if needs_web_search:
            st.info(
                "No relevant information was found in the uploaded documents."
            )
            st.write("Would you like to search online?")
            st.session_state.pending_web_search = True

        if st.session_state.pending_web_search:
            col1, col2 = st.columns(2)
            with col1:
                approve = st.button(
                    "Search Online",
                    use_container_width=True,
                )
            with col2:
                decline = st.button(
                    "Cancel",
                    use_container_width=True,
                )

            if approve:

                with st.spinner("Searching online..."):
                    try:
                        web_context, web_sources = web_search(query)
                    except Exception as e:
                        st.error(f"Web search failed:\n\n{e}")
                        st.session_state.pending_query = ""
                        st.session_state.pending_web_search = False
                        st.stop()
                    if not web_context:
                        st.error("No relevant online sources were found.")
                        st.session_state.pending_query = ""
                        st.session_state.pending_web_search = False
                        st.stop()
                    document_context = format_docs(retrieved_docs)
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
                            "Relevant information was found online, "
                            "but the available sources were not sufficient "
                            "to produce a reliable response."
                        )

                st.markdown(answer)
                references = []
                with st.expander("Web References"):
                    for source in web_sources:
                        st.markdown(f"**{source['title']}**")
                        st.caption(source["url"])
                        references.append(
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
                        "references": references,
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
            st.markdown(answer)
            references = []
            if retrieved_docs:
                with st.expander("References"):
                    for doc in retrieved_docs:
                        source = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "?")
                        snippet = doc.page_content[:300].strip() + "..."
                        st.markdown(f"**{source}** — Page {page}")
                        st.caption(snippet)
                        references.append(
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
                    "references": references,
                }
            )
            save_chat_history(st.session_state.messages)
            st.session_state.pending_query = ""
            st.rerun()