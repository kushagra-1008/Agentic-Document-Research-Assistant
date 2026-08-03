import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st

DATA_DIR = "data"
VECTOR_DB_DIR = "chroma_db"
COLLECTION_NAME = "documents"
PROMPT = ChatPromptTemplate.from_template("""
You are an Agentic Document Research Assistant.
You must answer ONLY using the supplied context.
If the answer cannot be found in the context, reply EXACTLY:
-"I couldn't find sufficient information in the uploaded documents."
-Do not use outside knowledge.
-Do not make assumptions or infer facts that are not explicitly supported by the context.
-When the context contains page metadata, cite the document name and page number in your answer.

Context:
{context}

Question:
{question}
""")
##################################################################################################

# -----------------------------
# Load LLM and Embedding model
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


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



def answer_question(question):

    # -----------------------------
    # Load Vector Store
    # -----------------------------

    retriever = create_retriever()

    if retriever is None:
        return (
            "No indexed documents were found.\n\nPlease upload one or more PDF files and click 'Save & Ingest PDFs' before asking questions.",
            [],
        )

    # -----------------------------
    # Retrieve Documents
    # -----------------------------
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return (
            "I couldn't find sufficient information in the uploaded documents.",
            [],
        )

    # -----------------------------
    # Format Context
    # -----------------------------
    context = format_docs(retrieved_docs)

    answer = generate_answer(
        context=context,
        question=question,
    )

    return answer, retrieved_docs


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
    st.session_state.messages = []

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

    # -----------------------------
    # Store User Message
    # -----------------------------
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(query)

    # -----------------------------
    # Generate Answer
    # -----------------------------
    with st.chat_message("assistant"):

        with st.spinner("Retrieving..."):

            answer, retrieved_docs = answer_question(query)

        st.markdown(answer)

        citations = []

        if retrieved_docs:

            with st.expander("📚 Sources"):

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

    # -----------------------------
    # Save Assistant Message
    # -----------------------------
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "citations": citations,
        }
    )

    st.rerun()
