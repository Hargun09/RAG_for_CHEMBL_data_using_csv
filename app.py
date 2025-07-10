import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import pandas as pd
import zipfile
import os
import shutil

# ========== Page Config & Styling ==========
st.set_page_config(page_title="🧬 ChEMBL QA", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #ffe6f0;
        }
        html, body, [class*="css"] {
            font-family: 'Times New Roman', Times, serif;
            color: #1a1a1a;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 ChEMBL QA Chatbot: Female Reproductive Tract Diseases")
st.write("Ask a biomedical question related to compounds, targets, or diseases:")

# ========== Hugging Face Token ==========
try:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    st.success("🔐 Hugging Face API token loaded.")
except Exception as e:
    st.warning("⚠️ Hugging Face API token not found.")
    print("Token error:", e)

# ========== Unzip and Prepare Index Folder ==========
if not os.path.exists("index"):
    os.makedirs("index", exist_ok=True)

    if os.path.exists("index.zip"):
        st.write("📦 Extracting `index.zip`...")
        with zipfile.ZipFile("index.zip", "r") as zip_ref:
            zip_ref.extractall("index")
        st.success("✅ Extracted `index.zip`.")

    if os.path.exists("pkl_file.zip"):
        st.write("📦 Extracting `pkl_file.zip`...")
        with zipfile.ZipFile("pkl_file.zip", "r") as zip_ref:
            zip_ref.extractall("index")
        st.success("✅ Extracted `pkl_file.zip`.")

# ========== Rebuild FAISS if index.faiss is Missing ==========
def rebuild_faiss():
    st.warning("⚠️ `index.faiss` not found. Rebuilding FAISS index from CSV...")

    df = pd.read_csv("final_final.csv")  # Use your CSV filename here
    documents = [Document(page_content=row["text"]) for _, row in df.iterrows()]  # Replace 'text' with your actual column name

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    db = FAISS.from_documents(documents, embeddings)

    db.save_local("index", index_name="index")
    st.success("✅ FAISS index rebuilt and saved.")

if not os.path.exists("index/index.faiss"):
    rebuild_faiss()

# ========== Load FAISS and LLM ==========
@st.cache_resource
def load_chain():
    st.write("⚙️ Loading FAISS index and LLM...")

    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    db = FAISS.load_local("index", embeddings=embedding, index_name="index")

    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    return qa_chain

qa_chain = load_chain()

# ========== User Input ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("🔎 Enter your question:")
if query:
    with st.spinner("🤖 Generating answer..."):
        result = qa_chain(query)
        answer = result["result"]
        sources = result["source_documents"]

        st.markdown(f"**💬 Answer:** {answer}")
        st.session_state.chat_history.append((query, answer))

        with st.expander("📚 Source Information"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {doc.page_content}")
