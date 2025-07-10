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

# ========== Unzip index.zip ==========
if not os.path.exists("index_pkl/index.faiss") or not os.path.exists("index_pkl/index.pkl"):
    if os.path.exists("index.zip"):
        st.write("📦 Extracting `index.zip`...")
        with zipfile.ZipFile("index.zip", "r") as zip_ref:
            zip_ref.extractall("index_pkl")
        st.success("✅ Extracted `index.zip`.")
    else:
        st.warning("⚠️ `index.zip` not found. Will attempt to rebuild FAISS from CSV.")

# ========== Rebuild FAISS if needed ==========
def rebuild_faiss():
    st.warning("⚠️ `index.faiss` not found. Rebuilding from CSV...")

    if not os.path.exists("final_final.csv"):
        st.error("❌ CSV file `final_final.csv` not found.")
        st.stop()

    df = pd.read_csv("final_final.csv")
    if "text" not in df.columns:
        st.error("❌ CSV must have a `text` column.")
        st.stop()

    documents = [Document(page_content=row["text"]) for _, row in df.iterrows()]
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    db = FAISS.from_documents(documents, embeddings)
    db.save_local("index_pkl", index_name="index")
    st.success("✅ FAISS index rebuilt and saved.")

# ========== Validate Index Presence ==========
if not (os.path.exists("index_pkl/index.faiss") and os.path.exists("index_pkl/index.pkl")):
    rebuild_faiss()

# ========== Load Chain ==========
def load_chain():
    st.write("🚧 Loading retrieval QA chain...")

    try:
        embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        st.write("✅ Embedding model loaded.")

        db = FAISS.load_local("index_pkl", embeddings=embedding, index_name="index")
        st.write("✅ FAISS index loaded.")

        llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
        )
        st.write("✅ LLM loaded.")

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        st.write("✅ RetrievalQA chain ready.")
        return qa_chain

    except Exception as e:
        st.error(f"❌ Error while loading chain: {e}")
        raise e

qa_chain = load_chain()

# ========== User Input ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("🔎 Enter your question:")
if query:
    with st.spinner("🤖 Generating answer..."):
        try:
            result = qa_chain(query)
            answer = result["result"]
            sources = result["source_documents"]

            st.markdown(f"**💬 Answer:** {answer}")
            st.session_state.chat_history.append((query, answer))

            with st.expander("📚 Source Information"):
                for i, doc in enumerate(sources, 1):
                    st.markdown(f"**Source {i}:** {doc.page_content}")
        except Exception as e:
            st.error(f"❌ Failed to generate answer: {e}")
