import streamlit as st
import pandas as pd
import zipfile, os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# ========== Page Setup ==========
st.set_page_config(page_title="🧬 ChEMBL QA", layout="centered")
st.markdown("""
<style>
    .stApp { background-color: #ffe6f0; }
    html, body, [class*="css"] { font-family: 'Times New Roman', serif; color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)
st.title("🧬 ChEMBL QA Chatbot: Female Reproductive Tract Diseases")

# ========== Hugging Face Token ==========
try:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    st.success("🔐 Hugging Face token loaded.")
except Exception as e:
    st.warning("⚠️ HF token not found.")
    print(e)

# ========== Unzip data.zip ==========
if not os.path.exists("final_final.csv") and os.path.exists("data.zip"):
    st.write("📦 Extracting `data.zip`...")
    with zipfile.ZipFile("data.zip", "r") as zip_ref:
        zip_ref.extractall()
    st.success("✅ Extracted `final_final.csv` from data.zip")

# ========== Build FAISS if needed ==========
def build_faiss():
    if not os.path.exists("final_final.csv"):
        st.error("❌ `final_final.csv` not found even after unzip.")
        st.stop()

    df = pd.read_csv("final_final.csv")

    if "text" not in df.columns:
        st.error("❌ CSV must contain a `text` column.")
        st.stop()

    st.info("📚 Building FAISS index from ChEMBL data...")
    docs = [Document(page_content=row["text"]) for _, row in df.iterrows()]
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    db = FAISS.from_documents(docs, embedding)
    db.save_local("index", index_name="index")
    st.success("✅ FAISS index built.")

if not os.path.exists("index/index.faiss"):
    build_faiss()

# ========== Load QA Chain ==========
def load_chain():
    st.write("🔄 Loading QA Chain...")
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    db = FAISS.load_local(
        "index",
        embeddings=embedding,
        index_name="index",
        allow_dangerous_deserialization=True
    )
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.5, "max_new_tokens": 512}
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

qa_chain = load_chain()

# ========== Input Box ==========
if "history" not in st.session_state: st.session_state.history = []

query = st.text_input("🔎 Ask a biomedical question:")
if query:
    with st.spinner("🤖 Thinking..."):
        result = qa_chain(query)
    st.markdown(f"**💬 Answer:** {result['result']}")
    with st.expander("📄 Sources"):
        for i, doc in enumerate(result["source_documents"], 1):
            st.markdown(f"**Source {i}:** {doc.page_content}")
    st.session_state.history.append((query, result["result"]))
