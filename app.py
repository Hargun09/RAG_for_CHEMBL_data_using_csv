import streamlit as st
import pandas as pd
import zipfile, os, shutil
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# --- Page Setup ---
st.set_page_config(page_title="🧬 ChEMBL QA", layout="centered")
st.markdown("""
<style>
  .stApp { background-color: #ffe6f0; }
  html, body, [class*="css"] { font-family: 'Times New Roman', Times, serif; color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)
st.title("🧬 ChEMBL QA Chatbot")

# --- HF Token ---
try:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    st.success("✅ Hugging Face token loaded.")
except:
    st.warning("⚠️ HF token not found.")

# --- Prepare Index Folder ---
if not os.path.exists("index"):
    os.makedirs("index", exist_ok=True)
    if os.path.exists("index.zip"):
        with zipfile.ZipFile("index.zip","r") as z: z.extractall("index")
    if os.path.exists("pkl_file.zip"):
        with zipfile.ZipFile("pkl_file.zip","r") as z: z.extractall("index")

# --- Load / Upload Data ---
uploaded = st.file_uploader("Upload CSV (with 'text' column) to build index:", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    df.to_csv("data.csv", index=False)
    st.success("CSV uploaded.")

def get_doc_df():
    if os.path.exists("data.csv"):
        df = pd.read_csv("data.csv")
    elif os.path.exists("data.zip"):
        with zipfile.ZipFile("data.zip","r") as z: z.extractall()
        df = pd.read_csv("data.csv")
    else:
        st.error("Please upload CSV or provide data.zip.")
        st.stop()
    return df

# --- Rebuild FAISS if needed ---
def rebuild():
    st.warning("🔁 Building FAISS index from data...")
    df = get_doc_df()
    if "text" not in df.columns:
        st.error("CSV must contain 'text' column.")
        st.stop()
    docs = [Document(page_content=r["text"]) for _,r in df.iterrows()]
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(docs, emb)
    db.save_local("index", index_name="index")
    st.success("✅ FAISS index saved.")

if not os.path.exists("index/index.faiss"):
    rebuild()

# --- Load the chain ---
def load_chain():
    st.write("🔧 Loading chain...")
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.load_local("index", embeddings=emb, index_name="index", allow_dangerous_deserialization=True)
    llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1", model_kwargs={"temperature":0.5,"max_new_tokens":512})
    return RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever(search_kwargs={"k":4}), return_source_documents=True)

qa = load_chain()
st.success("🔗 Chain ready.")

# --- Chat UI ---
if "history" not in st.session_state: st.session_state.history = []
q = st.text_input("Ask a question:")
if q:
    with st.spinner("Thinking..."):
        res = qa(q)
    st.write("**Answer:**", res["result"])
    st.session_state.history.append((q, res["result"]))
    with st.expander("Sources"):
        for i,doc in enumerate(res["source_documents"],1):
            st.markdown(f"📄 **Source {i}:** {doc.page_content}")
