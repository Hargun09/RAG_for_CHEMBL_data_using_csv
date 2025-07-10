import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from huggingface_hub import login

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="🧪 ChEMBL QA Chatbot", page_icon="🧬")
st.title("🧪 ChEMBL Biomedical Q&A Bot")
st.markdown("Ask me anything about ChEMBL-indexed biomedical data!")

# ================== EMBEDDING MODEL ==================
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ================== PATHS ==================
index_dir = "index_pkl"
index_faiss = os.path.join(index_dir, "index.faiss")
index_pkl = os.path.join(index_dir, "index_pkl.pkl")

# ================== VERIFY FILES EXIST ==================
if not os.path.exists(index_faiss) or not os.path.exists(index_pkl):
    st.error("❌ Required FAISS index files not found in `index_pkl/`.")
    st.stop()

# ================== LOAD VECTORSTORE ==================
try:
    db = FAISS.load_local(
        folder_path=index_dir,
        embeddings=embedding,
        index_name="index_pkl.pkl"  # must match actual .pkl filename
    )
    st.success("✅ FAISS vectorstore loaded.")
except Exception as e:
    st.error(f"❌ Failed to load FAISS index: {e}")
    st.stop()

# ================== LOGIN TO HUGGING FACE ==================
try:
    HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
    login(token=HUGGINGFACE_TOKEN)
except Exception as e:
    st.warning("⚠️ Hugging Face login failed.")
    print("Login error:", e)

# ================== LLM ==================
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

# ================== RETRIEVAL CHAIN ==================
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever()
)

# ================== USER QUERY ==================
query = st.text_input("🔎 Ask a biomedical question:")
if query:
    with st.spinner("🤖 Generating answer..."):
        result = qa_chain.run(query)
        st.write("✅ Answer:")
        st.write(result)
