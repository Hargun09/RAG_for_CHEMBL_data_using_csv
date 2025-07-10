import streamlit as st
import os
import zipfile
import faiss
import pickle
import traceback
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="🧪 ChEMBL QA Chatbot", page_icon="🧬")
st.title("🧪 ChEMBL Biomedical Q&A Bot")
st.markdown("Ask me anything about ChEMBL-indexed biomedical data!")

# ================== LOAD EMBEDDING MODEL ==================
try:
    embedder = SentenceTransformer("all-mpnet-base-v2")
    st.success("✅ SentenceTransformer loaded.")
except Exception as e:
    st.error("❌ Failed to load embedding model.")
    st.code(traceback.format_exc())
    st.stop()

# ================== UNZIP IF NEEDED ==================
if not all(os.path.exists(f) for f in ["index_pkl/index.faiss", "index_pkl/index.pkl"]):
    if os.path.exists("index_pkl.zip"):
        st.write("📦 Extracting `index_pkl.zip`...")
        os.makedirs("index_pkl", exist_ok=True)
        with zipfile.ZipFile("index_pkl.zip", "r") as zip_ref:
            zip_ref.extractall("index_pkl")
        st.success("✅ Extracted index.")
    else:
        st.error("❌ `index_pkl.zip` not found.")
        st.stop()

# ================== LOAD FAISS & METADATA ==================
try:
    index = faiss.read_index("index_pkl/index.faiss")
    with open("index_pkl/index.pkl", "rb") as f:
        documents = pickle.load(f)
    st.success("✅ FAISS index and metadata loaded.")
except Exception as e:
    st.error("❌ Failed to load FAISS or documents.")
    st.code(traceback.format_exc())
    st.stop()

# ================== LOAD FLAN LLM ==================
try:
    pipe = pipeline("text2text-generation", model="sshleifer/tiny-t5", max_length=256)

    st.success("✅ LLM pipeline loaded.")
except Exception as e:
    st.error("❌ Failed to load FLAN-T5 model.")
    st.code(traceback.format_exc())
    st.stop()

# ================== USER QUERY ==================
query = st.text_input("🔎 Ask a biomedical question:")

if query:
    try:
        query_vec = embedder.encode([query])
        D, I = index.search(query_vec, k=3)
        retrieved = [documents[i] for i in I[0]]
        context = "\n\n".join(retrieved)
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        with st.spinner("🤖 Generating answer..."):
            response = pipe(prompt)
            answer = response[0]['generated_text']
            st.success("✅ Answer:")
            st.write(answer)

    except Exception as e:
        st.error("❌ Error while generating answer:")
        st.code(traceback.format_exc())
