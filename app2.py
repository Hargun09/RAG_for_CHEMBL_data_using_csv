import streamlit as st
import os
import zipfile
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
import traceback
from langchain.chains.retrieval_qa.base import RetrievalQA

# ========== PAGE CONFIG ==========
st.set_page_config(page_title="🧪 ChEMBL QA Chatbot", page_icon="🧬")
st.title("🧪 ChEMBL Biomedical Q&A Bot")
st.markdown("Ask me anything about ChEMBL-indexed biomedical data!")

# ========== EMBEDDINGS ==========
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ========== UNZIP FAISS INDEX IF NEEDED ==========
if not all(os.path.exists(f) for f in ["index_pkl/index.faiss", "index_pkl/index.pkl"]):
    if os.path.exists("index_pkl.zip"):
        st.write("📦 Extracting `index_pkl.zip`...")
        os.makedirs("index_pkl", exist_ok=True)
        with zipfile.ZipFile("index_pkl.zip", "r") as zip_ref:
            zip_ref.extractall("index_pkl")
        st.success("✅ Extracted `index_pkl.zip`.")
    else:
        st.error("❌ `index_pkl.zip` not found.")
        st.stop()

# ========== DEBUG FILE CONTENTS ==========
try:
    st.write("📁 Contents of index_pkl:", os.listdir("index_pkl"))
except Exception as e:
    st.error(f"❌ Failed to read index_pkl/: {e}")
    st.stop()

# ========== LOAD VECTORSTORE ==========
try:
    db = FAISS.load_local(
        folder_path="index_pkl",
        embeddings=embedding,
        index_name="index",
        allow_dangerous_deserialization=True
    )
    st.success("✅ FAISS vectorstore loaded.")
except Exception as e:
    st.error(f"❌ Failed to load FAISS index:\n{e}")
    st.stop()

# ========== LOAD LLM PIPELINE ==========
try:
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=128,              # ✅ smaller max_length for stability
        temperature=0.3,
        device=-1                    # ✅ force CPU (safe for Hugging Face Spaces)
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    st.success("✅ LLM loaded (flan-t5-small).")
except Exception as e:
    st.error("❌ Failed to load LLM.")
    st.exception(e)
    st.stop()

# ========== RETRIEVER & QA CHAIN ==========
retriever = db.as_retriever(search_kwargs={"k": 3})  # ✅ fewer docs to avoid overload

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",        # ✅ safer than "stuff" for small models
    retriever=retriever
)

# ========== USER INPUT ==========
query = st.text_input("🔎 Ask a biomedical question:")

if query:
    try:
        with st.spinner("🤖 Generating answer..."):
            result = qa_chain.run(query)
            st.success("✅ Answer:")
            st.write(result)
    except Exception as e:
        st.error("❌ Error while generating the answer.")
        st.code(traceback.format_exc())
