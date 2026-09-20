import streamlit as st
import os
import zipfile
import traceback

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from transformers import pipeline

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
        max_length=128,
        temperature=0.3,
        device=-1  # force CPU (safe for Streamlit Cloud / HF Spaces)
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    st.success("✅ LLM loaded (flan-t5-small).")
except Exception as e:
    st.error("❌ Failed to load LLM.")
    st.exception(e)
    st.stop()

# ========== RETRIEVER & RAG CHAIN (modern LCEL replacement for RetrievalQA) ==========
retriever = db.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below. If the answer
isn't in the context, say you don't know.

Context:
{context}

Question: {input}

Answer:"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# ========== USER INPUT ==========
query = st.text_input("🔎 Ask a biomedical question:")

if query:
    try:
        with st.spinner("🤖 Generating answer..."):
            result = retrieval_chain.invoke({"input": query})
            st.success("✅ Answer:")
            st.write(result["answer"])
    except Exception as e:
        st.error("❌ Error while generating the answer.")
        st.code(traceback.format_exc())
