import streamlit as st
import os
import zipfile
import traceback
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="🧪 ChEMBL QA Chatbot", page_icon="🧬")
st.title("🧪 ChEMBL Biomedical Q&A Bot")
st.markdown("Ask me anything about ChEMBL-indexed biomedical data!")

# ================== EMBEDDING MODEL ==================
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ================== CHECK & UNZIP IF NEEDED ==================
if not all(os.path.exists(f) for f in ["index_pkl/index.faiss", "index_pkl/index.pkl"]):
    if os.path.exists("index_pkl.zip"):
        st.write("📦 Extracting `index_pkl.zip`...")
        os.makedirs("index_pkl", exist_ok=True)
        with zipfile.ZipFile("index_pkl.zip", "r") as zip_ref:
            zip_ref.extractall("index_pkl")
        st.success("✅ Extracted `index_pkl.zip`.")
    else:
        st.error("❌ `index_pkl.zip` not found. Cannot continue.")
        st.stop()

# ========== Debug: Confirm extraction ==========
try:
    st.write("📁 index_pkl/ contents:", os.listdir("index_pkl"))
except Exception as e:
    st.error(f"❌ Failed to read `index_pkl/`: {e}")

# ================== LOAD VECTORSTORE ==================
try:
    db = FAISS.load_local(
        folder_path="index_pkl",
        embeddings=embedding,
        index_name="index",
        allow_dangerous_deserialization=True
    )
    st.success("✅ FAISS vectorstore loaded.")
except Exception as e:
    st.error(f"❌ Failed to load FAISS index: {e}")
    st.stop()

# ================== LOAD LLM PIPELINE (NO TOKEN NEEDED) ==================
try:
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512, do_sample=False)
    llm = HuggingFacePipeline(pipeline=pipe)
    st.success("✅ LLM pipeline loaded.")
except Exception as e:
    st.error("❌ Could not load the LLM pipeline.")
    st.exception(e)
    st.stop()

# ================== USER QUERY WITHOUT RetrievalQA ==================
query = st.text_input("🔎 Ask a biomedical question:")

if query:
    try:
        with st.spinner("🤖 Retrieving context and generating answer..."):
            # Step 1: Retrieve relevant documents
            docs = db.similarity_search(query, k=3)
            if not docs:
                st.warning("⚠️ No relevant documents found.")
                st.stop()

            # Step 2: Build prompt from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

            # Step 3: Call the model
            response = pipe(prompt, max_length=512)
            answer = response[0]['generated_text'] if response else "⚠️ No answer generated."

            st.success("✅ Answer:")
            st.write(answer)
    except Exception as e:
        st.error("❌ An error occurred:")
        st.code(traceback.format_exc())
