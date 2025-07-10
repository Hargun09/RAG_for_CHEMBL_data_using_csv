import streamlit as st
import os
import zipfile
import traceback
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceInferenceAPI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from huggingface_hub import login

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

# ================== LOGIN TO HUGGING FACE ==================
try:
    HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
    login(token=HUGGINGFACE_TOKEN)
except Exception as e:
    st.warning("⚠️ Hugging Face login failed.")
    print("Login error:", e)

# ================== LLM ==================
llm = HuggingFaceInferenceAPI(
    api_key=st.secrets["HUGGINGFACE_TOKEN"],
    model_name="google/flan-t5-large",
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

if query and isinstance(query, str) and query.strip() != "":
    try:
        with st.spinner("🤖 Generating answer..."):
            result = qa_chain.invoke(query)
            st.write("✅ Answer:")
            st.write(result)
    except Exception as e:
        st.error("❌ An error occurred while generating the answer.")
        st.code(traceback.format_exc())
