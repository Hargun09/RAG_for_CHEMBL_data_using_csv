import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import os

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

# ========== Load FAISS & LLM ==========
@st.cache_resource
def load_chain():
    st.write("⚙️ Loading FAISS index and LLM...")
    
    embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    db = FAISS.load_local("faiss_index", embeddings=embedding, index_name="index")

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

        # Display Answer
        st.markdown(f"**💬 Answer:** {answer}")
        st.session_state.chat_history.append((query, answer))

        # Display Sources
        with st.expander("📚 Source Information"):
            for i, doc in enumerate(sources, 1):
                st.markdown(f"**Source {i}:** {doc.page_content}")
