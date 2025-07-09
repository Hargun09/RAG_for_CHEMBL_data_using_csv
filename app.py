import streamlit as st

# ========== Custom CSS ==========
st.markdown("""
    <style>
        .stApp {
            background-color: #ffe6f0;
        }
        html, body, [class*="css"] {
            font-family: 'Times New Roman', Times, serif;
            color: #1a1a1a;
        }
        .css-10trblm {
            font-size: 36px !important;
            font-weight: bold !important;
            color: #4a148c !important;
        }
    </style>
""", unsafe_allow_html=True)

# ========== UI ==========
st.title("🧬 ChEMBL Chatbot: Diseases of the Female Reproductive Tract")
st.write("Enter your biomedical question below:")

# ========== Libraries ==========
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.document import Document
from langchain.llms.base import LLM
import zipfile
import torch

# ========== Hugging Face Login ==========
try:
    HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
    login(token=HUGGINGFACE_TOKEN)
    st.success("🔐 Hugging Face login successful.")
except Exception as e:
    st.warning("⚠️ Hugging Face login failed. Proceeding without it.")
    print("Login error:", e)

# ========== Load Model ==========
st.write("⚙️ Loading model and tokenizer...")
model_id = "google/flan-t5-small"  # safe for CPU

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

def query_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=300, do_sample=True, temperature=0.3)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

class SimpleLLM(LLM):
    @property
    def _llm_type(self):
        return "transformer"

    def _call(self, prompt, stop=None):
        return query_model(prompt)

llm = SimpleLLM()
st.success("✅ Model loaded.")

# ========== Unzip Data ==========
with zipfile.ZipFile("data.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# ========== Load & Process ChEMBL CSV ==========
st.write("📄 Processing ChEMBL data...")
df = pd.read_csv('data/final_final.csv')

text = ""
for ind in df.index:
    row = df.loc[ind]
    sentence = (
        f"The compound {row['compound_name']} (ChEMBL ID: {row['molecule_chembl_id']}) "
        f"is associated with the disease {row['disease_name']} (MONDO ID: {row['mondo_id']}, "
        f"EFO ID: {row['efo_id']}, MeSH ID: {row['mesh_id']}). "
        f"It targets proteins with ChEMBL IDs: {row['target_chembl_ids']} "
        f"and UniProt IDs: {row['uniprot_ids']}.\n#####\n"
    )
    text += sentence

documents = Document(page_content=text, metadata={"source": "chembl_gene_disease"})
st.write("⚙️ documented")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["#####"])
st.write("⚙️ splitted...")

chunks = text_splitter.split_documents([documents])

st.write("⚙️ chunked...")


embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
st.write("⚙️ embedded...")

db = FAISS.from_documents(chunks, embedding=embeddings)
st.write("⚙️ faised...")

retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})
st.write("⚙️ retrieved")


qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, return_source_documents=True)
st.write("⚙️qa chained")

st.success("✅ Knowledge base ready.")

# ========== QA Interface ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("🔎 Ask a biomedical question:")
if query:
    result = qa_chain.invoke({'question': query, 'chat_history': st.session_state.chat_history})
    answer = result['answer']
    st.markdown(f"**💬 Answer:** {answer}")
    st.session_state.chat_history.append((query, answer))
