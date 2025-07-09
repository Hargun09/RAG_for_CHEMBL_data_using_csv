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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from huggingface_hub import login
import torch
import pandas as pd
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.schema.document import Document
import zipfile

# ========== Hugging Face Login ==========
try:
    HUGGINGFACE_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
    login(token=HUGGINGFACE_TOKEN, new_session=True)
    st.success("🔐 Hugging Face login successful.")
except KeyError:
    st.warning("⚠️ Hugging Face token not found in secrets. Proceeding without login.")

# ========== Load Model ==========
st.write("⚙️ Loading model and tokenizer...")

model_id = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=False,
    max_new_tokens=300,
    temperature=0.3,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=pipe)
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["#####"])
chunks = text_splitter.split_documents([documents])

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
db = FAISS.from_documents(chunks, embedding=embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 4})

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    return_source_documents=True
)

st.success("✅ Knowledge base ready.")

# ========== QA Interface ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("🔎 Ask a biomedical question:")
if query:
    result = qa_chain.invoke({'question': query, 'chat_history': st.session_state.chat_history})
    answer = result['answer']
    st.session_state.chat_history.append((query, answer))

# Show chat history
for i, (q, a) in enumerate(st.session_state.chat_history):
    st.markdown(f"**🧠 Q{i+1}:** {q}")
    st.markdown(f"**💬 A{i+1}:** {a}")
