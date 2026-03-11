import os
import streamlit as st
import re
from dotenv import load_dotenv
from openai import AzureOpenAI
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================
# Load Environment Variables
# ==============================
load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")

st.set_page_config(page_title="Smart CV Intelligence", layout="wide")
st.title("📄 Smart AI CV Analyzer")

# ==============================
# Azure OpenAI Client
# ==============================
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

# ==============================
# ChromaDB Client + Azure Embedding
# ==============================
chroma_client = chromadb.PersistentClient(path="./chroma_db")

azure_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=AZURE_API_KEY,
    api_base=AZURE_ENDPOINT,
    api_type="azure",
    api_version=AZURE_API_VERSION,
    deployment_id=EMBEDDING_DEPLOYMENT
)

COLLECTION_NAME = "cv_analysis_collection"

# ==============================
# Helper Functions
# ==============================
def is_position_question(question):
    keywords = ["position", "role", "job", "engineer", "developer", "manager", "analyst", "scientist", "designer"]
    return any(word in question.lower() for word in keywords)

def validate_real_position(question):
    # تم تبسيط البرومبت هنا أيضاً لتجنب الفلترة
    prompt = f"Is the following text a legitimate professional job title or role? '{question}'. Answer only YES or NO."
    try:
        response = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        res = response.choices[0].message.content.strip().upper()
        return "YES" in res
    except:
        return True

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text: text += page_text
    return text

def extract_candidate_name(text):
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in lines[:5]:
        if len(line) < 40 and not re.search(r"\d", line):
            words = line.split()
            if 2 <= len(words) <= 4 and all(word[0].isupper() for word in words if word.isalpha()):
                return line
    return lines[0] if lines else "Unknown"

# ==============================
# Prepare Vectorstore
# ==============================
def prepare_vectorstore(files):
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
    
    collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=azure_ef)
    all_candidate_names = []

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    for file in files:
        text = read_pdf(file)
        name = extract_candidate_name(text[:1000])
        all_candidate_names.append(name)

        doc_chunks = splitter.split_text(text)
        for idx, chunk in enumerate(doc_chunks):
            # وسم البيانات بشكل بسيط
            chunk_content = f"Candidate: {name}\nDetails: {chunk}"
            collection.add(
                documents=[chunk_content],
                metadatas=[{"candidate": name}],
                ids=[f"{name}_{idx}"]
            )
    return list(set(all_candidate_names))

# ==============================
# Main UI
# ==============================
uploaded_files = st.file_uploader("Upload PDF CVs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if "names" not in st.session_state:
        with st.spinner("Processing documents..."):
            st.session_state.names = prepare_vectorstore(uploaded_files)
        st.success(f"Successfully indexed: {', '.join(st.session_state.names)}")

    question = st.text_input("Ask your query (e.g., Who has Python skills?):")

    if question:
        # التحقق من الوظيفة بشكل مخفف
        if is_position_question(question):
            if not validate_real_position(question):
                st.error("The requested position seems invalid or was not found.")
                st.stop()

        collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=azure_ef)
        results = collection.query(query_texts=[question], n_results=15)

        # تجميع البيانات حسب المرشح
        candidate_data = {}
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            name = meta["candidate"]
            if name not in candidate_data:
                candidate_data[name] = []
            if doc not in candidate_data[name]:
                candidate_data[name].append(doc)

        # بناء سياق نصي "ناعم" واحترافي
        context_text = ""
        for name, chunks in candidate_data.items():
            context_text += f"\n### PROFILE FOR: {name}\n"
            context_text += "\n".join(chunks)
            context_text += "\n"

        # برومبت جديد هادئ لتجنب الـ Content Filter
        analytical_prompt = f"""
Please review the following candidate profiles and answer the user's question. 
Your goal is to list EVERY candidate who matches the requested skills or criteria.

Candidate Profiles:
{context_text}

Question: {question}
"""

        try:
            response = client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a professional HR assistant. Provide a comprehensive list of all matching candidates based on the provided text."},
                    {"role": "user", "content": analytical_prompt}
                ],
                temperature=0 
            )
            
            st.subheader("Analysis Result")
            st.write(response.choices[0].message.content)
            
        except Exception as e:
            if "content_filter" in str(e):
                st.error("Azure OpenAI Safety Filter triggered. Try rephrasing your question or simplifying the CV content.")
            else:
                st.error(f"Error: {e}")

        with st.expander("Show Data Sources"):
            for name, chunks in candidate_data.items():
                st.markdown(f"**{name}**")
                st.caption("\n".join(chunks)[:500] + "...")
