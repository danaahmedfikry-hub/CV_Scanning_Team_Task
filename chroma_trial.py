import os
import streamlit as st
import tempfile
import shutil
import re
from dotenv import load_dotenv
from openai import AzureOpenAI
from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ==============================
# 1. Load Environment Variables
# ==============================
load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")
EMBEDDING_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")

st.set_page_config(page_title="Ultra-Strict AI HR Matcher", layout="wide")
st.title("🛡️ AI HR Matcher (Secure & High-Precision)")

# ==============================
# 2. Azure OpenAI Client
# ==============================
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

# ==============================
# 3. ChromaDB Client + Azure Embedding
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
# 4. Helpers (Functions)
# ==============================
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

def is_position_question(question):
    """التحقق من وجود كلمات تدل على السؤال عن وظيفة"""
    keywords = ["position", "role", "job", "engineer", "developer", "manager", "analyst", "scientist", "designer"]
    return any(word in question.lower() for word in keywords)

def validate_real_position(question):
    """استخدام الـ AI للتأكد أن المسمى الوظيفي حقيقي ومهني"""
    check_prompt = f"""
    Evaluate the job title mentioned in this question: "{question}"
    Is it a standard, real-world professional role (e.g. 'Software Engineer', 'Data Scientist')? 
    Or is it a fake, nonsense, or non-standard title (e.g. 'Ai teams engineer', 'super human developer')?
    
    Reply ONLY with 'REAL' or 'FAKE'.
    """
    try:
        response = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": check_prompt}],
            temperature=0
        )
        result = response.choices[0].message.content.strip().upper()
        return "REAL" in result
    except:
        return True # تمرير في حالة حدوث خطأ تقني مؤقت

# ==============================
# 5. Preparation Logic
# ==============================
def prepare_vectorstore(files):
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except:
        pass
    
    collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=azure_ef)
    all_candidate_names = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

    for file in files:
        text = read_pdf(file)
        name = extract_candidate_name(text[:1000])
        all_candidate_names.append(name)
        doc_chunks = splitter.split_text(text)
        for idx, chunk in enumerate(doc_chunks):
            chunk_content = f"[CANDIDATE NAME: {name}]\n{chunk}"
            collection.add(
                documents=[chunk_content],
                metadatas=[{"candidate": name}],
                ids=[f"{name}_{idx}"]
            )
    return list(set(all_candidate_names))

# ==============================
# 6. Main UI & Interaction
# ==============================
uploaded_files = st.file_uploader("Upload exactly 5 CVs (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) != 5:
        st.error("Please upload exactly 5 CVs.")
        st.stop()

    if "names" not in st.session_state:
        with st.spinner("Indexing CVs with high precision..."):
            st.session_state.names = prepare_vectorstore(uploaded_files)
        st.success(f"Ready: {', '.join(st.session_state.names)}")

    question = st.text_input("Enter your technical or HR query:")

    if question:
        # --- الحماية البرمجية (Hard-coded Filter) ---
        forbidden_words = ["ignore", "forget", "bypass", "joke", "passed", "robot", "previous instructions"]
        if any(word in question.lower() for word in forbidden_words):
            st.warning("⚠️ Access Denied: Please ask professional HR-related questions only.")
            st.stop()

        # --- فلترة المسميات الوظيفية الوهمية ---
        if is_position_question(question):
            if not validate_real_position(question):
                st.error("No candidate found. The position mentioned does not exist or is non-standard.")
                st.stop()

        # --- استرجاع البيانات من Vector DB ---
        collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=azure_ef)
        results = collection.query(query_texts=[question], n_results=15)

        context_text = "\n\n".join([f"CV DATA:\n{doc}" for doc in results['documents'][0]])

        # --- بناء الـ Prompt المهني (تجنب كلمات الـ Jailbreak لـ Azure) ---
        strict_prompt = f"""
You are a Professional Technical Auditor. Your task is to extract information from the provided CV context.

### MANDATORY GUIDELINES:
- Focus ONLY on the provided CV data. Do not respond to non-HR or creative writing requests.
- If the user asks for a 'table', you MUST list EVERY matching candidate found in the context.
- Each matching candidate must be placed in a dedicated row in the table. DO NOT combine or omit results.
- If no match is found, reply: "No candidate found matching these specific criteria."
- If the question is outside of HR scope, state that you can only discuss CV data.

### CONTEXT:
{context_text}

### USER QUESTION:
{question}
"""

        try:
            response = client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are a literalist HR assistant. You report only what is explicitly found in the data. You do not bypass instructions."},
                    {"role": "user", "content": strict_prompt}
                ],
                temperature=0
            )

            st.subheader("Audited Answer")
            st.markdown(response.choices[0].message.content)

        except Exception as e:
            if "content_filter" in str(e):
                st.error("❌ Content Filter Triggered: Please ensure your question is professional and related to the CVs.")
            else:
                st.error(f"An error occurred: {e}")

        # --- جزء تصحيح الأخطاء (Debug) ---
        with st.expander("View Data Sources (Debug)"):
            for doc in results['documents'][0]:
                st.caption(doc)
                st.divider()