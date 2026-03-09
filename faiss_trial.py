import os
import streamlit as st
import numpy as np
import faiss_trial
from dotenv import load_dotenv
from openai import AzureOpenAI
from unstructured.partition.pdf import partition_pdf

load_dotenv()

# =========================
# Azure OpenAI Setup
# =========================
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")
EMBED_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(page_title="Document-Aware HR CV Screening", layout="wide")
st.title("📄 HR CV Screening System (Document-Aware + FAISS)")

uploaded_files = st.file_uploader(
    "Upload up to 5 CV PDFs",
    type="pdf",
    accept_multiple_files=True
)

# =========================
# Chunking Settings
# =========================
CHUNK_SIZE = 550
CHUNK_OVERLAP = 50

def create_chunks(elements):
    """Convert document elements into overlapping chunks"""
    texts = [el.text.strip() for el in elements if el.text]
    chunks = []
    current_chunk = ""

    for text in texts:
        if len(current_chunk) + len(text) < CHUNK_SIZE:
            current_chunk += "\n" + text
        else:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-CHUNK_OVERLAP:] + "\n" + text

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

@st.cache_data
def load_document_aware_cvs(uploaded_files):
    docs = []

    for file in uploaded_files:
        candidate = os.path.splitext(file.name)[0]
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        elements = partition_pdf(temp_path)
        chunks = create_chunks(elements)

        for i, chunk in enumerate(chunks):
            docs.append({
                "candidate": candidate,
                "chunk": chunk,
                "chunk_index": i
            })

        os.remove(temp_path)

    st.write(f"Total number of chunks in the dataset: {len(docs)}")
    return docs

def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model=EMBED_DEPLOYMENT,
            input=batch
        )
        all_embeddings.extend([x.embedding for x in response.data])
    return all_embeddings

@st.cache_resource
def build_index(chunks):
    texts = [c["chunk"] for c in chunks]
    embeddings = get_embeddings(texts)
    dim = len(embeddings[0])
    index = faiss_trial.IndexFlatIP(dim)
    vectors = np.array(embeddings).astype("float32")
    faiss_trial.normalize_L2(vectors)
    index.add(vectors)
    return index

def search(query, chunks, index, top_k=50, threshold=0.7):
    top_k = min(top_k, len(chunks))
    query_embedding = get_embeddings([query])[0]
    vector = np.array(query_embedding).astype("float32").reshape(1, -1)
    faiss_trial.normalize_L2(vector)

    scores, indices = index.search(vector, top_k)
    filtered_results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        if score >= threshold:
            filtered_results.append({
                "candidate": chunks[idx]["candidate"],
                "chunk": chunks[idx]["chunk"],
                "score": score
            })

    return sorted(filtered_results, key=lambda x: x["score"], reverse=True)

# =========================
# HR Prompt Template
# =========================
HR_TEMPLATE = """
You are a professional HR recruiter.
Only evaluate candidates strictly based on the context.

RULES:
- Only include candidates explicitly present in the context.
- Only include candidates matching ALL required skills.
- Do not hallucinate or include extra candidates.
- Do not joke or add commentary.
- Check the whole CV context for everything.
- Do not skip candidates unless they don't match criteria.

Context:
{context}

Question:
{question}

Final HR Evaluation:
"""

from langchain_core.prompts import PromptTemplate
PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=HR_TEMPLATE
)

QA_CHAIN_PROMPT = PROMPT

# =========================
# Main Streamlit Logic
# =========================
if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("Upload maximum 5 CVs")
        st.stop()

    chunks = load_document_aware_cvs(uploaded_files)
    index = build_index(chunks)

    question = st.text_input("Enter your Question:")

    if st.button("Evaluate Candidates") and question:
        with st.spinner("Analyzing all candidate CVs..."):
            results = search(question, chunks, index, top_k=20, threshold=0.2)

            if not results:
                st.warning("No candidates matched the criteria.")
            else:
                candidate_chunks = {}
                for r in results:
                    cand = r["candidate"]
                    candidate_chunks.setdefault(cand, []).append(r["chunk"])

                context_blocks = []
                for cand, texts in candidate_chunks.items():
                    combined_text = "\n\n".join(texts)
                    block = f"Candidate: {cand}\n{combined_text}"
                    context_blocks.append(block)

                context = "\n\n----------------------\n\n".join(context_blocks)

                formatted_prompt = QA_CHAIN_PROMPT.format(
                    context=context,
                    question=question
                )

                response = client.chat.completions.create(
                    model=CHAT_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "You are an HR assistant."},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0
                )

                st.subheader("HR Evaluation Result")
                st.markdown(response.choices[0].message.content)

                st.divider()
                st.subheader("Candidates Mentioned in Analysis")
                candidate_names = list(candidate_chunks.keys())
                st.markdown("**Candidates:** " + ", ".join(candidate_names))