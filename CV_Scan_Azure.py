import os
import re
import streamlit as st
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import AzureOpenAI
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import PromptTemplate

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")
EMBED_DEPLOYMENT = os.getenv("EMBEDDING_DEPLOYMENT")

st.set_page_config(page_title="Document-Aware HR CV Screening", layout="wide")
st.title("📄 HR CV Screening System (Document-Aware + FAISS)")

uploaded_files = st.file_uploader(
    "Upload up to 5 CV PDFs",
    type="pdf",
    accept_multiple_files=True
)

CHUNK_SIZE = 550
CHUNK_OVERLAP = 40

def extract_candidate_name(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:5]:
        if re.search(r"[a-zA-Z]", line) and not re.search(r"SUMMARY|PROFESSIONAL|CONTACT|CURRICULUM", line, re.I):
            words = line.split()
            if len(words) >= 2:
                return f"{words[0]} {words[1]}"
            elif words:
                return words[0]
    return "Unknown Candidate"

def is_position_question(question):
    keywords = ["position", "role", "job", "engineer", "developer", "manager", "analyst", "scientist", "designer"]
    return any(word in question.lower() for word in keywords)

def validate_real_position(question):
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
        return True

def generate_multi_queries(question):
    prompt = f"""
    Generate 4 alternative HR search queries 
    for retrieving relevant CV information.

    User query:
    {question}

    Return only the queries as a list.
    """
    try:
        response = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        queries = response.choices[0].message.content.split("\n")
        queries = [q.strip("- ").strip() for q in queries if q.strip()]
        queries.append(question)
        return list(set(queries))
    except:
        return [question]

def create_chunks(elements):
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
        temp_path = f"temp_{file.name}"
        with open(temp_path,"wb") as f:
            f.write(file.getbuffer())
        elements = partition_pdf(temp_path)
        full_text = "\n".join([el.text for el in elements if el.text])
        candidate = extract_candidate_name(full_text)
        chunks = create_chunks(elements)
        for i,chunk in enumerate(chunks):
            docs.append({
                "candidate": candidate,
                "chunk": chunk,
                "chunk_index": i
            })
        os.remove(temp_path)
    return docs

def get_embeddings(texts,batch_size=16):
    all_embeddings=[]
    for i in range(0,len(texts),batch_size):
        batch=texts[i:i+batch_size]
        response=client.embeddings.create(
            model=EMBED_DEPLOYMENT,
            input=batch
        )
        all_embeddings.extend([x.embedding for x in response.data])
    return all_embeddings

@st.cache_resource
def build_index(chunks):
    texts=[c["chunk"] for c in chunks]
    embeddings=get_embeddings(texts)
    dim=len(embeddings[0])
    index=faiss.IndexFlatIP(dim)
    vectors=np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)
    index.add(vectors)
    return index

def search(query,chunks,index,top_k=50,threshold=0.7):
    top_k=min(top_k,len(chunks))
    query_embedding=get_embeddings([query])[0]
    vector=np.array(query_embedding).astype("float32").reshape(1,-1)
    faiss.normalize_L2(vector)
    scores,indices=index.search(vector,top_k)
    filtered_results=[]
    for score,idx in zip(scores[0],indices[0]):
        if idx==-1: continue
        if score>=threshold:
            filtered_results.append({
                "candidate":chunks[idx]["candidate"],
                "chunk":chunks[idx]["chunk"],
                "score":score
            })
    return sorted(filtered_results,key=lambda x:x["score"],reverse=True)

HR_TEMPLATE="""
You are a professional HR recruiter.
Only evaluate candidates strictly based on the context.

RULES:
- If asked to list all candidates list them all with a simple summary for each.
- Only include candidates explicitly present in the context.
- Only include candidates matching ALL required skills.
- Do not hallucinate or include extra candidates.
- Check the whole CV context carefully.
- If the question asks about EXPERIENCE, only summarize actual experience sections, do not include skills or projects.
- Check the whole context before responding
- Check each candidate if theey match the criteria before responding

Context:
{context}

Question:
{question}

Final HR Evaluation:
"""

PROMPT=PromptTemplate(
    input_variables=["context","question"],
    template=HR_TEMPLATE
)

QA_CHAIN_PROMPT=PROMPT

if uploaded_files:

    if len(uploaded_files) > 5:
        st.warning("Upload maximum 5 CVs")
        st.stop()

    chunks = load_document_aware_cvs(uploaded_files)

    with st.expander("📂 All Chunks Preview", expanded=False):
        for c in chunks:
            with st.expander(f"Candidate: {c['candidate']} | Chunk Index: {c['chunk_index']}", expanded=False):
                st.code(c['chunk'], language="text")

    index = build_index(chunks)
    question = st.text_input("Enter your Question:")

    if st.button("Evaluate Candidates") and question:

        forbidden_words = ["ignore","forget","bypass","joke","passed","robot","previous instructions"]
        if any(word in question.lower() for word in forbidden_words):
            st.warning("⚠️ Access Denied: Ask professional HR questions only.")
            st.stop()

        if is_position_question(question):
            if not validate_real_position(question):
                st.error("❌ Invalid or non-standard job title. No candidate evaluation performed.")
                st.stop()

        generic_queries = ["list all candidates", "all candidates", "who are the candidates"]
        is_generic = any(g.lower() in question.lower() for g in generic_queries)

        if is_generic:
            candidate_names = list({c["candidate"] for c in chunks})
            st.subheader("All Candidates in Dataset")
            st.markdown(", ".join(candidate_names))
        else:
            queries = generate_multi_queries(question)
            all_results = []
            for q in queries:
                results = search(q, chunks, index, top_k=20, threshold=0.2)
                all_results.extend(results)

            unique_results = {r['chunk']: r for r in all_results}.values()
            candidate_chunks = {}
            for r in unique_results:
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

            st.subheader("✅ Final Unique Chunks for HR Evaluation")
            for cand, texts in candidate_chunks.items():
                st.markdown(f"**Candidate:** {cand}")
                for i, chunk in enumerate(texts, 1):
                    st.markdown(f"Chunk {i}:")
                    st.code(chunk, language="text")
                st.divider()