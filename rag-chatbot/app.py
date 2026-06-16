import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("🧠 Final Stable RAG Chatbot")

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load docs
with open("data/docs.txt", "r", encoding="utf-8") as f:
    docs = f.readlines()

st.success(f"Loaded {len(docs)} documents")

# Create embeddings (SAFE)
doc_embeddings = model.encode(docs)

# Convert properly for FAISS
doc_embeddings = np.array(doc_embeddings).astype("float32")

dimension = doc_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

query = st.text_input("Ask something:")

if query:
    try:
        query_vec = model.encode([query])
        query_vec = np.array(query_vec).astype("float32")

        distances, indices = index.search(query_vec, k=1)

        answer = docs[indices[0][0]]

        st.write("### Answer:")
        st.write(answer)

    except Exception as e:
        st.error(f"Error occurred: {e}")