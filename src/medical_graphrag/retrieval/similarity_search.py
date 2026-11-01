
import json
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================
# 1. LOAD CHUNKS AND EMBEDDINGS
# =====================================

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

with open(DATA / "medquad_chunks.json",
          "r", encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = np.load(
    DATA / "medquad_embeddings.npy",
    allow_pickle=False
)

assert embeddings.shape == (len(chunks), 384)

print("Chunks disponibles :", len(chunks))
print("Embeddings :", embeddings.shape)

# =====================================
# 2. LOAD THE SAME EMBEDDING MODEL
# =====================================

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# =====================================
# 3. USER QUERY
# =====================================

query = input("\nEnter your question: ").strip()

if not query:
    query = "What are the symptoms of acromegaly?"

print("\nQuestion :", query)

# =====================================
# 4. EMBED THE QUERY
# =====================================

query_vector = np.asarray(
    model.embed_query(query),
    dtype=np.float32
)

# =====================================
# 5. COMPUTE COSINE SIMILARITIES
# =====================================

# Embeddings are normalized, so dot
# product equals cosine similarity.
scores = embeddings @ query_vector

# =====================================
# 6. RETRIEVE TOP-K CHUNKS
# =====================================

top_k = 3
top_indices = np.argsort(scores)[::-1][:top_k]

# =====================================
# 7. DISPLAY RESULTS
# =====================================

print("\n--- TOP 3 RESULTS ---")

for rank, idx in enumerate(top_indices, start=1):
    chunk = chunks[int(idx)]

    print(f"\n===== RESULT {rank} =====")
    print("Similarity score :", round(float(scores[idx]), 4))
    print("Chunk ID :", chunk["chunk_id"])
    print("Original question :", chunk["question"])
    print("Content :", chunk["content"][:600])
    print("Source :", chunk["source_url"])
