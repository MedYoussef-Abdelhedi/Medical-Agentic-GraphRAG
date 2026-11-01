
import json
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings

# =====================================
# 1. PATHS
# =====================================

BASE = Path(__file__).resolve().parents[1]

INPUT = BASE / "data" / "medquad_chunks.json"
OUTPUT = BASE / "data" / "medquad_embeddings.npy"

# =====================================
# 2. LOAD CHUNKS
# =====================================

with open(INPUT, "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [
    chunk["text_for_embedding"]
    for chunk in chunks
]

print("Nombre de chunks :", len(texts))

# =====================================
# 3. INITIALIZE EMBEDDING MODEL
# =====================================

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,
        "batch_size": 32
    }
)

# =====================================
# 4. GENERATE EMBEDDINGS
# =====================================

print("Génération des embeddings...")

vectors = model.embed_documents(texts)

embeddings = np.asarray(
    vectors,
    dtype=np.float32
)

# =====================================
# 5. VALIDATE RESULTS
# =====================================

assert embeddings.shape == (len(chunks), 384)
assert np.isfinite(embeddings).all()

print("Forme des embeddings :", embeddings.shape)
print("Premier vecteur :", embeddings[0][:8])

# =====================================
# 6. SAVE EMBEDDINGS
# =====================================

np.save(OUTPUT, embeddings)

print("Embeddings sauvegardés :", OUTPUT)
