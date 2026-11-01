
import json
from pathlib import Path
from statistics import mean, median

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

# =====================================
# 1. PATHS
# =====================================

BASE = Path(__file__).resolve().parents[1]

INPUT = BASE / "data" / "medquad_sample.json"
OUTPUT = BASE / "data" / "medquad_chunks.json"

# =====================================
# 2. LOAD DATASET
# =====================================

with open(INPUT, "r", encoding="utf-8") as f:
    records = json.load(f)

print("Nombre de documents :", len(records))

# =====================================
# 3. ANALYZE ANSWER LENGTHS
# =====================================

lengths = [
    len(record["answer"])
    for record in records
]

print("\n--- ANALYSE DES RÉPONSES ---")
print("Longueur minimale :", min(lengths))
print("Longueur moyenne :", round(mean(lengths)))
print("Longueur médiane :", round(median(lengths)))
print("Longueur maximale :", max(lengths))

# =====================================
# 4. CONFIGURE TEXT SPLITTER
# =====================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    length_function=len
)

# =====================================
# 5. GENERATE CHUNKS
# =====================================

all_chunks = []

for doc_id, record in enumerate(records):

    question = record["question"]
    answer = record["answer"]

    # Split the medical answer
    chunks = text_splitter.split_text(answer)

    for chunk_id, chunk in enumerate(chunks):

        all_chunks.append({
            "document_id": doc_id,
            "chunk_id": f"{doc_id}_{chunk_id}",
            "question": question,
            "content": chunk,
            "text_for_embedding": (
                f"Question: {question}\n"
                f"Answer: {chunk}"
            ),
            "source_file": record["source_file"],
            "source_url": record["source_url"]
        })

# =====================================
# 6. SAVE CHUNKS
# =====================================

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(
        all_chunks,
        f,
        ensure_ascii=False,
        indent=2
    )

# =====================================
# 7. DISPLAY RESULTS
# =====================================

print("\n--- CHUNKING RESULTS ---")

print("Documents initiaux :", len(records))
print("Chunks générés :", len(all_chunks))

print("\n--- PREMIER CHUNK ---")

first = all_chunks[0]

print("ID :", first["chunk_id"])
print("Question :", first["question"])
print("Texte :", first["content"][:400])
print("Longueur :", len(first["content"]))

print("\nFichier généré :", OUTPUT)
