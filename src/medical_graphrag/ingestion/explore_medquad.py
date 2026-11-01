
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "medquad_sample.json"

with open(DATA, "r", encoding="utf-8") as f:
    records = json.load(f)

print("Nombre de paires :", len(records))

for i, record in enumerate(records[:3], start=1):
    print("\n" + "=" * 60)
    print("DOCUMENT", i)
    print("=" * 60)

    print("QUESTION:")
    print(record["question"])

    print("\nRÉPONSE (500 premiers caractères):")
    print(record["answer"][:500])

    print("\nSOURCE:")
    print(record["source_file"])
    print(record["source_url"])
