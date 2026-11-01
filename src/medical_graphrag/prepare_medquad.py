
from pathlib import Path
import xml.etree.ElementTree as ET
import json

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "data" / "MedQuAD" / "5_NIDDK_QA"
OUTPUT = BASE / "data" / "medquad_sample.json"

records = []
seen = set()

for file in sorted(SOURCE.rglob("*.xml")):
    root = ET.parse(file).getroot()
    source_url = root.attrib.get("url", "")

    for pair in root.findall(".//QAPair"):
        question = pair.findtext("Question", "").strip()
        answer = pair.findtext("Answer", "").strip()

        if not question or not answer:
            continue

        key = (question.lower(), answer.lower())

        if key in seen:
            continue

        seen.add(key)

        records.append({
            "question": question,
            "answer": answer,
            "source_file": file.name,
            "source_url": source_url
        })

        if len(records) >= 200:
            break

    if len(records) >= 200:
        break

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)

print("Documents préparés :", len(records))

if records:
    print("Première question :", records[0]["question"])
    print("Source :", records[0]["source_url"])
else:
    print("Aucune paire trouvée : vérifier SOURCE.")
