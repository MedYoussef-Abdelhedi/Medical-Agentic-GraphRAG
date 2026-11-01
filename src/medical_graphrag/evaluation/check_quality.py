
from pathlib import Path
from collections import defaultdict
import hashlib

# Localisation du dataset
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "medical_dialogues_50"

files = sorted(DATA_DIR.rglob("*.txt"))

hashes = defaultdict(list)
encoding_issues = []
missing_sections = []

for file in files:
    text = file.read_text(encoding="utf-8-sig")

    # Normalisation pour comparer les documents
    normalized = " ".join(text.split())

    # Empreinte numérique du document
    fingerprint = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()

    hashes[fingerprint].append(file.name)

    # Vérifier les sections patient / doctor
    if "<02>" not in text or "<03>" not in text:
        missing_sections.append(file.name)

    # Détecter des caractères suspects
    if any(x in text for x in ["Â", "Ã", "�"]):
        encoding_issues.append(file.name)

# Résultats
duplicates = [
    names for names in hashes.values()
    if len(names) > 1
]

print("Documents :", len(files))
print("Documents uniques :", len(hashes))

print("\n--- DOUBLONS ---")
for group in duplicates:
    print(group)

print("\n--- STRUCTURE ---")
print("Fichiers avec sections manquantes :",
      len(missing_sections))

print("\n--- ENCODAGE ---")
print("Fichiers avec caractères suspects :",
      len(encoding_issues))

print("\nAnalyse terminée.")
