from pathlib import Path

data_dir = Path(
    r"C:\Users\medyo\Desktop\Chat_Rag_Graph\Graph-rag-medical\medical_dialogues_50"
)

files = sorted(data_dir.rglob("*.txt"))

print("Dossier trouvé :", data_dir.exists())
print("Nombre de fichiers :", len(files))

total_characters = 0

for file in files:
    text = file.read_text(encoding="utf-8-sig")
    total_characters += len(text)

    print(f"{file.name} | {len(text)} caractères")

print("\nTotal caractères :", total_characters)