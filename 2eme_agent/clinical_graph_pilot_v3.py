"""Etape 13C - pilote de relations medicales sourcees, avec revue humaine obligatoire.

Mode par defaut: --preview (sans reseau, sans ecriture).
  python clinical_graph_pilot_v3.py --preview
  python clinical_graph_pilot_v3.py --prepare
  # relire et renseigner review_clinical_graph_v1.csv
  python clinical_graph_pilot_v3.py --check-review
  python clinical_graph_pilot_v3.py --apply
  python clinical_graph_pilot_v3.py --verify

Le pilote n'utilise aucun LLM, n'ecrit aucune donnee medicale non approuvee
et ne modifie pas l'index ni les nœuds documentaires MedQuAD existants.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

from import_medquad_v3_neo4j import DATA, PREFIX, load_payload, settings, verify

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "reports" / "clinical_graph_pilot_v1"
CANDIDATES_PATH = OUT_DIR / "clinical_candidates_v1.json"
REVIEW_PATH = OUT_DIR / "review_clinical_graph_v1.csv"
REVIEW_FIELDS = ["candidate_id", "disease", "symptom", "evidence_chunk_id", "evidence_phrase", "decision", "reviewer", "review_note"]

# Catalogue pilote de phrases historiques, expressément presentes dans les
# passages MedQuAD. Un candidat = un lien et une preuve textuelle precise.
# On ne deduit PAS de causalite, diagnostic, probabilite ou recommandations.
SEEDS = [
    ("acromegaly", "Acromegaly", "abnormal growth of hands and feet", "Abnormal growth of hands and feet", "1_0", "abnormal growth of the hands and feet"),
    ("acromegaly", "Acromegaly", "joint aches", "Joint aches", "1_1", "joint aches"),
    ("acromegaly", "Acromegaly", "sleep apnea", "Sleep apnea", "1_1", "sleep apnea"),
    ("acromegaly", "Acromegaly", "excessive sweating", "Excessive sweating", "1_1", "excessive sweating"),
    ("acromegaly", "Acromegaly", "headaches", "Headaches", "1_1", "headaches"),
    ("acromegaly", "Acromegaly", "impaired vision", "Impaired vision", "1_1", "impaired vision"),
    ("acromegaly", "Acromegaly", "fatigue and weakness", "Fatigue and weakness", "1_1", "fatigue and weakness"),
    ("acromegaly", "Acromegaly", "thick coarse oily skin", "Thick, coarse, oily skin", "1_1", "thick, coarse, oily skin"),
    ("adrenal insufficiency", "Adrenal insufficiency", "chronic fatigue", "Chronic fatigue", "9_0", "chronic, or long lasting, fatigue"),
    ("adrenal insufficiency", "Adrenal insufficiency", "muscle weakness", "Muscle weakness", "9_0", "muscle weakness"),
    ("adrenal insufficiency", "Adrenal insufficiency", "loss of appetite", "Loss of appetite", "9_0", "loss of appetite"),
    ("adrenal insufficiency", "Adrenal insufficiency", "weight loss", "Weight loss", "9_0", "weight loss"),
    ("adrenal insufficiency", "Adrenal insufficiency", "abdominal pain", "Abdominal pain", "9_0", "abdominal pain"),
    ("adrenal insufficiency", "Adrenal insufficiency", "low blood pressure", "Low blood pressure", "9_0", "low blood pressure"),
]

CONSTRAINTS = (
    "CREATE CONSTRAINT medquad_v3_disease_key_unique IF NOT EXISTS FOR (x:MedquadDisease) REQUIRE x.key IS UNIQUE",
    "CREATE CONSTRAINT medquad_v3_symptom_key_unique IF NOT EXISTS FOR (x:MedquadSymptom) REQUIRE x.key IS UNIQUE",
    "CREATE CONSTRAINT medquad_v3_claim_key_unique IF NOT EXISTS FOR (x:MedquadClinicalClaim) REQUIRE x.key IS UNIQUE",
)

# UNWIND + MATCH de la preuve existante : aucune creation si preuve absente.
# Execute dans UNE transaction, rejetee si le nombre de lignes est incorrect.
WRITE_CLAIMS = """
UNWIND $rows AS row
MATCH (chunk:MedquadChunk {key: row.chunk_key, index_version: 'V3', content_sha256: row.chunk_content_sha256})
MERGE (d:MedquadDisease {key: row.disease_key})
  ON CREATE SET d.name = row.disease, d.index_version = 'V3'
MERGE (s:MedquadSymptom {key: row.symptom_key})
  ON CREATE SET s.name = row.symptom, s.index_version = 'V3'
MERGE (claim:MedquadClinicalClaim {key: row.claim_key})
SET claim.candidate_id = row.candidate_id,
    claim.evidence_phrase = row.evidence_phrase,
    claim.evidence_chunk_id = row.evidence_chunk_id,
    claim.chunk_content_sha256 = row.chunk_content_sha256,
    claim.source_url = row.source_url,
    claim.corpus_sha256 = $corpus_sha256,
    claim.reviewed_by = row.reviewer,
    claim.review_note = row.review_note,
    claim.review_status = 'approved',
    claim.index_version = 'V3',
    claim.relation_type = 'HAS_SYMPTOM',
    claim.data_note = 'historical MedQuAD evidence; not current clinical guidance'
MERGE (d)-[:HAS_SYMPTOM]->(s)
MERGE (claim)-[:SUBJECT]->(d)
MERGE (claim)-[:OBJECT]->(s)
MERGE (claim)-[:SUPPORTED_BY]->(chunk)
RETURN count(*) AS matched_rows
"""

VERIFY_CLAIMS = """
MATCH (claim:MedquadClinicalClaim {index_version:'V3', corpus_sha256:$corpus_sha256})
OPTIONAL MATCH (claim)-[:SUBJECT]->(d:MedquadDisease)
OPTIONAL MATCH (claim)-[:OBJECT]->(s:MedquadSymptom)
OPTIONAL MATCH (claim)-[:SUPPORTED_BY]->(c:MedquadChunk)
RETURN claim.key AS claim_key, claim.candidate_id AS candidate_id,
       claim.evidence_phrase AS evidence_phrase, claim.evidence_chunk_id AS evidence_chunk_id,
       claim.chunk_content_sha256 AS chunk_content_sha256,
       claim.source_url AS source_url, claim.reviewed_by AS reviewer,
       claim.review_note AS review_note, claim.review_status AS review_status,
       collect(DISTINCT d.key) AS disease_keys,
       collect(DISTINCT s.key) AS symptom_keys,
       collect(DISTINCT c.key) AS chunk_keys
"""


def make_candidates(payload) -> list[dict]:
    """Verifier les preuves exactes contre les V3 existants, aucune inference."""
    by_chunk = {row["chunk_id"]: row for row in payload.chunks}
    candidates = []
    pairs: set[tuple[str, str]] = set()
    for i, (dslug, disease, sslug, symptom, cid, phrase) in enumerate(SEEDS, 1):
        if (dslug, sslug) in pairs:
            raise ValueError("Paire maladie/symptome duplicatee dans le catalogue")
        pairs.add((dslug, sslug))
        chunk = by_chunk.get(cid)
        if chunk is None:
            raise ValueError(f"Chunk requis {cid} absent du V3")
        if phrase.casefold() not in chunk["content"].casefold():
            raise ValueError(f"Phrase justificative absente ou modifiee dans {cid}: {phrase!r}")
        expected_doc_id = 1 if dslug == "acromegaly" else 9
        if chunk["document_id"] != expected_doc_id:
            raise ValueError(f"Provenance maladie/document invalide : {cid}")
        if disease.casefold() not in chunk["question"].casefold():
            raise ValueError(f"Titre du document non concordant pour {cid}")
        candidates.append({
            "candidate_id": f"C{i:03d}",
            "disease": disease,
            "symptom": symptom,
            "disease_key": f"{PREFIX}:clinical:disease:{dslug.replace(' ', '-')}",
            "symptom_key": f"{PREFIX}:clinical:symptom:{sslug.replace(' ', '-')}",
            "claim_key": f"{PREFIX}:clinical:claim:C{i:03d}",
            "evidence_chunk_id": cid,
            "chunk_key": f"{PREFIX}:chunk:{cid}",
            "chunk_content_sha256": hashlib.sha256(chunk["content"].encode("utf-8")).hexdigest(),
            "evidence_phrase": phrase,
            "source_url": chunk["source_url"],
            "source_file": chunk["source_file"],
        })
    if len(set(x["claim_key"] for x in candidates)) != len(candidates):
        raise ValueError("Claims dupliques")
    return candidates


def candidate_manifest(payload, candidates):
    return {
        "schema": "medquad-v3-clinical-pilot-v1",
        "corpus_sha256": payload.chunks_sha256,
        "status": "candidate_only_requires_manual_review",
        "method": "hand-curated exact phrases, no LLM, no clinical verification",
        "candidates": candidates,
    }


def make_review_csv(candidates) -> str:
    out = io.StringIO(newline="")
    w = csv.DictWriter(out, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    w.writeheader()
    for x in candidates:
        w.writerow({**{k: x.get(k, "") for k in REVIEW_FIELDS}, "decision": "pending", "reviewer": "", "review_note": ""})
    return out.getvalue()


def load_reviews(path: Path, candidates: list[dict]) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Fichier de revue manquant : {path}. Lancer --prepare.")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != REVIEW_FIELDS:
            raise ValueError("Colonnes du fichier de revue alterees")
        rows = list(reader)
    expected = {c["candidate_id"]: c for c in candidates}
    if len(rows) != len(expected):
        raise ValueError("Nombre de candidats dans la revue incorrect")
    seen: set[str] = set()
    reviewed = []
    for r in rows:
        cid = r["candidate_id"]
        if cid not in expected or cid in seen:
            raise ValueError(f"Candidat inconnu ou duplique dans la revue : {cid!r}")
        seen.add(cid)
        orig = expected[cid]
        for col in ("disease", "symptom", "evidence_chunk_id", "evidence_phrase"):
            if r[col] != orig[col]:
                raise ValueError(f"Preuve ou sujet modifie dans la revue : {cid}, {col}")
        decision = r["decision"].strip().lower()
        if decision not in ("pending", "approved", "rejected"):
            raise ValueError(f"Decision invalide {cid}: {decision!r}")
        reviewer = r["reviewer"].strip()
        note = r["review_note"].strip()
        if decision != "pending" and not reviewer:
            raise ValueError(f"Reviewer requis pour la decision {cid}")
        if len(reviewer) > 120 or len(note) > 1000:
            raise ValueError("Champ de revue trop long")
        reviewed.append({**orig, "decision": decision, "reviewer": reviewer, "review_note": note})
    return reviewed


def read_manifest_file(path: Path, payload, candidates) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Candidats manquants: {path}. Lancer --prepare.")
    observed = json.loads(path.read_text(encoding="utf-8"))
    expected = candidate_manifest(payload, candidates)
    if observed != expected:
        raise ValueError("Fichier de candidats altere, corpus modifie ou catalogue non concordant")


def approved_review(reviews):
    pending = [r["candidate_id"] for r in reviews if r["decision"] == "pending"]
    if pending:
        raise ValueError(f"Revue incomplete: {len(pending)} candidats pending (ex. {pending[:4]})")
    approved = [r for r in reviews if r["decision"] == "approved"]
    if not approved:
        raise ValueError("Aucune relation approuvee, rien a importer")
    return approved


def execute_write_transaction(tx, approved, corpus_sha256):
    result = tx.run(WRITE_CLAIMS, rows=approved, corpus_sha256=corpus_sha256)
    record = result.single()
    n = record["matched_rows"] if record is not None else -1
    if n != len(approved):
        raise RuntimeError(f"Import incomplet : {n}/{len(approved)} preuves trouvees. Transaction annulee.")
    return n


def verify_clinical(driver, database: str, approved: list[dict], corpus_sha256: str) -> None:
    records, _, _ = driver.execute_query(VERIFY_CLAIMS, database_=database, corpus_sha256=corpus_sha256)
    observed = {r["claim_key"]: r for r in records}
    expected = {r["claim_key"]: r for r in approved}
    if set(observed) != set(expected):
        raise RuntimeError(f"Claims Neo4j inattendus: {len(observed)} observes, {len(expected)} approuves")
    for key, src in expected.items():
        row = observed[key]
        simple_props = {
            "candidate_id": src["candidate_id"],
            "evidence_phrase": src["evidence_phrase"],
            "evidence_chunk_id": src["evidence_chunk_id"],
            "chunk_content_sha256": src["chunk_content_sha256"],
            "source_url": src["source_url"],
            "reviewer": src["reviewer"],
            "review_note": src["review_note"],
            "review_status": "approved",
            "disease_keys": [src["disease_key"]],
            "symptom_keys": [src["symptom_key"]],
            "chunk_keys": [src["chunk_key"]],
        }
        for prop, value in simple_props.items():
            if row[prop] != value:
                raise RuntimeError(f"Verification Neo4j : {key}, propriete {prop} incorrecte")
    # Controle de toutes les relations medicales (y compris liens sans claim).
    pairs = {(x["disease_key"], x["symptom_key"]) for x in approved}
    q = """MATCH (d:MedquadDisease)-[:HAS_SYMPTOM]->(s:MedquadSymptom)
           RETURN d.key AS dk, s.key AS sk"""
    edges, _, _ = driver.execute_query(q, database_=database)
    observed_pairs = [(r["dk"], r["sk"]) for r in edges]
    if set(observed_pairs) != pairs or len(observed_pairs) != len(pairs):
        raise RuntimeError("Liens HAS_SYMPTOM manquants, dupliques ou non approuves")


def run_cli(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--check-review", action="store_true")
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    payload = load_payload(DATA)
    candidates = make_candidates(payload)
    print(f"CORPUS V3: {len(payload.documents)} documents, {len(payload.chunks)} chunks | SHA256: {payload.chunks_sha256}")
    print(f"PILOTE: {len(candidates)} relations candidates, extraites par phrases exactes; aucune evaluation clinique")

    if args.preview:
        for c in candidates:
            print(f"  {c['candidate_id']} | {c['disease']} --HAS_SYMPTOM--> {c['symptom']} "
                  f"| preuve [{c['evidence_chunk_id']}] : {c['evidence_phrase']}")
        print("PREVIEW: PASS, aucune connexion Neo4j, aucun fichier ecrit.")
        return
    if args.prepare:
        if CANDIDATES_PATH.exists() or REVIEW_PATH.exists():
            raise FileExistsError("Fichiers de revue deja presents, --prepare ne les ecrase jamais")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CANDIDATES_PATH.write_text(json.dumps(candidate_manifest(payload, candidates), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        REVIEW_PATH.write_text("\ufeff" + make_review_csv(candidates), encoding="utf-8")
        print("Candidats:", CANDIDATES_PATH)
        print("Revue a renseigner:", REVIEW_PATH)
        print("IMPORTANT: toutes les decisions sont PENDING. Aucun nœud clinique importe.")
        return

    read_manifest_file(CANDIDATES_PATH, payload, candidates)
    reviews = load_reviews(REVIEW_PATH, candidates)
    counts = {state: sum(r["decision"] == state for r in reviews) for state in ("pending", "approved", "rejected")}
    print("REVUE:", counts)
    if args.check_review:
        if counts["pending"]:
            print("CHECK-REVIEW: INCOMPLETE - completer ou rejeter chaque candidat avant --apply")
        else:
            print("CHECK-REVIEW: PASS - decisions renseignees, sans verification medicale independante")
        return

    approved = approved_review(reviews)
    uri, username, password, database = settings()
    from neo4j import GraphDatabase
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        verify(driver, database, payload)  # Verifie le graphe documentaire integral avant ecriture.
        if args.apply:
            for cypher in CONSTRAINTS:
                driver.execute_query(cypher, database_=database)
            with driver.session(database=database) as session:
                n = session.execute_write(execute_write_transaction, approved, payload.chunks_sha256)
            print(f"APPLY: {n} relations candidates approuvees traitees")
        verify_clinical(driver, database, approved, payload.chunks_sha256)
    print(f"VERIFY: PASS - {len(approved)} claims et liens cliniques verifies, sources documentaires intactes")


if __name__ == "__main__":
    run_cli()
