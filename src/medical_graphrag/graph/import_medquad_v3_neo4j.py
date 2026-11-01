"""Etape 13B : importer seulement les liens documentaires MedQuAD V3 dans Neo4j.

Par defaut : previsualisation locale sans connexion ni ecriture.
  python import_medquad_v3_neo4j.py --dry-run
  python import_medquad_v3_neo4j.py --apply
  python import_medquad_v3_neo4j.py --verify

Ce script ne cree AUCUNE relation clinique. Aucun appel Groq/LLM.
Il n'efface jamais des donnees Neo4j. MERGE + contraintes -> relancable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
VERSION = "V3"
PREFIX = "medquad-v3"
BATCH_SIZE = 100

# SHA obtenu par le pipeline V3 valide pour CE projet; empeche un import
# accidentel d'un autre jeu de documents sous les memes identifiants.
EXPECTED_CHUNKS_SHA256 = "a0ff3068e591a8bc86f8206362fe65866d3e1f6b55477fab09b037c65d210656"

CONSTRAINTS = [
    "CREATE CONSTRAINT medquad_v3_source_key_unique IF NOT EXISTS FOR (s:MedquadSource) REQUIRE s.key IS UNIQUE",
    "CREATE CONSTRAINT medquad_v3_document_key_unique IF NOT EXISTS FOR (d:MedquadDocument) REQUIRE d.key IS UNIQUE",
    "CREATE CONSTRAINT medquad_v3_chunk_key_unique IF NOT EXISTS FOR (c:MedquadChunk) REQUIRE c.key IS UNIQUE",
]

WRITE_SOURCES = """
UNWIND $rows AS row
MERGE (s:MedquadSource {key: row.key})
SET s.source_url = row.source_url,
    s.source_file = row.source_file,
    s.index_version = 'V3',
    s.import_sha256 = $import_sha256
"""
WRITE_DOCUMENTS = """
UNWIND $rows AS row
MATCH (s:MedquadSource {key: row.source_key})
MERGE (d:MedquadDocument {key: row.key})
SET d.document_id = row.document_id,
    d.question = row.question,
    d.source_url = row.source_url,
    d.source_file = row.source_file,
    d.source_date_text = row.source_date_text,
    d.source_date_evidence_chunk_id = row.source_date_evidence_chunk_id,
    d.index_version = 'V3',
    d.import_sha256 = $import_sha256
MERGE (d)-[:FROM_SOURCE]->(s)
"""
WRITE_CHUNKS = """
UNWIND $rows AS row
MATCH (d:MedquadDocument {key: row.document_key})
MERGE (c:MedquadChunk {key: row.key})
SET c.chunk_id = row.chunk_id,
    c.document_id = row.document_id,
    c.question = row.question,
    c.content = row.content,
    c.content_sha256 = row.content_sha256,
    c.source_file = row.source_file,
    c.source_url = row.source_url,
    c.index_version = 'V3',
    c.import_sha256 = $import_sha256
MERGE (d)-[:HAS_CHUNK]->(c)
"""

READ_COUNTS = {
    "sources": "MATCH (s:MedquadSource {index_version: 'V3'}) RETURN count(s) AS n",
    "documents": "MATCH (d:MedquadDocument {index_version: 'V3'}) RETURN count(d) AS n",
    "chunks": "MATCH (c:MedquadChunk {index_version: 'V3'}) RETURN count(c) AS n",
    "source_links": "MATCH (d:MedquadDocument {index_version: 'V3'})-[:FROM_SOURCE]->(s:MedquadSource {index_version: 'V3'}) RETURN count(*) AS n",
    "chunk_links": "MATCH (d:MedquadDocument {index_version: 'V3'})-[:HAS_CHUNK]->(c:MedquadChunk {index_version: 'V3'}) RETURN count(*) AS n",
    "invalid_chunks": "MATCH (c:MedquadChunk {index_version: 'V3'}) WHERE c.import_sha256 IS NULL OR c.import_sha256 <> $expected RETURN count(c) AS n",
    "invalid_docs": "MATCH (d:MedquadDocument {index_version: 'V3'}) WHERE d.import_sha256 IS NULL OR d.import_sha256 <> $expected RETURN count(d) AS n",
    "invalid_sources": "MATCH (s:MedquadSource {index_version: 'V3'}) WHERE s.import_sha256 IS NULL OR s.import_sha256 <> $expected RETURN count(s) AS n",
    "dated_docs": "MATCH (d:MedquadDocument {index_version: 'V3'}) WHERE d.source_date_text IS NOT NULL RETURN count(d) AS n",
}


# Lecture exhaustive des proprietes et relations ; utile pour detecter une
# corruption post-import que des simples nombres de noeuds ne verraient pas.
READ_PROVENANCE_SOURCES = """
MATCH (s:MedquadSource {index_version: 'V3'})
RETURN s.key AS key, s.source_url AS source_url, s.source_file AS source_file
"""
READ_PROVENANCE_DOCUMENTS = """
MATCH (d:MedquadDocument {index_version: 'V3'})
OPTIONAL MATCH (d)-[:FROM_SOURCE]->(s:MedquadSource)
RETURN d.key AS key, d.document_id AS document_id, d.question AS question,
       d.source_url AS source_url, d.source_file AS source_file,
       d.source_date_text AS source_date_text,
       d.source_date_evidence_chunk_id AS source_date_evidence_chunk_id,
       collect(s.key) AS source_keys
"""
READ_PROVENANCE_CHUNKS = """
MATCH (c:MedquadChunk {index_version: 'V3'})
OPTIONAL MATCH (d:MedquadDocument)-[:HAS_CHUNK]->(c)
RETURN c.key AS key, c.chunk_id AS chunk_id,
       c.document_id AS document_id, c.question AS question,
       c.content AS content, c.content_sha256 AS content_sha256,
       c.source_url AS source_url, c.source_file AS source_file,
       collect(d.key) AS document_keys
"""


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def make_source_key(url: str) -> str:
    return f"{PREFIX}:source:{hashlib.sha256(url.encode('utf-8')).hexdigest()}"


@dataclass
class Payload:
    sources: list[dict]
    documents: list[dict]
    chunks: list[dict]
    chunks_sha256: str

    def counts(self) -> dict[str, int]:
        return {
            "sources": len(self.sources), "documents": len(self.documents),
            "chunks": len(self.chunks), "source_links": len(self.documents),
            "chunk_links": len(self.chunks),
        }


def load_payload(data_dir: Path, expected_sha: str | None = EXPECTED_CHUNKS_SHA256) -> Payload:
    chunks_path = data_dir / "medquad_chunks_v3.json"
    embeddings_path = data_dir / "medquad_embeddings_v3.npy"
    manifest_path = data_dir / "index_manifest_v3.json"
    for p in (chunks_path, embeddings_path, manifest_path):
        if not p.is_file():
            raise FileNotFoundError(f"Fichier V3 introuvable : {p}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks_sha = sha256_file(chunks_path)
    if chunks_sha != manifest.get("chunks_sha256"):
        raise ValueError("SHA256 des chunks different du manifeste V3")
    if expected_sha is not None and chunks_sha != expected_sha:
        raise ValueError("Corpus inattendu : SHA256 different du V3 valide dans ce projet")
    if sha256_file(embeddings_path) != manifest.get("embeddings_sha256"):
        raise ValueError("SHA256 des embeddings different du manifeste V3")

    rows = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != 750:
        raise ValueError("Le corpus V3 valide doit contenir exactement 750 chunks")
    if manifest.get("chunks_after") != 750 or manifest.get("source_documents") != 200:
        raise ValueError("Le manifeste V3 ne correspond pas au corpus valide (750/200)")

    source_by_key: dict[str, dict] = {}
    doc_by_id: dict[int, dict] = {}
    chunk_by_id: dict[str, dict] = {}
    for row in rows:
        docid = row.get("document_id")
        if type(docid) is not int or docid < 0:
            raise ValueError(f"document_id invalide : {docid!r}")
        cid = row.get("chunk_id")
        content = row.get("content")
        question = row.get("question")
        source_file = row.get("source_file")
        source_url = row.get("source_url")
        if not all(isinstance(v, str) and v.strip() for v in (cid, content, question, source_file, source_url)):
            raise ValueError(f"Ligne incompletement renseignee : {cid!r}")
        if row.get("text_for_embedding") != f"Question: {question}\nAnswer: {content}":
            raise ValueError(f"Texte d'embedding incoherent pour {cid}")
        if cid in chunk_by_id:
            raise ValueError(f"ID chunk duplique : {cid}")
        if not cid.startswith(f"{docid}_"):
            raise ValueError(f"chunk_id et document_id incoherents : {cid}")

        skey = make_source_key(source_url)
        sr = {"key": skey, "source_url": source_url, "source_file": source_file}
        if skey in source_by_key and source_by_key[skey] != sr:
            raise ValueError(f"Source contradictoire : {source_url}")
        source_by_key[skey] = sr

        doc = {
            "key": f"{PREFIX}:doc:{docid}",
            "document_id": docid,
            "question": question,
            "source_url": source_url,
            "source_file": source_file,
            "source_key": skey,
            "source_date_text": row.get("source_date_text"),
            "source_date_evidence_chunk_id": row.get("source_date_evidence_chunk_id"),
        }
        if docid in doc_by_id and doc_by_id[docid] != doc:
            raise ValueError(f"Metadonnees contradictoires pour document {docid}")
        doc_by_id[docid] = doc

        chunk_by_id[cid] = {
            "key": f"{PREFIX}:chunk:{cid}",
            "chunk_id": cid,
            "document_key": doc["key"],
            "document_id": docid,
            "question": question,
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "source_file": source_file,
            "source_url": source_url,
        }

    if len(doc_by_id) != 200:
        raise ValueError(f"200 documents attendus, obtenus {len(doc_by_id)}")
    if len(source_by_key) != 29:
        raise ValueError(f"29 sources attendues, obtenues {len(source_by_key)}")
    if doc_by_id[188]["source_date_text"] != "February 2014":
        raise ValueError("Date documentaire historique du document 188 non conservee")
    return Payload(
        sources=sorted(source_by_key.values(), key=lambda d: d["key"]),
        documents=sorted(doc_by_id.values(), key=lambda d: d["document_id"]),
        chunks=list(chunk_by_id.values()),
        chunks_sha256=chunks_sha,
    )


def batches(rows: list[dict], batch_size: int = BATCH_SIZE):
    if batch_size <= 0:
        raise ValueError("batch_size doit etre positif")
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def settings() -> tuple[str, str, str, str]:
    from dotenv import load_dotenv
    load_dotenv(BASE / ".env")
    keys = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE")
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise EnvironmentError("Variables Neo4j absentes du .env : " + ", ".join(missing))
    values = tuple(os.environ[k] for k in keys)
    if any("VOTRE_" in v or "MOT_DE_PASSE_REEL" in v for v in values):
        raise EnvironmentError("Le .env contient encore des valeurs d'exemple")
    return values  # type: ignore[return-value]


def query_count(driver, database: str, query: str, expected: str) -> int:
    records, _, _ = driver.execute_query(
        query, database_=database, routing_="r", parameters_={"expected": expected}
    )
    if len(records) != 1:
        raise RuntimeError("Requete de verification : nombre de resultats incorrect")
    result = records[0]["n"]
    if type(result) is not int or result < 0:
        raise RuntimeError("Requete de verification : comptage invalide")
    return result


def verify_row_provenance(driver, database: str, payload: Payload) -> None:
    """Read back every imported property/edge to detect mislinks or alterations."""
    for cypher, expected_rows, link_field, link_key in (
        (READ_PROVENANCE_SOURCES, payload.sources, None, None),
        (READ_PROVENANCE_DOCUMENTS, payload.documents, "source_keys", "source_key"),
        (READ_PROVENANCE_CHUNKS, payload.chunks, "document_keys", "document_key"),
    ):
        records, _, _ = driver.execute_query(cypher, database_=database, routing_="r")
        observed_rows = {}
        for record in records:
            row = record.data() if hasattr(record, "data") else dict(record)
            key = row.get("key")
            if key in observed_rows:
                raise RuntimeError(f"Cle dupliquee retournee par Neo4j : {key}")
            observed_rows[key] = row
        if len(observed_rows) != len(expected_rows):
            raise RuntimeError(
                f"Incoherence des lignes Neo4j : {len(observed_rows)} contre {len(expected_rows)}")
        for expected in expected_rows:
            row = observed_rows.get(expected["key"])
            if row is None:
                raise RuntimeError(f"Noeud manquant dans le graphe : {expected['key']}")
            for prop, value in expected.items():
                if prop in ("source_key", "document_key"):
                    continue
                if row.get(prop) != value:
                    raise RuntimeError(
                        f"Propriete Neo4j invalide pour {expected['key']} : {prop}")
            if link_field is not None:
                linked = row.get(link_field)
                if linked != [expected[link_key]]:
                    raise RuntimeError(
                        f"Lien de provenance incorrect pour {expected['key']} : {link_field}")


def verify(driver, database: str, payload: Payload) -> dict[str, int]:
    observed = {
        name: query_count(driver, database, cypher, payload.chunks_sha256)
        for name, cypher in READ_COUNTS.items()
    }
    expected = payload.counts()
    failures = {
        name: {"expected": target, "observed": observed[name]}
        for name, target in expected.items() if observed[name] != target
    }
    for name in ("invalid_chunks", "invalid_docs", "invalid_sources"):
        if observed[name] != 0:
            failures[name] = {"expected": 0, "observed": observed[name]}
    if observed["dated_docs"] != 1:
        failures["dated_docs"] = {"expected": 1, "observed": observed["dated_docs"]}
    if failures:
        raise RuntimeError("Verification Neo4j non conforme : " + json.dumps(failures, ensure_ascii=False))
    verify_row_provenance(driver, database, payload)
    return observed


def import_graph(driver, database: str, payload: Payload) -> None:
    # Les contraintes et MERGE evitent les doublons sur une nouvelle execution.
    # Si un probleme reseau interrompt l'import, relancer --apply est sans doublons.
    for query in CONSTRAINTS:
        driver.execute_query(query, database_=database)
    for category, query, rows in (
        ("sources", WRITE_SOURCES, payload.sources),
        ("documents", WRITE_DOCUMENTS, payload.documents),
        ("chunks", WRITE_CHUNKS, payload.chunks),
    ):
        for index, batch in enumerate(batches(rows), start=1):
            driver.execute_query(
                query, parameters_={"rows": batch, "import_sha256": payload.chunks_sha256},
                database_=database,
            )
            print(f"  {category}: lot {index}, {len(batch)} lignes confirmees")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="controle local uniquement (mode par defaut)")
    mode.add_argument("--apply", action="store_true", help="AUTORISER les ecritures dans Neo4j")
    mode.add_argument("--verify", action="store_true", help="verifier seulement le graphe existant")
    args = parser.parse_args()
    payload = load_payload(DATA)
    expected = payload.counts()
    print("MEDQUAD V3 : corpus valide | SHA256 :", payload.chunks_sha256)
    print("Nœuds attendus: sources={sources}, documents={documents}, chunks={chunks}".format(**expected))
    print("Relations attendues: FROM_SOURCE={source_links}, HAS_CHUNK={chunk_links}".format(**expected))
    print("Aucune entite clinique (Disease/Symptom/Treatment) ne sera creee.")
    if not (args.apply or args.verify):
        print("DRY RUN: PASS. Aucune connexion Neo4j, aucune ecriture.")
        return

    uri, username, password, database = settings()
    from neo4j import GraphDatabase
    # Ne jamais afficher les secrets de connexion.
    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        if args.apply:
            print("IMPORT Neo4j : debut des ecritures idempotentes")
            import_graph(driver, database, payload)
        result = verify(driver, database, payload)
    print("VERIFY: PASS", json.dumps(result, ensure_ascii=False))
    if args.apply:
        print("IMPORT: PASS. Une seconde execution de --apply doit conserver les memes nombres.")


if __name__ == "__main__":
    main()
