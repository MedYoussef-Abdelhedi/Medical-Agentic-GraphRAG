# Étape 13B — Import documentaire Neo4j AuraDB

## Objectif

Créer un **graphe de provenance**, pas encore un graphe clinique :

- 29 `:MedquadSource`, identifiées par l'URL source historique ;
- 200 `:MedquadDocument`, contenant notamment la question et l'origine ;
- 750 `:MedquadChunk`, contenant les passages textuels indexés en V3 ;
- 200 liens `(:MedquadDocument)-[:FROM_SOURCE]->(:MedquadSource)` ;
- 750 liens `(:MedquadDocument)-[:HAS_CHUNK]->(:MedquadChunk)`.

**Total attendu dans une base vide : 979 nœuds et 950 relations.** Ce ne sont pas des faits médicaux : les relations `HAS_SYMPTOM`, `TREATED_BY` et autres ne sont pas encore extraites ni validées.

## Installation

Décompresser `import_medquad_v3_neo4j.py` et `test_import_medquad_v3_neo4j.py` directement dans le dossier `2eme_agent/`, à côté de `agent_langgraph_v3_neo4j.py` et du dossier `data/`.

Vérifier le `.env` existant : `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, et `NEO4J_DATABASE`. Ne jamais publier les valeurs ni ajouter `.env` au dépôt Git.

Le driver Neo4j et `python-dotenv` sont réutilisés ; aucun modèle ne sera téléchargé et aucun appel Groq ne sera effectué. Un import nécessitera les droits d'écriture uniquement pendant cette opération ; idéalement l'agent d'inférence utilise un compte distinct en lecture seule (si les rôles de l'offre le permettent).

## Exécution sous PowerShell (à partir de `2eme_agent/`)

### 1. Tests, sans API

```powershell
python -m unittest -v test_import_medquad_v3_neo4j.py
```

### 2. Prévisualisation sans connexion ni écriture

```powershell
python import_medquad_v3_neo4j.py --dry-run
```

Le script vérifie les empreintes des chunks **et** des embeddings via le manifeste V3, la structure des métadonnées, les 750 chunks, 200 documents, 29 URLs, et la provenance de la date historique conservée. L'empreinte attendue des chunks est verrouillée sur le résultat validé localement : `a0ff3068e591a8bc86f8206362fe65866d3e1f6b55477fab09b037c65d210656`.

### 3. Import réel dans l'instance Neo4j

```powershell
python import_medquad_v3_neo4j.py --apply
```

**ATTENTION : cette commande écrit réellement dans Neo4j** : création ou mise à jour de nœuds et relations documentaires, et de 3 contraintes d'unicité. Elle n'efface pas les autres données et ne met pas d'embeddings dans le graphe. Le script utilise `MERGE`, ce qui permet une reprise après interruption et empêche la duplication de ces identifiants. L'import se fait par lots ; un échec en cours de route laisse potentiellement des lots validés. Dans ce cas, corriger l'erreur et relancer `--apply`.

### 4. Vérification après import

```powershell
python import_medquad_v3_neo4j.py --verify
python agent_langgraph_v3_neo4j.py --check-neo4j
```

Résultat attendu, si la base ne contient pas d'autres jeux de données :

```text
Nœuds : 979
Relations : 950
Labels : MedquadSource, MedquadDocument, MedquadChunk
Types : FROM_SOURCE, HAS_CHUNK
```

Si l'instance avait d'autres données, les totaux globaux peuvent être supérieurs. `--verify` compare **seulement** les types MedQuAD V3 et compte les liens entre eux. Il vérifie aussi l'empreinte de l'import, et la conservation de la date source pour le document 188.

### 5. Vérifier un lien dans Neo4j Query

```cypher
MATCH (d:MedquadDocument {document_id: 0})-[:HAS_CHUNK]->(c:MedquadChunk)
RETURN d.question AS question, c.chunk_id AS chunk_id, left(c.content, 160) AS extrait
ORDER BY chunk_id
LIMIT 5;
```

```cypher
MATCH (d:MedquadDocument {document_id: 188})-[:FROM_SOURCE]->(s:MedquadSource)
RETURN d.source_date_text AS date_conservee, d.source_date_evidence_chunk_id AS provenance_date,
       s.source_url AS url_historique;
```

## Critères de réussite

- Commandes tests et dry-run sans échec.
- Import `--apply` termine avec `IMPORT: PASS`.
- Un second `--verify` indique `VERIFY: PASS`.
- Total des **nœuds MedQuAD V3** = 979 ; **relations MedQuAD V3** = 950.
- Aucun lien maladie-symptôme n'est inféré ou prétendu validé.

## Étape 13C

Construction d'une couche **entités et relations médicales sourcées**, avec extraction contrôlée, normalisation des termes, validation des triplets, et connexion au moteur de recherche de l'agent. Les liens existants Document→Chunk→Source permettront de tracer chaque relation médicale à son passage justificatif.
