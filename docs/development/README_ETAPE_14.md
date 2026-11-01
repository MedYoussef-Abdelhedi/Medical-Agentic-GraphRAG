# Étape 14 — GraphRAG hybride contrôlé : Neo4j + Vector RAG V3 + LangGraph

## Objectif

Nous avons déjà importé et vérifié dans Neo4j :

- 29 sources, 200 documents, 750 chunks ;
- 2 maladies, 14 symptômes, 14 claims revus **documentairement** ;
- 14 liens `HAS_SYMPTOM`, 14 `SUBJECT`, 14 `OBJECT` et 14 `SUPPORTED_BY`.

À cette étape, **aucune nouvelle donnée n'est importée**. Nous lisons les relations existantes et leurs preuves, puis nous effectuons une recherche vectorielle complémentaire. Le workflow LangGraph transmet les deux résultats à Groq pour une réponse sourcée.

Les résultats proviennent de documents MedQuAD historiques : ni avis individuel, ni recommandation clinique actuelle, ni validation indépendante d'un professionnel médical.

## Installation

Placez `agent_graphrag_hybride_v3.py` et `test_agent_graphrag_hybride_v3.py` **à la racine** du dossier `2eme_agent`, aux côtés de ces fichiers existants :

- `agent_langgraph_v3.py`
- `agent_langgraph_v3_guarded.py`
- `agent_langgraph_v3_neo4j.py`
- `.env`
- `data/medquad_chunks_v3.json`
- `data/medquad_embeddings_v3.npy`
- `data/index_manifest_v3.json`

Aucun paquet supplémentaire n'est requis si les étapes 11 à 13 sont déjà fonctionnelles (`neo4j`, `langgraph`, `langchain-groq`, `langchain-huggingface`, `python-dotenv`, `numpy`). **Ne modifiez pas `.env`.**

## Exécution progressive (PowerShell)

Dans `2eme_agent` :

```powershell
python -m unittest -v test_agent_graphrag_hybride_v3.py
python agent_graphrag_hybride_v3.py --check
```

Ces tests sont locaux, sans connexion Neo4j ou Groq. Le test d'intégration LangGraph ne s'exécute que si LangGraph est installé.

Vérifiez ensuite les preuves réellement présentes sur AuraDB **sans LLM et sans écriture** :

```powershell
python agent_graphrag_hybride_v3.py --check-graph
python agent_graphrag_hybride_v3.py --graph "Acromegaly"
python agent_graphrag_hybride_v3.py --graph "Adrenal insufficiency"
```

`--check-graph` doit trouver huit claims pour l'acromégalie et six pour l'insuffisance surrénalienne, avec références SHA256 des passages V3 identiques aux données locales. Il vérifie aussi la traversée `Document → Chunk` et `Document → Source`. Lancer ceci avant d'appeler Groq.

Enfin exécutez le vrai workflow **LangGraph en trois nœuds** :

```powershell
python agent_graphrag_hybride_v3.py --question "Quels sont les symptômes de l'acromégalie ?"
python agent_graphrag_hybride_v3.py --question "What are the symptoms of adrenal insufficiency?"
```

L'enchaînement est **déterministe**, volontairement : `neo4j_relations → recherche_vectorielle → synthese_avec_groq`. Contrairement à l'étape 11 (agent choisissant des outils), ce pilote LangGraph impose **les deux sources** afin de vérifier la valeur de l'hybridation. Une réponse sans citation de chunk du graphe sera bloquée. Les citations ne prouvent pas la fidélité sémantique : la validation humaine et les mesures comparatives restent nécessaires.

Test du blocage de fraîcheur clinique (sans Groq) :

```powershell
python agent_graphrag_hybride_v3.py --question "What do the latest 2026 clinical guidelines recommend for acromegaly?"
```

## Sécurité et limites

- La requête Cypher est **fixe et paramétrée**, jamais écrite par un LLM.
- Le code ne contient aucune instruction Cypher d'écriture et ne modifie ni les index, ni Neo4j.
- La route `routing_="r"` organise l'accès en lecture ; elle **ne remplace pas** un compte Neo4j ayant de vrais droits en lecture seule.
- Les deux maladies sont les seules couvertes à ce stade. L'absence de relations pour d'autres maladies ne signifie pas que les symptômes n'existent pas médicalement.
- Le contrôle des citations ne teste que leur existence dans les outils ; il ne prouve pas que chaque assertion est correctement soutenue par la citation.
- Certaines `reviewer` existantes peuvent signaler une pré-revue assistée par IA ; ne les présenter ni comme revue médicale ni comme revue humaine indépendante.
- Le filtre simple des questions médicales actuelles / personnelles n'est pas exhaustif.
- Pas de scores quantitatifs supplémentaires avant une évaluation sur un jeu de questions figé. Pour comparer Vector RAG et GraphRAG hybride, utilisez exactement les mêmes questions et des jugements de pertinence indépendants.

## Suite

Étape 15 : construire un mini-benchmark apparié Vector RAG vs GraphRAG hybride (sources utilisées, omissions, fidélité, latence, cas hors graphe), puis ajouter d'autres types de relations **uniquement** après revue documentaire.
