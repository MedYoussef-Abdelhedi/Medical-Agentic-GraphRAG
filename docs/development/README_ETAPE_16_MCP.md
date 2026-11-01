# Étape 16 — Ajouter MCP à LangGraph + GraphRAG MedQuAD V3

## Objectif et précautions

**MCP (Model Context Protocol)** est un protocole qui permet à un programme (l'agent LangGraph) de découvrir et d'appeler les outils d'un autre programme (serveur MCP). Ce n'est **pas** un nouveau modèle de langage et MCP n'améliore pas automatiquement la qualité des réponses.

**Avant :** LangGraph appelle directement des fonctions Python de recherche.

**Après :** LangGraph → client MCP → serveur MCP local → fonctions existantes (MiniLM / Neo4j) → preuves → Groq.

Cette étape **ne reconstruit pas** MedQuAD V3, ne modifie pas Neo4j, ne recrée pas les 14 relations et ne résout pas à elle seule les réserves qualitatives de l'étape 15C.

Le serveur expose exactement trois outils en lecture seule :

| Outil MCP | Usage | Appels externes |
|---|---|---|
| `statistiques_medquad_v3` | Taille de l'index V3 contrôlé | Aucun |
| `rechercher_passages_v3` | Top-k de 1 à 5 chunks avec provenance | Modèle d'embedding local, parfois téléchargement initial HF |
| `symptomes_neo4j_v3` | 8 ou 6 relations maladie-symptôme et preuves V3 vérifiées | Neo4j Aura, requête Cypher **fixe** et paramétrée |

Le serveur MCP **n'appelle pas Groq**. La génération Groq reste dans LangGraph. Les questions hors du pilote ne sont pas traitées comme des maladies compatibles. La recherche et les faits sont documentaires, non validés cliniquement.

## Préparer l'environnement (Windows PowerShell)

Dans `C:\Users\medyo\Desktop\Chat_Rag_Graph\Graph-rag-medical\2eme_agent` avec `.venv` activé :

```powershell
python --version
python -m pip show mcp
```

Le pack est écrit pour **MCP Python SDK v2** (API : `from mcp.server import MCPServer`, `from mcp import Client`). La version 1 nécessite d'autres imports et ne convient pas à ces scripts. Installer, si absent ou en v1 :

```powershell
python -m pip freeze > requirements-before-mcp.txt
python -m pip install "mcp>=2,<3"
python -m pip check
python -c "import importlib.metadata; print('MCP SDK:', importlib.metadata.version('mcp'))"
```

Python 3.11+ est requis; Python 3.12 du projet convient. La validation `pip check` peut révéler des conflits déjà présents ; ne pas continuer s'il en trouve. **Ne pas réinstaller LangChain ni Neo4j.**

## Copier les fichiers

Les cinq fichiers Python et ce README doivent être placés dans `2eme_agent` avec les modules déjà présents :

- `mcp_medquad_service_v3.py`
- `mcp_medquad_server_v3.py`
- `mcp_medquad_client_v3.py`
- `agent_graphrag_mcp_v3.py`
- `test_mcp_medquad_v3.py`
- `README_ETAPE_16_MCP.md`

Ne pas toucher aux fichiers `data/medquad_*_v3` ni à `index_manifest_v3.json`. Les modules existants doivent aussi rester :

- `agent_langgraph_v3.py`, `agent_langgraph_v3_guarded.py`, `agent_langgraph_v3_neo4j.py`
- `agent_graphrag_hybride_v3.py`, `compare_vector_graphrag_v3.py`, `quality_generation_v3.py`

Le `.env` de `2eme_agent` doit déjà contenir `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` (dans ton cas le nom de base fourni par Aura), et `GROQ_API_KEY`. **Ne jamais publier `.env` ni le fichier de credentials Aura.** Vérifier que `.env` est dans `.gitignore`.

## Ordre de validation (ne pas sauter les étapes)

### 1. Tests locaux (pas d'API)

```powershell
python -m unittest -v test_mcp_medquad_v3.py
python agent_graphrag_mcp_v3.py --check
```

Les tests de fonctions marchent hors ligne. Après installation de MCP v2, **un test supplémentaire** vérifie le protocole MCP en mémoire (listing des tools, appel de statistiques simulées).

### 2. Réelle connexion client MCP ↔ serveur MCP local (`stdio`)

```powershell
python mcp_medquad_client_v3.py --list
python mcp_medquad_client_v3.py --stats
```

Attendu : `HANDSHAKE MCP: PASS`, 3 outils exacts; puis 200 documents / 750 chunks / 384 dimensions si le projet V3 est inchangé.

**Ne pas exécuter `python mcp_medquad_server_v3.py` directement pour tester** : il écoutera sur stdin et semblera bloqué. Le client démarre et termine automatiquement le processus serveur. Les logs doivent aller vers stderr, sinon le flux MCP `stdio` sera corrompu.

### 3. Vérifier Neo4j et la recherche vectorielle par MCP

```powershell
python mcp_medquad_client_v3.py --graph "Acromegaly"
python mcp_medquad_client_v3.py --search "What are the symptoms of acromegaly?"
```

Le premier doit retourner 8 `claims` vérifiés, provenant du graphe médical déjà construit ; 6 pour `Adrenal insufficiency`. Le second doit retourner des passages V3 sourcés, avec `chunk_id` et `source_url_historical`. L'avertissement HF sur les requêtes non authentifiées n'implique pas un échec du protocole si la recherche aboutit.

### 4. Contrôler la connexion MCP de LangGraph

```powershell
python agent_graphrag_mcp_v3.py --list-tools
```

Ce contrôle lance le serveur en processus enfant, négocie le protocole et vérifie les trois outils. **Aucun appel Groq, Neo4j ou embedding.**

### 5. Exécuter le vrai workflow LangGraph → MCP → Groq

```powershell
python agent_graphrag_mcp_v3.py --question "Quels sont les symptômes de l'acromégalie ?"
```

Attendu : `MCP HANDSHAKE PASS`, noms des 8 claims, ids des chunks V3, génération Groq avec `finish_reason=stop` et marqueur de fin contrôlé, citations connues. **Le rendu précis de la réponse peut varier.**

Tester ensuite le garde-fou :

```powershell
python agent_graphrag_mcp_v3.py --question "What do the latest 2026 clinical guidelines recommend for acromegaly?"
```

Ce test doit se terminer **sans appel MCP ni Groq**, par un message indiquant la limite documentaire.

## Principes de sécurité et limites

- Le serveur utilise **stdio local**, pas de port HTTP ouvert sur Internet. Un outil MCP local reste accessible aux clients du même utilisateur qui le lancent ; n'exposer le serveur qu'à des applications fiables.
- Le code Cypher est fixé et paramétré. Il n'offre **aucune écriture** ni exécution de Cypher libre. Le routage `routing_='r'` **n'est pas une autorisation** : employer un compte Neo4j aux droits minimums, idéalement lecture seule.
- Aucune donnée clinique à caractère personnel ne doit être introduite dans ce pilote de démonstration.
- Les réponses MCP sont contrôlées (`status`, structures attendues, hashes et preuves existantes). La vérification des IDs de citations ne garantit pas que chaque affirmation soit justifiée.
- La liste de symptômes Neo4j est un pilote **non exhaustif** sur 2 maladies. Les 14 relations restent issues de documents historiques et de pré-revue IA, pas d'une validation médicale indépendante.
- Les fichiers JSON/CSV d'analyse précédente sont conservés. Ne pas conclure à un gain de qualité lié à MCP sans expérience comparative dédiée.

## Dépannage

| Message | Cause probable | Action |
|---|---|---|
| `ModuleNotFoundError: mcp` | SDK absent | Installer `mcp>=2,<3` dans `.venv` |
| `cannot import name MCPServer` | SDK 1.x | Vérifier `python -m pip show mcp` et installer 2.x |
| `No module named agent_graphrag_hybride_v3` | Fichiers dans le mauvais dossier | Déplacer les scripts dans `2eme_agent` |
| `Neo4j configuration ...` | `.env` non rempli ou incorrect | Vérifier les paramètres **sans les afficher en public** |
| `MCP tool failed` | Une fonction a échoué côté serveur | Lire le message, tester d'abord les stats et le graphe |
| `SDK incompatible / pip check` | Conflit de dépendances | Sauvegarder l'état et corriger le conflit, pas de downgrade aveugle |
| `Respuesta trunca / IncompleteResponseError` | Groq n'a pas achevé la génération | Signaler l'échec, ne pas diffuser comme réponse validée |

## Documentation officielle consultée

- MCP Python SDK v2 : https://py.sdk.modelcontextprotocol.io/
- Serveur : https://py.sdk.modelcontextprotocol.io/get-started/first-steps/
- Client : https://py.sdk.modelcontextprotocol.io/client/
- Exécution stdio : https://py.sdk.modelcontextprotocol.io/run/

## Étape suivante après validation

Tester les performances du même workflow en **appel direct** et en **appel MCP** : mêmes questions, mêmes V3, mêmes faits Neo4j, mêmes prompts, métriques de latence et de fiabilité ; puis compléter la documentation, les tests de non-régression et l'évaluation clinique indépendante si un usage réel est envisagé.
