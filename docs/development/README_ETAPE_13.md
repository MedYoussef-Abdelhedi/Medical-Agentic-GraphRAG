# Étape 13A — Raccorder Neo4j en lecture seule à l'agent LangGraph V3

**Objectif limité et vérifiable :** ajouter un troisième outil qui interroge le nombre de nœuds, de relations, les *labels* et les types de liens présents dans la base Neo4j. Ce n'est **pas encore** une recherche GraphRAG maladie–symptôme : aucun schéma de connaissances MedQuAD n'a été vérifié, et le corpus V3 n'a pas été importé automatiquement dans Neo4j.

## Prérequis

- Les fichiers des étapes 11 et 12 fonctionnent déjà : `agent_langgraph_v3.py`, `agent_langgraph_v3_guarded.py`.
- Index V3 présent dans `data/`.
- Python 3.12 et venv activé.
- Instance Neo4j existante (locale ou Aura). Ne pas mettre de mot de passe dans les scripts ou dans une capture d'écran.

Depuis `2eme_agent`, installer uniquement le driver officiel si absent :

```powershell
python -m pip show neo4j
python -m pip install "neo4j>=5,<7"
python -m pip check
```

Dans le `.env` **existant**, ajouter les variables adaptées à ton instance (ne pas inclure les chevrons) :

```dotenv
NEO4J_URI=neo4j+s://VOTRE_INSTANCE.databases.neo4j.io
NEO4J_USERNAME=VOTRE_UTILISATEUR_LECTURE
NEO4J_PASSWORD=VOTRE_MOT_DE_PASSE
NEO4J_DATABASE=neo4j
```

Pour Neo4j en local, utiliser plutôt une URI telle que `bolt://localhost:7687` et la configuration correspondante. **Ne pas publier `.env` dans Git**. Idéalement utiliser un utilisateur Neo4j dont le rôle interdit toute écriture. `routing_="r"` sert au routage, et **n'est pas une autorisation de sécurité**.

## Installation des fichiers

Copier `agent_langgraph_v3_neo4j.py` et `test_agent_langgraph_v3_neo4j.py` à la racine de `2eme_agent`. Ne pas remplacer les étapes précédentes.

## Commandes (ordre de vérification)

1. Tests unitaires hors réseau :

```powershell
python -m unittest -v test_agent_langgraph_v3_neo4j.py
```

2. Vérification de l'index V3, sans réseau :

```powershell
python agent_langgraph_v3_neo4j.py --check
```

3. Connexion Neo4j **sans Groq** et inspection de la base existante :

```powershell
python agent_langgraph_v3_neo4j.py --check-neo4j
```

**Si la base est vide**, le script affichera 0 nœud / 0 relation. C'est normal : V3 reste dans des fichiers `.json` et `.npy`. Ne pas inventer de résultats médicaux Neo4j.

4. Vérifier le troisième outil via LangGraph + Groq :

```powershell
python agent_langgraph_v3_neo4j.py --question "How many nodes and relationships are in Neo4j?"
```

5. Test de non-régression :

```powershell
python agent_langgraph_v3_neo4j.py --question "How many documents and chunks are indexed?"
python agent_langgraph_v3_neo4j.py --question "What are the symptoms of acromegaly?"
python agent_langgraph_v3_neo4j.py --question "What do the latest 2026 clinical guidelines recommend for acromegaly?"
```

## Critères de validation de l'étape 13A

- Tests offline passent.
- Intégrité V3 toujours correcte (750 chunks, 200 documents).
- `--check-neo4j` réussit et affiche les statistiques de la **vraie** base sélectionnée, pas des valeurs fictives.
- Le LLM choisit `statistiques_graphe_neo4j` pour la question Neo4j.
- Le résultat final Neo4j est **formaté à partir du JSON fourni par l'outil**, et non d'un nombre écrit librement par le LLM.
- Les questions MedQuAD et le blocage documentaire restent fonctionnels.

## Sécurité et limites

- Quatre requêtes Cypher **fixes**, aucune génération de Cypher libre, aucune écriture prévue.
- Les comptages concernent toute la base sélectionnée, **pas nécessairement MedQuAD**.
- Les labels et types de relations sont limités aux 20 premiers ; aucune propriété de nœud n'est extraite.
- `MATCH (n)` et `MATCH ()-[r]->()` peuvent être coûteux sur une base très volumineuse. Sur des données de production, prévoir limites, monitoring et droits adaptés avant exécution.
- Pas de diagnostic médical, pas de validation clinique.
- Étape suivante **13B** : inspecter ton schéma Neo4j réel, choisir les nœuds/relations et la provenance document/chunk, puis construire le véritable outil de récupération de relations médicales et une évaluation GraphRAG contre V3.
