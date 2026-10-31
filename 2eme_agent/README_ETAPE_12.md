# Étape 12 — Contrôles qualité du premier agent LangGraph V3

Les sorties de l'étape 11 montrent le bon usage des outils de statistique et de recherche, mais la réponse aux recommandations 2026 a été rejetée uniquement parce que le LLM n'avait appelé aucun outil. Ce n'est pas une politique d'abstention suffisante.

Ce pack fournit un **wrapper** qui réutilise **sans le modifier** `agent_langgraph_v3.py` :

1. Bloque **avant appel Groq** certaines demandes explicites de recommandations médicales actuelles ou de prescription personnelle, avec une explication.
2. Vérifie que l'outil adapté a été appelé (`statistiques_corpus` ou `rechercher_medquad`).
3. Pour une réponse documentaire, refuse une sortie ne citant aucun chunk, ou citant un chunk absent des traces de recherche.
4. Conserve le contrôle SHA256 déjà présent dans l'agent d'origine.
5. Ne modifie aucun fichier de l'index, aucune dépendance et n'enregistre aucune question personnelle sur disque.

## Installer

Dans le dossier `2eme_agent`, à côté de `agent_langgraph_v3.py`, placer :

- `agent_langgraph_v3_guarded.py`
- `test_agent_langgraph_v3_guarded.py`

Ne pas écraser `agent_langgraph_v3.py`.

## Contrôles sans API

```powershell
python -m unittest -v test_agent_langgraph_v3_guarded.py
python agent_langgraph_v3_guarded.py --check
```

## Tests Groq

```powershell
python agent_langgraph_v3_guarded.py --question "How many documents and chunks are indexed?"
python agent_langgraph_v3_guarded.py --question "What are the symptoms of acromegaly?"
python agent_langgraph_v3_guarded.py --question "What do the latest 2026 clinical guidelines recommend for acromegaly?"
```

Pour la troisième question, le garde déterministe doit expliquer l'indisponibilité des recommandations actuelles **sans aucun appel API**.

## Important — limites du garde

- C'est un filtre illustratif fondé sur des règles simples en français et en anglais, **pas un système de sûreté exhaustif**. Il ne reconnaît pas toutes les formes de demande de conseil clinique.
- La vérification des identifiants de chunks **ne démontre pas** que chaque affirmation est soutenue par la source. Il faut une évaluation de fidélité et une revue humaine.
- Les sources MedQuAD indexées sont historiques et les URL peuvent avoir changé.
- Le modèle peut produire une réponse sans appel de l'outil requis : elle sera alors rejetée plutôt que réparée automatiquement.
- Ne pas utiliser en prise de décision clinique sans gouvernance, validation et sources actualisées.

**Étape suivante :** tests d'intégration et d'abstention sur un jeu de cas figé, puis ajout du troisième outil Neo4j avec des requêtes en lecture seule.
