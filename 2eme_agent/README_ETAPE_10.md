# Étape 10 — Évaluation des réponses du RAG V3

## Objectif

Contrôler séparément **la génération** (Groq) et **l'évaluation** (RAGAS, avec un LLM juge). Aucun index V1/V2/V3 n'est modifié. Les documents MedQuAD sont historiques ; les évaluations ne mesurent pas la sûreté clinique.

## Installation

Copier les quatre fichiers à côté de `rag_pipeline.py`, dans `2eme_agent/` :

- `rag_eval_cases_v1.json` : quatre questions avec réponses de référence, deux sondes de recommandations récentes sans source ; annotations candidates, à revoir humainement.
- `generate_rag_eval_v3.py` : recherche dans V3 + génération des réponses, résultat JSON versionné.
- `score_ragas_v3.py` : quatre métriques RAGAS v0.4 pour les questions avec référence.
- `test_rag_eval_v3.py` : tests unitaires sans API.

Fichiers V3 requis sous `2eme_agent/data/` : `medquad_chunks_v3.json`, `medquad_embeddings_v3.npy`. Clé `GROQ_API_KEY` dans `2eme_agent/.env` (ne jamais publier la clé).

## Exécution progressive — depuis le dossier `2eme_agent`

```powershell
python -m unittest -v test_rag_eval_v3.py
python generate_rag_eval_v3.py --dry-run
python generate_rag_eval_v3.py --max-cases 3
```

Cette troisième commande appelle Groq 3 fois, et crée `reports/rag_generation_v3/<run_id>/evaluation_inputs.json`. Conserver **ce chemin exact**.

Puis installer RAGAS v0.4 et le SDK Groq (sans changer le modèle d'embedding du retriever) :

```powershell
python -m pip install "ragas>=0.4,<0.5" groq
python -m pip check
python -c "import ragas; print('RAGAS:', ragas.__version__)"
```

Faire une vérification **sans appels réseau** du JSON généré :

```powershell
python score_ragas_v3.py --input "reports/rag_generation_v3/<run_id>/evaluation_inputs.json" --dry-run
```

Puis lancer 2 évaluations au maximum pour commencer (l'évaluateur effectue plusieurs appels Groq par exemple et peut consommer le quota API) :

```powershell
python score_ragas_v3.py --input "reports/rag_generation_v3/<run_id>/evaluation_inputs.json" --max-cases 2
```

Remplacer `<run_id>` par le dossier **réel** du résultat ; ne pas recopier les chevrons littéralement. Le script sauvegarde `ragas_scores.json` à côté de son entrée. Un `null` signifie que le calcul a échoué, **pas** que le score vaut 0. Ne jamais inventer de score.

## Qualité et limites

- `Faithfulness` mesure le soutien de la réponse par les passages récupérés (et non la vérité médicale).
- `AnswerRelevancy` juge la correspondance question-réponse, avec des embeddings locaux MiniLM.
- `ContextPrecision` et `ContextRecall` exploitent des réponses de référence candidates, non validées indépendamment.
- Les deux cas `unsupported_current` ne reçoivent **aucun score RAGAS de rappel/précision**, car aucune réponse de référence clinique récente n'est disponible ; on examine manuellement l'abstention.
- Si Groq refuse des sorties structurées demandées par RAGAS, les erreurs sont consignées (pas de réussite simulée). Les API Groq et RAGAS doivent alors être adaptées à leurs versions réellement installées.
- Le nouveau script utilise directement V3. Le fichier d'origine `rag_pipeline.py` continue de lire V1 tant que nous ne le modifions pas explicitement.
- `rag_eval_cases_v1.json` est un **test de fumée** (6 cas), et ne suffit pas pour une validation entreprise. Il faudra ensuite étiqueter un benchmark plus large, idéalement relu indépendamment.
- Toute note de protocole doit préciser le modèle, les versions, les hash des données, la taille du test, la date, le coût/API, les échecs de calcul et les limites des sources.

## Sources officielles des API utilisées

- Ragas v0.4 migration : https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/
- Ragas Groq adapter : https://docs.ragas.io/en/stable/howtos/llm-adapters/
- Ragas metrics : https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/
