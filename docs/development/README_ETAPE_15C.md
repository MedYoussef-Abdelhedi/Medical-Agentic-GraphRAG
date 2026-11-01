# Étape 15C — Corrections de qualité et tests de régression

## Objectif

Corriger les trois faiblesses détectées : réponses interrompues, qualifications documentaires perdues (Addison vs secondaire, symptômes fréquents vs autres symptômes, « in women ») et couverture artificiellement limitée par les 14 claims Neo4j.

**Aucune modification** des fichiers `medquad_chunks_v3.json` / `.npy`, de Neo4j, des scripts des étapes 14/15 ou des précédents rapports.

## Installation

Copier **quatre fichiers Python** dans `2eme_agent/`, au même niveau que les scripts existants :

- `quality_generation_v3.py` : prompt et détection de réponse incomplète, relance contrôlée
- `agent_graphrag_hybride_v3_quality.py` : agent hybride avec contrôles, sans modification des anciens agents
- `compare_vector_graphrag_v3_quality.py` : comparaison appariée V2 qualité
- `test_quality_generation_v3.py` : 11 tests sans réseau

Prérequis : `agent_graphrag_hybride_v3.py`, `compare_vector_graphrag_v3.py`, `benchmark_questions_v1.json`, `.env` et les fichiers V3 déjà présents.

## Commandes PowerShell (depuis `2eme_agent`)

```powershell
python -m unittest -v test_quality_generation_v3.py
python agent_graphrag_hybride_v3_quality.py --check
python compare_vector_graphrag_v3_quality.py --dry-run
```

Si ces trois contrôles réussissent, tester une question :

```powershell
python agent_graphrag_hybride_v3_quality.py --question "What are the symptoms of adrenal insufficiency?"
```

Puis refaire la comparaison, à partir des mêmes questions et contextes :

```powershell
python compare_vector_graphrag_v3_quality.py --max-cases 2 --sleep-between 20
```

Rapports enregistrés dans `reports/hybrid_comparison_v2_quality/<run_id>/`, sans écrasement des anciens résultats.

## Quelles garanties ?

- `finish_reason=length` (ou équivalent) : réponse refusée.
- Le marqueur explicite `[[END_RESPONSE]]` doit être à la fin, puis est retiré.
- Si le marqueur manque, un deuxième essai avec un budget de tokens plus élevé est tenté ; une réponse interrompue n'est pas comptée comme succès.
- L'information `finish_reason` et les tokens consommés sont enregistrés pour les réponses réussies.
- Les deux modes reçoivent le **même prompt**, les mêmes trois passages vectoriels et le même modèle. Seuls les claims Neo4j sont ajoutés à `hybrid`.
- `manual_review.csv` est intentionnellement initialisé avec `pending`: l'amélioration du prompt **ne valide pas** la fidélité médicale.

## Vérification finale

Comparer le nouveau rapport au précédent `rapport_qualitatif_etape15.md` : absence d'interruption, préservation explicite des distinctions Addison/secondaire, fréquence, mention des femmes, couverture des passages vectoriels, citations, latences et éventuelles erreurs Groq.

L'achèvement structurel (marqueur + `finish_reason`) **ne garantit pas** l'absence d'omissions ou d'erreurs cliniques. Toutes les revendications médicales exigent une évaluation indépendante.

En cas de 429, de coupure réseau ou de réponse non terminée, consulter le message d'erreur; ne pas le traiter comme une réussite et ne pas relancer en boucle.
