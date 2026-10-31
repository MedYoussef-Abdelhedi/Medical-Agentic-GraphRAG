# Étape 15 — Comparer Vector RAG et GraphRAG hybride (pilote contrôlé)

## Ce qui est déjà validé

Votre journal confirme la réussite du GraphRAG sur **une** question d'acromégalie en français : 13 tests, index V3, 8/6 claims Neo4j, trois opérations (Neo4j, vector, Groq), citations existantes. Cela prouve une **intégration fonctionnelle**, pas une supériorité sur Vector RAG.

## Objectif expérimental

- Même question et même modèle `openai/gpt-oss-20b` ; même prompt système.
- Une seule recherche vectorielle `top_k=3` par question, réutilisée **à l'identique** dans les deux modalités.
- Mode `vector` : contexte des 3 passages. Mode `hybrid` : mêmes 3 passages + relations documentaires Neo4j vérifiées.
- Comparer **temps de génération**, **latence approximative bout-à-bout** et **présence de citations traçables**.
- Générer `manual_review.csv` avec des champs de revue **laissés à pending** pour les omissions, hallucinations, fidélité des citations. Aucune note clinique automatique.

Les quatre requêtes sont des paraphrases sur **les deux maladies du pilote uniquement**. Elles ne sont ni un jeu de test clinique indépendant ni suffisamment nombreuses pour prouver un gain statistique. Elles sont intentionnellement proches du périmètre graphe. La construction des relations a pu influencer le choix des questions : il faut les considérer comme une évaluation exploratoire biaisée.

## Installation

Extraire les trois fichiers du ZIP au niveau de `2eme_agent`, à côté de `agent_graphrag_hybride_v3.py` et `.env`. Le programme ne modifie aucun script préexistant, aucune donnée, ni la base Neo4j. Aucune nouvelle dépendance nécessaire après l'étape 14.

## Commandes PowerShell

Tests sans services externes :

```powershell
python -m unittest -v test_compare_vector_graphrag_v3.py
python compare_vector_graphrag_v3.py --dry-run
```

Petit essai avec les **2 premières questions** (une par maladie, donc 4 générations Groq) :

```powershell
python compare_vector_graphrag_v3.py --max-cases 2 --sleep-between 12
```

Toutes les 4 questions (8 générations) seulement si le petit essai fonctionne et si les limites Groq le permettent :

```powershell
python compare_vector_graphrag_v3.py --max-cases 4 --sleep-between 20
```

**Ne pas relancer en boucle après une erreur 429.** Les limites de tokens par minute dépendent du compte. Le programme garde les erreurs comme telles, sans fabriquer de réponse. Il effectue seulement les lectures Neo4j déjà validées et les appels Groq nécessaires à l'évaluation.

## Fichiers produits

Dans `reports/hybrid_comparison_v1/<run_id>/` :

- `results.json` : questions, identifiants des preuves, citations, latences, erreurs, SHA256 du corpus, modèles.
- `summary.md` : métriques **techniques** descriptives pour les deux modalités.
- `manual_review.csv` : **revue humaine à effectuer** de la pertinence et du support de chaque réponse.

**Précision des chronométrages** : `generation_latency_s` mesure l'appel Groq depuis Python; `total_latency_s` est une somme indicative temps de recherche vectorielle + requête Neo4j éventuelle + génération, sans parallélisme. Le premier chargement du modèle HF et les fluctuations de réseau peuvent fausser une comparaison sur 2-4 questions. Les cas en erreur ne sont pas inclus dans les moyennes, mais restent dénombrés.

## Ce que l'on ne prétend pas mesurer

- `citations_structurally_valid=True` ne prouve **pas** qu'une affirmation est vraie ou correctement appuyée par la citation.
- L'outil `HAS_SYMPTOM` se limite à **14 claims** (8 acromégalie + 6 insuffisance surrénalienne) provenant de MedQuAD historique ; il ne couvre pas la médecine générale.
- Pas de mesure RAGAS ici : ses erreurs précédentes restent documentées séparément.
- Aucun diagnostic individuel, recommandation de traitement ou validation clinique.

## Suite après les résultats

Examiner les réponses des deux modes dans `manual_review.csv` (support des phrases, symptômes omis, données historiques). Une fois la revue humaine faite, élargir le benchmark à des questions indépendantes, négatives et hors graphe ; mesurer aussi les coûts d'exploitation, puis documenter un test de régression avant d'ajouter des outils MCP.
