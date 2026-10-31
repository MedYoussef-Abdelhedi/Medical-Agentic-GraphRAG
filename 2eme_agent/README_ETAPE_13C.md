# Étape 13C — Premier pilote de graphe médical **avec revue humaine**

## Objectif et périmètre

Le graphe documentaire V3 a été importé et vérifié : 29 sources, 200 documents, 750 chunks, 950 relations documentaires.
Le présent module propose **14 relations candidates** de type `HAS_SYMPTOM` pour **deux maladies** (acromégalie et insuffisance surrénalienne), à partir de phrases **exactement présentes** dans certains chunks V3.

**Ce n'est ni une extraction LLM automatique, ni une base de recommandations médicales.** L'extraction initiale emploie un catalogue pilote de phrases explicitement sélectionnées, sans interpréter les textes. Les documents historiques peuvent être incomplets ou obsolètes.

## Fichiers à installer

Mettre `clinical_graph_pilot_v3.py` et `test_clinical_graph_pilot_v3.py` dans `2eme_agent/`, **à côté de** `import_medquad_v3_neo4j.py`, sans remplacer les fichiers précédents.

## Commandes (à partir de `2eme_agent`)

1. Exécuter les tests sans réseau :

   ```powershell
   python -m unittest -v test_clinical_graph_pilot_v3.py
   ```

2. Examiner les candidats et leurs sources, sans toucher aux fichiers :

   ```powershell
   python clinical_graph_pilot_v3.py --preview
   ```

3. Générer le fichier des candidats et la feuille de revue (aucun accès réseau) :

   ```powershell
   python clinical_graph_pilot_v3.py --prepare
   ```

   Sorties dans `reports/clinical_graph_pilot_v1/` :
   - `clinical_candidates_v1.json`, version figée des 14 candidats et de leurs preuves ;
   - `review_clinical_graph_v1.csv`, décisions initiales `pending`.

4. **Étape obligatoire : revue manuelle.** Ouvrir le CSV avec Excel ou VS Code, examiner `evidence_phrase`, `evidence_chunk_id` et la sémantique du lien. Mettre `approved` ou `rejected` pour chaque ligne, remplir `reviewer` pour chaque décision. En cas de doute, choisir `rejected` ou laisser `pending` (l'import sera refusé). Conserver les autres colonnes intactes. Ne pas se limiter à la présence des mots : vérifier le sens dans le contexte et l'ancienneté de la source. Enregistrer le CSV en UTF-8. La revue documentaire ne remplace pas l'avis d'un professionnel de santé.

5. Valider la feuille de revue, sans écriture :

   ```powershell
   python clinical_graph_pilot_v3.py --check-review
   ```

6. **Uniquement après la revue** : importer les faits approuvés (cette commande écrit dans Neo4j) :

   ```powershell
   python clinical_graph_pilot_v3.py --apply
   ```

   Le programme commence par relire le graphe documentaire en entier et contrôle son intégrité. Les relations cliniques sont créées dans une transaction via `MERGE`, avec des identifiants uniques. S'il manque une preuve source, la transaction échoue sans écrire de faits cliniques.

7. Vérifier :

   ```powershell
   python clinical_graph_pilot_v3.py --verify
   python agent_langgraph_v3_neo4j.py --check-neo4j
   ```

## Schéma créé après approbation

```text
(:MedquadDisease {name})-[:HAS_SYMPTOM]->(:MedquadSymptom {name})
(:MedquadClinicalClaim {candidate_id, evidence_phrase, reviewed_by})
      -[:SUBJECT]->(:MedquadDisease)
      -[:OBJECT]->(:MedquadSymptom)
      -[:SUPPORTED_BY]->(:MedquadChunk)
(:MedquadDocument)-[:HAS_CHUNK]->(:MedquadChunk)
```

On peut remonter de chaque fait approuvé à son chunk original et sa source historique. `HAS_SYMPTOM` ne signifie ni une relation causale certaine ni une recommandation clinique actuelle.

## Précautions

- La vérification des preuves garantit seulement une correspondance textuelle, **pas une validation médicale**.
- Le mode `--prepare` ne réécrit jamais un CSV de revue existant.
- Ne pas utiliser `--apply` avant la revue : le programme refusera toutes les lignes encore `pending`.
- La commande `--apply` ne supprime pas des faits précédemment approuvés. Pour révoquer une approbation déjà importée, il faut prévoir un processus séparé avec audit.
- On ne fait pas encore de recherche GraphRAG libre : la prochaine étape sera un **outil Cypher fixe en lecture** qui retourne les liens **avec** leurs citations, puis des tests de routing LangGraph.
- Pour augmenter la couverture au-delà de ces 14 exemples, on élaborera une véritable extraction structurée et son benchmark de précision avec revue humaine, plutôt que d'inférer de nouvelles relations non vérifiées.
