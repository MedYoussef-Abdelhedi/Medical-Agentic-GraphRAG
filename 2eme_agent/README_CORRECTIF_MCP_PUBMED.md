# Étape 16B — Correctif de validation des références PubMed

## Constat observé

Le serveur MCP annonce 4 outils, la recherche PubMed retourne 5 notices et les 18 tests initiaux passent. Cependant, après deux générations Groq, l'agent lève `ValueError: No complete response with permitted PMID citations ... citation_ids_valid=False`.

La fonction initiale n'acceptait que des citations **strictement** formées comme `[PMID:42050227]`, sans espace, et son exception sortait du contexte asynchrone MCP sous la forme d'un `ExceptionGroup`. Les traces **ne contiennent pas le texte des générations rejetées** : il n'est donc pas possible de savoir si les deux erreurs proviennent d'une ponctuation différente, d'une absence de citation ou d'un identifiant inventé.

## Fichiers

- `agent_graphrag_mcp_web_v3.py` : **remplace seulement** la version précédente de ce fichier, après en avoir fait une sauvegarde.
- `test_mcp_pubmed_citations_hotfix.py` : nouveau fichier, à copier à côté de l'agent.

Conserver les anciens clients, serveurs, index V3, données MedQuAD, graphe Neo4j et le fichier de tests `test_mcp_medquad_web_v1.py`.

## Correctifs

1. Accepte les formes explicitement identifiées `[PMID:123]`, `[PMID: 123]`, `(PMID:123)` en les normalisant. La référence doit toujours correspondre à un PMID réellement retourné par PubMed.
2. Rejette les citations non vérifiées : `[1]` ne devient **pas** automatiquement un PMID, et les PMIDs inconnus restent interdits.
3. Le second essai contient une consigne explicite avec la liste exacte des références autorisées.
4. Journalise les identifiants observés et les formats de crochets non reconnus pour diagnostiquer d'éventuels nouveaux écarts.
5. Pour la recherche de recommandations sur une maladie, seuls les articles dont **le titre** mentionne la maladie (ou le synonyme connu pour l'insuffisance surrénalienne) sont fournis à la synthèse. Les résultats connexes sont signalés, mais ne sont plus traités comme des recommandations spécifiques. Ce filtre est volontairement conservateur et **non exhaustif**.
6. Si aucune synthèse entièrement terminée avec des citations permises n'est produite après deux essais, l'agent indique `BIBLIOGRAPHY ONLY` et ne montre **pas** une réponse non vérifiée : il ne plante plus lors de la sortie du contexte MCP.

## Sur Windows PowerShell

Avec le venv déjà activé, dans `2eme_agent` :

1. Faire une sauvegarde du script d'origine :

   ```powershell
   Copy-Item .\agent_graphrag_mcp_web_v3.py .\agent_graphrag_mcp_web_v3_backup.py
   ```

2. Décompresser le ZIP puis copier les deux fichiers Python dans `2eme_agent` (remplacer **uniquement** `agent_graphrag_mcp_web_v3.py`).

3. Exécuter les tests :

   ```powershell
   python -m unittest -v test_mcp_medquad_web_v1.py test_mcp_pubmed_citations_hotfix.py
   python agent_graphrag_mcp_web_v3.py --check
   ```

4. Vérifier le chemin web :

   ```powershell
   python agent_graphrag_mcp_web_v3.py --question "What are the latest 2026 clinical guidelines for treating acromegaly?"
   ```

Le succès technique est `ANSWER (PubMed abstracts only, not certified current guidance)` accompagné de véritables `[PMID:...]` autorisés. Si le LLM refuse de fournir des références vérifiables, `BIBLIOGRAPHY ONLY` est une sortie volontaire et sûre, pas un test de génération réussi.

## Limites restantes

Les références PMID sont validées **structurellement** : la correction ne confirme **ni la fidélité de chaque affirmation au résumé, ni la validité médicale, ni la pertinence d'une publication comme dernière recommandation clinique officielle**. PubMed ne garantit pas l'exhaustivité des documents émis par les sociétés savantes ou les autorités de santé. Le correctif ne remplace pas une vérification des textes intégrals et recommandations officielles en vigueur.

### Test local effectué lors de la préparation

`30 tests`: 18 tests historiques + 12 nouveaux tests de régression **réussis hors réseau**. Aucun accès réel à ton compte Groq, à PubMed ou au serveur MCP Windows pendant la préparation.
