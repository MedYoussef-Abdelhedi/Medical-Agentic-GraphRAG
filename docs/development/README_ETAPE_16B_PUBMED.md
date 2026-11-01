# Étape 16B — Recherche biomédicale actuelle avec MCP (PubMed)

## Pourquoi ce choix ?

L'agent historique utilise 750 chunks MedQuAD V3 et un graphe pilote Neo4j. Ces données **ne permettent pas de justifier des recommandations cliniques de 2026**.

Ce pack ajoute un **quatrième outil MCP** qui interroge **en direct** l'API PubMed de la National Library of Medicine (ESearch et EFetch) pour récupérer les notices de publications de type *guideline* ou *consensus*, avec PMID, titre, date et résumé disponible. Le contenu n'est pas ajouté à Neo4j ni aux embeddings V3.

**Important : PubMed n'est pas un registre exhaustif des recommandations officielles en vigueur.** Un article récent n'est pas forcément une recommandation à jour ; les notices ne remplacent pas la lecture du texte intégral ni la vérification sur le site de l'organisme émetteur (NICE, HAS, société savante, etc.). Le système doit refuser de conclure quand les résumés ne suffisent pas. Ce prototype n'est pas un outil de conseil médical.

## À installer

Copier les **5 fichiers Python** du ZIP directement dans le dossier déjà existant `2eme_agent`, à côté des scripts de l'étape 16. Ajouter le README facultativement. **Ne pas remplacer** les fichiers `mcp_medquad_client_v3.py`, `mcp_medquad_server_v3.py`, `agent_graphrag_mcp_v3.py`. Le correctif de lecture MCP de l'étape 16 doit rester en place.

Fichiers :

- `mcp_medquad_web_service_v1.py` : recherche PubMed E-utilities en HTTPS ; aucune dépendance externe.
- `mcp_medquad_server_web_v3.py` : les **trois** outils du serveur existant + le quatrième outil `rechercher_guidelines_pubmed_actuelles_v1`.
- `mcp_medquad_web_client_v3.py` : CLI de test du nouveau serveur MCP.
- `agent_graphrag_mcp_web_v3.py` : routage historique vs publications actuelles + Groq.
- `test_mcp_medquad_web_v1.py` : tests hors réseau.

## Vérifier sans réseau

Dans PowerShell (venv activé) :

```powershell
python -m unittest -v test_mcp_medquad_web_v1.py
python agent_graphrag_mcp_web_v3.py --check
```

Les 18 tests couvrent : parsing XML PubMed, construction d'URL fixe NCBI, gestion des erreurs, contrôles d'entrée, routage, références PMID et fin de génération. Le SDK MCP, Neo4j et Groq n'ont **pas été exécutés** dans l'environnement de préparation du pack ; une exécution réelle sur ton PC est requise.

## Tester le quatrième outil

```powershell
python mcp_medquad_web_client_v3.py --list
python mcp_medquad_web_client_v3.py --pubmed "Acromegaly"
```

Pour `--list`, attendre **4 outils**.

La commande `--pubmed` interroge Internet. Elle doit retourner des identifiants `PMID`, des URLs `https://pubmed.ncbi.nlm.nih.gov/.../`, des `publication_date` et les résumés s'ils existent. Si PubMed ne répond pas, `status=not_supported` avec la raison ; ne pas inventer de recommandations.

## Tester le routage de l'agent

Question historique — le workflow LangGraph + Neo4j + Vector V3 + Groq existant est réutilisé, avec le nouveau serveur :

```powershell
python agent_graphrag_mcp_web_v3.py --question "What are the symptoms of acromegaly?"
```

Question actuelle — le routeur appelle PubMed via MCP, **pas les anciens chunks MedQuAD** :

```powershell
python agent_graphrag_mcp_web_v3.py --question "What are the latest 2026 clinical guidelines for treating acromegaly?"
```

Le résultat devra indiquer les dates réelles de publication des articles retrouvés et les citer sous la forme `[PMID:12345678]`. Si aucune recommandation de 2026 ne figure dans les **résultats retournés**, le système doit signaler cette limitation, sans déclarer qu'il n'existe aucune recommandation de 2026.

## Fonctionnement (premier prototype)

1. Le routeur reconnaît une demande de *recommandations actuelles* ou récentes.
2. Il recherche des publications sur les **deux maladies du pilote** : `Acromegaly` ou `Adrenal insufficiency` (pas de diagnostic implicite).
3. L'outil MCP interroge l'index PubMed et récupère titres, dates et abstracts.
4. Si le contenu est suffisamment documenté, Groq produit une **synthèse bibliographique prudente**, avec des références PMID vérifiées structurellement et une fin contrôlée ; sinon seules les références ou la raison de l'échec sont affichées.
5. Pour les questions historiques sur les symptômes, le workflow V3 est inchangé.

## Sécurité et qualité

- **Aucune clé PubMed requise pour les tests ponctuels.** Une clé optionnelle `NCBI_API_KEY` peut être définie dans l'environnement si nécessaire ; jamais dans les fichiers du projet.
- Limitation des requêtes à l'hôte officiel NCBI et aux deux endpoints fixes. Pas de navigation arbitraire, pas d'URL fournie par l'utilisateur, pas de code généré par LLM.
- Réseau HTTPS + timeout ; aucune persistance des résultats dans le graphe.
- L'API NCBI limite par défaut à **3 requêtes/seconde sans clé**. Cette extension fait deux appels successifs par question et n'est pas conçue pour des recherches intensives.
- Ne pas envoyer de dossier patient ou d'informations personnelles à PubMed ; seules des thématiques médicales génériques sont transmises.
- La validation structurelle des PMID **ne vérifie pas** que chaque phrase de la réponse est réellement justifiée par un passage. Une revue documentaire/humaine reste requise.
- Si le besoin est la conformité aux recommandations cliniques **en vigueur**, il faudra ensuite ajouter des sources primaires officielles avec vérification de leurs versions et dates, et tester la recherche externe sur un benchmark spécifique.

### Sources techniques

- SDK MCP v2 : https://py.sdk.modelcontextprotocol.io/
- API NCBI E-utilities : https://www.ncbi.nlm.nih.gov/books/NBK25497/
- Politique NCBI et limites de débit : https://support.nlm.nih.gov/kbArticle/?pn=KA-05510
