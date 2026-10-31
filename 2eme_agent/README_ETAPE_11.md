# Étape 11 — Premier agent médical LangGraph + MedQuAD V3

**Objectif :** un agent qui choisit entre **rechercher un passage** et **consulter les statistiques**, avec traçabilité des résultats. C'est un prototype documentaire ; il ne fournit pas d'avis médical personnalisé.

## Installation

Copier `agent_langgraph_v3.py` et `test_agent_langgraph_v3.py` dans `2eme_agent`, à côté de `rag_pipeline.py` et du répertoire `data/`.

Exécuter dans PowerShell déjà ouvert dans `2eme_agent` :

```powershell
python -m pip show langgraph langchain-groq langchain-huggingface
python -m pip check
python -m unittest -v test_agent_langgraph_v3.py
python agent_langgraph_v3.py --check
```

Si les bibliothèques LangGraph ou LangChain Groq sont absentes, installer celles qui manquent :

```powershell
python -m pip install "langgraph>=1,<2" "langchain-groq>=1,<2"
python -m pip check
```

Un conflit signalé par `pip check` doit être résolu avant de changer d'autres paquets. Ne jamais exposer la clé `GROQ_API_KEY`.

## Premiers essais (appels Groq)

```powershell
python agent_langgraph_v3.py --question "How many documents and chunks are indexed?"
python agent_langgraph_v3.py --question "What are the symptoms of acromegaly?"
python agent_langgraph_v3.py --question "What do the latest 2026 clinical guidelines recommend for acromegaly?"
```

Vérifier pour chaque essai :
- l'outil appelé est cohérent (`statistiques_corpus` ou `rechercher_medquad`),
- l'agent répond avec des identifiants de chunks existants lorsque nécessaire,
- il ne présente pas MedQuAD comme des recommandations médicales actuelles,
- si aucun outil n'a été appelé, le script bloque la réponse.

**Limites :** un appel à l'outil ne garantit pas à lui seul l'exactitude de la réponse. Les références affichées sous « PASSAGES RETROUVÉS » servent à l'audit, pas à valider toutes les affirmations. L'agent n'a pas de mémoire persistante, de recherche web, de Neo4j ni de MCP dans cette version. Le modèle peut appeler un outil inadéquat : vérifier ses traces. Les documents NIDDK sont historiques. Pour la suite, ajouter un test automatisé sur les sorties réelles et une politique d'abstention plus stricte.

## Références

- https://reference.langchain.com/python/langgraph.prebuilt/tool_node/tools_condition
- https://reference.langchain.com/python/langchain-groq/chat_models/ChatGroq/bind_tools
- https://console.groq.com/docs/tool-use/local-tool-calling
