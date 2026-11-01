"""MCP v2 stdio server adding the live PubMed tool to the THREE original V3 tools.

Never run manually in terminal; use a stdio MCP client. No stdout debug prints.
"""
from __future__ import annotations

from medical_graphrag.mcp_tools.mcp_medquad_server_v3 import mcp  # Reuse existing server and three tools unchanged.
from medical_graphrag.mcp_tools.mcp_medquad_web_service_v1 import search_pubmed_guidelines


@mcp.tool()
def rechercher_guidelines_pubmed_actuelles_v1(topic: str, max_results: int = 5) -> dict:
    """Search PubMed live for guideline/consensus records; return dates, PMIDs and abstracts.

    This is NOT a verifier of 2026 clinical recommendations; PubMed indexing is not
    a complete official guideline source. Do not send patient-identifying text.
    """
    return search_pubmed_guidelines(topic, max_results)


if __name__ == '__main__':
    mcp.run(transport='stdio')
