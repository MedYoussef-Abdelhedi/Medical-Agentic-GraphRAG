"""Live biomedical literature retrieval via the public NCBI PubMed API.

Only NCBI E-utilities host, fixed endpoints, encoded parameters, no generic URL fetch.
Published literature != an official currently applicable guideline. Never merge results
with the old MedQuAD corpus or treat titles alone as medical recommendations.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Callable

BASE = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
MAX_BODY_BYTES = 2_000_000


def _download(url: str) -> bytes:
    if not url.startswith(BASE):
        raise ValueError('Only NCBI E-utilities allowed')
    req = urllib.request.Request(url, headers={'User-Agent': 'MedQuADResearchPrototype/1.0 (academic document search)', 'Accept': 'application/json, application/xml'})
    with urllib.request.urlopen(req, timeout=12) as response:
        data = response.read(MAX_BODY_BYTES + 1)
    if len(data) > MAX_BODY_BYTES:
        raise ValueError('NCBI response too large')
    return data


def _as_text(node: ET.Element | None) -> str:
    return ' '.join(' '.join(node.itertext()).split()) if node is not None else ''


def _first_text(node: ET.Element, *paths: str) -> str:
    for p in paths:
        value = _as_text(node.find(p))
        if value:
            return value
    return ''


def _publication_date(article: ET.Element) -> str:
    # ArticleDate is electronic-publication date; PubDate is journal issue date.
    node = article.find('.//ArticleDate')
    if node is None:
        node = article.find('.//JournalIssue/PubDate')
    if node is None:
        return 'unknown'
    year = _first_text(node, 'Year')
    month = _first_text(node, 'Month')
    day = _first_text(node, 'Day')
    if year:
        return '-'.join(x for x in (year, month, day) if x)
    return _first_text(node, 'MedlineDate') or 'unknown'


def parse_pubmed_xml(xml: bytes) -> list[dict[str, Any]]:
    root = ET.fromstring(xml)
    results = []
    for item in root.findall('.//PubmedArticle'):
        medline = item.find('MedlineCitation')
        if medline is None:
            continue
        pmid = _first_text(medline, 'PMID')
        if not re.fullmatch(r'\d{1,15}', pmid):
            continue
        article = medline.find('Article')
        if article is None:
            continue
        title = _first_text(article, 'ArticleTitle')
        abstract_node = article.find('Abstract')
        pieces = []
        if abstract_node is not None:
            for part in abstract_node.findall('AbstractText'):
                label = part.attrib.get('Label', '')
                value = _as_text(part)
                if value:
                    pieces.append(f'{label}: {value}' if label else value)
        abstract = '\n'.join(pieces)[:4500]
        types = [t.strip() for t in (_as_text(x) for x in article.findall('.//PublicationType')) if t.strip()]
        results.append({
            'pmid': pmid,
            'title': title,
            'publication_date': _publication_date(article),
            'publication_types': types,
            'abstract': abstract,
            'has_abstract': bool(abstract),
            'url': f'https://pubmed.ncbi.nlm.nih.gov/{pmid}/',
            'source': 'PubMed/NLM indexed literature',
        })
    return results


def search_pubmed_guidelines(topic: str, max_results: int = 5, *, downloader: Callable[[str], bytes] | None = None) -> dict:
    """Search current PubMed bibliographic index and fetch the indexed abstracts.

    This is NOT exhaustive web search and NOT verification of current practice standards.
    This tool cannot establish that a 2026 guideline exists or that results supersede
    prior society guidelines. Results are informational literature snippets only.
    """
    if not isinstance(topic, str) or not (3 <= len(topic.strip()) <= 100):
        raise ValueError('Medical topic must contain 3..100 characters')
    topic = ' '.join(topic.split())
    if not re.fullmatch(r"[\w\s\-.'’]+", topic, re.UNICODE) or not re.search(r'\w', topic):
        raise ValueError('Medical topic contains unsupported characters')
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 8:
        raise ValueError('max_results must be 1..8')
    downloader = downloader or _download
    today = datetime.now(timezone.utc).date().isoformat()
    # Exact topic phrase in title/abstract, PubMed-indexed MeSH, plus guideline/consensus filters.
    escaped = topic.replace('"', '')
    term = (f'("{escaped}"[Title/Abstract] OR "{escaped}"[MeSH Terms]) '
            'AND ("Guideline"[Publication Type] OR "Practice Guideline"[Publication Type] '
            'OR guideline[Title] OR consensus[Title] OR "clinical practice"[Title])')
    params = {'db': 'pubmed', 'term': term, 'sort': 'pub date', 'retmax': str(max_results), 'retmode': 'json', 'tool': 'MedQuADResearchPrototype'}
    if os.getenv('NCBI_API_KEY'):
        params['api_key'] = os.environ['NCBI_API_KEY']
    try:
        search_url = BASE + 'esearch.fcgi?' + urllib.parse.urlencode(params)
        data = json.loads(downloader(search_url))
        if not isinstance(data, dict) or not isinstance(data.get('esearchresult'), dict):
            raise ValueError('Unexpected PubMed search format')
        ids = data['esearchresult'].get('idlist', [])
        if not isinstance(ids, list) or not all(re.fullmatch(r'\d{1,15}', str(i)) for i in ids):
            raise ValueError('Unexpected PubMed IDs')
        ids = ids[:max_results]
        if not ids:
            return {'status': 'not_supported', 'topic': topic, 'searched_on_utc': today,
                    'results': [], 'reason': 'No PubMed-indexed guideline/consensus matches returned; not proof that no guideline exists.'}
        fetch_params = {'db': 'pubmed', 'id': ','.join(ids), 'retmode': 'xml', 'tool': 'MedQuADResearchPrototype'}
        if os.getenv('NCBI_API_KEY'):
            fetch_params['api_key'] = os.environ['NCBI_API_KEY']
        fetch_url = BASE + 'efetch.fcgi?' + urllib.parse.urlencode(fetch_params)
        articles = parse_pubmed_xml(downloader(fetch_url))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, ET.ParseError, json.JSONDecodeError) as exc:
        # No fallback to unsourced model knowledge or historic MedQuAD for a current-guideline request.
        return {'status': 'not_supported', 'topic': topic, 'searched_on_utc': today,
                'results': [], 'reason': f'PubMed retrieval unavailable or invalid ({type(exc).__name__}).'}
    indexed = {a['pmid']: a for a in articles}
    articles = [indexed[i] for i in ids if i in indexed]
    return {
        'status': 'ok' if articles else 'not_supported',
        'topic': topic, 'searched_on_utc': today,
        'source_type': 'live_pubmed_indexed_literature_not_necessarily_current_official_guidance',
        'results': articles,
        'count': len(articles),
        'warning': ('Publication dates are not verification dates. PubMed can omit guidelines published on professional society sites. '
                    'Abstracts may be incomplete. Do not infer official 2026 recommendations from a title or indexed record.'),
    }
