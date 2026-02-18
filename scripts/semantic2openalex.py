import os
import time
import json
import re
from typing import Dict, Any, List, Optional

import requests

SEMANTIC_SCHOLAR_API_KEY = "D09UYXN5hYS4qiVnYIPl9dtFOcL3FzQ3pmesXHub"
OPENALEX_MAILTO = "yuancarrieyjy@gmail.com"
S2_BASE = "https://api.semanticscholar.org/graph/v1"
OA_BASE = "https://api.openalex.org"

conference_full_names = {
    "NeurIPS": "Neural Information Processing Systems",
    "ICLR": "International Conference on Learning Representations",
    "ICML": "International Conference on Machine Learning",
    "ICCV": "International Conference on Computer Vision",
    "ECCV": "European Conference on Computer Vision",
    "CVPR": "Conference on Computer Vision and Pattern Recognition",
    "AAAI": "AAAI Conference on Artificial Intelligence",
    "AISTATS": "International Conference on Artificial Intelligence and Statistics",
    "CHI": "ACM Conference on Human Factors in Computing Systems",
    "HRI": "ACM/IEEE International Conference on Human-Robot Interaction",
    "COLM": "Conference on Language, Models and Learning",
    "ACL": "Annual Meeting of the Association for Computational Linguistics",
    "EMNLP": "Conference on Empirical Methods in Natural Language Processing",
    "NAACL": "North American Chapter of the Association for Computational Linguistics",
}



# Optional but recommended for Semantic Scholar higher rate limits:
S2_API_KEY = os.getenv(SEMANTIC_SCHOLAR_API_KEY)  # put your key in env var
# Optional: OpenAlex asks for a contact email via mailto (polite pool)
OPENALEX_MAILTO = os.getenv(OPENALEX_MAILTO)  # e.g., "you@university.edu"

SESSION = requests.Session()

def _headers_s2() -> Dict[str, str]:
    h = {"User-Agent": "paper-keywords-demo/0.1"}
    if S2_API_KEY:
        h["x-api-key"] = S2_API_KEY
    return h

def _params_openalex() -> Dict[str, str]:
    # Not required, but polite + may improve pooling behavior.
    return {"mailto": OPENALEX_MAILTO} if OPENALEX_MAILTO else {}

def normalize_doi(doi: str) -> str:
    """
    Normalize DOI to a bare form like: 10.1145/1234567.1234568
    Accepts inputs like:
      - "10.1145/..."
      - "https://doi.org/10.1145/..."
      - "doi:10.1145/..."
    """
    doi = doi.strip()
    doi = re.sub(r"^(doi:\s*)", "", doi, flags=re.IGNORECASE)
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi

def s2_bulk_search(
    query: str,
    year: Optional[str] = "2010-",
    fields: str = "paperId,title,year,venue,externalIds,url",
    max_papers: Optional[int] = None,
    publication_types: Optional[str] = None,
    fields_of_study: Optional[str] = None,
    venue: Optional[str] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    publication_date_or_year: Optional[str] = None,
    sort: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Uses Semantic Scholar /paper/search/bulk with token pagination.
    The bulk endpoint is recommended for large retrieval in their tutorial.

    Args:
        query: Text query matched against title/abstract (required).
        year: Publication year or range, e.g. "2020", "2016-2020", "2010-", "-2015".
        fields: Comma-separated fields to return.
        max_papers: Max papers to fetch; None = fetch all matching papers.
        publication_types: Comma-separated types, e.g. "Conference", "JournalArticle".
        fields_of_study: Comma-separated fields, e.g. "Computer Science", "Physics".
        open_access_pdf: If True, restrict to papers with public PDF.
        min_citation_count: Minimum citation count filter.
        publication_date_or_year: Date range, e.g. "2019", "2016-03-05:2020-06-06".
        venue: Comma-separated venue names or ISO4 abbreviations.
        sort: Sort order, e.g. "publicationDate:asc", "citationCount:desc".
    """
    url = f"{S2_BASE}/paper/search/bulk"
    params: Dict[str, Any] = {"query": query, "fields": fields}
    if year is not None:
        params["year"] = year
    if publication_types is not None:
        params["publicationTypes"] = publication_types
    if fields_of_study is not None:
        params["fieldsOfStudy"] = fields_of_study
    if open_access_pdf:
        params["openAccessPdf"] = ""
    if min_citation_count is not None:
        params["minCitationCount"] = str(min_citation_count)
    if publication_date_or_year is not None:
        params["publicationDateOrYear"] = publication_date_or_year
    if venue is not None:
        params["venue"] = venue
    if sort is not None:
        params["sort"] = sort

    out: List[Dict[str, Any]] = []
    token: Optional[str] = None

    while True:
        if token:
            params["token"] = token

        r = SESSION.get(url, params=params, headers=_headers_s2(), timeout=60)
        r.raise_for_status()
        payload = r.json()

        data = payload.get("data", [])
        out.extend(data)
        if max_papers is not None and len(out) >= max_papers:
            return out[:max_papers]

        token = payload.get("token")
        if not token:
            return out

        # Be nice to the API (tune as needed)
        time.sleep(0.2)


def s2_paper_batch(
    paper_ids: List[str],
    fields: str = "paperId,title,abstract,year,venue,embedding.specter_v2",
) -> List[Dict[str, Any]]:
    """
    Fetch paper details by ID via POST /paper/batch. Supports embedding.specter_v2.
    Max 500 IDs per request.
    """
    url = f"{S2_BASE}/paper/batch"
    out: List[Dict[str, Any]] = []
    for i in range(0, len(paper_ids), 500):
        chunk = paper_ids[i : i + 500]
        r = SESSION.post(
            url,
            params={"fields": fields},
            json={"ids": chunk},
            headers=_headers_s2(),
            timeout=120,
        )
        r.raise_for_status()
        out.extend(r.json())
        time.sleep(0.2)
    return out


def openalex_work_by_doi(doi_bare: str) -> Optional[Dict[str, Any]]:
    """
    Looks up an OpenAlex work by DOI using:
      /works/https://doi.org/<DOI>  :contentReference[oaicite:4]{index=4}
    """
    doi_url = f"https://doi.org/{doi_bare}"
    url = f"{OA_BASE}/works/{doi_url}"
    r = SESSION.get(url, params=_params_openalex(), timeout=60)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def top_openalex_keywords(work: Dict[str, Any], k: int = 15) -> List[str]:
    """
    OpenAlex Work.keywords is a list of objects including display_name + score. :contentReference[oaicite:5]{index=5}
    """
    kws = work.get("keywords") or []
    kws_sorted = sorted(kws, key=lambda x: x.get("score", 0.0), reverse=True)
    return [kw["display_name"] for kw in kws_sorted[:k] if "display_name" in kw]

def build_dataset_from_s2_to_openalex(
    s2_query: str,
    year: str = "2010-",
    keywords_k: int = 15,
    max_papers: int = 500,
    publication_types: Optional[str] = None,
    fields_of_study: Optional[str] = None,
    venue: Optional[str] = None,
    fields_to_return: str = "paperId,title,year,fieldsOfStudy,venue,externalIds,url",
    **s2_search_kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Fetches ALL papers matching the query from Semantic Scholar, enriches with
    OpenAlex keywords, and returns up to max_papers records.

    max_papers: Maximum number of records to return/save (not the S2 fetch limit).

    Returns items like:
      {
        "paperId": "...",
        "title": "...",
        "year": 2021,
        "venue": "...",
        "doi": "10....",
        "openalex_id": "https://openalex.org/W....",
        "keywords": ["...", "...", ...]
      }
    """
    print(f"Searching for papers with query: {s2_query}, year: {year}, publication_types: {publication_types}, fields_of_study: {fields_of_study}, venue: {venue}, other kwargs: {s2_search_kwargs}, max_papers (to save): {max_papers}")
    if venue:
        venue = ",".join([conference_full_names[v] for v in venue.split(",")])
    else:
        venue = None
    # Fetch ALL matching papers from S2 (no limit)
    papers = s2_bulk_search(
        query=s2_query,
        year=year,
        venue=venue,
        max_papers=None,  # fetch all; we limit output by max_papers below
        fields=fields_to_return,
        publication_types=publication_types,
        fields_of_study=fields_of_study,
        **s2_search_kwargs,
    )
    print(f"Fetched {len(papers)} papers from Semantic Scholar")

    results: List[Dict[str, Any]] = []
    for i, p in enumerate(papers, start=1):
        if len(results) >= max_papers:
            break

        ext = p.get("externalIds") or {}
        doi_raw = ext.get("DOI") or ext.get("doi")
        if not doi_raw:
            # No DOI => OpenAlex matching is possible by title search, but less reliable.
            continue

        doi = normalize_doi(doi_raw)
        oa_work = openalex_work_by_doi(doi)
        if not oa_work:
            continue

        keywords = top_openalex_keywords(oa_work, k=keywords_k)

        results.append({
            **{k: v for k, v in p.items() if k in fields_to_return},
            "keywords": keywords,
        })

        # Be nice to OpenAlex (tune as needed)
        time.sleep(0.1)

    return results

if __name__ == "__main__":
    # Example: all papers in 2020, conference only, Computer Science
    dataset = build_dataset_from_s2_to_openalex(
        s2_query="InvestESG",  # broad match; use specific terms for narrower search
        #year="2022",
        #fields_of_study="Computer Science",
        keywords_k=20,
        max_papers=200,
        venue=None,
    )

    print(f"Built {len(dataset)} labeled records")
    print(json.dumps(dataset[:3], indent=2))

    # Save JSONL (one paper per line)
    with open("papers_with_openalex_keywords.jsonl", "w", encoding="utf-8") as f:
        for row in dataset:
            f.write(json.dumps(row) + "\n")
