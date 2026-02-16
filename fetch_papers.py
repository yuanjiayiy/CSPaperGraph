#!/usr/bin/env python3
"""
Fetch papers from ICML, NeurIPS, ICLR, AAAI using Semantic Scholar API.
Saves raw paper data for building content and context hypergraphs.
"""

import json
import time
from pathlib import Path

import requests

from config import (
    S2_API_BASE,
    S2_BATCH_SIZE,
    S2_RATE_LIMIT_DELAY,
    VENUES,
    YEAR_END,
    YEAR_START,
)

# Fields for search (bulk search does not return references)
SEARCH_FIELDS = "paperId,title,year,venue,publicationVenue,fieldsOfStudy,s2FieldsOfStudy"


def search_papers_by_venue(venue: str, year: str) -> list[dict]:
    """Search papers from a venue for a given year using Semantic Scholar bulk search."""
    url = f"{S2_API_BASE}/paper/search/bulk"
    all_papers = []
    token = None
    # Broad query - bulk search requires a query; "learning" matches ML/AI papers
    query = "learning"

    while True:
        params = {
            "query": query,
            "year": year,
            "venue": venue,
            "fields": SEARCH_FIELDS,
            "limit": 1000,
        }
        if token:
            params["token"] = token

        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"  Error fetching {venue} {year}: {e}")
            break

        papers = data.get("data", [])
        # Filter to papers actually from this venue (venue filter can be fuzzy)
        venue_lower = venue.lower()
        year_int = int(year)
        filtered = [
            p
            for p in papers
            if p.get("year") == year_int
            and (
                (p.get("venue") or "").lower().find(venue_lower) >= 0
                or (p.get("publicationVenue", {}).get("name") or "").lower().find(venue_lower) >= 0
            )
        ]
        all_papers.extend(filtered)

        token = data.get("token")
        if not token or len(papers) < 1000:
            break

        time.sleep(S2_RATE_LIMIT_DELAY)

    return all_papers


def fetch_papers_for_year(year: int) -> list[dict]:
    """Fetch all papers from target venues for a given year."""
    all_papers = []
    seen_ids = set()

    for venue in VENUES:
        papers = search_papers_by_venue(venue, str(year))
        for p in papers:
            pid = p.get("paperId")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                all_papers.append(p)
        print(f"  {venue} {year}: {len(papers)} papers")
        time.sleep(S2_RATE_LIMIT_DELAY)

    return all_papers


def fetch_references_batch(paper_ids: list[str]) -> dict[str, list]:
    """Fetch references for a batch of papers (max 500)."""
    url = f"{S2_API_BASE}/paper/batch"
    params = {
        "fields": "references.paperId,references.venue,references.publicationVenue"
    }
    try:
        resp = requests.post(
            url,
            params=params,
            json={"ids": paper_ids},
            timeout=120,
        )
        resp.raise_for_status()
        return {p["paperId"]: p.get("references", []) for p in resp.json()}
    except requests.RequestException as e:
        print(f"  Batch reference fetch error: {e}")
        return {}


def enrich_papers_with_references(papers: list[dict], output_dir: Path) -> None:
    """Fetch references for all papers (bulk search doesn't return them) and save."""
    for i in range(0, len(papers), S2_BATCH_SIZE):
        batch = papers[i : i + S2_BATCH_SIZE]
        refs = fetch_references_batch([p["paperId"] for p in batch])
        for p in batch:
            p["references"] = refs.get(p["paperId"], [])
        time.sleep(S2_RATE_LIMIT_DELAY)
        if (i + S2_BATCH_SIZE) % 5000 == 0 or i + S2_BATCH_SIZE >= len(papers):
            print(f"  Fetched references for {min(i + S2_BATCH_SIZE, len(papers))}/{len(papers)} papers")

    # Update saved files by year
    by_year = {}
    for p in papers:
        y = p.get("year")
        if y is not None:
            by_year.setdefault(y, []).append(p)
    for year, year_papers in by_year.items():
        out_path = output_dir / f"papers_{year}.json"
        with open(out_path, "w") as f:
            json.dump(year_papers, f, indent=0)


def main():
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in range(YEAR_START, YEAR_END + 1):
        print(f"Fetching papers for {year}...")
        papers = fetch_papers_for_year(year)
        print(f"  Total unique: {len(papers)}")

        if papers:
            out_path = output_dir / f"papers_{year}.json"
            with open(out_path, "w") as f:
                json.dump(papers, f, indent=0)
            print(f"  Saved to {out_path}")

    # Enrich with references for context hypergraph
    print("Fetching references for papers missing them...")
    all_papers = []
    for p in output_dir.glob("papers_*.json"):
        with open(p) as f:
            all_papers.extend(json.load(f))
    enrich_papers_with_references(all_papers, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
