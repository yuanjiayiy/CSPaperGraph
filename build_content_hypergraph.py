#!/usr/bin/env python3
"""
Build content hypergraph from paper data.
Each node = keyword (from fieldsOfStudy / s2FieldsOfStudy).
Each hyperedge = tuple of keywords for a paper (one hyperedge per paper).
Output: list of tuples per year, where each tuple = (keyword1, keyword2, ...).
"""

import json
from pathlib import Path

from config import YEAR_END, YEAR_START


def extract_keywords(paper: dict) -> tuple[str, ...]:
    """
    Extract content keywords from a paper.
    Uses s2FieldsOfStudy (category) for finer granularity, falls back to fieldsOfStudy.
    """
    keywords = set()

    # s2FieldsOfStudy: [{"category": "Computer Science", "source": "..."}, ...]
    for fos in paper.get("s2FieldsOfStudy") or []:
        cat = fos.get("category")
        if cat:
            keywords.add(cat.strip())

    # fieldsOfStudy: ["Computer Science", "Mathematics", ...]
    for fos in paper.get("fieldsOfStudy") or []:
        if fos:
            keywords.add(fos.strip())

    return tuple(sorted(keywords))


def build_content_hyperedges_for_year(papers: list[dict]) -> list[tuple]:
    """
    Build content hyperedges for a year.
    Each hyperedge = tuple of keywords. Papers with < 2 keywords are skipped.
    """
    hyperedges = []
    for paper in papers:
        keywords = extract_keywords(paper)
        if len(keywords) >= 2:  # Hyperedge needs at least 2 nodes
            hyperedges.append(keywords)
    return hyperedges


def main():
    data_dir = Path("data/raw")
    output_dir = Path("data/hypergraphs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for year in range(YEAR_START, YEAR_END + 1):
        path = data_dir / f"papers_{year}.json"
        if not path.exists():
            print(f"Skipping {year}: no data file")
            continue

        with open(path) as f:
            papers = json.load(f)

        hyperedges = build_content_hyperedges_for_year(papers)
        print(f"{year}: {len(hyperedges)} content hyperedges from {len(papers)} papers")

        out_path = output_dir / f"content_hyperedges_{year}.json"
        with open(out_path, "w") as f:
            json.dump(hyperedges, f, indent=0)
        print(f"  Saved to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
