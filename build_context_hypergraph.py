#!/usr/bin/env python3
"""
Build context hypergraph from paper data.
Each node = venue (journal/conference) cited by the paper.
Each hyperedge = tuple of cited venues for a paper (one hyperedge per paper).
Output: list of tuples per year, where each tuple = (venue1, venue2, ...).
"""

import json
from pathlib import Path
from re import T

from pandas._libs.lib import tuples_to_object_array

from config import YEAR_END, YEAR_START


def extract_cited_venues(paper: dict) -> tuple[str, ...]:
    """
    Extract venues from a paper's references (cited papers).
    Uses publicationVenue.name or venue field from each reference.
    """
    venues = set()
    for ref in paper.get("references") or []:
        # Prefer publicationVenue.name (canonical), fallback to venue
        pv = ref.get("publicationVenue") or {}
        name = pv.get("name") or ref.get("venue")
        if name and str(name).strip():
            venues.add(str(name).strip())
    return tuple(sorted(venues))


def build_context_tuples_for_year(papers: list[dict]) -> list[tuple]:
    """
    Build context tuples for a year.
    Each tuple = (paper_id, cited venue).
    """
    tuples = []
    for paper in papers:
        venues = extract_cited_venues(paper)
        tuples.extend((paper["paperId"], venue) for venue in venues)
    return tuples


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

        tuples = build_context_tuples_for_year(papers)
        print(f"{year}: {len(tuples)} context tuples from {len(papers)} papers")

        out_path = output_dir / f"context_tuples_{year}.json"
        with open(out_path, "w") as f:
            json.dump(tuples, f, indent=0)
        print(f"  Saved to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
