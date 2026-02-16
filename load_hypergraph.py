"""
Load hypergraph data as list of tuples.
Each tuple represents a hyperedge (set of nodes).
"""

import json
from pathlib import Path


def load_content_hyperedges(year: int) -> list[tuple]:
    """Load content hyperedges for a year as list of tuples."""
    path = Path("data/hypergraphs") / f"content_hyperedges_{year}.json"
    with open(path) as f:
        data = json.load(f)
    return [tuple(h) for h in data]


def load_context_hyperedges(year: int) -> list[tuple]:
    """Load context hyperedges for a year as list of tuples."""
    path = Path("data/hypergraphs") / f"context_hyperedges_{year}.json"
    with open(path) as f:
        data = json.load(f)
    return [tuple(h) for h in data]


# Example usage:
# from load_hypergraph import load_content_hyperedges, load_context_hyperedges
# content_edges = load_content_hyperedges(2020)  # list[tuple[str, ...]]
# context_edges = load_context_hyperedges(2020)  # list[tuple[str, ...]]
