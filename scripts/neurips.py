#!/usr/bin/env python3
"""
Retrieve NeurIPS, ICML, ICLR 2024 papers from Semantic Scholar with SPECTER 2.0 embeddings.
Uses API embedding when available; falls back to local SPECTER2 computation otherwise.
"""
import json
import sys
from pathlib import Path

# Add parent for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from semantic2openalex import (
    s2_bulk_search,
    s2_paper_batch,
    conference_full_names,
)
from specter2_embed import specter2_embed_papers


def _extract_embedding(paper: dict) -> list | None:
    """Extract SPECTER 2.0 embedding from paper dict. Returns None if missing."""
    emb = paper.get("embedding")
    if emb is None:
        return None
    # API may return embedding.specter_v2 as nested or flat
    if isinstance(emb, list):
        return emb
    if isinstance(emb, dict):
        v2 = emb.get("specter_v2")
        if v2 is not None:
            vec = v2.get("vector") if isinstance(v2, dict) else v2
            return vec if isinstance(vec, list) else None
        vec = emb.get("vector")
        return vec if isinstance(vec, list) else None
    return None


def fetch_neurips_icml_iclr_2024_with_embeddings(
    output_path: str = "neurips_icml_iclr_2024_embeddings.jsonl",
    batch_size: int = 32,
) -> list[dict]:
    """
    Fetch all NeurIPS, ICML, ICLR 2024 papers and attach SPECTER 2.0 embeddings.
    Saves to JSONL: one paper per line with paperId, title, abstract, year, venue, embedding.
    """
    # S2 venue names: try both short and full forms for better matching
    venue_aliases = [
        "NeurIPS",
        "Neural Information Processing Systems",
        "Advances in Neural Information Processing Systems",
        "ICML",
        "International Conference on Machine Learning",
        "ICLR",
        "International Conference on Learning Representations",
    ]
    venue_full = ",".join(venue_aliases)

    print("Fetching papers from Semantic Scholar (NeurIPS, ICML, ICLR 2024)...")
    papers = s2_bulk_search(
        query="*",
        year="2024",
        publication_types="Conference",
        fields_of_study="Computer Science",
        venue=venue_full,
        max_papers=None,
        fields="paperId,title,abstract,year,venue,externalIds",
    )
    print(f"Fetched {len(papers)} papers")

    if not papers:
        print("No papers found.")
        return []

    # Fetch embeddings from batch API (embedding.specter_v2)
    paper_ids = [p["paperId"] for p in papers]
    print("Fetching embeddings from Semantic Scholar batch API...")
    try:
        batch_papers = s2_paper_batch(
            paper_ids,
            fields="paperId,title,abstract,year,venue,embedding.specter_v2",
        )
    except Exception as e:
        print(f"Batch API failed ({e}), will compute all embeddings locally")
        batch_papers = []

    # Build paper_id -> paper map from batch response
    batch_by_id = {p["paperId"]: p for p in batch_papers if "paperId" in p}

    # Attach embeddings; compute locally for missing ones
    need_local = []
    results = []
    for p in papers:
        pid = p["paperId"]
        batch_p = batch_by_id.get(pid, p)
        emb = _extract_embedding(batch_p)
        if emb is not None:
            p["embedding"] = emb
            p["embedding_source"] = "api"
            results.append(p)
        else:
            need_local.append(p)

    if need_local:
        print(f"Computing SPECTER 2.0 locally for {len(need_local)} papers...")
        local_embs = specter2_embed_papers(need_local, batch_size=batch_size)
        for p, emb in zip(need_local, local_embs):
            p["embedding"] = emb
            p["embedding_source"] = "local"
            results.append(p)

    # Sort by venue, then title for stable output
    def _key(r):
        v = r.get("venue") or {}
        vname = v.get("name", "") if isinstance(v, dict) else str(v)
        return (vname, r.get("title", ""))

    results.sort(key=_key)

    # Save JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(results)} papers to {output_path}")
    return results


if __name__ == "__main__":
    fetch_neurips_icml_iclr_2024_with_embeddings()
