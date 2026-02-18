#!/usr/bin/env python3
"""
For papers from neurips.py: estimate LLM use per paper, bin into 5 groups,
and compute the diameter of the embedding space for each bin.

Diameter = max pairwise Euclidean distance between embeddings in the bin.

Usage:
  1. Run neurips.py first to create neurips_icml_iclr_2024_embeddings.jsonl
  2. Ensure distribution/CS.parquet exists (from Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers)
  3. python neurips_llm_bins_diameter.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_quantification import quantify_llm_use

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PAPERS_JSONL = SCRIPT_DIR / "neurips_icml_iclr_2024_embeddings.jsonl"
DISTRIBUTION_PATH = (
    SCRIPT_DIR
    / "Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers"
    / "distribution"
    / "CS.parquet"
)
N_BINS = 5
N_BOOTSTRAP_PER_PAPER = 100  # Lower for speed; increase for stability


def load_papers(path: Path) -> list[dict]:
    """Load papers from JSONL."""
    papers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


def get_abstract(paper: dict) -> str:
    """Extract abstract text from paper."""
    abstract = paper.get("abstract")
    if abstract is None or (isinstance(abstract, str) and not abstract.strip()):
        # Fallback to title if no abstract
        return paper.get("title") or ""
    return str(abstract)


def get_embedding(paper: dict) -> np.ndarray | None:
    """Extract embedding vector from paper."""
    emb = paper.get("embedding")
    if emb is None:
        return None
    if isinstance(emb, list):
        return np.array(emb, dtype=np.float64)
    if isinstance(emb, dict):
        v2 = emb.get("specter_v2")
        if v2 is not None:
            vec = v2.get("vector") if isinstance(v2, dict) else v2
            return np.array(vec, dtype=np.float64) if vec is not None else None
        vec = emb.get("vector")
        return np.array(vec, dtype=np.float64) if vec is not None else None
    return None


def compute_diameter(embeddings: np.ndarray, metric: str = "euclidean") -> float:
    """
    Compute diameter = max pairwise distance between vectors.
    embeddings: (n, d) array
    metric: 'euclidean' (L2) or 'cosine'
    """
    if len(embeddings) < 2:
        return 0.0

    if metric == "euclidean":
        # Max L2 distance between any two points
        from scipy.spatial.distance import pdist

        pairwise = pdist(embeddings, metric="euclidean")
        return float(np.max(pairwise))
    elif metric == "cosine":
        from scipy.spatial.distance import pdist

        pairwise = pdist(embeddings, metric="cosine")
        return float(np.max(pairwise))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def run(
    papers_path: Path = PAPERS_JSONL,
    distribution_path: Path = DISTRIBUTION_PATH,
    n_bins: int = N_BINS,
    n_bootstrap: int = N_BOOTSTRAP_PER_PAPER,
    diameter_metric: str = "euclidean",
) -> dict:
    """
    Load papers, estimate LLM use per paper, bin by alpha, compute diameter per bin.
    Returns dict with bin info and results.
    """
    if not papers_path.exists():
        raise FileNotFoundError(
            f"Papers file not found: {papers_path}. Run neurips.py first."
        )
    if not distribution_path.exists():
        raise FileNotFoundError(
            f"Distribution not found: {distribution_path}. "
            "Clone Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers and ensure distribution/CS.parquet exists."
        )

    papers = load_papers(papers_path)
    print(f"Loaded {len(papers)} papers")

    # Filter papers with abstract and embedding
    valid = []
    for p in papers:
        abstract = get_abstract(p)
        emb = get_embedding(p)
        if abstract.strip() and emb is not None:
            valid.append(p)
    print(f"Papers with abstract and embedding: {len(valid)}")

    if len(valid) == 0:
        return {"error": "No valid papers"}

    # Estimate LLM use (alpha) per paper
    print("Estimating LLM use per paper (this may take a while)...")
    alphas = []
    for i, p in enumerate(valid):
        abstract = get_abstract(p)
        alpha, _ = quantify_llm_use(
            [abstract],
            distribution_path=str(distribution_path),
            n_bootstrap=n_bootstrap,
        )
        alphas.append(alpha)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(valid)} papers")

    alphas = np.array(alphas)

    # Bin by alpha (equal count per bin via sorted order)
    order = np.argsort(alphas)
    n = len(alphas)
    bin_indices = np.zeros(n, dtype=int)
    for b in range(n_bins):
        start = b * n // n_bins
        end = (b + 1) * n // n_bins
        bin_indices[order[start:end]] = b

    bin_results = []
    for b in range(n_bins):
        indices = np.where(bin_indices == b)[0]
        bin_papers = [valid[i] for i in indices]
        bin_alphas = alphas[indices]
        embeddings = np.array([get_embedding(p) for p in bin_papers])
        diameter = compute_diameter(embeddings, metric=diameter_metric)
        bin_results.append({
            "bin": b + 1,
            "alpha_range": (float(np.min(bin_alphas)), float(np.max(bin_alphas))),
            "n_papers": len(bin_papers),
            "alpha_mean": float(np.mean(bin_alphas)),
            "alpha_std": float(np.std(bin_alphas)) if len(indices) > 1 else 0.0,
            "diameter": diameter,
        })

    return {
        "n_papers": len(valid),
        "bins": bin_results,
        "alpha_overall": {
            "mean": float(np.mean(alphas)),
            "std": float(np.std(alphas)),
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bin papers by LLM use, compute diameter per bin")
    parser.add_argument(
        "--papers",
        type=Path,
        default=PAPERS_JSONL,
        help="Path to papers JSONL (from neurips.py)",
    )
    parser.add_argument(
        "--distribution",
        type=Path,
        default=DISTRIBUTION_PATH,
        help="Path to distribution parquet (CS.parquet)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=N_BINS,
        help="Number of bins",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=N_BOOTSTRAP_PER_PAPER,
        help="Bootstrap samples per paper (lower=faster, higher=more stable)",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidean", "cosine"],
        default="euclidean",
        help="Distance metric for diameter",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    results = run(
        papers_path=args.papers,
        distribution_path=args.distribution,
        n_bins=args.bins,
        n_bootstrap=args.n_bootstrap,
        diameter_metric=args.metric,
    )
    if "error" in results:
        print(results["error"])
        sys.exit(1)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    print("\n" + "=" * 70)
    print("LLM use bins and embedding space diameter")
    print("=" * 70)
    print(f"Total papers: {results['n_papers']}")
    print(f"Overall alpha: {results['alpha_overall']['mean']:.3f} ± {results['alpha_overall']['std']:.3f}")
    print()
    print(f"{'Bin':>4} {'Alpha range':>20} {'N':>6} {'Mean α':>8} {'Diameter':>12}")
    print("-" * 70)
    for br in results["bins"]:
        lo, hi = br["alpha_range"]
        print(
            f"{br['bin']:>4} [{lo:.3f}, {hi:.3f}]"
            f" {br['n_papers']:>6} {br['alpha_mean']:>8.3f} {br['diameter']:>12.4f}"
        )


if __name__ == "__main__":
    main()
