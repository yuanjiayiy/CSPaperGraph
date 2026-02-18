#!/usr/bin/env python3
"""
Demo: Quantify LLM use in scientific abstracts.

Based on: https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers

Usage:
  1. Install: pip install -r requirements-llm-quantify.txt
  2. Download spaCy model: python -m spacy download en_core_web_sm
  3. (Optional) Clone the reference repo for pre-built distributions and validation data:
     git clone https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers.git ref_repo
  4. Run this script
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_quantification import quantify_llm_use, estimate_text_distribution, tokenize
import pandas as pd

REF_REPO_PATH = "/home/ymwan/Sci4Sci/Carrie/scripts/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers"
def demo_with_validation_data():
    """
    Validate against the reference repo's ground-truth data.
    Requires: git clone ... Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers ref_repo
    """
    ref = Path(REF_REPO_PATH)  # or path to cloned repo
    if not (ref / "distribution" / "CS.parquet").exists():
        print("Skipping validation: clone the repo and set ref_repo path")
        return

    print("Validation (Computer Science abstracts, known alpha):")
    print(f"{'Ground Truth':>12} {'Prediction':>12} {'CI':>10} {'Error':>10}")
    for alpha in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:
        path = ref / f"data/validation_data/CS/ground_truth_alpha_{alpha}.parquet"
        if not path.exists():
            print(f"Skipping validation: {path} does not exist")
            continue
        est, ci = quantify_llm_use(
            str(path),
            distribution_path=str(ref / "distribution/CS.parquet"),
            n_bootstrap=100,  # reduce for quick demo
            exploded_data=True,
        )
        err = abs(est - alpha)
        print(f"{alpha:12.3f} {est:12.3f} {ci:10.3f} {err:10.3f}")


def demo_with_raw_text():
    """Demo with raw abstract text."""
    abstracts = [
        "We propose a novel method for estimating the distribution of AI-generated "
        "content in scientific corpora. Our approach uses maximum likelihood estimation "
        "without requiring instance-level classification.",
        "Large language models have demonstrated remarkable capabilities across "
        "diverse tasks. In this work, we explore their application to scientific writing.",
    ]
    alpha, ci = quantify_llm_use(
        abstracts,
        distribution_path=f"{REF_REPO_PATH}/distribution/CS.parquet",  # or build from human+AI data
        exploded_data=True,
    )
    print(f"Estimated LLM use: {alpha:.1%} ± {ci:.1%}")
    return alpha, ci


def demo_build_distribution_and_infer():
    """
    Build distribution from human + AI parquet, then run inference.
    You need to provide human and AI text parquet files.
    """
    human_path = "human_abstracts.parquet"  # column: human_sentence (list of words)
    ai_path = "ai_abstracts.parquet"        # column: ai_sentence (list of words)
    target_path = "target_abstracts.parquet"  # column: inference_sentence

    if not Path(human_path).exists():
        print("Create human_abstracts.parquet and ai_abstracts.parquet first")
        return

    alpha, ci = quantify_llm_use(
        target_path,
        human_parquet_path=human_path,
        ai_parquet_path=ai_path,
        distribution_save_path="my_distribution.parquet",
    )
    print(f"Estimated LLM use: {alpha:.1%} ± {ci:.1%}")


if __name__ == "__main__":
    print("LLM Quantification Demo")
    print("-" * 50)
    demo_with_validation_data()
    print()
    demo_with_raw_text()
