"""
Estimate text distributions of human vs AI-generated content.
Based on: https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers
"""
import pandas as pd
import numpy as np
from collections import Counter


def get_vocabulary_intersection(human_counts: dict, ai_counts: dict) -> set:
    """Words present in both human and AI vocabularies."""
    return set(human_counts.keys()).intersection(ai_counts.keys())


def filter_frequent_words(word_counts: dict, min_occurrences: int) -> dict:
    """Filter words by minimum occurrence threshold."""
    return {w: c for w, c in word_counts.items() if c >= min_occurrences}


def count_human_binary_word_occurrences(human_data: pd.DataFrame) -> dict:
    """
    Count how many sentences each word appears in (binary per sentence).
    human_data must have 'human_sentence' column (list of words per row).
    """
    word_counts = Counter(
        word for sent in human_data["human_sentence"] for word in set(sent)
    )
    return dict(word_counts)


def count_ai_binary_word_occurrences(ai_data: pd.DataFrame) -> dict:
    """
    Count how many sentences each word appears in (binary per sentence).
    ai_data must have 'ai_sentence' column (list of words per row).
    """
    word_counts = Counter(
        word for sent in ai_data["ai_sentence"] for word in set(sent)
    )
    return dict(word_counts)


def estimate_log_probabilities(word_counts: dict, total_sents: int) -> dict:
    """Log probability of each word appearing in any sentence."""
    return {
        word: np.log(count / total_sents)
        for word, count in word_counts.items()
    }


def calculate_log_probability(
    human_probs: dict, ai_probs: dict, common_vocab: set
) -> pd.DataFrame:
    """
    Compute log(p), log(1-p), log(q), log(1-q) for common vocabulary.
    p = P(word in human sentence), q = P(word in AI sentence).
    """
    data = []
    for word in common_vocab:
        log_human_prob = human_probs.get(word, -np.inf)
        log_ai_prob = ai_probs.get(word, -np.inf)

        log_one_minus_human_prob = np.log1p(-np.exp(log_human_prob))
        log_one_minus_ai_prob = np.log1p(-np.exp(log_ai_prob))

        human_log_odds = log_human_prob - log_one_minus_human_prob
        ai_log_odds = log_ai_prob - log_one_minus_ai_prob
        log_odds_ratio = human_log_odds - ai_log_odds

        if np.isinf(log_odds_ratio) or np.isnan(log_odds_ratio):
            continue

        data.append({
            "Word": word,
            "logP": log_human_prob,
            "log1-P": log_one_minus_human_prob,
            "logQ": log_ai_prob,
            "log1-Q": log_one_minus_ai_prob,
        })

    df = pd.DataFrame(data)
    return df


def estimate_text_distribution(
    human_source_path: str,
    ai_source_path: str,
    save_file_path: str = "Word.parquet",
) -> None:
    """
    Estimate text distribution of human vs AI content and save to parquet.

    Parameters:
        human_source_path: Parquet with 'human_sentence' column (list of words per row).
        ai_source_path: Parquet with 'ai_sentence' column (list of words per row).
        save_file_path: Output path for distribution parquet.
    """
    human_data = pd.read_parquet(human_source_path)
    ai_data = pd.read_parquet(ai_source_path)

    if "human_sentence" not in human_data.columns:
        raise ValueError("human_sentence column not found in human data")
    if "ai_sentence" not in ai_data.columns:
        raise ValueError("ai_sentence column not found in ai data")

    human_data = human_data[human_data["human_sentence"].apply(len) > 1]
    ai_data = ai_data[ai_data["ai_sentence"].apply(len) > 1]
    human_data.dropna(subset=["human_sentence"], inplace=True)
    ai_data.dropna(subset=["ai_sentence"], inplace=True)

    human_word_counts = count_human_binary_word_occurrences(human_data)
    ai_word_counts = count_ai_binary_word_occurrences(ai_data)

    total_human_sentences = len(human_data)
    total_ai_sentences = len(ai_data)

    human_log_probs = estimate_log_probabilities(
        human_word_counts, total_human_sentences
    )
    ai_log_probs = estimate_log_probabilities(
        ai_word_counts, total_ai_sentences
    )

    common_vocab = get_vocabulary_intersection(human_word_counts, ai_word_counts)
    frequent_human_words = filter_frequent_words(human_word_counts, 5)
    frequent_ai_words = filter_frequent_words(ai_word_counts, 3)
    frequent_common_vocab = common_vocab.intersection(
        frequent_human_words.keys(), frequent_ai_words.keys()
    )

    log_likelihood_df = calculate_log_probability(
        human_log_probs, ai_log_probs, frequent_common_vocab
    )
    log_likelihood_df.to_parquet(save_file_path, index=False)
