"""
MLE-based inference for estimating fraction of AI-generated text (alpha).
Based on: https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Tuple, Union


def _safe_apply(series, func):
    """Apply with optional swifter for parallelization."""
    try:
        import swifter
        return series.swifter.progress_bar(False).apply(func)
    except ImportError:
        return series.apply(func)


class MLE:
    """
    Maximum-likelihood estimator for the mixing parameter alpha between
    human-generated and AI-generated text distributions.
    """

    def __init__(self, word_df_path: str):
        """
        Load word distribution from parquet.

        Expected columns: Word, logP, logQ, log1-P, log1-Q
        - logP: log P(word in human sentence)
        - logQ: log P(word in AI sentence)
        """
        df = pd.read_parquet(word_df_path)
        word_df = df.copy()
        self.all_tokens_set = set(word_df["Word"].tolist())
        self.log_p_hat = {row["Word"]: row["logP"] for _, row in word_df.iterrows()}
        self.log_q_hat = {row["Word"]: row["logQ"] for _, row in word_df.iterrows()}
        self.log_one_minus_p_hat = {
            row["Word"]: row["log1-P"] for _, row in word_df.iterrows()
        }
        self.log_one_minus_q_hat = {
            row["Word"]: row["log1-Q"] for _, row in word_df.iterrows()
        }

    def optimized_log_likelihood(
        self,
        alpha: np.ndarray,
        log_p_values: np.ndarray,
        log_q_values: np.ndarray,
    ) -> float:
        """Negative log likelihood of mixture model. Minimize to find alpha."""
        alpha = alpha[0]
        ll = np.mean(
            np.log((1 - alpha) + alpha * np.exp(log_q_values - log_p_values))
        )
        return -ll

    def precompute_log_probabilities(
        self, data: Union[pd.Series, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Precompute log P(sentence) and log Q(sentence) for each sentence.
        Uses -13.8 for out-of-vocabulary words.
        """
        if isinstance(data, np.ndarray):
            data = pd.Series(list(data))

        total_log_one_minus_p = sum(self.log_one_minus_p_hat.values())
        total_log_one_minus_q = sum(self.log_one_minus_q_hat.values())

        def log_p_fn(x):
            return sum(self.log_p_hat.get(t, -13.8) for t in x) + (
                total_log_one_minus_p
                - sum(
                    self.log_one_minus_p_hat[t]
                    for t in x
                    if t in self.all_tokens_set
                )
            )

        def log_q_fn(x):
            return sum(self.log_q_hat.get(t, -13.8) for t in x) + (
                total_log_one_minus_q
                - sum(
                    self.log_one_minus_q_hat[t]
                    for t in x
                    if t in self.all_tokens_set
                )
            )

        log_p_values = _safe_apply(data, log_p_fn)
        log_q_values = _safe_apply(data, log_q_fn)
        return np.array(log_p_values), np.array(log_q_values)

    def bootstrap_alpha_inference(
        self,
        data: Union[pd.Series, np.ndarray, list],
        n_bootstrap: int = 1000,
    ) -> np.ndarray:
        """
        Infer alpha via bootstrap. Returns 95% CI as [2.5th, 97.5th] percentiles.
        """
        if isinstance(data, list):
            data = pd.Series(data)
        full_log_p, full_log_q = self.precompute_log_probabilities(data)
        alpha_values = []
        n = len(data)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            sample_log_p = full_log_p[idx]
            sample_log_q = full_log_q[idx]
            result = minimize(
                self.optimized_log_likelihood,
                x0=[0.5],
                args=(sample_log_p, sample_log_q),
                method="L-BFGS-B",
                bounds=[(0, 1)],
            )
            if result.success:
                alpha_values.append(result.x[0])
        return np.percentile(alpha_values, [2.5, 97.5])

    def inference(
        self,
        inference_file_path: str,
        exploded_data: bool = False,
        n_bootstrap: int = 1000,
    ) -> Tuple[float, float]:
        """
        Estimate alpha and 95% CI for a target corpus.

        Parameters:
            inference_file_path: Parquet with 'inference_sentence' column
                (list of words per row, or list of sentences per row if not exploded).
            exploded_data: If False, explode so each row is one sentence.

        Returns:
            (estimated_alpha, half_width_of_95_CI)
        """
        inference_data = pd.read_parquet(inference_file_path)
        if "inference_sentence" not in inference_data.columns:
            raise ValueError("inference_sentence column not found")

        if not exploded_data:
            inference_data = inference_data.explode("inference_sentence")
        inference_data.dropna(subset=["inference_sentence"], inplace=True)
        inference_data = inference_data[
            inference_data["inference_sentence"].apply(len) > 1
        ]
        inference_data.dropna(subset=["inference_sentence"], inplace=True)
        inference_data.reset_index(drop=True, inplace=True)

        def filter_tokens(x):
            return set(t for t in x if t in self.all_tokens_set)

        data = _safe_apply(
            inference_data["inference_sentence"],
            filter_tokens,
        )

        ci = self.bootstrap_alpha_inference(data, n_bootstrap=n_bootstrap)
        solution = round(np.mean(ci), 3)
        half_width = round((ci[1] - ci[0]) / 2, 3)
        return solution, half_width
