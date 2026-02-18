"""
Tokenization for LLM quantification. Uses spaCy for sentence splitting.
Based on: https://github.com/Weixin-Liang/Mapping-the-Increasing-Use-of-LLMs-in-Scientific-Papers
"""
import re
from typing import List


def tokenize(text: str, nlp=None) -> List[List[str]]:
    """
    Process text into tokenized sentences for LLM quantification.

    Splits into sentences, extracts words (lowercased, non-numeric) per sentence.

    Parameters:
        text: Input text (e.g., abstract).
        nlp: Optional spaCy model. If None, will lazy-load en_core_web_sm.

    Returns:
        List of sentences, each a list of words.
    """
    if nlp is None:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            ) from None

    text = text.replace("\n", " ")
    sentence_list = []
    doc = nlp(text)
    for sent in doc.sents:
        words = re.findall(r"\b\w+\b", sent.text.lower())
        words_without_digits = [w for w in words if not w.isdigit()]
        if len(words_without_digits) != 0:
            sentence_list.append(words_without_digits)
    return sentence_list
