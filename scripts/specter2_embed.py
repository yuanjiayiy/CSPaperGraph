"""
SPECTER 2.0 embedding computation for papers without API embeddings.
Uses allenai/specter2_base + allenai/specter2 (proximity adapter).
"""
from typing import Dict, Any, List, Optional

_SPECTER2_MODEL = None
_SPECTER2_TOKENIZER = None


def _get_specter2_model():
    """Lazy-load SPECTER2 model and tokenizer."""
    global _SPECTER2_MODEL, _SPECTER2_TOKENIZER
    if _SPECTER2_MODEL is None:
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel

        _SPECTER2_TOKENIZER = AutoTokenizer.from_pretrained("allenai/specter2_base")
        _SPECTER2_MODEL = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        _SPECTER2_MODEL.load_adapter(
            "allenai/specter2", source="hf", load_as="proximity", set_active=True
        )
    return _SPECTER2_MODEL, _SPECTER2_TOKENIZER


def specter2_embed_papers(
    papers: List[Dict[str, Any]],
    batch_size: int = 32,
    device: Optional[str] = None,
) -> List[List[float]]:
    """
    Compute SPECTER 2.0 embeddings for papers. Each paper must have 'title' and optionally 'abstract'.

    Args:
        papers: List of dicts with 'title' and 'abstract' (or None)
        batch_size: Batch size for inference
        device: 'cuda' or 'cpu'; auto-detect if None

    Returns:
        List of embedding vectors (each is list of floats)
    """
    import torch

    model, tokenizer = _get_specter2_model()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    text_batch = [
        (p.get("title") or "") + tokenizer.sep_token + (p.get("abstract") or "")
        for p in papers
    ]

    all_embeddings = []
    for i in range(0, len(text_batch), batch_size):
        batch = text_batch[i : i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        # [CLS] token embedding
        emb = output.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.extend([e.tolist() for e in emb])
    return all_embeddings
