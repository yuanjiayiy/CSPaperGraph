# Hypergraph Data Preparation for CS Conference Papers

Replication of [Shi & Evans (2023)](https://www.nature.com/articles/s41467-023-36741-4) on CS top conference papers (ICML, NeurIPS, ICLR, AAAI).

## Setup

```bash
pip install -r requirements.txt
```

**Note**: Semantic Scholar API allows ~100 requests per 5 minutes without an API key. For large-scale fetching, consider [requesting an API key](https://www.semanticscholar.org/product/api).

## Pipeline

### 1. Fetch papers from Semantic Scholar

```bash
python fetch_papers.py
```

- Queries Semantic Scholar API for papers from ICML, NeurIPS, ICLR, AAAI
- Saves raw paper data to `data/raw/papers_{year}.json`
- Fetches references for context hypergraph (cited venues)

### 2. Build content hypergraph

```bash
python build_content_hypergraph.py
```

- **Nodes**: Keywords from `fieldsOfStudy` / `s2FieldsOfStudy` (e.g., Computer Science, Mathematics)
- **Hyperedges**: Each paper → tuple of its keywords
- **Output**: `data/hypergraphs/content_hyperedges_{year}.json` — list of tuples, each tuple = (keyword1, keyword2, ...)

### 3. Build context hypergraph

```bash
python build_context_hypergraph.py
```

- **Nodes**: Venues (journals/conferences) cited by the paper
- **Hyperedges**: Each paper → tuple of cited venues
- **Output**: `data/hypergraphs/context_hyperedges_{year}.json` — list of tuples, each tuple = (venue1, venue2, ...)

## Output format

Each output file is a JSON array of arrays. In Python, load as list of tuples:

```python
import json
with open("data/hypergraphs/content_hyperedges_2020.json") as f:
    content_hyperedges = [tuple(h) for h in json.load(f)]
# content_hyperedges = [("Computer Science", "Mathematics"), ...]
```

Or use the helper:

```python
from load_hypergraph import load_content_hyperedges, load_context_hyperedges
content_edges = load_content_hyperedges(2020)
context_edges = load_context_hyperedges(2020)
```

## Configuration

Edit `config.py` to change:
- `VENUES`: Target conferences
- `YEAR_START`, `YEAR_END`: Year range (default 2013–2023 to include ICLR)
# CSPaperGraph
