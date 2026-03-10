"""Configuration for CS conference paper data preparation."""

# Target venues (Semantic Scholar venue filter - one per conference to avoid duplicates)
VENUES = [
    "ICML",
    "NeurIPS",
    "ICLR",
    "AAAI",
]

# Venue name normalization for matching (publicationVenue.name or venue field)
VENUE_ALIASES = {
    "icml": ["ICML", "International Conference on Machine Learning"],
    "neurips": ["NeurIPS", "Neural Information Processing Systems", "NIPS"],
    "iclr": ["ICLR", "International Conference on Learning Representations"],
    "aaai": ["AAAI", "AAAI Conference on Artificial Intelligence"],
}

# Year range for analysis (ICML started 1980, NeurIPS 1987, ICLR 2013, AAAI 1980)
YEAR_START = 2020  # ICLR start - ensures all 4 venues have papers
YEAR_END = 2020

# Semantic Scholar API
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_BATCH_SIZE = 500  # Max 500 papers per batch request
S2_RATE_LIMIT_DELAY = 1.0  # Seconds between API calls
