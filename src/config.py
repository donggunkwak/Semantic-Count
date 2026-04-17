"""Central configuration for the semantic counting pipeline."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
SENTENCES_PATH = DATA_DIR / "sentences.json"
CLUSTER_ASSIGNMENTS_PATH = DATA_DIR / "cluster_assignments.json"
CLUSTER_SUMMARIES_PATH = DATA_DIR / "cluster_summaries.json"
RESULTS_PATH = OUTPUT_DIR / "results.json"

# ── Embedding model ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ── HDBSCAN clustering ───────────────────────────────────────────────────────
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 5

# ── Query engine ──────────────────────────────────────────────────────────────
TOP_K_CLUSTERS = 5
SUMMARY_SAMPLE_SIZE = 15  # sentences sampled per cluster for summarization

# ── LLM (Google Colab AI — built-in, no API key needed) ──────────────────────
LLM_MODEL = "google/gemini-2.0-flash-lite"

# ── Ensure directories exist ──────────────────────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
