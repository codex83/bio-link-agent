"""
Configuration file for Bio-Link Agent

Modify these settings based on your setup.
"""

# Ollama Configuration
OLLAMA_MODEL = "gemma3:12b"  # Options: "gemma3:12b", "llama3.1:8b", "gemma3:4b"
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_TEMPERATURE = 0.1  # Lower = more deterministic

# Embedding Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Options: "all-MiniLM-L6-v2", "dmis-lab/biobert-base-cased-v1.1"

# Data Directories
DATA_DIR = "./data"
VECTOR_STORE_DIR = "./data/indexed_trials"

# API Configuration
PUBMED_EMAIL = None  # Optional: your email for higher rate limits (e.g., "your.email@example.com")
PUBMED_API_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
CLINICALTRIALS_API_BASE = "https://clinicaltrials.gov/api/v2/"

# Rate Limiting (requests per second)
PUBMED_RATE_LIMIT = 0.34  # ~3 requests/sec (without API key)
CLINICALTRIALS_RATE_LIMIT = 0.5  # 2 requests/sec

# Agent Configuration
DEFAULT_MAX_PAPERS = 25  # Number of papers to analyze (Landscape Agent)
DEFAULT_MAX_TRIALS = 20  # Number of trials to analyze (Landscape Agent)
DEFAULT_TRIALS_TO_INDEX = 100  # Number of trials to index (Semantic Matcher)
DEFAULT_TOP_K_MATCHES = 10  # Number of trial matches to return

# Logging
LOG_LEVEL = "INFO"  # Options: "DEBUG", "INFO", "WARNING", "ERROR"

# Demo Configuration
DEMO_MAX_PAPERS = 15  # Reduced for faster demos
DEMO_MAX_TRIALS = 10
DEMO_TRIALS_TO_INDEX = 75

