# Bio-Link Agent

**Unified Clinical Research & Precision Trial Matching using MCP and GraphRAG**

---

## Overview

Bio-Link Agent addresses the critical disconnection between **retrospective medical knowledge** (published literature) and **prospective research** (active clinical trials). By leveraging Model Context Protocol (MCP) architecture and GraphRAG techniques, this system provides two complementary solutions:

1. **Therapeutic Landscape Agent** - Identifies misalignments between published literature and active clinical trials
2. **Semantic Matcher Agent** - Matches patients to eligible trials using semantic search and ontology-based filtering

---

## The Problem

### Problem A: The "Macro" Gap (For Researchers)

Medical researchers struggle to validate whether active clinical trials align with the latest scientific consensus. When searching for "Glioblastoma Immunotherapy," they encounter thousands of papers and hundreds of trials. Manually cross-referencing this information to identify contradictions—such as trials targeting mechanisms that literature suggests are ineffective—is impractical at scale.

**Missing Link**: Multi-hop reasoning to connect mechanisms of action from literature to interventions in trials.

### Problem B: The "Micro" Mismatch (For Clinicians)

Clinicians face vocabulary mismatches when matching patients to trials. Patient notes use colloquial language ("short of breath," "worsening cough"), while trial protocols employ rigid medical taxonomies ("Dyspnea Grade 3," "COPD Exacerbation"). Standard keyword searches fail to identify eligible patients despite conceptual alignment.

**Missing Link**: Semantic understanding to bridge vocabulary gaps between clinical notes and trial eligibility criteria.

---

## The Solution

### Architecture

Bio-Link Agent uses a **Model Context Protocol (MCP)** architecture that treats PubMed and ClinicalTrials.gov as agentic tools enhanced by GraphRAG and vector search.

```
┌─────────────────────────────────────────────────────────────┐
│                     Bio-Link Agent System                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────────────┐  ┌──────────────────────────┐   │
│  │ Therapeutic Landscape │  │   Semantic Matcher       │   │
│  │  Agent (Macro)        │  │   Agent (Micro)          │   │
│  └───────────┬───────────┘  └──────────┬───────────────┘   │
│              │                          │                   │
│              ├──────────┬───────────────┤                   │
│              │          │               │                   │
│  ┌───────────▼──┐  ┌───▼────┐  ┌──────▼──────┐            │
│  │ PubMed MCP   │  │Clinical│  │  Vector DB  │            │
│  │   Server     │  │Trials  │  │  (ChromaDB) │            │
│  │              │  │MCP Srvr│  │             │            │
│  └──────────────┘  └────────┘  └─────────────┘            │
│                                                             │
│  ┌────────────────┐  ┌──────────────────────────┐          │
│  │ Knowledge Graph│  │   Ollama LLM             │          │
│  │  (NetworkX)    │  │   (Gemma/Llama)          │          │
│  └────────────────┘  └──────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Solution A: Therapeutic Landscape Agent

**Approach**: On-the-fly knowledge graph construction with LLM-based reasoning.

**Process**:
1. Query PubMed for recent literature on a condition
2. Extract entities (drugs, targets, pathways, outcomes) using Ollama
3. Build a NetworkX knowledge graph with relationships
4. Query ClinicalTrials.gov for active trials
5. Compare literature findings to trial designs using LLM reasoning
6. Identify and highlight discrepancies

**Example Output**: "Literature suggests 'Pathway B' shows limited efficacy in late-stage glioblastoma, yet Trial NCT123456 is actively recruiting patients for this approach."

### Solution B: Semantic Matcher Agent

**Approach**: Vector search with ontology-based constraint validation.

**Process**:
1. Index trial eligibility criteria into ChromaDB vector database
2. Encode patient descriptions using sentence transformers
3. Perform semantic similarity search
4. Apply ontology-based filters (age, sex, exclusion criteria)
5. Return ranked, eligible trials with match reasoning

**Example**: Patient description "experiencing shortness of breath and fatigue" semantically matches trials requiring "Dyspnea Grade 2-3" despite terminology differences.

---

## Installation

### Prerequisites

- Python 3.9+
- Mac M-series (M1/M2/M3/M4) or GPU-enabled system recommended
- Ollama installed with Gemma 3 or Llama 3 model
- 16GB+ RAM (24GB recommended)

### Setup

1. **Clone the repository**
```bash
cd /path/to/Final-Project
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install Ollama and download model** (if not already installed)
```bash
# Install Ollama (Mac)
brew install ollama

# Download model
ollama pull gemma3:12b
# OR
ollama pull llama3.1:8b
```

5. **Verify installation**
```bash
ollama run gemma3:12b "What is glioblastoma?"
```

---

## Usage

### Option 1: Web Interface (Recommended)

Launch the Streamlit web interface for an interactive experience:

```bash
# Activate virtual environment
source biolink_env/bin/activate

# Run the app
streamlit run app.py

# OR use the convenience script
./run_app.sh
```

The web interface provides:
- **Semantic Matcher**: Interactive patient-trial matching with real-time results
- **Landscape Agent**: Literature vs. trials comparison with visualizations
- **System Status**: Check data availability and system health
- **Results Display**: Formatted outputs with graphs and metrics

The app will open in your browser at `http://localhost:8501`

### Option 2: Command-Line Demos

#### Demo 1: Therapeutic Landscape Analysis
```bash
python demos/demo_landscape.py
```

This demo analyzes three conditions:
- Glioblastoma + Immunotherapy
- Lung Cancer + Targeted Therapy  
- Type 2 Diabetes

**Output**: Analysis reports, knowledge graphs, and JSON results in `demo_outputs/`

#### Demo 2: Semantic Patient Matching
```bash
python demos/demo_matcher.py
```

This demo matches patient cases to trials for:
- Lung Cancer (2 cases)
- Glioblastoma (2 cases)
- Type 2 Diabetes (2 cases)

**Output**: Ranked trial matches with similarity scores and reasoning

### Using the Agents Programmatically

#### Therapeutic Landscape Agent
```python
from agents.landscape_agent import TherapeuticLandscapeAgent

agent = TherapeuticLandscapeAgent(
    ollama_model="gemma3:12b",
    max_papers=25,
    max_trials=20
)

results = agent.analyze_condition(
    condition="glioblastoma",
    intervention="immunotherapy"
)

print(results['comparison_analysis'])
agent.visualize_graph("output.png")
```

#### Semantic Matcher Agent
```python
from agents.semantic_matcher import SemanticMatcher

# Uses EMBEDDING_MODEL from config.py (can be switched to a biomedical model like BioBERT)
matcher = SemanticMatcher()

# Index trials (one-time setup)
matcher.index_trials_for_condition("lung cancer", max_trials=100)

# Match patient
matches = matcher.match_patient_to_trials(
    patient_description="65-year-old with NSCLC, stage III, shortness of breath",
    collection_name="lung_cancer",
    patient_age=65,
    patient_sex="MALE",
    top_k=5
)

for match in matches:
    print(f"{match['nct_id']}: {match['score']:.3f} - {match['reasoning']}")
```

---

## Project Structure

```
Final-Project/
├── README.md              # Main documentation
├── requirements.txt       # Python dependencies
├── config.py             # Configuration settings
├── app.py                # Streamlit web application
│
├── agents/               # Core agent implementations
│   ├── landscape_agent.py      # Therapeutic Landscape Agent
│   └── semantic_matcher.py     # Semantic Matcher Agent
│
├── mcp_servers/          # Model Context Protocol servers
│   ├── pubmed_server.py        # PubMed API wrapper
│   └── clinicaltrials_server.py # ClinicalTrials.gov API wrapper
│
├── utils/                # Utility modules
│   ├── ollama_client.py         # LLM interaction layer
│   ├── graph_builder.py         # Knowledge graph construction
│   └── embeddings.py            # Vector search and ontology
│
├── scripts/              # Utility scripts
│   ├── data_pipeline.py         # Data collection and indexing
│   ├── train.py                 # Entity extraction and graph building
│   ├── verify_setup.py           # Environment verification
│   └── run_app.sh               # Streamlit launcher
│
├── demos/                # Demo scripts
│   ├── demo_landscape.py        # Landscape agent demo
│   └── demo_matcher.py          # Matcher agent demo
│
└── data/                 # Data storage
    ├── papers/                  # Cached PubMed papers
    ├── trials/                  # Cached clinical trials
    ├── processed/               # Extracted entities
    ├── graphs/                  # Knowledge graphs
    └── indexed_trials/          # Vector database (ChromaDB)
```

---

## Technical Details

### Technologies Used

- **MCP (Model Context Protocol)**: Agentic tool framework for API integration
- **Ollama**: Local LLM inference (Gemma 3 / Llama 3)
- **NetworkX**: Knowledge graph construction and analysis
- **ChromaDB**: Vector database for semantic search
- **Sentence Transformers**: Embedding generation for semantic similarity
- **PubMed E-utilities**: Literature retrieval
- **ClinicalTrials.gov API v2**: Trial data access

### Key Features

1. **Zero External API Costs**: Runs entirely on local LLM (Ollama)
2. **Graph-Based Reasoning**: Multi-hop inference across entity relationships
3. **Semantic Search**: Overcomes vocabulary mismatch in medical text using biomedical domain embeddings (BioBERT)
4. **Ontology Validation**: Rule-based filters for hard eligibility constraints
5. **Scalable Indexing**: Persistent vector database with incremental updates
6. **Explainable Results**: Provides reasoning for matches and discrepancies
7. **Biomedical Domain Embeddings**: Uses BioBERT for improved medical terminology understanding

---

## Embedding Model Selection

The system uses **BioBERT** (`dmis-lab/biobert-base-cased-v1.1`) as the default embedding model for semantic patient-trial matching. This choice was made after a systematic comparison with general-purpose embeddings (MiniLM).

### Comparison Results

An evaluation script (`evaluation/compare_embeddings.py`) was used to compare:
- **MiniLM** (`all-MiniLM-L6-v2`): General-purpose sentence transformer (384 dimensions)
- **BioBERT** (`dmis-lab/biobert-base-cased-v1.1`): Biomedical domain-specific model (768 dimensions)

**Evaluation Methodology:**
- Tested on 6 patient cases across 3 conditions (Lung Cancer, Glioblastoma, Type 2 Diabetes)
- Each model indexed 300 trials per condition (900 total trials)
- Compared average match scores and top-k trial relevance

**Key Findings:**
- **BioBERT significantly outperformed MiniLM** across all patient cases:
  - **Average Match Score**: BioBERT 0.9398 vs MiniLM 0.6690 (+0.2708, ~40% improvement)
  - **Top Match Score**: BioBERT consistently achieved 0.93-0.95 vs MiniLM 0.66-0.74
- BioBERT demonstrated superior semantic understanding of medical terminology and abbreviations
- More clinically relevant trial matches for patient descriptions (e.g., better matching of "NSCLC" to lung cancer trials, "EGFR mutation" to targeted therapy trials)
- Improved handling of domain-specific vocabulary and clinical concepts

**Per-Condition Results:**
- **Lung Cancer**: BioBERT +0.27-0.33 improvement in match scores
- **Glioblastoma**: BioBERT +0.24 improvement in match scores  
- **Type 2 Diabetes**: BioBERT +0.20-0.29 improvement in match scores

**Model Configuration:**
- The embedding model can be changed via `config.EMBEDDING_MODEL` in `config.py`
- Both models are supported; BioBERT is recommended for medical applications
- BioBERT automatically uses GPU (MPS on Mac) when available for faster inference
- Full evaluation results available in `evaluation/outputs/comparison_report.txt`

See `evaluation/compare_embeddings.py` to run your own comparison or reproduce the evaluation.

---

## Performance Considerations

### Speed
- **Landscape Agent**: 5-10 minutes per condition (depends on paper count)
- **Matcher Agent**: 
  - Indexing: 2-3 minutes per 100 trials (BioBERT on GPU), 3-4 minutes (CPU)
  - Querying: <1 second per patient

### Resource Usage
- **RAM**: 8-12GB during active processing
- **Storage**: ~500MB per 1000 indexed trials
- **GPU**: Recommended for BioBERT embeddings (2-3x faster on M-series Macs via MPS)

---

## Limitations & Future Work

### Current Limitations
1. **Entity Extraction**: LLM-based extraction may miss domain-specific abbreviations
2. **Ontology**: Simplified hierarchy; production would use UMLS/SNOMED-CT
3. **Relationship Extraction**: Limited to direct relationships (no multi-document synthesis)
4. **Trial Parsing**: Eligibility criteria parsing is keyword-based, not fully structured
5. **Evaluation**: No ground-truth validation dataset included

### Future Enhancements
1. ~~Integrate BioNLP models (BioBERT, PubMedBERT) for specialized entity recognition~~ ✅ **Completed**: BioBERT integrated as default embedding model
2. Add UMLS Metathesaurus for comprehensive medical ontology
3. Implement multi-document reasoning for cross-paper synthesis
4. Develop structured eligibility criteria parser using biomedical NLP
5. Create evaluation dataset with expert-labeled trial-patient matches
6. Add real-time monitoring of newly published papers and trials
7. Build web interface for clinician/researcher access

---

## Citation

If you use this work, please cite:

```
Bio-Link Agent: Unified Clinical Research & Precision Trial Matching
University of Chicago, Next Generation NLP Course
2025
```

---

## License

This project is for educational purposes as part of a university course.

---

## Contact

For questions or issues, please contact the development team.

---

## Acknowledgments

- PubMed E-utilities API
- ClinicalTrials.gov database
- Ollama project for local LLM inference
- Sentence Transformers library
- ChromaDB vector database

---

**Note**: This system is a proof-of-concept for educational purposes. It is not intended for clinical decision-making without expert oversight.

