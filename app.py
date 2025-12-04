"""
Bio-Link Agent - Streamlit Web Interface

A user-friendly web interface for the Bio-Link Agent system.
"""

import streamlit as st
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.semantic_matcher import SemanticMatcher
from agents.landscape_agent import TherapeuticLandscapeAgent
from utils.embeddings import EmbeddingManager

# Page configuration
st.set_page_config(
    page_title="Bio-Link Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'matcher' not in st.session_state:
    st.session_state.matcher = None
if 'landscape_agent' not in st.session_state:
    st.session_state.landscape_agent = None
if 'trials_indexed' not in st.session_state:
    st.session_state.trials_indexed = False

# Header
st.markdown('<p class="main-header">🔬 Bio-Link Agent</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Unified Clinical Research & Precision Trial Matching using MCP and GraphRAG</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Agent",
        ["🏠 Home", "🔍 Semantic Matcher", "📊 Landscape Agent", "📈 About"]
    )
    
    st.markdown("---")
    st.header("System Status")
    
    # Check if data exists
    data_dir = Path("./data")
    has_trials = (data_dir / "trials").exists() and len(list((data_dir / "trials").glob("*.json"))) > 0
    has_papers = (data_dir / "papers").exists() and len(list((data_dir / "papers").glob("*.json"))) > 0
    has_processed = (data_dir / "processed").exists() and len(list((data_dir / "processed").glob("*.json"))) > 0
    
    if has_trials:
        st.success("✓ Trial data available")
    else:
        st.warning("⚠ No trial data - run data_pipeline.py")
    
    if has_papers:
        st.success("✓ Paper data available")
    else:
        st.warning("⚠ No paper data - run data_pipeline.py")
    
    if has_processed:
        st.success("✓ Processed entities available")
    else:
        st.warning("⚠ No processed data - run train.py")
    
    st.markdown("---")
    st.markdown("### Quick Links")
    st.markdown("- [GitHub](https://github.com)")
    st.markdown("- [Documentation](README.md)")

# Home Page
if page == "🏠 Home":
    st.header("Welcome to Bio-Link Agent")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔍 Semantic Matcher Agent
        **Solves the "Micro Mismatch" Problem**
        
        Match patients to clinical trials using semantic search, overcoming vocabulary mismatches between:
        - Patient descriptions ("shortness of breath")
        - Trial criteria ("Dyspnea Grade 3")
        
        **Features:**
        - Vector-based semantic similarity
        - Ontology-based filtering
        - Real-time patient matching
        """)
        
        if st.button("Try Semantic Matcher →", use_container_width=True):
            st.session_state.page = "semantic"
            st.rerun()
    
    with col2:
        st.markdown("""
        ### 📊 Therapeutic Landscape Agent
        **Solves the "Macro Gap" Problem**
        
        Compare published literature to active clinical trials to identify:
        - Discrepancies between research and trials
        - Missing safety information
        - Research gaps
        
        **Features:**
        - Knowledge graph construction
        - Multi-hop reasoning
        - Discrepancy detection
        """)
        
        if st.button("Try Landscape Agent →", use_container_width=True):
            st.session_state.page = "landscape"
            st.rerun()
    
    st.markdown("---")
    
    # Statistics
    st.header("System Statistics")
    
    if has_trials and has_papers:
        try:
            stats_file = data_dir / "dataset_statistics.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                col1, col2, col3, col4 = st.columns(4)
                
                total_trials = sum(c['trials']['total_trials'] for c in stats['conditions'].values())
                total_papers = sum(c['papers']['total_papers'] for c in stats['conditions'].values())
                
                with col1:
                    st.metric("Total Trials", total_trials)
                with col2:
                    st.metric("Total Papers", total_papers)
                with col3:
                    st.metric("Conditions", len(stats['conditions']))
                with col4:
                    if has_processed:
                        processed = len(list((data_dir / "processed").glob("*.json")))
                        st.metric("Processed", processed)
                    else:
                        st.metric("Processed", 0)
        except Exception as e:
            st.warning(f"Could not load statistics: {e}")

# Semantic Matcher Page
elif page == "🔍 Semantic Matcher":
    st.header("Semantic Matcher Agent")
    st.markdown("Match patients to clinical trials using semantic search")
    
    # Initialize matcher
    if st.session_state.matcher is None:
        with st.spinner("Initializing Semantic Matcher..."):
            try:
                st.session_state.matcher = SemanticMatcher()
                st.success("Semantic Matcher initialized!")
            except Exception as e:
                st.error(f"Error initializing matcher: {e}")
                st.stop()
    
    # Condition selection
    condition = st.selectbox(
        "Select Medical Condition",
        ["lung_cancer", "glioblastoma", "type_2_diabetes"],
        format_func=lambda x: x.replace("_", " ").title()
    )
    
    # Check if trials are indexed
    collection_name = condition
    trials_file = Path(f"data/trials/{condition}_trials.json")
    
    try:
        stats = st.session_state.matcher.get_collection_info(collection_name)
        trial_count = stats.get('count', 0)
        
        if trial_count == 0:
            # Try to load from cached data first (300 trials from data pipeline)
            if trials_file.exists():
                st.info(f"Loading {condition} trials from cached data...")
                with st.spinner("Indexing trials from cache (this may take a minute)..."):
                    try:
                        with open(trials_file, 'r') as f:
                            cached_trials = json.load(f)
                        
                        st.session_state.matcher.embeddings.index_trials(
                            trials=cached_trials,
                            collection_name=collection_name,
                            overwrite=True
                        )
                        st.success(f"✓ Indexed {len(cached_trials)} trials from cache!")
                        st.rerun()
                    except Exception as e:
                        st.warning(f"Error loading from cache: {e}. Fetching from API...")
                        # Fall back to API
                        with st.spinner("Fetching trials from API (this may take a few minutes)..."):
                            st.session_state.matcher.index_trials_for_condition(
                                condition.replace("_", " "),
                                max_trials=300,  # Match data pipeline size
                                collection_name=collection_name
                            )
                        st.success("Trials indexed!")
                        st.rerun()
            else:
                # No cached data, fetch from API
                st.warning(f"No cached data found. Fetching trials from API...")
                with st.spinner("Fetching and indexing trials (this may take a few minutes)..."):
                    st.session_state.matcher.index_trials_for_condition(
                        condition.replace("_", " "),
                        max_trials=300,  # Match data pipeline size
                        collection_name=collection_name
                    )
                st.success("Trials indexed!")
                st.rerun()
        else:
            st.success(f"✓ {trial_count} trials indexed for {condition}")
    except Exception as e:
        # Collection doesn't exist, try to load from cache
        if trials_file.exists():
            st.info(f"Loading {condition} trials from cached data...")
            with st.spinner("Indexing trials from cache (this may take a minute)..."):
                try:
                    with open(trials_file, 'r') as f:
                        cached_trials = json.load(f)
                    
                    st.session_state.matcher.embeddings.index_trials(
                        trials=cached_trials,
                        collection_name=collection_name,
                        overwrite=True
                    )
                    st.success(f"✓ Indexed {len(cached_trials)} trials from cache!")
                    st.rerun()
                except Exception as e2:
                    st.error(f"Error loading from cache: {e2}")
                    st.stop()
        else:
            st.warning(f"Collection not found and no cached data. Fetching from API...")
            with st.spinner("Fetching trials from API (this may take a few minutes)..."):
                try:
                    st.session_state.matcher.index_trials_for_condition(
                        condition.replace("_", " "),
                        max_trials=300,  # Match data pipeline size
                        collection_name=collection_name
                    )
                    st.success("Trials indexed!")
                    st.rerun()
                except Exception as e2:
                    st.error(f"Error indexing: {e2}")
                    st.stop()
    
    st.markdown("---")
    
    # Patient input
    st.subheader("Patient Information")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=65)
    with col2:
        patient_sex = st.selectbox("Sex", ["MALE", "FEMALE", "ALL"])
    with col3:
        patient_conditions = st.text_input(
            "Conditions (comma-separated)",
            placeholder="e.g., lung cancer, NSCLC"
        )
    
    patient_description = st.text_area(
        "Patient Description",
        height=150,
        placeholder="""Example: 65-year-old male with progressive shortness of breath and persistent cough. 
Recent CT scan shows mass in right upper lobe. Biopsy confirmed non-small cell lung cancer (NSCLC), 
adenocarcinoma subtype. Stage IIIA. Patient is a former smoker (quit 5 years ago). 
No prior cancer treatment. Performance status good."""
    )
    
    top_k = st.slider("Number of matches to return", 1, 10, 5)
    
    if st.button("🔍 Find Matching Trials", type="primary", use_container_width=True):
        if not patient_description.strip():
            st.error("Please enter a patient description")
        else:
            with st.spinner("Searching for matching trials..."):
                try:
                    conditions_list = [c.strip() for c in patient_conditions.split(",")] if patient_conditions else None
                    
                    matches = st.session_state.matcher.match_patient_to_trials(
                        patient_description=patient_description,
                        collection_name=collection_name,
                        patient_conditions=conditions_list,
                        patient_age=patient_age if patient_age > 0 else None,
                        patient_sex=patient_sex if patient_sex != "ALL" else None,
                        top_k=top_k
                    )
                    
                    if matches:
                        st.success(f"Found {len(matches)} matching trials!")
                        st.markdown("---")
                        
                        for i, match in enumerate(matches, 1):
                            with st.expander(f"Trial {i}: {match['metadata']['title'][:80]}...", expanded=(i==1)):
                                col1, col2 = st.columns([2, 1])
                                
                                with col1:
                                    st.markdown(f"**NCT ID:** {match['nct_id']}")
                                    st.markdown(f"**Status:** {match['metadata']['status']}")
                                    st.markdown(f"**Match Score:** {match['score']:.3f}")
                                    st.markdown(f"**Reasoning:** {match['reasoning']}")
                                
                                with col2:
                                    st.markdown(f"**Age Range:** {match['metadata'].get('min_age', 'N/A')} - {match['metadata'].get('max_age', 'N/A')}")
                                    st.markdown(f"**Sex:** {match['metadata'].get('sex', 'ALL')}")
                                
                                st.markdown("**Conditions:**")
                                st.write(match['metadata'].get('conditions', 'N/A'))
                                
                                # Progress bar for match score
                                st.progress(match['score'])
                    else:
                        st.warning("No matching trials found. Try adjusting the patient description or conditions.")
                
                except Exception as e:
                    st.error(f"Error matching patient: {e}")
                    st.exception(e)

# Landscape Agent Page
elif page == "📊 Landscape Agent":
    st.header("Therapeutic Landscape Agent")
    st.markdown("Compare literature findings to active clinical trials")
    
    # Initialize agent
    if st.session_state.landscape_agent is None:
        with st.spinner("Initializing Landscape Agent..."):
            try:
                st.session_state.landscape_agent = TherapeuticLandscapeAgent(
                    ollama_model="gemma3:12b",
                    max_papers=60,  # Fallback limit if cache not available
                    max_trials=300,  # Fallback limit if cache not available
                    use_cache=True  # Use cached data from data_pipeline.py and train.py
                )
                st.success("Landscape Agent initialized!")
            except Exception as e:
                st.error(f"Error initializing agent: {e}")
                st.stop()
    
    st.info("💡 This agent uses cached data: 60 papers and 300 trials per condition (from data_pipeline.py and train.py)")
    
    # Analysis parameters
    col1, col2 = st.columns(2)
    with col1:
        condition = st.selectbox(
            "Medical Condition",
            ["glioblastoma", "lung cancer", "type 2 diabetes"],
            format_func=lambda x: x.replace("_", " ").title()
        )
    with col2:
        intervention = st.text_input(
            "Intervention (optional)",
            value="",
            placeholder="e.g., immunotherapy, targeted therapy"
        )
    
    if st.button("🔬 Analyze Therapeutic Landscape", type="primary", use_container_width=True):
        if not condition.strip():
            st.error("Please enter a medical condition")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("Loading cached papers and trials...")
                progress_bar.progress(10)
                
                results = st.session_state.landscape_agent.analyze_condition(
                    condition=condition,
                    intervention=intervention
                )
                
                if "error" in results:
                    st.error(results["error"])
                else:
                    progress_bar.progress(100)
                    status_text.text("Analysis complete!")
                    
                    st.success("Analysis complete!")
                    st.markdown("---")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Articles", results['articles_analyzed'])
                    with col2:
                        st.metric("Trials", results['trials_analyzed'])
                    with col3:
                        st.metric("Graph Nodes", results['graph_summary']['num_nodes'])
                    with col4:
                        st.metric("Graph Edges", results['graph_summary']['num_edges'])
                    
                    st.markdown("---")
                    
                    # Literature Summary
                    st.subheader("📚 Literature Summary")
                    st.markdown(results['literature_summary'])
                    
                    st.markdown("---")
                    
                    # Comparison Analysis
                    st.subheader("🔍 Comparison Analysis")
                    st.markdown(results['comparison_analysis'])
                    
                    st.markdown("---")
                    
                    # Top Trials
                    st.subheader("🏥 Top Active Trials")
                    for i, trial in enumerate(results.get('trials', [])[:5], 1):
                        with st.expander(f"Trial {i}: {trial['brief_title'][:80]}...", expanded=(i==1)):
                            st.markdown(f"**NCT ID:** {trial['nct_id']}")
                            st.markdown(f"**Status:** {trial['status']}")
                            st.markdown(f"**Conditions:** {', '.join(trial['conditions'])}")
                            
                            if trial.get('interventions'):
                                st.markdown("**Interventions:**")
                                for interv in trial['interventions'][:3]:
                                    st.markdown(f"- {interv['name']} ({interv['type']})")
                            
                            st.markdown("**Summary:**")
                            st.write(trial.get('brief_summary', 'N/A')[:500] + "...")
                    
                    # Graph visualization
                    if st.button("📊 View Knowledge Graph"):
                        graph_path = f"demo_outputs/graph_{condition.replace(' ', '_')}.png"
                        if os.path.exists(graph_path):
                            st.image(graph_path, use_container_width=True)
                        else:
                            st.info("Generating graph visualization...")
                            st.session_state.landscape_agent.visualize_graph(graph_path)
                            st.image(graph_path, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.exception(e)
            
            finally:
                progress_bar.empty()
                status_text.empty()

# About Page
elif page == "📈 About":
    st.header("About Bio-Link Agent")
    
    st.markdown("""
    ### Project Overview
    
    Bio-Link Agent addresses the critical disconnection between **retrospective medical knowledge** 
    (published literature) and **prospective research** (active clinical trials).
    
    ### The Two Problems We Solve
    
    #### Problem A: The "Macro" Gap (For Researchers)
    - **Issue**: Researchers cannot easily validate if active clinical trials align with latest scientific consensus
    - **Solution**: Therapeutic Landscape Agent compares literature to trials, identifying discrepancies
    
    #### Problem B: The "Micro" Mismatch (For Clinicians)
    - **Issue**: Vocabulary mismatch between patient descriptions and trial eligibility criteria
    - **Solution**: Semantic Matcher Agent uses vector embeddings to match patients to trials semantically
    
    ### Technology Stack
    
    - **MCP (Model Context Protocol)**: Agentic tool framework
    - **GraphRAG**: Knowledge graph construction from literature
    - **Vector Search**: Semantic similarity matching (ChromaDB)
    - **LLM**: Local inference via Ollama (Gemma 3:12b)
    - **APIs**: PubMed E-utilities, ClinicalTrials.gov API v2
    
    ### Key Features
    
    ✅ Zero external API costs (runs on local LLM)
    ✅ Graph-based multi-hop reasoning
    ✅ Semantic search overcomes vocabulary gaps
    ✅ Ontology-based constraint validation
    ✅ Explainable results with reasoning
    
    ### Dataset
    
    - **900 clinical trials** indexed across 3 conditions
    - **180 research papers** analyzed
    - **500+ knowledge graph nodes** with relationships
    - **3 medical conditions**: Lung Cancer, Glioblastoma, Type 2 Diabetes
    
    ### Getting Started
    
    1. Ensure Ollama is running with `gemma3:12b` model
    2. Run data collection: `python data_pipeline.py --auto`
    3. Run training: `python train.py`
    4. Use this interface or run demos: `python demos/demo_matcher.py`
    
    ### Documentation
    
    - **README.md**: Full project documentation
    - **explanation.md**: Technical deep-dive
    - **QUICKSTART.md**: 5-minute setup guide
    """)
    
    st.markdown("---")
    st.markdown("**Version**: 1.0")
    st.markdown("**Date**: December 2025")

# Run the app
if __name__ == "__main__":
    pass

