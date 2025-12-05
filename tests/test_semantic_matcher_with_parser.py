"""
Test SemanticMatcher with integrated eligibility parser on real patient cases.
"""

import sys
import os
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.semantic_matcher import SemanticMatcher

def test_semantic_matcher_with_parser():
    """Test SemanticMatcher with eligibility parser on real patient cases."""
    
    print("="*80)
    print("SEMANTIC MATCHER WITH ELIGIBILITY PARSER - REAL PATIENT TEST")
    print("="*80)
    
    # Initialize matcher (will use eligibility parser)
    print("\nInitializing SemanticMatcher with eligibility parser...")
    matcher = SemanticMatcher(embedding_model=config.EMBEDDING_MODEL)
    
    # Load trial data to check if indexed
    trials_file = Path("data/trials/lung_cancer_trials.json")
    if not trials_file.exists():
        print(f"ERROR: Trial file not found: {trials_file}")
        print("Please run data pipeline first.")
        return
    
    with open(trials_file, 'r') as f:
        trials = json.load(f)
    
    print(f"Loaded {len(trials)} trials from cache")
    
    # Check if collection exists - use model-specific collection name
    collection_name = f"lung_cancer_{config.EMBEDDING_MODEL.split('/')[-1].replace('-', '_')}"
    print(f"Using collection: {collection_name}")
    
    try:
        stats = matcher.get_collection_info(collection_name)
        print(f"Collection '{collection_name}' exists with {stats.get('count', 0)} trials")
        
        # If collection doesn't have enough trials or doesn't exist, re-index
        if stats.get('count', 0) < 10:
            print(f"Re-indexing trials for {collection_name}...")
            matcher.embeddings.index_trials(trials[:50], collection_name, overwrite=True)
            print("Indexing complete")
    except Exception as e:
        print(f"Collection not found or error: {e}")
        print(f"Indexing trials for {collection_name}...")
        matcher.embeddings.index_trials(trials[:50], collection_name, overwrite=True)
        print("Indexing complete")
    
    # Test Case 1: Eligible patient
    print(f"\n{'='*80}")
    print("TEST CASE 1: Eligible Patient")
    print(f"{'='*80}")
    patient_1 = {
        "description": "67-year-old male with progressive shortness of breath and persistent cough. Recent CT scan shows mass in right upper lobe. Biopsy confirmed non-small cell lung cancer (NSCLC), adenocarcinoma subtype. Stage IIIA. Patient is a former smoker (quit 5 years ago). No prior cancer treatment. Performance status good.",
        "conditions": ["lung cancer", "non-small cell lung cancer"],
        "age": 67,
        "sex": "MALE"
    }
    
    print(f"Patient: {patient_1['description'][:150]}...")
    print(f"Age: {patient_1['age']}, Sex: {patient_1['sex']}")
    print(f"Conditions: {', '.join(patient_1['conditions'])}")
    
    print("\nSearching for matching trials...")
    matches_1 = matcher.match_patient_to_trials(
        patient_description=patient_1["description"],
        collection_name=collection_name,
        patient_conditions=patient_1["conditions"],
        patient_age=patient_1["age"],
        patient_sex=patient_1["sex"],
        top_k=5
    )
    
    print(f"\nFound {len(matches_1)} eligible trials:")
    for i, match in enumerate(matches_1, 1):
        print(f"\n{i}. {match['metadata']['title'][:70]}...")
        print(f"   NCT ID: {match['nct_id']}")
        print(f"   Match Score: {match['score']:.3f}")
        print(f"   Reasoning: {match['reasoning']}")
        if 'parsed_criteria' in match:
            pc = match['parsed_criteria']
            print(f"   Parsed Criteria: Age {pc.get('min_age', 'N/A')}-{pc.get('max_age', 'N/A')}, "
                  f"Sex: {pc.get('sex', 'N/A')}, "
                  f"Lab constraints: {pc.get('lab_constraints_count', 0)}")
    
    # Test Case 2: Ineligible patient (too young)
    print(f"\n{'='*80}")
    print("TEST CASE 2: Ineligible Patient (Too Young)")
    print(f"{'='*80}")
    patient_2 = {
        "description": "16-year-old male with lung cancer, stage II. Diagnosed recently. No prior treatment.",
        "conditions": ["lung cancer"],
        "age": 16,
        "sex": "MALE"
    }
    
    print(f"Patient: {patient_2['description']}")
    print(f"Age: {patient_2['age']}, Sex: {patient_2['sex']}")
    
    print("\nSearching for matching trials...")
    matches_2 = matcher.match_patient_to_trials(
        patient_description=patient_2["description"],
        collection_name=collection_name,
        patient_conditions=patient_2["conditions"],
        patient_age=patient_2["age"],
        patient_sex=patient_2["sex"],
        top_k=5
    )
    
    print(f"\nFound {len(matches_2)} eligible trials (should be fewer due to age filtering):")
    for i, match in enumerate(matches_2, 1):
        print(f"\n{i}. {match['metadata']['title'][:70]}...")
        print(f"   NCT ID: {match['nct_id']}")
        print(f"   Match Score: {match['score']:.3f}")
        if 'parsed_criteria' in match:
            pc = match['parsed_criteria']
            print(f"   Parsed Criteria: Age {pc.get('min_age', 'N/A')}-{pc.get('max_age', 'N/A')}")
    
    # Test Case 3: Female patient (to test sex filtering)
    print(f"\n{'='*80}")
    print("TEST CASE 3: Female Patient")
    print(f"{'='*80}")
    patient_3 = {
        "description": "52-year-old female with metastatic lung cancer, previously treated with chemotherapy. Now experiencing worsening fatigue and back pain. Imaging shows progression with new bone metastases. EGFR mutation positive. Seeking new treatment options.",
        "conditions": ["metastatic lung cancer"],
        "age": 52,
        "sex": "FEMALE"
    }
    
    print(f"Patient: {patient_3['description'][:150]}...")
    print(f"Age: {patient_3['age']}, Sex: {patient_3['sex']}")
    
    print("\nSearching for matching trials...")
    matches_3 = matcher.match_patient_to_trials(
        patient_description=patient_3["description"],
        collection_name=collection_name,
        patient_conditions=patient_3["conditions"],
        patient_age=patient_3["age"],
        patient_sex=patient_3["sex"],
        top_k=5
    )
    
    print(f"\nFound {len(matches_3)} eligible trials:")
    for i, match in enumerate(matches_3, 1):
        print(f"\n{i}. {match['metadata']['title'][:70]}...")
        print(f"   NCT ID: {match['nct_id']}")
        print(f"   Match Score: {match['score']:.3f}")
        if 'parsed_criteria' in match:
            pc = match['parsed_criteria']
            print(f"   Parsed Criteria: Age {pc.get('min_age', 'N/A')}-{pc.get('max_age', 'N/A')}, "
                  f"Sex: {pc.get('sex', 'N/A')}")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")
    print("\nSummary:")
    print(f"  Test 1 (Eligible): {len(matches_1)} matches")
    print(f"  Test 2 (Too Young): {len(matches_2)} matches (should be fewer)")
    print(f"  Test 3 (Female): {len(matches_3)} matches")

if __name__ == "__main__":
    test_semantic_matcher_with_parser()

