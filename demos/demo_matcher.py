"""
Demo: Semantic Matcher Agent

Demonstrates the Micro problem solution by matching patients to trials for:
1. Lung Cancer
2. Glioblastoma  
3. Type 2 Diabetes

Shows how semantic search overcomes vocabulary mismatch between patient
descriptions and trial eligibility criteria.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.semantic_matcher import SemanticMatcher
from loguru import logger


# Example patient cases for each condition
PATIENT_CASES = {
    "lung_cancer": {
        "patient_1": {
            "description": """
            67-year-old male with progressive shortness of breath and persistent cough.
            Recent CT scan shows mass in right upper lobe. Biopsy confirmed non-small cell 
            lung cancer (NSCLC), adenocarcinoma subtype. Stage IIIA. Patient is a former 
            smoker (quit 5 years ago). No prior cancer treatment. Performance status good.
            """,
            "conditions": ["lung cancer", "non-small cell lung cancer"],
            "age": 67,
            "sex": "MALE"
        },
        "patient_2": {
            "description": """
            52-year-old female with metastatic lung cancer, previously treated with 
            chemotherapy. Now experiencing worsening fatigue and back pain. Imaging shows 
            progression with new bone metastases. EGFR mutation positive. Seeking new 
            treatment options.
            """,
            "conditions": ["metastatic lung cancer"],
            "age": 52,
            "sex": "FEMALE"
        }
    },
    "glioblastoma": {
        "patient_1": {
            "description": """
            45-year-old male with recent onset headaches, vision problems, and memory issues.
            MRI revealed large mass in frontal lobe. Surgical biopsy confirmed glioblastoma
            multiforme (GBM), grade IV. Patient completed standard radiation and temozolomide.
            Disease progression detected on follow-up scan. Karnofsky score 80.
            """,
            "conditions": ["glioblastoma", "brain cancer"],
            "age": 45,
            "sex": "MALE"
        },
        "patient_2": {
            "description": """
            58-year-old female with recurrent glioblastoma. Original diagnosis 14 months ago,
            treated with surgery, radiation, and chemotherapy. Recent MRI shows tumor 
            recurrence. Patient experiencing increased seizure frequency and cognitive decline.
            IDH wild-type, MGMT unmethylated. Interested in immunotherapy trials.
            """,
            "conditions": ["recurrent glioblastoma"],
            "age": 58,
            "sex": "FEMALE"
        }
    },
    "type_2_diabetes": {
        "patient_1": {
            "description": """
            62-year-old overweight male with poorly controlled blood sugar levels. Diagnosed
            with Type 2 diabetes 8 years ago. Current HbA1c is 9.2%. Taking metformin and
            glipizide but still experiencing hyperglycemia. Patient reports frequent urination,
            increased thirst, and blurry vision. BMI 34. No diabetic complications yet.
            """,
            "conditions": ["type 2 diabetes", "hyperglycemia"],
            "age": 62,
            "sex": "MALE"
        },
        "patient_2": {
            "description": """
            55-year-old female with Type 2 diabetes and early diabetic kidney disease.
            Current medications include metformin and insulin. Difficulty maintaining stable
            glucose levels despite treatment. Patient is motivated to try new therapies.
            Recent labs show elevated creatinine and microalbuminuria. No cardiovascular disease.
            """,
            "conditions": ["type 2 diabetes", "diabetic nephropathy"],
            "age": 55,
            "sex": "FEMALE"
        }
    }
}


def index_all_conditions(matcher: SemanticMatcher):
    """Index trials for all three conditions."""
    import json
    from pathlib import Path
    
    conditions = {
        "lung_cancer": "lung cancer",
        "glioblastoma": "glioblastoma",
        "type_2_diabetes": "type 2 diabetes"
    }
    
    print("\n" + "="*80)
    print("INDEXING CLINICAL TRIALS")
    print("="*80)
    print("\nLoading trials from cache or fetching from ClinicalTrials.gov...")
    print("This may take a few minutes...\n")
    
    for collection_name, condition in conditions.items():
        print(f"Indexing {condition} trials...")
        try:
            # Try to load from cached data first (300 trials from data pipeline)
            trials_file = Path(f"data/trials/{collection_name}_trials.json")
            
            if trials_file.exists():
                print(f"  Loading from cache: {trials_file}")
                with open(trials_file, 'r') as f:
                    cached_trials = json.load(f)
                
                print(f"  Found {len(cached_trials)} trials in cache")
                matcher.embeddings.index_trials(
                    trials=cached_trials,
                    collection_name=collection_name,
                    overwrite=True
                )
                print(f"  ✓ Indexed {len(cached_trials)} trials from cache for {condition}\n")
            else:
                # No cache, fetch from API
                print(f"  No cache found, fetching from API...")
                matcher.index_trials_for_condition(
                    condition=condition,
                    max_trials=300,  # Match data pipeline size
                    collection_name=collection_name
                )
                
                # Show collection stats
                stats = matcher.get_collection_info(collection_name)
                print(f"  ✓ Indexed {stats['count']} trials for {condition}\n")
        
        except Exception as e:
            logger.error(f"Error indexing {condition}: {e}")
            print(f"  ✗ Error indexing {condition}: {e}\n")
    
    print("Indexing complete!\n")


def demo_condition(matcher: SemanticMatcher, condition_name: str, collection_name: str):
    """Run demo for a specific condition."""
    print("\n" + "="*80)
    print(f"CONDITION: {condition_name.upper()}")
    print("="*80)
    
    patients = PATIENT_CASES[collection_name]
    
    for patient_id, patient_data in patients.items():
        print(f"\n{'-'*80}")
        print(f"Patient Case: {patient_id.replace('_', ' ').title()}")
        print(f"{'-'*80}")
        print(f"\nPatient Profile:")
        print(f"  Age: {patient_data['age']}")
        print(f"  Sex: {patient_data['sex']}")
        print(f"  Conditions: {', '.join(patient_data['conditions'])}")
        print(f"\nClinical Description:")
        print(patient_data['description'].strip())
        
        print(f"\n{'>'*80}")
        print("SEARCHING FOR MATCHING TRIALS...")
        print(f"{'>'*80}\n")
        
        try:
            matches = matcher.match_patient_to_trials(
                patient_description=patient_data['description'],
                collection_name=collection_name,
                patient_conditions=patient_data['conditions'],
                patient_age=patient_data['age'],
                patient_sex=patient_data['sex'],
                top_k=5
            )
            
            if matches:
                print(f"Found {len(matches)} matching trials:\n")
                
                for i, match in enumerate(matches, 1):
                    metadata = match['metadata']
                    print(f"{i}. {metadata['title']}")
                    print(f"   NCT ID: {match['nct_id']}")
                    print(f"   Status: {metadata['status']}")
                    print(f"   Match Score: {match['score']:.3f}")
                    print(f"   Reasoning: {match['reasoning']}")
                    print(f"   Age Range: {metadata.get('min_age', 'N/A')} - {metadata.get('max_age', 'N/A')}")
                    print(f"   Sex: {metadata.get('sex', 'ALL')}")
                    print()
            else:
                print("No matching trials found for this patient.\n")
        
        except Exception as e:
            logger.error(f"Error matching patient: {e}")
            print(f"Error: {e}\n")


def main():
    """Run semantic matcher demos for all conditions."""
    
    print("\n" + "="*80)
    print("SEMANTIC MATCHER AGENT - DEMO")
    print("="*80)
    print("\nThis demo showcases semantic matching of patients to clinical trials.")
    print("\nThe Problem:")
    print("  Patient descriptions use colloquial language ('shortness of breath')")
    print("  Trial criteria use medical terminology ('Dyspnea Grade 3')")
    print("  Traditional keyword search fails due to vocabulary mismatch")
    print("\nThe Solution:")
    print("  Vector embeddings capture semantic meaning")
    print("  Patients match to trials even with different terminology")
    print("  Ontology-based filters enforce hard constraints (age, sex, etc.)")
    print("\nConditions Covered:")
    print("  1. Lung Cancer (2 patient cases)")
    print("  2. Glioblastoma (2 patient cases)")
    print("  3. Type 2 Diabetes (2 patient cases)")
    print("\n")
    
    input("Press Enter to begin...")
    
    # Initialize matcher
    matcher = SemanticMatcher()
    
    # Index trials for all conditions
    index_all_conditions(matcher)
    
    input("Press Enter to start patient matching demos...")
    
    # Run demos for each condition
    conditions = [
        ("Lung Cancer", "lung_cancer"),
        ("Glioblastoma", "glioblastoma"),
        ("Type 2 Diabetes", "type_2_diabetes")
    ]
    
    for i, (condition_name, collection_name) in enumerate(conditions, 1):
        demo_condition(matcher, condition_name, collection_name)
        
        if i < len(conditions):
            print("\n" + "="*80)
            input("Press Enter to continue to next condition...")
    
    # Summary
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nKey Takeaways:")
    print("  ✓ Semantic search successfully matches patients to trials")
    print("  ✓ Works despite vocabulary differences")
    print("  ✓ Ontology filters enforce eligibility constraints")
    print("  ✓ Provides reasoning for each match")
    print("\nVector database persisted at: ./data/indexed_trials/")
    print("You can rerun demos without re-indexing.\n")


if __name__ == "__main__":
    main()

