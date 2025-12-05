"""
Embedding Model Comparison Script

Compares MiniLM vs BioBERT embeddings for semantic patient-trial matching.
Tests both models on the same patient cases and generates a comparison report.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.semantic_matcher import SemanticMatcher
from loguru import logger
import config


# Test patient cases (same as demo_matcher.py)
TEST_CASES = {
    "lung_cancer": [
        {
            "description": "67-year-old male with progressive shortness of breath and persistent cough. Recent CT scan shows mass in right upper lobe. Biopsy confirmed non-small cell lung cancer (NSCLC), adenocarcinoma subtype. Stage IIIA. Patient is a former smoker (quit 5 years ago). No prior cancer treatment. Performance status good.",
            "age": 67,
            "sex": "MALE",
            "conditions": ["lung cancer", "non-small cell lung cancer"]
        },
        {
            "description": "52-year-old female with metastatic lung cancer, previously treated with chemotherapy. Now experiencing worsening fatigue and back pain. Imaging shows progression with new bone metastases. EGFR mutation positive. Seeking new treatment options.",
            "age": 52,
            "sex": "FEMALE",
            "conditions": ["metastatic lung cancer"]
        }
    ],
    "glioblastoma": [
        {
            "description": "45-year-old male with recent onset headaches, vision problems, and memory issues. MRI revealed large mass in frontal lobe. Surgical biopsy confirmed glioblastoma multiforme (GBM), grade IV. Patient completed standard radiation and temozolomide. Disease progression detected on follow-up scan. Karnofsky score 80.",
            "age": 45,
            "sex": "MALE",
            "conditions": ["glioblastoma", "brain cancer"]
        },
        {
            "description": "58-year-old female with recurrent glioblastoma. Original diagnosis 14 months ago, treated with surgery, radiation, and chemotherapy. Recent MRI shows tumor recurrence. Patient experiencing increased seizure frequency and cognitive decline. IDH wild-type, MGMT unmethylated. Interested in immunotherapy trials.",
            "age": 58,
            "sex": "FEMALE",
            "conditions": ["recurrent glioblastoma"]
        }
    ],
    "type_2_diabetes": [
        {
            "description": "62-year-old overweight male with poorly controlled blood sugar levels. Diagnosed with Type 2 diabetes 8 years ago. Current HbA1c is 9.2%. Taking metformin and glipizide but still experiencing hyperglycemia. Patient reports frequent urination, increased thirst, and blurry vision. BMI 34. No diabetic complications yet.",
            "age": 62,
            "sex": "MALE",
            "conditions": ["type 2 diabetes", "hyperglycemia"]
        },
        {
            "description": "55-year-old female with Type 2 diabetes and early diabetic kidney disease. Current medications include metformin and insulin. Difficulty maintaining stable glucose levels despite treatment. Patient is motivated to try new therapies. Recent labs show elevated creatinine and microalbuminuria. No cardiovascular disease.",
            "age": 55,
            "sex": "FEMALE",
            "conditions": ["type 2 diabetes", "diabetic nephropathy"]
        }
    ]
}


def run_evaluation_with_model(model_name: str, model_display_name: str) -> Dict:
    """
    Run evaluation with a specific embedding model.
    
    Args:
        model_name: Model name (e.g., "all-MiniLM-L6-v2" or "dmis-lab/biobert-base-cased-v1.1")
        model_display_name: Display name for reports (e.g., "MiniLM" or "BioBERT")
    
    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"EVALUATING: {model_display_name} ({model_name})")
    logger.info(f"{'='*80}\n")
    
    # Temporarily override config
    original_model = config.EMBEDDING_MODEL
    config.EMBEDDING_MODEL = model_name
    
    try:
        # Initialize matcher with the specified model
        matcher = SemanticMatcher(embedding_model=model_name)
        
        results = {
            "model_name": model_name,
            "model_display_name": model_display_name,
            "conditions": {}
        }
        
        # Test each condition
        for condition, cases in TEST_CASES.items():
            logger.info(f"\nTesting {condition}...")
            
            # Ensure trials are indexed for this condition
            # Use model-specific collection name to avoid dimension conflicts
            collection_name = f"{condition}_{model_display_name.lower()}"
            trials_file = Path(f"data/trials/{condition}_trials.json")
            
            if trials_file.exists():
                with open(trials_file, 'r') as f:
                    trials = json.load(f)
                
                # Always re-index for this model (different models have different embedding dimensions)
                logger.info(f"  Indexing {len(trials)} trials for {condition} with {model_display_name}...")
                matcher.embeddings.index_trials(trials, collection_name, overwrite=True)
            else:
                logger.warning(f"  No cached trials found for {condition}. Skipping.")
                continue
            
            condition_results = {
                "cases": []
            }
            
            # Test each patient case
            for i, case in enumerate(cases, 1):
                logger.info(f"  Patient {i}/{len(cases)}...")
                
                try:
                    matches = matcher.match_patient_to_trials(
                        patient_description=case["description"],
                        collection_name=collection_name,
                        patient_conditions=case["conditions"],
                        patient_age=case["age"],
                        patient_sex=case["sex"],
                        top_k=5
                    )
                    
                    if matches:
                        avg_score = sum(m["score"] for m in matches) / len(matches)
                        top_score = matches[0]["score"]
                        
                        case_result = {
                            "patient_id": f"{condition}_patient_{i}",
                            "num_matches": len(matches),
                            "avg_score": avg_score,
                            "top_score": top_score,
                            "top_trial": {
                                "nct_id": matches[0]["nct_id"],
                                "title": matches[0]["metadata"]["title"][:100],
                                "score": matches[0]["score"]
                            },
                            "all_scores": [m["score"] for m in matches]
                        }
                    else:
                        case_result = {
                            "patient_id": f"{condition}_patient_{i}",
                            "num_matches": 0,
                            "avg_score": 0.0,
                            "top_score": 0.0,
                            "error": "No matches found"
                        }
                    
                    condition_results["cases"].append(case_result)
                    
                except Exception as e:
                    logger.error(f"    Error matching patient: {e}")
                    condition_results["cases"].append({
                        "patient_id": f"{condition}_patient_{i}",
                        "error": str(e)
                    })
            
            results["conditions"][condition] = condition_results
        
        # Restore original config
        config.EMBEDDING_MODEL = original_model
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating {model_display_name}: {e}")
        config.EMBEDDING_MODEL = original_model
        raise


def generate_comparison_report(minilm_results: Dict, biobert_results: Dict) -> str:
    """
    Generate a human-readable comparison report.
    
    Args:
        minilm_results: Results from MiniLM evaluation
        biobert_results: Results from BioBERT evaluation
    
    Returns:
        Formatted comparison report string
    """
    report = []
    report.append("="*80)
    report.append("EMBEDDING MODEL COMPARISON REPORT")
    report.append("="*80)
    report.append("")
    report.append(f"MiniLM Model: {minilm_results['model_name']}")
    report.append(f"BioBERT Model: {biobert_results['model_name']}")
    report.append("")
    report.append("="*80)
    report.append("")
    
    # Overall statistics
    minilm_scores = []
    biobert_scores = []
    
    for condition in ["lung_cancer", "glioblastoma", "type_2_diabetes"]:
        if condition not in minilm_results["conditions"] or condition not in biobert_results["conditions"]:
            continue
        
        report.append(f"\n{condition.upper().replace('_', ' ')}")
        report.append("-"*80)
        
        minilm_cases = minilm_results["conditions"][condition]["cases"]
        biobert_cases = biobert_results["conditions"][condition]["cases"]
        
        for i, (minilm_case, biobert_case) in enumerate(zip(minilm_cases, biobert_cases), 1):
            report.append(f"\n  Patient Case {i}:")
            
            if "error" in minilm_case or "error" in biobert_case:
                report.append(f"    ⚠ Error in one or both models")
                continue
            
            minilm_avg = minilm_case.get("avg_score", 0.0)
            biobert_avg = biobert_case.get("avg_score", 0.0)
            minilm_top = minilm_case.get("top_score", 0.0)
            biobert_top = biobert_case.get("top_score", 0.0)
            
            minilm_scores.append(minilm_avg)
            biobert_scores.append(biobert_avg)
            
            diff_avg = biobert_avg - minilm_avg
            diff_top = biobert_top - minilm_top
            
            report.append(f"    MiniLM - Avg Score: {minilm_avg:.4f}, Top Score: {minilm_top:.4f}")
            report.append(f"    BioBERT - Avg Score: {biobert_avg:.4f}, Top Score: {biobert_top:.4f}")
            report.append(f"    Difference: {diff_avg:+.4f} (avg), {diff_top:+.4f} (top)")
            
            if diff_avg > 0.01:
                report.append(f"    ✓ BioBERT performs better")
            elif diff_avg < -0.01:
                report.append(f"    ✓ MiniLM performs better")
            else:
                report.append(f"    ≈ Similar performance")
            
            # Show top trial titles
            if "top_trial" in minilm_case and "top_trial" in biobert_case:
                report.append(f"    MiniLM Top Trial: {minilm_case['top_trial']['title']}")
                report.append(f"    BioBERT Top Trial: {biobert_case['top_trial']['title']}")
    
    # Overall summary
    report.append("\n" + "="*80)
    report.append("OVERALL SUMMARY")
    report.append("="*80)
    
    if minilm_scores and biobert_scores:
        minilm_overall_avg = sum(minilm_scores) / len(minilm_scores)
        biobert_overall_avg = sum(biobert_scores) / len(biobert_scores)
        
        report.append(f"\nAverage Match Score Across All Cases:")
        report.append(f"  MiniLM:  {minilm_overall_avg:.4f}")
        report.append(f"  BioBERT: {biobert_overall_avg:.4f}")
        report.append(f"  Difference: {biobert_overall_avg - minilm_overall_avg:+.4f}")
        
        if biobert_overall_avg > minilm_overall_avg + 0.01:
            report.append(f"\n✓ RECOMMENDATION: Use BioBERT (better average match scores)")
        elif minilm_overall_avg > biobert_overall_avg + 0.01:
            report.append(f"\n✓ RECOMMENDATION: Use MiniLM (better average match scores)")
        else:
            report.append(f"\n≈ RECOMMENDATION: Both models perform similarly")
    
    report.append("\n" + "="*80)
    
    return "\n".join(report)


def main():
    """Run comparison evaluation."""
    print("\n" + "="*80)
    print("EMBEDDING MODEL COMPARISON EVALUATION")
    print("="*80)
    print("\nThis script will:")
    print("  1. Test MiniLM embeddings on patient cases")
    print("  2. Test BioBERT embeddings on patient cases")
    print("  3. Compare results and generate a report")
    print("\nThis may take 5-10 minutes...\n")
    
    input("Press Enter to start evaluation...")
    
    # Create output directory
    output_dir = Path("evaluation/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Evaluate MiniLM
        minilm_results = run_evaluation_with_model(
            "all-MiniLM-L6-v2",
            "MiniLM"
        )
        
        # Save MiniLM results
        with open(output_dir / "minilm_results.json", 'w') as f:
            json.dump(minilm_results, f, indent=2)
        
        # Evaluate BioBERT
        biobert_results = run_evaluation_with_model(
            "dmis-lab/biobert-base-cased-v1.1",
            "BioBERT"
        )
        
        # Save BioBERT results
        with open(output_dir / "biobert_results.json", 'w') as f:
            json.dump(biobert_results, f, indent=2)
        
        # Generate comparison report
        report = generate_comparison_report(minilm_results, biobert_results)
        
        # Save report
        report_file = output_dir / "embedding_model_comparison_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Print report
        print("\n" + report)
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print("="*80)
        print(f"\nResults saved to: {output_dir}/")
        print(f"  - minilm_results.json")
        print(f"  - biobert_results.json")
        print(f"  - embedding_model_comparison_report.txt")
        print("\n")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()

