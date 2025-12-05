"""
Quick test to see the eligibility parser in action.
Run this to see how the parser extracts conditions from eligibility criteria.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.eligibility_parser import EligibilityParser

def quick_test():
    """Quick demonstration of the eligibility parser."""
    
    print("="*70)
    print("ELIGIBILITY PARSER - QUICK TEST")
    print("="*70)
    
    # Initialize parser (uses NER by default)
    print("\nInitializing EligibilityParser with NER...")
    parser = EligibilityParser(use_ner=True)
    print("✓ Parser ready!\n")
    
    # Sample eligibility criteria
    sample_criteria = """
    INCLUSION CRITERIA:
    1. Age >= 18 years
    2. Histologically confirmed non-small cell lung cancer (NSCLC), adenocarcinoma subtype
    3. Stage IIIA or IIIB disease
    4. ECOG performance status 0-1
    5. Adequate organ function:
       - Hemoglobin >= 9 g/dL
       - Platelet count >= 100,000/mcL
       - Creatinine <= 1.5 mg/dL
    
    EXCLUSION CRITERIA:
    1. Pregnant or breastfeeding women
    2. Active infection requiring treatment
    3. History of autoimmune disease (e.g., rheumatoid arthritis, lupus)
    4. Prior treatment with immunotherapy within 6 months
    5. Known hypersensitivity to study medications
    """
    
    print("Sample Eligibility Criteria:")
    print("-" * 70)
    print(sample_criteria)
    print("-" * 70)
    
    # Parse the criteria
    print("\nParsing eligibility criteria...")
    criteria = parser.parse(
        eligibility_text=sample_criteria,
        min_age_str="18 Years",
        max_age_str="",
        sex_str="ALL"
    )
    
    # Display results
    print("\n" + "="*70)
    print("PARSED RESULTS")
    print("="*70)
    
    print(f"\n📋 Basic Information:")
    print(f"  • Min Age: {criteria.min_age}")
    print(f"  • Max Age: {criteria.max_age or 'None'}")
    print(f"  • Sex: {criteria.sex}")
    
    print(f"\n📝 Criteria Items:")
    print(f"  • Inclusion Criteria: {len(criteria.inclusion_criteria)} items")
    print(f"  • Exclusion Criteria: {len(criteria.exclusion_criteria)} items")
    
    print(f"\n🏥 Medical Conditions (NER-extracted):")
    if criteria.conditions:
        for i, cond in enumerate(criteria.conditions, 1):
            print(f"  {i}. {cond}")
    else:
        print("  (None found)")
    
    print(f"\n❌ Excluded Conditions (NER-extracted):")
    if criteria.excluded_conditions:
        for i, cond in enumerate(criteria.excluded_conditions, 1):
            print(f"  {i}. {cond}")
    else:
        print("  (None found)")
    
    print(f"\n🔬 Lab Value Constraints:")
    if criteria.lab_constraints:
        for i, lab in enumerate(criteria.lab_constraints, 1):
            print(f"  {i}. {lab.parameter}: {lab.operator} {lab.value} {lab.unit or ''}")
    else:
        print("  (None found)")
    
    print(f"\n⚡ Performance Status:")
    print(f"  • {criteria.performance_status or 'None specified'}")
    
    print(f"\n💊 Medication Exclusions:")
    if criteria.medication_exclusions:
        for i, med in enumerate(criteria.medication_exclusions, 1):
            print(f"  {i}. {med}")
    else:
        print("  (None found)")
    
    # Test patient validation
    print("\n" + "="*70)
    print("PATIENT VALIDATION TEST")
    print("="*70)
    
    test_patients = [
        {
            "name": "Eligible Patient",
            "age": 55,
            "sex": "MALE",
            "conditions": ["non-small cell lung cancer"],
            "labs": {"hemoglobin": 12.0, "platelet": 150000, "creatinine": 1.2}
        },
        {
            "name": "Too Young",
            "age": 16,
            "sex": "MALE",
            "conditions": ["non-small cell lung cancer"]
        },
        {
            "name": "Low Hemoglobin",
            "age": 55,
            "sex": "MALE",
            "conditions": ["non-small cell lung cancer"],
            "labs": {"hemoglobin": 7.0}  # Below 9 g/dL requirement
        }
    ]
    
    for patient in test_patients:
        is_eligible, reasons = parser.validate_patient(
            criteria=criteria,
            patient_age=patient["age"],
            patient_sex=patient["sex"],
            patient_conditions=patient.get("conditions"),
            patient_labs=patient.get("labs")
        )
        
        status = "✓ ELIGIBLE" if is_eligible else "✗ INELIGIBLE"
        print(f"\n{patient['name']}: {status}")
        if reasons:
            print(f"  Reasons: {', '.join(reasons)}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\n💡 The parser uses NER (Named Entity Recognition) to extract")
    print("   medical conditions accurately, without regex patterns!")

if __name__ == "__main__":
    quick_test()

