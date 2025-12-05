# Biomedical NER Model Comparison

## Current Implementation

**Model Used:** `raynardj/ner-disease-ncbi-bionlp-bc5cdr-pubmed` (Fine-tuned NER Model)

**What We Implemented:**
- ✅ **True NER**: Uses fine-tuned BioBERT model trained on BC5CDR, NCBI-disease, BioNLP, and PubMed datasets
- ✅ **Token Classification Pipeline**: Uses HuggingFace transformers pipeline for entity recognition
- ✅ **Disease Entity Extraction**: Identifies disease/condition entities with proper boundaries
- ✅ **No Regex Patterns**: Pure NER-based extraction (as requested)

**Why This Model:**
1. **Trained on Multiple Datasets**: BC5CDR, NCBI-disease, BioNLP, PubMed - comprehensive training
2. **Disease-Specific**: Specifically fine-tuned for disease/condition recognition
3. **Available on HuggingFace**: Publicly available, easy to load
4. **Good Performance**: Trained on biomedical literature, handles medical terminology well
5. **Fallback Options**: Automatically tries alternative models if primary fails

**Model Details:**
- **Base Model**: BioBERT
- **Training Datasets**: BC5CDR, NCBI-disease, BioNLP, PubMed
- **Task**: Token classification for disease entity recognition
- **Labels**: Disease entities (B-DISEASE, I-DISEASE format)

## Proper Biomedical NER Models

### Option 1: BC5CDR Fine-tuned Models

**Models:**
- `alvaroalon2/biobert_chemical_ner` - BioBERT fine-tuned on BC5CDR
- `alvaroalon2/biobert_disease_ner` - BioBERT fine-tuned for disease NER
- `alvaroalon2/biobert_chemical_disease_ner` - Combined chemical and disease NER

**Dataset:** BC5CDR (BioCreative V CDR) - Chemical and Disease Mention Recognition

**Pros:**
- ✅ Specifically trained for biomedical entity recognition
- ✅ Handles both chemicals and diseases
- ✅ Good performance on medical text
- ✅ Pre-trained and ready to use

**Cons:**
- ❌ May not be optimized for clinical trial eligibility criteria
- ❌ Trained on PubMed abstracts, not trial protocols
- ❌ May miss some condition-specific terminology

**Performance:**
- F1 Score: ~0.85-0.90 on BC5CDR test set
- Good at identifying disease mentions in scientific literature

---

### Option 2: NCBI-Disease Fine-tuned Models

**Models:**
- Models fine-tuned on NCBI-disease corpus
- Often based on BioBERT or SciBERT

**Dataset:** NCBI-disease corpus - Disease mention recognition

**Pros:**
- ✅ Focused specifically on disease/condition recognition
- ✅ Good for identifying disease names and synonyms
- ✅ Handles abbreviations well (e.g., "NSCLC" → "Non-Small Cell Lung Cancer")

**Cons:**
- ❌ Only handles diseases, not other medical entities
- ❌ May not capture treatment-related entities
- ❌ Less common on HuggingFace (may need to train yourself)

**Performance:**
- F1 Score: ~0.80-0.85 on NCBI-disease test set
- Good at disease normalization and linking

---

### Option 3: SciBERT-based Models

**Models:**
- `allenai/scibert_scivocab_uncased` (base)
- Fine-tuned versions for NER tasks

**Dataset:** Scientific papers (computer science, biomedicine)

**Pros:**
- ✅ Trained on scientific literature
- ✅ Good vocabulary for technical terms
- ✅ Handles abbreviations and acronyms well

**Cons:**
- ❌ Not specifically biomedical-focused
- ❌ May need fine-tuning for medical NER
- ❌ Less specialized than BioBERT for medical text

**Performance:**
- Comparable to BioBERT on some tasks
- Better on general scientific text

---

### Option 4: PubMedBERT-based Models

**Models:**
- `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
- Fine-tuned versions for NER

**Dataset:** PubMed abstracts and full-text articles

**Pros:**
- ✅ Trained on extensive biomedical literature
- ✅ Large vocabulary of medical terms
- ✅ Good performance on biomedical tasks
- ✅ Microsoft-maintained, well-documented

**Cons:**
- ❌ Larger model size (more memory)
- ❌ May be slower than BioBERT
- ❌ Fine-tuned NER versions may not be readily available

**Performance:**
- Often outperforms BioBERT on biomedical tasks
- F1 Score: ~0.88-0.92 on various biomedical NER tasks

---

### Option 5: ClinicalBERT / Clinical BioBERT

**Models:**
- Models fine-tuned on clinical notes (MIMIC-III, etc.)
- `emilyalsentzer/Bio_ClinicalBERT`

**Dataset:** Clinical notes, discharge summaries

**Pros:**
- ✅ Trained on clinical text (closer to eligibility criteria style)
- ✅ Better at understanding clinical terminology
- ✅ Handles informal medical language

**Cons:**
- ❌ May not be publicly available
- ❌ Requires access to clinical datasets
- ❌ May need additional fine-tuning

**Performance:**
- Best for clinical text (notes, summaries)
- May not generalize well to trial protocols

---

## Comparison Table

| Model | Type | Training Data | Best For | F1 Score | Size | Availability |
|-------|------|---------------|----------|----------|------|--------------|
| **BioBERT (current)** | Base | PubMed abstracts | General biomedical | N/A (base) | ~110M | ✅ Easy |
| **BC5CDR BioBERT** | Fine-tuned | BC5CDR corpus | Disease/Chemical NER | ~0.87 | ~110M | ✅ Available |
| **NCBI-Disease** | Fine-tuned | NCBI-disease | Disease recognition | ~0.82 | ~110M | ⚠️ Limited |
| **SciBERT** | Base | Scientific papers | Scientific text | N/A (base) | ~110M | ✅ Easy |
| **PubMedBERT** | Base | PubMed abstracts | Biomedical text | N/A (base) | ~110M | ✅ Easy |
| **ClinicalBERT** | Fine-tuned | Clinical notes | Clinical text | ~0.85 | ~110M | ⚠️ Limited |

---

## Recommendation for This Project

### Best Option: BC5CDR Disease NER Model

**Model:** `alvaroalon2/biobert_disease_ner` or similar

**Why:**
1. **Specificity**: Trained specifically for disease/condition recognition
2. **Availability**: Readily available on HuggingFace
3. **Performance**: Good F1 scores on biomedical text
4. **Compatibility**: Based on BioBERT (similar to our embedding model)
5. **Use Case Match**: Designed for extracting medical conditions from text

**Implementation:**
```python
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_name = "alvaroalon2/biobert_disease_ner"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# Extract conditions
results = ner_pipeline(text)
conditions = [entity["word"] for entity in results if entity["entity_group"] == "DISEASE"]
```

### Alternative: Train Custom Model

If BC5CDR models don't work well for eligibility criteria:

1. **Fine-tune BioBERT on eligibility criteria**
   - Annotate eligibility criteria with condition labels
   - Fine-tune BioBERT for token classification
   - Better domain-specific performance

2. **Use Zero-shot NER**
   - Use BioBERT embeddings with a classifier
   - Train a small classifier on labeled eligibility criteria
   - More control over what's extracted

---

## Why Current Implementation Uses Base BioBERT

**Reasons:**
1. **No Dependency on External NER Models**: Works with what we have
2. **Fast Implementation**: Can be done immediately
3. **No Model Download**: Uses existing BioBERT tokenizer
4. **Fallback Strategy**: Works even if NER models fail to load

**Limitations:**
- Not true NER (uses keyword matching)
- Less accurate than fine-tuned models
- May miss conditions not in keyword list
- Doesn't use learned entity boundaries

---

## Next Steps

To implement proper NER:

1. **Try BC5CDR Disease NER Model:**
   ```python
   model_name = "alvaroalon2/biobert_disease_ner"
   ```

2. **If that doesn't work, use BC5CDR Chemical+Disease:**
   ```python
   model_name = "alvaroalon2/biobert_chemical_disease_ner"
   ```

3. **Test on eligibility criteria** to see if it captures conditions well

4. **Fine-tune if needed** on a small set of annotated eligibility criteria

Would you like me to implement one of these proper NER models?

