# 02. Extraction Methods and Metrics in NLP

## Introduction

Extraction tasks in NLP involve identifying and extracting structured information from unstructured text. Common extraction tasks include named entity recognition (NER), relation extraction, and information extraction. This guide covers extraction methods, evaluation metrics, and practical code examples.

## 1. Extraction Methods

### a. Rule-Based Methods
- Use regular expressions or pattern matching to extract entities or relations.
- Simple but limited in handling linguistic variability.

### b. Sequence Labeling Models
- Assign a label to each token (e.g., BIO tagging for NER).
- Models: CRF, BiLSTM-CRF, Transformer-based models (BERT, RoBERTa).

### c. Span-Based and End-to-End Models
- Predict start and end positions of entities or relations in text.
- Used in modern QA and extraction systems.

## 2. Mathematical Formulation: Sequence Labeling
Given input sequence $`x = (x_1, ..., x_n)`$ and label sequence $`y = (y_1, ..., y_n)`$:

```math
P(y|x) = \prod_{i=1}^n P(y_i | x, y_{<i})
```

- $`y_i`$: label for token $`x_i`$

## 3. Evaluation Metrics

### a. Precision, Recall, F1 for Extraction
```math
\text{Precision} = \frac{\text{Correctly extracted}}{\text{Extracted}}
```
```math
\text{Recall} = \frac{\text{Correctly extracted}}{\text{Relevant}}
```
```math
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```

### b. Exact Match and Partial Match
- **Exact Match:** Entity boundaries and types must match exactly.
- **Partial Match:** Overlapping or partially correct extractions are counted.

## 4. Python Example: Named Entity Recognition with spaCy
```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying U.K. startup for $1 billion.")
for ent in doc.ents:
    print(ent.text, ent.label_)
```

## 5. Error Analysis in Extraction
- Analyze false positives (spurious extractions) and false negatives (missed entities).
- Use confusion matrices and manual review for deeper insights.

## Summary
- Extraction methods include rule-based, sequence labeling, and span-based models.
- Evaluation metrics focus on precision, recall, F1, and exact/partial match.
- Careful error analysis is key to improving extraction systems. 