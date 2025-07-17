# 03. Analysis of Model Outputs

## Introduction

Analyzing the outputs of Natural Language Understanding (NLU) models is crucial for understanding their strengths, weaknesses, and real-world behavior. Output analysis helps diagnose errors, assess quality, and guide model improvements.

## 1. Output Quality Metrics

### a. Accuracy, Precision, Recall, F1
- Standard metrics for classification tasks.

### b. BLEU, ROUGE, METEOR
- Metrics for evaluating text generation and summarization.

### c. Perplexity
- Measures how well a language model predicts a sample.

```math
\text{Perplexity} = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P(w_i) \right )
```

## 2. Error and Confusion Analysis
- Examine confusion matrices to identify common misclassifications.
- Analyze error types (e.g., false positives, false negatives).

### Python Example: Confusion Matrix
```python
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

y_true = [1, 0, 1, 1, 0, 0, 1]
y_pred = [1, 0, 0, 1, 0, 1, 1]
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

## 3. Qualitative Output Analysis
- Manually inspect model outputs for correctness, fluency, and relevance.
- Identify patterns in errors or unexpected behaviors.

### Example: Manual Review Table
| Input | Model Output | Expected Output | Notes |
|-------|--------------|----------------|-------|
| "The cat sat on the mat." | "A cat is on a mat." | "A cat is on a mat." | Correct |
| "He didn't like the food." | "He liked the food." | "He did not like the food." | Negation error |

## 4. Calibration and Confidence Analysis
- Assess whether model confidence scores reflect true likelihood of correctness.
- Use reliability diagrams and calibration curves.

### Mathematical Formulation: Expected Calibration Error (ECE)
```math
\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} | \text{acc}(B_m) - \text{conf}(B_m) |
```
- $`B_m`$: set of predictions in bin $`m`$
- $`\text{acc}(B_m)`$: accuracy in bin $`m`$
- $`\text{conf}(B_m)`$: average confidence in bin $`m`$

## 5. Output Diversity and Bias Analysis
- Measure diversity in generated outputs (e.g., Self-BLEU).
- Analyze outputs for social bias, stereotypes, or offensive content.

## 6. Python Example: Output Diversity with Self-BLEU
```python
from nltk.translate.bleu_score import sentence_bleu
outputs = [
    "The cat sat on the mat.",
    "A cat is sitting on a mat.",
    "There is a cat on the mat."
]
self_bleu = [sentence_bleu([outputs[:i] + outputs[i+1:]], outputs[i]) for i in range(len(outputs))]
print(f"Self-BLEU scores: {self_bleu}")
```

## Summary
- Output analysis includes quantitative metrics, error/confusion analysis, qualitative review, calibration, and diversity/bias checks.
- These methods provide a comprehensive understanding of NLU model behavior and guide further improvements. 