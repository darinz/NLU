# 01. Best Practices for Designing NLP Experiments

## Introduction

Careful experimental design is essential for reliable and reproducible NLP research. This guide covers best practices for formulating hypotheses, selecting datasets, defining metrics, and planning experiments.

## 1. Formulating Research Questions and Hypotheses
- Clearly state the research question or hypothesis.
- Define the scope and objectives of the experiment.

### Example
- **Research Question:** Does data augmentation improve sentiment classification accuracy?
- **Hypothesis:** Models trained with augmented data achieve higher accuracy than those trained on original data only.

## 2. Dataset Selection and Preparation
- Choose datasets that are appropriate, diverse, and representative of the target task.
- Split data into training, validation, and test sets.
- Document dataset sources and preprocessing steps.

### Python Example: Data Splitting
```python
from sklearn.model_selection import train_test_split
X = ["sample1", "sample2", "sample3", "sample4"]
y = [0, 1, 0, 1]
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("Train:", X_train)
print("Val:", X_val)
print("Test:", X_test)
```

## 3. Experimental Controls and Baselines
- Include strong baselines for comparison (e.g., majority class, simple models).
- Control for confounding variables (e.g., random seeds, data splits).

## 4. Metric Selection and Statistical Significance
- Choose appropriate evaluation metrics (accuracy, F1, BLEU, etc.).
- Use statistical tests to assess significance of results.

### Mathematical Formulation: Paired t-test
Given paired results $`x_i, y_i`$ for $`n`$ samples:
```math
t = \frac{\bar{d}}{s_d / \sqrt{n}}
```
- $`\bar{d}`$: mean of differences $`d_i = x_i - y_i`$
- $`s_d`$: standard deviation of differences

## 5. Reproducibility Considerations
- Set random seeds for all libraries and frameworks.
- Log all hyperparameters, code versions, and environment details.
- Share code and data when possible.

### Python Example: Setting Random Seeds
```python
import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
```

## Summary
- Good experimental design starts with clear questions, strong baselines, and careful controls.
- Proper data handling, metric selection, and reproducibility practices are essential for reliable NLP research. 