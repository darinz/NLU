# 02. Fairness Evaluation of NLU Models

## Introduction

Fairness evaluation assesses whether Natural Language Understanding (NLU) models treat different groups equitably and avoid biased or discriminatory behavior. Ensuring fairness is crucial for ethical and trustworthy AI systems.

## 1. What is Fairness in NLU?

- **Definition:** Fairness means that model predictions are not unduly influenced by sensitive attributes (e.g., gender, race, age).
- **Goal:** Minimize disparate impact and ensure equal treatment across groups.

## 2. Types of Fairness

### a. Demographic Parity
- The prediction rate should be similar across groups.

```math
P(\hat{Y} = 1 | A = 0) \approx P(\hat{Y} = 1 | A = 1)
```
- $`A`$: sensitive attribute (e.g., gender)

### b. Equalized Odds
- The true positive and false positive rates should be similar across groups.

```math
P(\hat{Y} = 1 | Y = y, A = 0) \approx P(\hat{Y} = 1 | Y = y, A = 1), \quad \forall y
```
- $`Y`$: true label

### c. Counterfactual Fairness
- The prediction should not change if a sensitive attribute is altered, holding all else constant.

## 3. Measuring Fairness

- **Statistical Parity Difference:**

```math
\text{SPD} = P(\hat{Y} = 1 | A = 0) - P(\hat{Y} = 1 | A = 1)
```

- **Equal Opportunity Difference:**

```math
\text{EOD} = P(\hat{Y} = 1 | Y = 1, A = 0) - P(\hat{Y} = 1 | Y = 1, A = 1)
```

- **Average Odds Difference:**

```math
\text{AOD} = \frac{1}{2} \left( [P(\hat{Y} = 1 | Y = 1, A = 0) - P(\hat{Y} = 1 | Y = 1, A = 1)] + [P(\hat{Y} = 1 | Y = 0, A = 0) - P(\hat{Y} = 1 | Y = 0, A = 1)] \right)
```

## 4. Python Example: Measuring Statistical Parity Difference
```python
import numpy as np

y_pred = np.array([1, 0, 1, 1, 0, 0, 1, 0])  # Model predictions
A = np.array([0, 1, 0, 1, 0, 1, 0, 1])        # Sensitive attribute (e.g., gender)

spd = np.mean(y_pred[A == 0]) - np.mean(y_pred[A == 1])
print(f"Statistical Parity Difference: {spd:.3f}")
```

## 5. Fairness Benchmarks and Datasets
- **Bias in Bios, WinoBias, Jigsaw Toxicity:** Datasets for evaluating bias and fairness in NLU models.
- **Fairness Indicators:** Tools for systematic fairness evaluation.

## 6. Improving Fairness
- Data balancing and augmentation
- Debiasing algorithms (e.g., adversarial training, reweighting)
- Post-processing (e.g., threshold adjustment)

## Summary
- Fairness evaluation ensures NLU models do not propagate or amplify bias.
- Metrics like statistical parity and equalized odds quantify fairness.
- Improving fairness is essential for ethical and responsible AI deployment. 