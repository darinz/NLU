# 02. Diagnostic Techniques for NLU Models

## Introduction

Diagnostic techniques help identify, understand, and address errors or weaknesses in Natural Language Understanding (NLU) models. These methods are essential for improving model reliability, robustness, and performance.

## 1. Error Analysis
- Systematically examine incorrect predictions to find common failure modes.
- Categorize errors (e.g., lexical, syntactic, semantic, pragmatic).

### Python Example: Error Analysis by Category
```python
y_true = [1, 0, 1, 1, 0]
y_pred = [1, 1, 0, 1, 0]
errors = [(i, t, p) for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
print("Error indices and values:", errors)
```

## 2. Probing Tasks
- Train simple classifiers (probes) on model representations to test for encoded linguistic properties (e.g., part-of-speech, syntax).

### Mathematical Formulation: Probing Classifier
Given model representation $`h(x)`$ and probe $`g_\phi`$:

```math
\hat{y} = g_\phi(h(x))
```

- $`h(x)`$: hidden representation from NLU model
- $`g_\phi`$: probe classifier

## 3. Counterfactual and Contrastive Testing
- Modify input minimally and observe changes in output.
- Test for consistency and sensitivity to specific features.

### Example: Counterfactual Test
- Original: "The doctor said he will help."
- Counterfactual: "The doctor said she will help."

## 4. Adversarial Testing
- Generate inputs designed to expose model weaknesses (e.g., synonym swaps, paraphrasing).
- Use adversarial attack frameworks (e.g., TextAttack, TextFooler).

## 5. Behavioral Testing Frameworks
- **CheckList:** Create test cases for invariance, directional expectation, and minimum functionality.
- **NLU Diagnostics:** Use curated test suites for linguistic phenomena (e.g., negation, coreference).

## 6. Python Example: Using CheckList for Behavioral Testing
```python
from checklist.test_suite import TestSuite
from checklist.perturb import Perturb

suite = TestSuite()
sentences = ["The movie was great!", "I did not like the food."]
perturbed = Perturb.add_typos(sentences)
print("Perturbed sentences:", perturbed)
# Add more tests and evaluate model predictions
```

## Summary
- Diagnostic techniques include error analysis, probing, counterfactual/contrastive testing, adversarial testing, and behavioral test suites.
- These methods help uncover model weaknesses and guide improvements in NLU systems. 