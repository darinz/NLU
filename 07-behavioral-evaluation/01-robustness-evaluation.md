# 01. Robustness Evaluation of NLU Models

## Introduction

Robustness evaluation assesses how well Natural Language Understanding (NLU) models perform under challenging, noisy, or adversarial conditions. Robust models maintain high performance even when inputs are perturbed or unexpected.

## 1. What is Robustness?

- **Definition:** The ability of a model to maintain accuracy and reliability when faced with input variations, noise, or adversarial attacks.
- **Importance:** Ensures models are reliable in real-world, unpredictable environments.

## 2. Types of Robustness Tests

### a. Noise Robustness
- Add random noise (e.g., typos, misspellings) to input data.
- Evaluate model performance on noisy vs. clean data.

### b. Adversarial Robustness
- Apply targeted perturbations designed to fool the model (e.g., synonym substitution, paraphrasing).
- Use adversarial attack algorithms (e.g., TextFooler, HotFlip).

### c. Out-of-Distribution (OOD) Robustness
- Test model on data from a different distribution than training data.
- Example: Training on news articles, testing on tweets.

## 3. Mathematical Formulation: Robustness Metric

Let $`f_\theta`$ be the model, $`x`$ the input, $`\tilde{x}`$ a perturbed input, and $`y`$ the true label.

- **Robustness Score:**

```math
\text{Robustness} = \mathbb{E}_{(x, y)} [ \mathbb{I}(f_\theta(x) = y) - \mathbb{I}(f_\theta(\tilde{x}) = y) ]
```

- $`\mathbb{I}`$: indicator function (1 if true, 0 otherwise)
- Lower difference indicates higher robustness.

## 4. Python Example: Evaluating Noise Robustness
```python
import random

def add_typo(text):
    if len(text) < 2:
        return text
    idx = random.randint(0, len(text) - 2)
    return text[:idx] + text[idx+1] + text[idx] + text[idx+2:]

texts = ["the quick brown fox", "jumps over the lazy dog"]
noisy_texts = [add_typo(t) for t in texts]
print(noisy_texts)

# Evaluate model predictions on noisy_texts vs. texts
```

## 5. Robustness Benchmarks and Datasets
- **TextFlint, Robustness Gym:** Frameworks for systematic robustness evaluation.
- **HANS, ANLI:** Datasets designed to test robustness to specific phenomena.

## 6. Improving Robustness
- Data augmentation (e.g., paraphrasing, back-translation)
- Adversarial training
- Regularization techniques

## Summary
- Robustness evaluation tests NLU models under noise, adversarial, and OOD conditions.
- Metrics compare performance on clean vs. perturbed data.
- Improving robustness is key for real-world deployment of NLU systems. 