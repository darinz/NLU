# 01. Interpretability Methods for NLU Models

## Introduction

Interpretability methods help us understand how Natural Language Understanding (NLU) models make decisions. These techniques provide insights into model reasoning, feature importance, and potential biases, making models more transparent and trustworthy.

## 1. Why Interpretability Matters
- **Trust:** Users and stakeholders need to trust model predictions.
- **Debugging:** Helps identify errors and unexpected behaviors.
- **Fairness:** Reveals potential biases in model decisions.

## 2. Feature Importance Methods

### a. Attention Visualization
- Visualize attention weights in transformer models to see which tokens the model focuses on.

### b. Saliency Maps
- Compute gradients of the output with respect to input tokens to measure their influence.

### c. LIME (Local Interpretable Model-agnostic Explanations)
- Perturb input and observe changes in output to estimate feature importance.

## 3. Mathematical Formulation: Saliency Maps

Given model $`f_\theta`$, input $`x`$, and output $`y`$:

```math
\text{Saliency}(x_i) = \left| \frac{\partial f_\theta(x)}{\partial x_i} \right|
```

- $`x_i`$: $i$-th input token or feature

## 4. Python Example: Attention Visualization with Transformers
```python
from transformers import BertTokenizer, BertModel
import torch
import matplotlib.pyplot as plt

# Load model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentence = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(sentence, return_tensors='pt')
outputs = model(**inputs)
attentions = outputs.attentions  # Tuple of (num_layers, batch, num_heads, seq_len, seq_len)

# Visualize attention from the last layer, first head
import seaborn as sns
import numpy as np
attn = attentions[-1][0, 0].detach().numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, cmap='viridis')
plt.title('Attention Heatmap (Last Layer, Head 0)')
plt.show()
```

## 5. Other Interpretability Techniques
- **Integrated Gradients:** Attribute prediction to input features by integrating gradients along a path.
- **SHAP (SHapley Additive exPlanations):** Uses game theory to assign importance values to each feature.
- **Probing Classifiers:** Train simple classifiers on model representations to test for encoded information.

## Summary
- Interpretability methods reveal how NLU models make decisions.
- Techniques include attention visualization, saliency maps, LIME, SHAP, and probing.
- These methods improve trust, debugging, and fairness in NLU systems. 