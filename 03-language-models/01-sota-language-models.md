# 01. State-of-the-Art Language Models

## Introduction

Language models (LMs) are at the core of modern NLP, enabling machines to understand and generate human language. State-of-the-art (SOTA) LMs leverage deep learning architectures and massive datasets to achieve remarkable performance on a wide range of tasks.

## 1. What is a Language Model?

A language model estimates the probability of a sequence of words:

```math
P(w_1, w_2, ..., w_N) = \prod_{t=1}^N P(w_t | w_1, ..., w_{t-1})
```

- $`w_t`$: the $`t`$-th word in the sequence

## 2. Evolution of Language Models

- **N-gram Models:** Use fixed-length context. Limited by sparsity and context window.
- **Neural Language Models:** Use neural networks to capture longer dependencies.
- **Recurrent Neural Networks (RNNs):** Model sequences with hidden states.
- **Long Short-Term Memory (LSTM):** Address vanishing gradients in RNNs.
- **Transformers:** Use self-attention to model global dependencies efficiently.

## 3. State-of-the-Art Models

### a. GPT (Generative Pre-trained Transformer)
- **Architecture:** Transformer decoder
- **Training:** Unsupervised, next-token prediction
- **Applications:** Text generation, summarization, dialogue

### b. BERT (Bidirectional Encoder Representations from Transformers)
- **Architecture:** Transformer encoder
- **Training:** Masked language modeling, next sentence prediction
- **Applications:** Classification, QA, NER

### c. T5 (Text-to-Text Transfer Transformer)
- **Architecture:** Encoder-decoder transformer
- **Training:** All tasks cast as text-to-text
- **Applications:** Translation, summarization, QA, more

### d. Other Notable Models
- **XLNet:** Permutation-based training for better context
- **RoBERTa:** Robustly optimized BERT
- **ALBERT:** Parameter-efficient BERT
- **GPT-3/4:** Large-scale autoregressive models

## 4. Mathematical Formulation: Transformer Self-Attention

Given input $`X \in \mathbb{R}^{n \times d}`$:

```math
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
```

Self-attention:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

## 5. Python Example: Using a Pre-trained Language Model

Below is an example using HuggingFace Transformers to generate text with GPT-2:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=30, num_return_sequences=1)
generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated)
```

## 6. Key Takeaways
- SOTA language models use deep architectures and large datasets.
- Transformers are the foundation of most modern LMs.
- Pre-trained models can be fine-tuned for a variety of NLP tasks. 