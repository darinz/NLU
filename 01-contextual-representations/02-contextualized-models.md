# 02. Contextualized Models: ELMo, BERT, and Beyond

## Introduction

Traditional word embeddings assign a single vector to each word, regardless of context. Contextualized models generate word representations that depend on the surrounding text, capturing polysemy and richer semantics. Key models include ELMo and BERT.

## 1. Motivation for Contextualized Representations

- **Polysemy:** Words like "bank" have different meanings in different contexts.
- **Static embeddings** (Word2Vec, GloVe) cannot distinguish these meanings.
- **Contextualized models** produce different vectors for the same word in different sentences.

## 2. ELMo: Embeddings from Language Models

ELMo (Peters et al., 2018) uses deep, bidirectional LSTM language models to generate context-sensitive embeddings.

### Architecture
- **Input:** Sequence of tokens
- **Layers:** Stacked bidirectional LSTMs
- **Output:** For each word, concatenate hidden states from all layers

### Mathematical Formulation

Given a sequence $`(w_1, w_2, ..., w_N)`$:

- Forward LSTM computes $`\overrightarrow{h}_t = \text{LSTM}_f(w_1, ..., w_t)`$
- Backward LSTM computes $`\overleftarrow{h}_t = \text{LSTM}_b(w_N, ..., w_t)`$
- ELMo representation for word $`t`$:

```math
\text{ELMo}_t = \gamma \sum_{j=0}^L s_j h_{t,j}
```

Where:
- $`h_{t,j}`$ is the hidden state at layer $`j`$
- $`s_j`$ are softmax-normalized weights
- $`\gamma`$ is a scaling parameter

### Example: Using Pre-trained ELMo (via AllenNLP)

```python
from allennlp.commands.elmo import ElmoEmbedder

elmo = ElmoEmbedder()
sentence = ["The", "cat", "sat", "on", "the", "mat"]
embeddings = elmo.embed_sentence(sentence)
print(embeddings.shape)  # (num_layers, num_tokens, embedding_dim)
```

## 3. BERT: Bidirectional Encoder Representations from Transformers

BERT (Devlin et al., 2018) uses the Transformer architecture to produce deeply bidirectional, context-dependent representations.

### Transformer Encoder
- **Self-attention:** Each word attends to all others in the sequence.
- **Layers:** Stacked transformer blocks

### Mathematical Formulation

#### Self-Attention
Given input matrix $`X \in \mathbb{R}^{n \times d}`$:

- Compute queries $`Q`$, keys $`K`$, values $`V`$:

```math
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
```

- Attention weights:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

#### BERT Pre-training Objectives
- **Masked Language Modeling (MLM):** Randomly mask tokens and predict them.
- **Next Sentence Prediction (NSP):** Predict if one sentence follows another.

### Example: Using Pre-trained BERT (via HuggingFace Transformers)

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

sentence = "The cat sat on the mat."
inputs = tokenizer(sentence, return_tensors='pt')
outputs = model(**inputs)
# outputs.last_hidden_state: (batch_size, sequence_length, hidden_size)
print(outputs.last_hidden_state.shape)
```

## 4. Comparison: ELMo vs. BERT

| Feature         | ELMo                | BERT                |
|-----------------|---------------------|---------------------|
| Architecture    | BiLSTM              | Transformer         |
| Context         | Bidirectional       | Deeply bidirectional|
| Pre-training    | Language modeling   | MLM + NSP           |
| Tokenization    | Word-level          | Subword (WordPiece) |

## 5. Applications

- Named Entity Recognition (NER)
- Question Answering
- Sentiment Analysis
- Text Classification

## 6. Fine-tuning Contextualized Models

Contextualized models can be fine-tuned on downstream tasks for improved performance.

### Example: Fine-tuning BERT for Classification

```python
from transformers import BertForSequenceClassification, Trainer, TrainingArguments

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
# Prepare dataset and training arguments...
# trainer = Trainer(model=model, ...)
# trainer.train()
```

## Summary
- Contextualized models generate word vectors that depend on context.
- ELMo uses BiLSTMs; BERT uses Transformers and self-attention.
- These models achieve state-of-the-art results on many NLP tasks. 