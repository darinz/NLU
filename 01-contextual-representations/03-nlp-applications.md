# 03. Applications of Contextual Representations in NLP Tasks

## Introduction

Contextual word representations have revolutionized NLP by enabling models to better understand meaning in context. This section explores how contextual embeddings (from models like ELMo and BERT) are applied to various NLP tasks, improving performance and interpretability.

## 1. Named Entity Recognition (NER)

NER involves identifying and classifying entities (e.g., people, organizations, locations) in text.

### Traditional Approach
- Use static embeddings as input to sequence models (e.g., BiLSTM-CRF).

### With Contextual Embeddings
- Replace static embeddings with contextualized vectors from ELMo/BERT.
- Each token’s representation adapts to its context, improving disambiguation.

#### Example: Using BERT for NER

```python
from transformers import BertTokenizer, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=9)

sentence = "Hugging Face is based in New York City."
inputs = tokenizer(sentence, return_tensors='pt')
outputs = model(**inputs)
# outputs.logits: (batch_size, sequence_length, num_labels)
```

## 2. Question Answering (QA)

QA tasks require models to extract or generate answers from a given context.

### With Contextual Embeddings
- BERT-style models are fine-tuned on QA datasets (e.g., SQuAD).
- The model predicts start and end positions of the answer span.

#### Example: Using BERT for Extractive QA

```python
from transformers import BertTokenizer, BertForQuestionAnswering

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased')

question = "Where is Hugging Face based?"
context = "Hugging Face is based in New York City."
inputs = tokenizer(question, context, return_tensors='pt')
outputs = model(**inputs)
start_scores = outputs.start_logits
end_scores = outputs.end_logits
```

## 3. Sentiment Analysis

Sentiment analysis classifies the sentiment (positive, negative, neutral) of text.

### With Contextual Embeddings
- Fine-tune BERT or similar models for sequence classification.
- Contextual representations help distinguish sentiment in ambiguous cases.

#### Example: Sentiment Classification with BERT

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

sentence = "I love this product!"
inputs = tokenizer(sentence, return_tensors='pt')
outputs = model(**inputs)
logits = outputs.logits
```

## 4. Textual Entailment (Natural Language Inference)

Textual entailment determines if a hypothesis is entailed by, contradicts, or is neutral with respect to a premise.

### With Contextual Embeddings
- BERT and similar models are fine-tuned on NLI datasets (e.g., SNLI, MNLI).
- The model encodes both premise and hypothesis, capturing nuanced relationships.

#### Example: NLI with BERT

```python
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

premise = "A man is playing a guitar."
hypothesis = "A person is making music."
inputs = tokenizer(premise, hypothesis, return_tensors='pt')
outputs = model(**inputs)
logits = outputs.logits
```

## 5. Mathematical Formulation: Sequence Labeling with Contextual Embeddings

Given a sequence $`(w_1, ..., w_N)`$, let $`x_t`$ be the contextual embedding for token $`t`$.

- For classification (e.g., NER):

```math
P(y_t | x_t) = \text{softmax}(Wx_t + b)
```

- For span prediction (e.g., QA):

```math
P_{start}(t) = \text{softmax}(w_{start}^T x_t)
P_{end}(t) = \text{softmax}(w_{end}^T x_t)
```

## 6. Advantages of Contextual Representations

- **Disambiguation:** Different senses of a word get different vectors.
- **Transfer Learning:** Pre-trained models can be fine-tuned for many tasks.
- **State-of-the-art Performance:** Significant improvements across NLP benchmarks.

## Summary
- Contextual embeddings are widely used in NER, QA, sentiment analysis, and NLI.
- They enable models to better understand meaning in context, leading to superior results. 