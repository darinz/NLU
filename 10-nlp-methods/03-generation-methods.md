# 03. Generation Methods and Metrics in NLP

## Introduction

Text generation is a fundamental NLP task where the goal is to produce coherent, relevant, and fluent text. Applications include machine translation, summarization, dialogue, and creative writing. This guide covers generation methods, evaluation metrics, and practical code examples.

## 1. Generation Methods

### a. Rule-Based and Template Methods
- Use predefined rules or templates to generate text.
- Limited flexibility and diversity.

### b. Statistical Language Models
- N-gram models generate text by sampling from learned probabilities.
- Limited context and fluency.

### c. Neural Language Models
- RNNs, LSTMs, and Transformers (e.g., GPT, T5, BART) generate text by predicting the next token given previous context.
- Can be fine-tuned for specific tasks (translation, summarization, etc.).

### d. Decoding Strategies
- **Greedy Decoding:** Select the most probable token at each step.
- **Beam Search:** Keep top $`k`$ sequences at each step for more diverse outputs.
- **Sampling:** Randomly sample from the probability distribution (e.g., top-$`k`$, nucleus sampling).

## 2. Mathematical Formulation: Sequence Generation
Given input $`x`$ and model $`f_\theta`$:

```math
P(y|x) = \prod_{t=1}^T P(y_t | y_{<t}, x)
```

- $`y_t`$: generated token at step $`t`$
- $`x`$: optional input (e.g., source sentence for translation)

## 3. Evaluation Metrics

### a. BLEU (Bilingual Evaluation Understudy)
- Measures $n$-gram overlap between generated and reference texts.

### b. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- Measures overlap of $n$-grams, word sequences, and word pairs.

### c. METEOR
- Considers synonymy and stemming in addition to $n$-gram overlap.

### d. Perplexity
- Measures how well a model predicts a sample.

```math
\text{Perplexity} = \exp\left( -\frac{1}{N} \sum_{i=1}^N \log P(w_i) \right )
```

## 4. Python Example: Text Generation with GPT-2
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

## 5. Error and Diversity Analysis
- Analyze generated outputs for repetition, incoherence, or lack of diversity.
- Use Self-BLEU to measure diversity among generated samples.

## Summary
- Generation methods include rule-based, statistical, and neural models with various decoding strategies.
- Evaluation metrics include BLEU, ROUGE, METEOR, and perplexity.
- Careful analysis of output quality and diversity is essential for effective generation systems. 