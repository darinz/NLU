# 03. Building and Fine-tuning Language Models for NLP Applications

## Introduction

Building and fine-tuning language models enables you to adapt powerful pre-trained models to specific NLP tasks, such as classification, question answering, and summarization. This guide covers the steps, concepts, and code for building and fine-tuning language models.

## 1. Pre-training vs. Fine-tuning

- **Pre-training:** Train a language model on a large, generic corpus (e.g., Wikipedia, Common Crawl) using self-supervised objectives (e.g., next-token prediction, masked language modeling).
- **Fine-tuning:** Adapt the pre-trained model to a specific task using a smaller, labeled dataset.

## 2. Fine-tuning Process Overview

1. **Select a pre-trained model** (e.g., BERT, GPT, T5).
2. **Prepare your dataset** for the target task (classification, QA, etc.).
3. **Modify the model head** (e.g., add a classification layer).
4. **Train the model** on your dataset, typically with a lower learning rate.
5. **Evaluate and iterate** to optimize performance.

## 3. Mathematical Formulation: Fine-tuning for Classification

Given input $`x`$ and label $`y`$:

- The model outputs logits $`z = f_\theta(x)`$.
- The loss is typically cross-entropy:

```math
L = -\sum_{i=1}^C y_i \log \text{softmax}(z)_i
```

where $`C`$ is the number of classes.

## 4. Python Example: Fine-tuning BERT for Text Classification

Below is an example using HuggingFace Transformers to fine-tune BERT for sentiment analysis:

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load pre-trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

dataset = load_dataset('imdb')

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'].shuffle(seed=42).select(range(2000)),
    eval_dataset=tokenized_datasets['test'].shuffle(seed=42).select(range(1000)),
)

trainer.train()
```

## 5. Other Fine-tuning Scenarios

- **Question Answering:** Use models like BERT or T5 with span prediction heads.
- **Sequence-to-Sequence Tasks:** Fine-tune encoder-decoder models (e.g., T5, BART) for translation, summarization.
- **Token Classification:** For NER, POS tagging, etc.

## 6. Tips for Effective Fine-tuning

- Use a small learning rate (e.g., $`2 \times 10^{-5}`$).
- Use early stopping and validation to prevent overfitting.
- Monitor for catastrophic forgetting (loss of pre-trained knowledge).
- Data augmentation and regularization can help with small datasets.

## Summary
- Fine-tuning adapts pre-trained language models to specific NLP tasks.
- The process involves modifying the model head, preparing data, and training with task-specific objectives.
- HuggingFace Transformers provides tools to streamline fine-tuning for many applications. 