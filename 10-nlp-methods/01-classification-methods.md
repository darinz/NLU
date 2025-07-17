# 01. Classification Methods and Metrics in NLP

## Introduction

Classification is a core task in NLP, where the goal is to assign predefined labels to text inputs. This guide covers common classification methods, evaluation metrics, and practical code examples.

## 1. Classification Methods

### a. Logistic Regression
- Linear model for binary or multiclass classification.
- Uses features such as bag-of-words, TF-IDF, or embeddings.

### b. Support Vector Machines (SVM)
- Finds a hyperplane that separates classes with maximum margin.
- Effective for high-dimensional, sparse data.

### c. Neural Networks
- Feedforward, CNN, RNN, and Transformer-based models for text classification.
- Can learn complex, non-linear decision boundaries.

### d. Pre-trained Language Models
- Fine-tune models like BERT, RoBERTa, or GPT for classification tasks.

## 2. Mathematical Formulation: Logistic Regression
Given input $`x`$ and weights $`w`$:

```math
P(y=1|x) = \sigma(w^T x + b)
```

- $`\sigma(z) = 1 / (1 + e^{-z})`$ is the sigmoid function.

## 3. Evaluation Metrics

### a. Accuracy
```math
\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total predictions}}
```

### b. Precision, Recall, F1
```math
\text{Precision} = \frac{TP}{TP + FP}
```
```math
\text{Recall} = \frac{TP}{TP + FN}
```
```math
\text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
```
- $`TP`$: true positives, $`FP`$: false positives, $`FN`$: false negatives

### c. ROC-AUC
- Measures the area under the receiver operating characteristic curve.

## 4. Python Example: Text Classification with Logistic Regression
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

texts = ["I love this movie!", "This was a terrible experience.", "Great food and service.", "I will not come back."]
labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
clf = LogisticRegression()
clf.fit(X, labels)
preds = clf.predict(X)

print("Accuracy:", accuracy_score(labels, preds))
print("Precision:", precision_score(labels, preds))
print("Recall:", recall_score(labels, preds))
print("F1:", f1_score(labels, preds))
```

## Summary
- Classification in NLP can be performed with linear models, SVMs, neural networks, and pre-trained transformers.
- Evaluation metrics include accuracy, precision, recall, F1, and ROC-AUC.
- Proper evaluation and model selection are key to effective classification systems. 