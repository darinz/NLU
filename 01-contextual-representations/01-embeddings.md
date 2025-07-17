# 01. Embeddings: Representing Words as Vectors

## Introduction

Word embeddings are dense vector representations of words that capture semantic and syntactic information. Unlike one-hot encoding, embeddings allow similar words to have similar representations, enabling models to generalize better in NLP tasks.

## 1. One-Hot Encoding vs. Embeddings

### One-Hot Encoding
Each word is represented as a vector with all zeros except for a single one at the index corresponding to that word.

```python
import numpy as np

vocab = ['cat', 'dog', 'apple']
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

# One-hot encoding for 'dog'
one_hot = np.zeros(len(vocab))
one_hot[word_to_idx['dog']] = 1
print(one_hot)  # Output: [0. 1. 0.]
```

- **Drawbacks:**
  - High dimensionality for large vocabularies
  - No notion of similarity between words

### Embeddings
Embeddings map words to dense, low-dimensional vectors. These vectors are learned from data and capture relationships between words.

```python
import torch
import torch.nn as nn

embedding_dim = 5
embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim)
word_idx = torch.tensor([word_to_idx['dog']])
word_vec = embedding(word_idx)
print(word_vec)
```

## 2. Learning Embeddings: Skip-Gram and CBOW

### Skip-Gram Model
Given a center word, predict the context words.

- **Objective:** Maximize the probability of context words given the center word.

```math
J = \frac{1}{T} \sum_{t=1}^T \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)
```

Where $`c`$ is the context window size, $`w_t`$ is the center word, and $`w_{t+j}`$ are context words.

### CBOW Model
Given context words, predict the center word.

- **Objective:** Maximize the probability of the center word given the context.

```math
J = \frac{1}{T} \sum_{t=1}^T \log P(w_t | w_{t-c}, ..., w_{t+c})
```

### Softmax and Negative Sampling
The probability $`P(w_O | w_I)`$ is computed using softmax:

```math
P(w_O | w_I) = \frac{\exp({v'_{w_O}}^T v_{w_I})}{\sum_{w=1}^W \exp({v'_w}^T v_{w_I})}
```

- $`v_{w_I}`$: input vector for word $`w_I`$
- $`v'_{w_O}`$: output vector for word $`w_O`$

Negative sampling simplifies training by only updating a small subset of weights per step.

## 3. Properties of Embeddings

- **Semantic Similarity:** Words with similar meanings have similar vectors (e.g., $`\text{king} - \text{man} + \text{woman} \approx \text{queen}`$).
- **Linear Structure:** Embeddings capture analogies and relationships.

## 4. Pre-trained Embeddings

Popular pre-trained embeddings:
- Word2Vec
- GloVe
- FastText

### Example: Using GloVe Embeddings

```python
import numpy as np

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

glove = load_glove_embeddings('glove.6B.50d.txt')
print(glove['cat'])  # Example vector for 'cat'
```

## 5. Visualizing Embeddings

Embeddings can be visualized using dimensionality reduction techniques like t-SNE or PCA.

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

words = ['cat', 'dog', 'apple']
vecs = np.array([glove[word] for word in words])
pca = PCA(n_components=2)
vecs_2d = pca.fit_transform(vecs)

plt.scatter(vecs_2d[:,0], vecs_2d[:,1])
for i, word in enumerate(words):
    plt.annotate(word, (vecs_2d[i,0], vecs_2d[i,1]))
plt.show()
```

## Summary
- Embeddings are dense vector representations of words.
- They capture semantic and syntactic relationships.
- Learned via models like Skip-Gram and CBOW.
- Pre-trained embeddings can be used for various NLP tasks. 