# 02. Search Algorithms in Information Retrieval

## Introduction

Search algorithms are at the heart of information retrieval systems, enabling efficient and effective retrieval of relevant documents from large corpora. This guide covers classic and modern search algorithms, their mathematical foundations, and practical code examples.

## 1. Inverted Index

An inverted index maps terms to the list of documents in which they appear, enabling fast lookup.

### Python Example: Building a Simple Inverted Index
```python
def build_inverted_index(documents):
    index = {}
    for doc_id, text in enumerate(documents):
        for word in set(text.split()):
            index.setdefault(word, set()).add(doc_id)
    return index

documents = [
    "the quick brown fox",
    "jumped over the lazy dog",
    "the dog barked"
]
index = build_inverted_index(documents)
print(index)
```

## 2. Boolean Retrieval Model

- Uses Boolean logic (AND, OR, NOT) to match documents to queries.
- Returns documents that exactly match the query conditions.

### Example Query
- Query: `dog AND barked`
- Returns documents containing both "dog" and "barked".

## 3. Vector Space Model (VSM)

- Represents documents and queries as vectors in a high-dimensional space.
- Computes similarity (often cosine similarity) between query and document vectors.

### Mathematical Formulation: Cosine Similarity

Given vectors $`\vec{q}`$ (query) and $`\vec{d}`$ (document):

```math
\text{sim}(\vec{q}, \vec{d}) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}
```

### Python Example: Cosine Similarity
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "the quick brown fox",
    "jumped over the lazy dog",
    "the dog barked"
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
query = ["dog barked"]
Q = vectorizer.transform(query)
similarities = cosine_similarity(Q, X)
print(similarities)
```

## 4. Probabilistic Retrieval Models

- Estimate the probability that a document is relevant to a query.
- Classic example: Binary Independence Model (BIM).

### Mathematical Formulation: BIM

Given query $`q`$ and document $`d`$:

```math
P(R=1|d, q) = \frac{P(d|R=1, q)P(R=1|q)}{P(d|q)}
```

- $`R=1`$: document is relevant

## 5. Modern Search Algorithms

- **BM25:** A ranking function based on the probabilistic retrieval framework.
- **Neural Search:** Uses deep learning to encode queries and documents into dense vectors for retrieval (e.g., DPR, ColBERT).

### Python Example: BM25 (Using rank_bm25)
```python
from rank_bm25 import BM25Okapi

documents = [
    "the quick brown fox",
    "jumped over the lazy dog",
    "the dog barked"
]
tokenized_docs = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)
query = "dog barked".split()
scores = bm25.get_scores(query)
print(scores)
```

## Summary
- Search algorithms include Boolean, vector space, probabilistic, and neural models.
- Inverted indexes enable efficient search.
- Cosine similarity and BM25 are widely used for ranking.
- Neural search is advancing the field with dense representations. 