# 02. Language Model Architectures

## Introduction

Language model architectures have evolved from simple statistical models to complex deep learning frameworks. This guide covers the main architectures used in modern NLP, their mathematical foundations, and practical code examples.

## 1. N-gram Models

N-gram models estimate the probability of a word based on the previous $`n-1`$ words:

```math
P(w_t | w_{t-n+1}, ..., w_{t-1})
```

- **Limitation:** Suffer from data sparsity and limited context window.

## 2. Neural Language Models

### a. Feedforward Neural Networks
- Use a fixed window of previous words, embed them, and pass through dense layers.

### b. Recurrent Neural Networks (RNNs)
- Maintain a hidden state that summarizes all previous words.

#### Mathematical Formulation

```math
h_t = \sigma(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
```

- $`h_t`$: hidden state at time $`t`$
- $`x_t`$: input embedding at time $`t`$
- $`\sigma`$: activation function (e.g., tanh)

#### Python Example: Simple RNN Cell
```python
import numpy as np

def rnn_cell(x_t, h_prev, W_xh, W_hh, b_h):
    return np.tanh(np.dot(W_xh, x_t) + np.dot(W_hh, h_prev) + b_h)
```

### c. Long Short-Term Memory (LSTM)
- Addresses vanishing gradients in RNNs with gating mechanisms.

#### Mathematical Formulation

```math
\begin{align*}
i_t &= \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f) \\
o_t &= \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c) \\
h_t &= o_t \odot \tanh(c_t)
\end{align*}
```

- $`i_t, f_t, o_t`$: input, forget, and output gates
- $`c_t`$: cell state
- $`h_t`$: hidden state

#### Python Example: LSTM Cell (Simplified)
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lstm_cell(x_t, h_prev, c_prev, params):
    W_xi, W_hi, b_i = params['W_xi'], params['W_hi'], params['b_i']
    W_xf, W_hf, b_f = params['W_xf'], params['W_hf'], params['b_f']
    W_xo, W_ho, b_o = params['W_xo'], params['W_ho'], params['b_o']
    W_xc, W_hc, b_c = params['W_xc'], params['W_hc'], params['b_c']
    i_t = sigmoid(np.dot(W_xi, x_t) + np.dot(W_hi, h_prev) + b_i)
    f_t = sigmoid(np.dot(W_xf, x_t) + np.dot(W_hf, h_prev) + b_f)
    o_t = sigmoid(np.dot(W_xo, x_t) + np.dot(W_ho, h_prev) + b_o)
    c_t = f_t * c_prev + i_t * np.tanh(np.dot(W_xc, x_t) + np.dot(W_hc, h_prev) + b_c)
    h_t = o_t * np.tanh(c_t)
    return h_t, c_t
```

## 3. Transformer Architecture

Transformers use self-attention to model dependencies between all words in a sequence, enabling parallel computation and long-range context.

### Self-Attention Mechanism

Given input $`X \in \mathbb{R}^{n \times d}`$:

```math
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
```

Self-attention:

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

### Multi-Head Attention
- Multiple attention heads allow the model to focus on different parts of the sequence.

### Python Example: Self-Attention (Simplified)
```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def self_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    weights = softmax(scores)
    return np.dot(weights, V)
```

## 4. Encoder, Decoder, and Encoder-Decoder Architectures

- **Encoder-only:** (e.g., BERT) Good for understanding tasks (classification, NER).
- **Decoder-only:** (e.g., GPT) Good for generation tasks (text completion).
- **Encoder-Decoder:** (e.g., T5, BART) Good for sequence-to-sequence tasks (translation, summarization).

## 5. Summary
- Language model architectures have evolved from n-grams to RNNs, LSTMs, and Transformers.
- Transformers are the foundation of most modern LMs due to their efficiency and effectiveness.
- Understanding these architectures is key to building and fine-tuning powerful NLP models. 