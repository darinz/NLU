# 01. Principles of In-Context Learning

## Introduction

In-Context Learning (ICL) is a paradigm where models learn and adapt to new tasks by conditioning on examples and instructions provided at inference time, without explicit retraining. This enables flexible adaptation to new tasks and domains.

## 1. What is In-Context Learning?

- **Definition:** The ability of a model to perform new tasks by conditioning on a prompt containing task instructions and/or examples, rather than updating model parameters.
- **Key Feature:** No gradient updates or retraining; adaptation happens through context.

## 2. Prompt-Based Learning

- **Prompt:** A sequence of instructions and/or input-output examples provided to the model.
- **Few-Shot Learning:** Providing a few examples in the prompt to guide the model.
- **Zero-Shot Learning:** Providing only instructions, no examples.

### Example Prompt (Few-Shot)
```
Translate English to French:
cat -> chat
dog -> chien
bird ->
```

## 3. Mathematical Formulation

Given a pre-trained language model $`f_\theta`$, a prompt $`P`$ (containing instructions and/or examples), and a new input $`x`$:

- The model predicts output $`y`$ as:

```math
y = f_\theta(P, x)
```

- The model's parameters $`\theta`$ are fixed; adaptation is via $`P`$.

## 4. Mechanisms Enabling In-Context Learning

- **Large-Scale Pre-training:** Models are trained on diverse data, enabling generalization.
- **Attention Mechanisms:** Transformers can attend to all parts of the prompt, enabling flexible conditioning.
- **Implicit Meta-Learning:** The model learns to use context as a form of task specification during pre-training.

## 5. Python Example: In-Context Learning with GPT-3/4 (OpenAI API)
```python
import openai

prompt = """
Translate English to French:
cat -> chat
dog -> chien
bird ->
"""

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=5
)
print(response.choices[0].text.strip())
```

## 6. Advantages and Challenges

- **Advantages:**
  - No retraining required for new tasks
  - Flexible and adaptable
  - Enables few-shot and zero-shot learning
- **Challenges:**
  - Prompt design is critical
  - Performance can be sensitive to prompt phrasing and order
  - May require large models and compute

## Summary
- In-Context Learning allows models to adapt to new tasks using only context and examples at inference time.
- No parameter updates are needed; adaptation is achieved through prompt engineering.
- This paradigm underpins the flexibility of modern large language models. 