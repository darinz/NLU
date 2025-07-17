# 02. Roles of Prompters and Promptees

## Introduction

Prompt engineering involves two key roles: the prompter (the entity designing the prompt) and the promptee (the model responding to the prompt). Understanding these roles is essential for effective interaction and control of language models.

## 1. The Prompter

- **Definition:** The prompter is the user, developer, or system that crafts prompts to elicit desired behaviors from the model.
- **Responsibilities:**
  - Specify the task or question clearly
  - Provide context, instructions, or examples
  - Refine prompts based on model responses

### Example: Prompter Crafting a Prompt
```
You are a helpful assistant. Summarize the following text:
The cat sat on the mat and purred contentedly.
```

## 2. The Promptee

- **Definition:** The promptee is the language model (e.g., GPT-4) that interprets and responds to the prompt.
- **Responsibilities:**
  - Parse the prompt and extract task requirements
  - Generate a response conditioned on the prompt
  - Adapt output style and content based on prompt cues

### Example: Promptee Response
```
A cat sat happily on a mat.
```

## 3. Interaction Dynamics

- **Iterative Refinement:** The prompter may adjust the prompt based on the promptee's output to achieve better results.
- **Role Assignment:** Prompters can assign roles to the promptee (e.g., "You are a math tutor...") to influence behavior.
- **Feedback Loop:** Prompters learn from model responses and improve prompt design over time.

## 4. Mathematical Perspective: Prompt-Response Mapping

Given a prompt $`P`$ from the prompter and a model $`f_\theta`$ (promptee):

```math
\text{Response} = f_\theta(P)
```

- The prompter controls $`P`$; the promptee generates the response.

## 5. Python Example: Prompter and Promptee Interaction
```python
import openai

# Prompter crafts the prompt
prompt = "You are a math tutor. Explain why the square root of 4 is 2."

# Promptee (model) generates the response
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt,
    max_tokens=50
)
print(response.choices[0].text.strip())
```

## 6. Best Practices for Prompters
- Be explicit and clear in instructions
- Use examples to guide the model
- Experiment with prompt phrasing and structure
- Assign roles or context as needed

## Summary
- The prompter designs prompts to guide the model (promptee).
- Effective prompt engineering is an interactive, iterative process.
- Understanding both roles leads to better model control and outcomes. 