# Reliable Question Answering with LLM Fine-Tuning

This project explores fine-tuning a small instruction-following Large Language Model (LLM) to perform **reliable question answering** by learning when to answer and when to abstain.

The main goal is to improve model reliability by making it output **`UNANSWERABLE`** when the provided context does not contain enough information to answer the question, instead of generating unsupported or hallucinated responses.

## Project Overview

Large language models can generate fluent answers even when the required information is missing from the input context. This project investigates a simple but important reliability capability:

> **Answer only when the context supports the answer; otherwise, refuse.**

The model is fine-tuned on the **SQuAD v2** dataset, which contains both answerable and unanswerable questions.

### Example

**Input**

```
Context:
Marie Curie was born in Warsaw, Poland.

Question:
Where was Marie Curie born?
```

**Output**

```
Warsaw, Poland
```

---

**Input**

```
Context:
Marie Curie was a physicist and chemist who conducted research on radioactivity.

Question:
Where was Marie Curie born?
```

**Output**

```
UNANSWERABLE
```

## Approach

* Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`
* Dataset: SQuAD v2
* Framework: PyTorch + Hugging Face Transformers
* Training method:

  * Supervised Fine-Tuning (SFT)
  * Parameter-Efficient Fine-Tuning (LoRA)

The training pipeline includes:

* Custom PyTorch Dataset and DataLoader
* Prompt-based instruction formatting
* Label masking for supervised fine-tuning
* Validation-based checkpoint selection
* Reliability-focused evaluation metrics

## Evaluation

The model is evaluated using:

* Overall Accuracy
* Answerable Question Accuracy
* Unanswerable Question Accuracy
* Exact Match
* Token-level F1 Score

The evaluation focuses not only on generating correct answers, but also on avoiding hallucinations when the context is insufficient.

## Technologies

* Python
* PyTorch
* Hugging Face Transformers
* PEFT (LoRA)
* NumPy

## Future Improvements

Possible extensions include:

* Scaling to larger instruction-tuned models
* Improving calibration of answer confidence
* Training with more challenging adversarial unanswerable examples
* Comparing LoRA fine-tuning with full-parameter fine-tuning

## Motivation

This project was developed to study practical techniques for improving the reliability and trustworthiness of language models, especially in settings where unsupported generation can reduce model usefulness.
