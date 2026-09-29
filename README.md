# FInetuning-unsloth-vs-huggingface
# LLM Fine-Tuning Lab

A hands-on learning repository exploring **LLM fine-tuning with LoRA and SFT**, implemented using both **Unsloth** and the standard **Hugging Face ecosystem**.

The goal of this repository is not simply to run a fine-tuning script.

The goal is to understand **what is actually happening at each stage of the pipeline**.

---

## Why this repository?

LLM fine-tuning tutorials often hide a lot of complexity behind high-level APIs.

For example, a single function can configure:

* model loading
* quantization
* PEFT
* LoRA
* dataset processing
* training
* memory optimization

That's convenient, but it can make it difficult to understand what is happening underneath.

This repository takes a different approach.

I first implemented the workflow using **Unsloth**, then reproduced the same conceptual workflow using the **standard Hugging Face stack**.

The comparison makes it easier to understand both the optimized workflow and the underlying components.

---

## What I learned

### 1. LLM model loading

Understanding how pretrained causal language models are loaded and how configuration such as:

* model architecture
* sequence length
* dtype
* quantization

affects training and inference.

---

### 2. 4-bit Quantization

Explored why quantization is useful when fine-tuning large language models on limited GPU memory.

This also helped clarify the difference between:

* model weights
* quantized weights
* training representation
* inference/deployment formats
* GGUF

---

### 3. LoRA / PEFT

Learned how LoRA avoids updating the entire pretrained model.

Instead of directly modifying:

```text
W
```

LoRA learns a low-rank update:

```text
W' = W + BA
```

where `A` and `B` are much smaller matrices.

The notebooks explore:

* LoRA rank
* LoRA alpha
* dropout
* target modules
* trainable vs frozen parameters
* PEFT

---

### 4. Attention and MLP target modules

The LoRA configuration targets modules such as:

```text
q_proj
k_proj
v_proj
o_proj

gate_proj
up_proj
down_proj
```

Understanding these modules helped connect LoRA configuration to the actual Transformer architecture rather than treating `target_modules` as arbitrary configuration.

---

### 5. Chat Templates

Learned how structured conversations are converted into the format expected by an instruction-tuned model.

The conceptual pipeline is:

```text
messages
    ↓
chat template
    ↓
formatted conversation
    ↓
tokenizer
    ↓
input IDs
```

This also demonstrated why the same chat template needs to be handled correctly during inference.

---

### 6. Dataset Preparation

The notebooks walk through:

```text
Raw conversation
       ↓
Standardized conversation
       ↓
Chat template
       ↓
Training text
       ↓
Tokenization
```

This clarified the difference between:

* dataset schema
* conversation structure
* chat template
* tokenization

---

### 7. Supervised Fine-Tuning

Explored the SFT training process:

```text
Input tokens
     ↓
Transformer
     ↓
Logits
     ↓
Cross-entropy loss
     ↓
Backpropagation
     ↓
Optimizer
     ↓
Parameter update
```

This was particularly useful for connecting LLM training to the standard deep-learning workflow used in CNNs.

The major difference is that a causal language model predicts the next token rather than a class label.

---

### 8. Response-Only Loss

For instruction tuning, the user message can remain as context while the loss is calculated only over the assistant's response.

Conceptually:

```text
User tokens       → -100 → ignored
Assistant tokens  → target IDs → included in loss
```

This helped clarify how conversational SFT teaches the model to produce assistant
