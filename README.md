# PA5: Advanced Topics in Machine Learning - LLM Alignment & Interpretability

## Overview

This programming assignment explores two major frontiers in modern machine learning:

1. **LLM Alignment** (Tasks 1-2): Training language models to generate helpful, harmless, and honest responses
2. **Mechanistic Interpretability** (Task 3): Understanding what neural networks learn through sparse autoencoders

---

## Task Summaries

### Task 1: Decoding Strategies for Language Models

📁 `task1/`

Implements and compares different text generation strategies:

- **Greedy Search**: Always picks the most likely token
- **Beam Search**: Explores multiple candidate sequences
- **Top-K Sampling**: Samples from top K tokens
- **Top-P (Nucleus) Sampling**: Samples from dynamic vocabulary based on cumulative probability

**Key Findings**: Explores the quality-diversity tradeoff - deterministic methods (greedy/beam) produce coherent but repetitive text, while sampling methods with higher temperature produce more creative but potentially incoherent outputs.

---

### Task 2: LLM Alignment with RLHF Techniques

📁 `task2/`

Implements four alignment algorithms to train models for human preferences:

- **Reward Model (RM)**: Learns to score responses based on preference data
- **DPO (Direct Preference Optimization)**: Directly optimizes policy without explicit RL
- **PPO (Proximal Policy Optimization)**: Classic RL with clipped surrogate objective
- **GRPO (Group Relative Policy Optimization)**: Uses group-based relative rankings

**Key Findings**: Compares alignment effectiveness, reveals potential reward hacking behaviors, and analyzes verbosity bias in reward models.

---

### Task 3: Universal Sparse Autoencoders

📁 `task3/`

Discovers shared interpretable features across different neural network architectures:

- **USAE Architecture**: Joint sparse autoencoder for ResNet50 and MobileNetV3
- **Universality Analysis**: Entropy-based classification of model-specific vs universal features
- **Feature Visualization**: Activation maximization to understand learned concepts

**Key Findings**: Some features are truly universal (shared concepts like textures, shapes) while others are architecture-specific artifacts. Cross-reconstruction training improves feature alignment.

---

## Quick Start

### Prerequisites

```
Python 3.8+
PyTorch >= 2.0.0
transformers >= 4.35.0
CUDA-compatible GPU (recommended)
```

### Running Each Task

```bash
# Task 1: Decoding Strategies
cd task1/
# Run task1.ipynb cells sequentially

# Task 2: LLM Alignment
cd task2/
# Run task2.ipynb cells sequentially

# Task 3: Universal SAE
cd task3/
# Run task3.ipynb cells sequentially
```

---

## Directory Structure

```
pa5/
├── README.md                 # This file
├── task1/
│   ├── task1.ipynb          # Decoding strategies notebook
│   └── README.md            # Detailed task documentation
├── task2/
│   ├── task2.ipynb          # LLM alignment notebook
│   └── README.md            # Detailed task documentation
└── task3/
    ├── task3.ipynb          # Universal SAE notebook
    └── README.md            # Detailed task documentation
```

---

## Models & Datasets

| Task   | Model                 | Dataset               |
| ------ | --------------------- | --------------------- |
| Task 1 | SmolLM2-135M-Instruct | N/A (generation only) |
| Task 2 | SmolLM2-135M-Instruct | Intel/orca_dpo_pairs  |
| Task 3 | ResNet50, MobileNetV3 | CIFAR-100             |

---

## Key Dependencies

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
trl>=0.7.0
datasets>=2.14.0
torchvision>=0.15.0
overcomplete>=0.1.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

---
