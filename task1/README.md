# Task 1: Decoding Strategies for Language Models

## Overview

This task implements and compares different decoding strategies for text generation using a small language model (SmolLM2-135M-Instruct). The goal is to understand how different decoding methods affect the quality, diversity, and coherence of generated text.

### Decoding Strategies Implemented

1. **Greedy Search** - Always selects the most probable next token
2. **Beam Search** - Maintains multiple candidate sequences and selects the best
3. **Top-K Sampling** - Samples from the top K most probable tokens
4. **Top-P (Nucleus) Sampling** - Samples from tokens whose cumulative probability exceeds P

### Models Used

- **Generation Model**: `HuggingFaceTB/SmolLM2-135M-Instruct` (135M parameter instruction-tuned model)
- **Reward Model**: `OpenAssistant/reward-model-deberta-v3-large-v2` (DeBERTa-based reward model for quality scoring)

---

## Notebook Structure

### Cell 1: Model Loading (PART 1)

Loads the required models and tokenizers:

- SmolLM2-135M-Instruct as the generative model
- OpenAssistant reward model for scoring generated outputs
- Sets up CUDA device if available

### Cell 2: Decoding Strategies

Implements four decoding algorithms:

**Greedy Search**

- Simple argmax selection at each step
- Deterministic output (same input → same output)
- Fast but can produce repetitive text

**Beam Search**

- Maintains `beam_width` candidate sequences
- Uses log-probability scoring
- More exploration than greedy, still deterministic

**Top-K Sampling**

- Filters to top K tokens by probability
- Samples from filtered distribution
- Temperature controls randomness

**Top-P (Nucleus) Sampling**

- Dynamically selects tokens until cumulative probability ≥ P
- Adapts vocabulary size based on distribution shape
- Temperature controls randomness

### Cell 3: Metric Functions

Defines evaluation metrics:

**Distinct-N**

- Measures n-gram diversity in generated text
- Higher = more diverse vocabulary usage
- Calculated as: unique n-grams / total n-grams

**Reward Score**

- Uses OpenAssistant reward model to score quality
- Higher = more helpful/aligned response
- Input format: "Question: {prompt}\nAnswer: {generated}"

### Cell 4: Setup and Evaluation

Runs comprehensive experiments:

- Tests 5 diverse prompts
- Evaluates each strategy at multiple temperatures (0.2, 0.5, 0.8, 1.0, 1.2)
- Collects reward scores, diversity metrics, and response lengths
- Generates visualization plots
- Prints summary statistics with validation warnings

---

## How to Run

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- ~4GB GPU memory for both models

### Step-by-Step Execution

1. **Run Cell 1** (Model Loading)

   ```
   Loads SmolLM2 and reward model
   Takes ~1-2 minutes to download on first run
   ```

2. **Run Cell 2** (Decoding Strategies)

   ```
   Defines all decoding functions
   No output expected
   ```

3. **Run Cell 3** (Metric Functions)

   ```
   Defines evaluation metrics
   No output expected
   ```

4. **Run Cell 4** (Evaluation)
   ```
   Runs all experiments and generates plots
   Takes ~5-10 minutes depending on hardware
   ```

### Expected Runtime

- **GPU (T4/V100)**: ~5-10 minutes
- **CPU**: ~30-45 minutes

---

## Test Prompts

The models are evaluated on five diverse prompts:

1. "Explain gravity to a 5-year-old."
2. "Write a short poem about a robot."
3. "List three healthy breakfast ideas."
4. "How do I change a tire on a car?"
5. "Why is the sky blue?"

---

## Hyperparameters

| Parameter      | Greedy | Beam | Top-K   | Top-P   |
| -------------- | ------ | ---- | ------- | ------- |
| Max New Tokens | 50     | 50   | 50      | 50      |
| Beam Width     | -      | 3    | -       | -       |
| K              | -      | -    | 50      | -       |
| P              | -      | -    | -       | 0.9     |
| Temperatures   | -      | -    | 0.2-1.2 | 0.2-1.2 |

---

## Expected Outputs

### Console Output

- Progress bar for each prompt
- Summary statistics table grouped by strategy and temperature
- Validation warning if greedy outputs are too short

### Visualizations

1. **Temperature vs Diversity** - Line plot showing how temperature affects Distinct-2 scores
2. **Temperature vs Reward** - Line plot showing quality-diversity tradeoff

### Summary Table Columns

- `Strategy`: Decoding method used
- `Temp`: Temperature value (0.0 for deterministic methods)
- `Reward`: Mean reward score from quality model
- `Distinct-2`: Bigram diversity metric
- `Length`: Average response length in words

---

## Key Observations

1. **Greedy/Beam**: High quality but low diversity, deterministic
2. **Low Temperature (0.2)**: Outputs similar to greedy, more focused
3. **High Temperature (1.0+)**: More creative but potentially incoherent
4. **Top-P**: Adapts better to different distribution shapes than Top-K
5. **Quality-Diversity Tradeoff**: Higher diversity often comes at cost of coherence

---

## Dependencies

```
torch>=2.0.0
transformers>=4.35.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
tqdm>=4.65.0
```

---
