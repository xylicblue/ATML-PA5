# Task 3: Universal Sparse Autoencoders for Cross-Model Feature Discovery

## Overview

This task implements **Universal Sparse Autoencoders (USAE)** to discover shared interpretable features across different neural network architectures. The goal is to learn a common latent space where features from different models (ResNet50 and MobileNetV3) can be compared and analyzed for universality.

### Core Concept

Different neural networks trained on the same task may learn similar abstract concepts (e.g., "fur texture", "wheel shape") but represent them differently. USAE learns to align these representations into a shared sparse feature space, enabling:

- **Feature Universality Analysis**: Which features are model-specific vs. universal?
- **Interpretability**: Visualize what each learned feature represents
- **Cross-Model Comparison**: Measure correlation between model representations

### Models Used

- **ResNet50**: Deep residual network (2048-dim features from avgpool)
- **MobileNetV3-Large**: Efficient architecture (1280-dim features from classifier)
- **Dataset**: CIFAR-100 (100 classes, 50K train / 10K test images)

---

## Notebook Structure

### Part 1: Model Preparation

#### Cell 1: Data Loading

- Loads CIFAR-100 dataset with ImageNet normalization
- Applies data augmentation (flip, rotation) for training
- Creates train/test dataloaders

#### Cell 2: Model Initialization

- Loads pretrained ResNet50, VGG16, MobileNetV3 from ImageNet
- Modifies final layers to output 100 classes (CIFAR-100)

#### Cell 3: Test Function

- Evaluates model accuracy on test set
- Reports classification accuracy

#### Cell 4: Training Function

- Fine-tunes models on CIFAR-100
- Uses CrossEntropyLoss + Adam optimizer
- Saves checkpoints after each epoch

#### Cell 5: Fine-Tuning Execution

- Trains selected models (ResNet, VGG, MobileNet)
- ~6-10 epochs per model

### Part 2: Feature Extraction & USAE

#### Cell 6: Hook Setup

- Loads fine-tuned model weights
- Registers forward hooks to capture intermediate features
- ResNet: avgpool output (2048-dim)
- MobileNet: classifier layer 2 output (1280-dim)
- Applies L2 normalization to features

#### Cell 7: USAE Architecture

Defines the Universal Sparse Autoencoder:

```python
class USAE(nn.Module):
    - sae_res: TopKSAE for ResNet features
    - sae_mobile: TopKSAE for MobileNet features

    Forward pass returns:
    - recon_zt_res: ResNet reconstructed from ResNet codes
    - recon_zt_mobile: MobileNet reconstructed from MobileNet codes
    - recon_zc_res: ResNet reconstructed from MobileNet codes (cross)
    - recon_zc_mobile: MobileNet reconstructed from ResNet codes (cross)
```

#### Cell 8: USAE Training

- Joint training with 4 reconstruction losses:
  - Self-reconstruction for both models
  - Cross-reconstruction (universal alignment) with 1.5x weight
- 14 epochs, learning rate 1e-4

### Part 3: Universality Analysis

#### Cell 9: Entropy-Based Universality

- Computes firing patterns for each feature across models
- Calculates normalized entropy:
  - **Entropy ≈ 0**: Model-specific (only fires for one model)
  - **Entropy ≈ 1**: Universal (fires equally for both models)
- Plots histogram of entropy distribution

#### Cell 10: Independent SAE Training (Baseline)

- Trains separate SAEs for each model (no cross-reconstruction)
- Serves as comparison baseline

#### Cell 11: R² Score Comparison

- Computes explained variance (R²) for USAE vs independent SAEs
- Measures reconstruction quality

#### Cell 12: Entropy for Independent SAEs

- Same entropy analysis for baseline SAEs
- Comparison of universality between approaches

### Part 4: Feature Visualization

#### Cell 13: Activation Hook Setup

- Re-registers hooks for gradient-based visualization

#### Cell 14: Feature Maximization Function

- Implements activation maximization via gradient ascent
- Uses random jittering for regularization
- Optimizes input image to maximize specific feature activation

#### Cell 15: Correlation Analysis

- Collects sparse codes from both models on test set
- Filters "dead" features (< 10 activations)
- Computes Pearson correlation between corresponding features
- Identifies top correlated (universal) features

#### Cell 16: Model-Specific Feature Visualization

- Visualizes low-entropy (model-specific) features
- Shows what ResNet vs MobileNet "see" for same feature index

### Part 5: Advanced Analysis

#### Cells 17-19: Additional Visualizations

- High-entropy (universal) feature visualization
- Side-by-side comparison of model views

#### Cells 20-22: Quantitative Metrics

- Feature activation statistics
- Cross-model alignment scores

#### Cells 23-25: Final Comparisons

- USAE vs Independent SAE performance
- Universal vs specific feature ratio analysis

---

## How to Run

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (required, ~6GB VRAM)
- `overcomplete` library for TopKSAE

### Step-by-Step Execution

1. **Run Cells 1-5** (Model Preparation)

   ```
   Downloads CIFAR-100 and fine-tunes models
   Skip if you have pre-trained weights:
   - resnet_fine_tuned.pth
   - mobile_fine_tuned.pth
   ```

2. **Run Cell 6** (Load Models + Hooks)

   ```
   Loads saved weights and sets up feature extraction
   ```

3. **Run Cells 7-8** (USAE Training)

   ```
   Trains universal sparse autoencoder
   Takes ~10-15 minutes on GPU
   ```

4. **Run Cell 9** (Universality Analysis)

   ```
   Generates entropy histogram
   ```

5. **Run Cells 10-12** (Baseline Comparison)

   ```
   Trains independent SAEs and compares
   ```

6. **Run Cells 13-16** (Visualization)

   ```
   Feature maximization (slow, ~2-3 min per feature)
   ```

7. **Run Remaining Cells** (Advanced Analysis)
   ```
   Additional metrics and visualizations
   ```

### Expected Runtime

- **Model Fine-tuning**: ~30-60 min per model
- **USAE Training**: ~15-20 min
- **Feature Visualization**: ~5-10 min per feature

---

## Hyperparameters

| Parameter               | Value  | Description                |
| ----------------------- | ------ | -------------------------- |
| Feature Dim (ResNet)    | 2048   | avgpool output size        |
| Feature Dim (MobileNet) | 1280   | classifier[2] output size  |
| Latent Dim (z)          | 10,000 | Number of SAE concepts     |
| Top-K                   | 32     | Sparsity constraint        |
| Cross-Recon Weight      | 1.5    | Weight for alignment loss  |
| USAE Learning Rate      | 1e-4   | Adam optimizer             |
| USAE Epochs             | 14     | Training iterations        |
| Vis Opt Steps           | 500    | Feature maximization steps |

---

## Expected Outputs

### Saved Models

```
resnet_fine_tuned.pth   # Fine-tuned ResNet50
mobile_fine_tuned.pth   # Fine-tuned MobileNetV3
vgg_fine_tuned.pth      # Fine-tuned VGG16 (optional)
```

### Visualizations

1. **Entropy Histogram**: Distribution of feature universality
2. **Feature Maximization Images**: What each feature "looks for"
3. **R² Comparison**: Reconstruction quality metrics

### Console Output

- Training losses per epoch
- Top correlated features with correlation values
- R² scores for USAE vs independent SAEs

---

## Key Observations

1. **Universal Features**: Features with high entropy (≈1) represent abstract concepts shared across architectures (e.g., textures, shapes)

2. **Model-Specific Features**: Low entropy features capture architecture-specific artifacts or biases

3. **Cross-Reconstruction**: USAE with cross-reconstruction loss learns more aligned representations than independent SAEs

4. **Correlation Analysis**: Top correlated features between models indicate truly universal representations

5. **Visualization Insight**: Same feature index may produce similar (universal) or different (specific) maximization images across models

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
matplotlib>=3.6.0
overcomplete>=0.1.0  # For TopKSAE
```

### Installing overcomplete

```bash
pip install overcomplete
```

---
