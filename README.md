# Vision Transformer (ViT)

# Architecture

![vit](https://viso.ai/wp-content/uploads/2021/09/vision-transformer-vit.png)

## Overview

This repository contains an implementation of the Vision Transformer (ViT) architecture for image classification tasks. ViT represents a departure from traditional convolutional neural networks (CNNs) by utilizing a transformer architecture originally designed for natural language processing.

## Vision Transformer Architecture

The Vision Transformer architecture consists of the following key components:

### 1. Input Embedding

The input image is divided into fixed-size non-overlapping patches. Each patch is linearly embedded to obtain flattened feature vectors.

### 2. Positional Encoding

Since transformers lack inherent spatial information, positional encodings are added to the input embeddings to preserve the spatial relationships between patches.

### 3. Transformer Encoder

The transformer encoder consists of multiple layers, each comprising a multi-head self-attention mechanism and a feedforward neural network. This enables the model to capture both local and global dependencies in the image.

### 4. Classification Head

The final output of the transformer encoder is fed into a classification head (typically a simple MLP) to produce class probabilities.

# Usage

```python
from vit import ViT

patch = ViT()
x = torch.randn((1, 3, 224, 224))
out = patch(x)
```