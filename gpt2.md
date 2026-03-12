# GPT-2 Mathematical Architecture (V1)

This document serves as the single mathematical source of truth for the `model_gpt2` architecture. This version utilizes classic Absolute Position Embeddings (APE) and standard Transformer components as defined in the 2017 *Attention Is All You Need* paper.

## 1. Embeddings
Tokens are given fixed positional awareness before entering the Transformer blocks.
$$ x_0 = \text{TokenEmbedding}(idx) + \text{PositionEmbedding}(pos) $$

## 2. Self-Attention
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

## 3. Multi-Head Attention
The outputs of the individual attention heads are concatenated and linearly projected.
$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O $$
$$ \text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

## 4. FeedForward Network
Computation layer passing through a standard non-linear Rectified Linear Unit (ReLU).
$$ \text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2 $$

## 5. Layer Normalization
The stabilizing function used before Attention and FeedForward math.
$$ \text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta $$
