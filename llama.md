# LLaMA Mathematical Architecture (V2)

This document serves as the single mathematical source of truth for the modern `model_llama` architecture. It replaces static positional logic with dynamic rotations (RoPE), leading to improved performance.

## 1. Embeddings
No absolute position embeddings are used. The initial vector is purely semantic.
$$ x_0 = \text{TokenEmbedding}(idx) $$

## 2. Rotary Position Embedding (RoPE)
Instead of adding position information to the initial token, $Q$ and $K$ vectors are dynamically rotated during the attention calculation based on their relative distance. 

For a given position $m$ and frequency array $\theta$:
$$ q_m = q_m \odot \cos(m\theta) + \text{rotate}(q_m) \odot \sin(m\theta) $$
$$ k_m = k_m \odot \cos(m\theta) + \text{rotate}(k_m) \odot \sin(m\theta) $$

## 3. Self-Attention (with RoPE)
The standard attention mathematical formula, but utilizing the dynamically rotated $Q_{rope}$ and $K_{rope}$ vectors instead of raw projections.
$$ \text{Attention}(Q_{rope}, K_{rope}, V) = \text{softmax}\left(\frac{Q_{rope}K_{rope}^T}{\sqrt{d_k}}\right)V $$

## 4. Multi-Head Attention
$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O $$
$$ \text{where } \text{head}_i = \text{Attention}(Q_{rope}W_i^Q, K_{rope}W_i^K, VW_i^V) $$

## 5. FeedForward Network (SwiGLU)
LLaMA abandons the simple ReLU activation in favor of a Swish-Gated Linear Unit (SwiGLU). This requires calculating a "Gate" and a "Value" in parallel, applying the `SiLU` (Swish) non-linearity to the gate, and multiplying them together.
$$ \text{Swish}(x) = x \cdot \text{Sigmoid}(x) $$
$$ \text{SwiGLU}(x) = (\text{Swish}(x W_1) \odot (x W_3)) W_2 $$

## 6. Normalization (RMSNorm)
LLaMA abandons standard LayerNorm because calculating the mean across every vector wastes GPU cycles. It instead calculates only the Root Mean Square, which stabilizes the network faster and much more efficiently.
$$ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma $$
