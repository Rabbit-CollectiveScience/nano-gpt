# Nano-GPT: Research Paper Syllabus

By building this repository from scratch, we have practically implemented the history of modern Large Language Models step-by-step. The mathematical concepts and architectures in this codebase are directly derived from the following 7 major AI research papers:

## 1. The Foundations

* **"Attention Is All You Need" (Vaswani et al., 2017)** The Holy Grail paper from Google that invented the `Transformer` architecture. Every time we write `Q @ K.T`, or build the `MultiHeadAttention` class, we are directly transcribing the math from this paper.
* **"Language Models are Unsupervised Multitask Learners" (Radford et al., 2019)** The famous OpenAI **GPT-2** paper. They took Google's Transformer, ripped out the Encoder, and proved that predicting the "next word" recursively creates intelligent behavior. This powers our `train_gpt.py` loop and `model_gpt2` architecture.

## 2. The LLaMA Upgrades (Modernization)

When we duplicated the GPT-2 folder into `model_llama`, we implemented three distinct papers that Meta combined to create **LLaMA (Touvron et al., 2023)**:

* **"RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)** The paper that proved we shouldn't use Absolute Position Embeddings (APE), but instead mathematically *rotate* the Q and K vectors to understand relative distance. This is our `apply_rotary_emb` function.
* **"GLU Variants Improve Transformer" (Shazeer, 2020)** This paper proved that replacing simple ReLU curves with a dual-path mathematical gate (`SwiGLU`) makes the model exponentially smarter. We coded this into `step2a1_feedforward.py`.
* **"Root Mean Square Layer Normalization" (Zhang and Sennrich, 2019)** This paper proved that calculating the "Mean" when normalizing data wastes precious GPU cycles. We coded their `RMSNorm` formula to speed up our LLaMA architecture.

## 3. The Theory & Benchmarking

When building our `benchmarks/` arena and analyzing the results, we relied on these foundational principles:

* **"Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022 - DeepMind)** Better known as the **"Chinchilla Laws"**. DeepMind proved the golden rule: *For every 1 parameter in your model, you must feed it 20 tokens of data.* This explains why our 87M parameter LLaMA model mathematically needed 1.7 Billion tokens to win, and lost to GPT-2 on our tiny 1MB Shakespeare dataset.
* **"The Pile: An 800GB Dataset of Diverse Text" (Gao et al., 2020)** The open-source standard for how the community actually evaluates models using massive, diverse, high-quality datasets instead of toy datasets.
