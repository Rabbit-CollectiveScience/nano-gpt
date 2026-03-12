# Step 2a: The Transformer Block

The `step2a_block.py` file is the repeating factory floor of the GPT. Vectors enter this block, get refined, and leave. 

There are two major phases inside the block: **Communication** and **Computation**.

## The Block Architecture

```mermaid
flowchart LR
    Start[Input Vectors] --> Add1((+))
    
    %% Path A: Communications
    Add1 --> LN1[LayerNorm]
    LN1 --> SA[step2a2_multihead.py]
    SA --> Add2((+))
    
    %% Residual Math
    Start -.->|Skip Connection| Add2
    
    %% Path B: Computation
    Add2 --> LN2[LayerNorm]
    LN2 --> FF[step2a1_feedforward.py]
    FF --> Add3((+))
    
    %% Residual Math
    Add2 -.->|Skip Connection| Add3
    
    Add3 --> Finish[Output Vectors]
```

### 1. Communication (Multi-Head Attention)
The first thing that happens is the vectors look at each other. The word " బ్యాంకు" (Bank) needs to look around the sentence to figure out if it means a "River Bank" or a "Money Bank". This happens inside `step2a2_multihead.py`.

### 2. Computation (FeedForward)
Once "Bank" realizes it is sitting next to "River", it needs time to process what that means. The FeedForward layer (`step2a1_feedforward.py`) allows every single token to think about its new context in total isolation. 

## The Crucial Helpers

### Layer Normalization (`LayerNorm`)
Notice how the data passes through `LayerNorm` before doing any heavy math. Normalization simply squashes the numbers so they aren't astronomically huge or microscopically small. This keeps the network stable during training.

### Residual Connections (`+`)
Notice the dotted lines skipping past the heavy math? These are **Residual Connections**. 
If a Transformer Block accidentally ruins a vector during computation, the original vector is simply added back on at the end `(+)`. This ensures the model never "forgets" what the word originally was!
