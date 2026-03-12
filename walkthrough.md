# Nano-GPT Architecture Walkthrough

This document visually breaks down how the Generative Pre-trained Transformer predicts the next word, avoiding complex mathematics in favor of logic flow. 

## 1. The High-Level Flow (How a word travels)

When you run `generate.py` and give it a context, here is exactly how that text becomes a new word prediction:

```mermaid
flowchart TD
    %% Base Data Processing
    A([Input Text]) -->|1. Tokenizer| B(shared/step1_tokenizer.py)
    B -->|Integers| C[Active Model Brain]
    
    %% The hierarchical nesting of the models
    subgraph "Active Model Brain (model_gpt2 or model_llama)"
        direction TB
        
        subgraph "Transformer Layers"
            direction TB
            D[Multihead Attention] 
            E[Single Attention Heads]
            F[FeedForward Computation]
            
            D -->|Contains| E
            D -->|Passes Output to| F
        end
        
        C -->|Tokens enter| D
        F -->|Tokens exit| G(Final Layer Norm)
    end
    
    %% The Output
    G -->|Context Vectors| H(shared/step3_output.py: Logits & Softmax)
    H --> I([Next Word Prediction])
```

## 2. Inside the Transformer Block

The true magic happens inside the Transformer Block layer. Each Block is a repeated factory floor. The tokens enter the floor, perform two separate operations, and leave.

```mermaid
flowchart LR
    Start[Tokens Enter Block] --> Add1((+))
    
    %% Path A: Communications
    Add1 --> LN1[LayerNorm]
    LN1 --> SA[Attention & Multihead]
    SA --> Add2((+))
    
    %% Residual Math
    Start -.->|Skip Connection| Add2
    
    %% Path B: Computation
    Add2 --> LN2[LayerNorm]
    LN2 --> FF[FeedForward Processing]
    FF --> Add3((+))
    
    %% Residual Math
    Add2 -.->|Skip Connection| Add3
    
    Add3 --> Finish[Tokens Leave Block]
```
* **Self-Attention** is where the tokens *communicate* with each other to gather context.
* **FeedForward** is where each token takes exactly what it just learned and *computes* it individually before passing it to the next block.
* **The `(+)` signs** represent "Residual Connections." These are crucial for keeping the model stable. We add the raw input back onto the processed output so the original meaning is never lost!

## 3. How Self-Attention Thinks (Pseudo-Code)

If we examine the Attention Head, the core math relies on three concepts: `Query (Q)`, `Key (K)`, and `Value (V)`. Here is what that actually means:

```python
def explain_self_attention(sentence):
    """
    Every token in the sentence plays three roles simultaneously:
    """
    for token in sentence:
        # ROLE 1: The Query (What am I looking for?)
        token.query = "I am an adjective, I need to look for the noun I describe."
        
        # ROLE 2: The Key (Who am I?)
        token.key = "I am a noun, my name is 'Apple'."
        
        # ROLE 3: The Value (If you find me, what do I actually mean?)
        token.value = [0.12, 0.44, -0.9] # The actual mathematical meaning

    for token in sentence:
        # 1. Match my Query against every past token's Key
        scores = get_match_scores(my_query=token.query, all_past_keys=past.keys)
        
        # 2. We don't want to look at future tokens (that's cheating!)
        scores = apply_autoregressive_mask(scores) # Hides the future
        
        # 3. Grab the Values of the tokens that matched me strongly
        my_new_context = aggregate(scores * past.values)
        
    return my_new_context
```

## 4. The Training Loop (How it learns)

When you run `train_gpt.py`, it loops over your dataset repeatedly. 

```mermaid
flowchart TD
    A[Grab 8 chunks of text] --> B[Model predicts next words]
    B --> C{Calculate Loss: How wrong was it?}
    C -->|It was very wrong| D[Backward Pass: Calculate Gradients]
    D --> E[Optimizer changes the 87 Million Weights]
    E --> F{Next Iteration!}
    
    C -->|Loss is near 0| G[Done Training. Save nano_gpt.pt]
```
