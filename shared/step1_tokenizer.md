# Step 1: The Tokenizer

Before a Deep Learning model can do any math, we must convert human text into integers. This model uses OpenAI's `tiktoken` (specifically the `cl100k_base` vocabulary used in GPT-4).

## The Core Concept: Subword Tokenization

Instead of mapping every single English word to a number (which would create a dictionary of millions of words), or mapping every single character (which forces the model to learn spelling from scratch), GPT uses **Byte-Pair Encoding (BPE)**.

It breaks words down into common "chunks" (subwords).

### Visualizing the Split

If we feed the tokenizer the sentence: `"The alien superhero exploded."`

```mermaid
flowchart LR
    A["The alien superhero exploded."] -->|Chunking| B["The"]
    A -->|Chunking| C[" alien"]
    A -->|Chunking| D[" super"]
    A -->|Chunking| E["hero"]
    A -->|Chunking| F[" expl"]
    A -->|Chunking| G["oded"]
    A -->|Chunking| H["."]
```
*Notice how "superhero" was split into two common chunks, and "exploded" was split into two common chunks. The leading spaces are also mathematically important parts of the chunk!*

### Translating to Integers

Once the text is chunked, the tokenizer looks up each chunk in its massive dictionary of 100,277 known pieces, and returns the matching Integer ID:

```mermaid
flowchart TD
    B["The"] -->|Lookup| B_ID["(ID: 791)"]
    C[" alien"] -->|Lookup| C_ID["(ID: 35140)"]
    D[" super"] -->|Lookup| D_ID["(ID: 3042)"]
    E["hero"] -->|Lookup| E_ID["(ID: 25419)"]
    F[" expl"] -->|Lookup| F_ID["(ID: 5547)"]
    G["oded"] -->|Lookup| G_ID["(ID: 4184)"]
    H["."] -->|Lookup| H_ID["(ID: 13)"]
    
    B_ID --> Final["Final Tensor: [791, 35140, 3042, 25419, 5547, 4184, 13]"]
    C_ID --> Final
    D_ID --> Final
    E_ID --> Final
    F_ID --> Final
    G_ID --> Final
    H_ID --> Final
```

## How `step1_tokenizer.py` works

This file is essentially a translation dictionary. 
* `.encode()` takes a String and returns the List of Integers.
* `.decode()` takes a List of Integers (outputted by the model) and stitches the string chunks back together so humans can read it.

Once the integers are generated, they are passed directly into `step2_gpt.py` to be embedded!
