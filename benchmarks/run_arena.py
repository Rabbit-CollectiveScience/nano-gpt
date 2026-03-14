import os
import sys
import torch
import time
import json

# Ensure we can import from root
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

import config
from shared.step3_output import OutputHead

@torch.no_grad()
def evaluate_loss(model, head, data_tensor, eval_iters=50):
    """Calculates the average loss on the unseen dataset without training."""
    model.eval()
    head.eval()
    
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        # Grab random chunks of Alice in Wonderland
        ix = torch.randint(len(data_tensor) - config.block_size, (config.batch_size,))
        x = torch.stack([data_tensor[i:i+config.block_size] for i in ix])
        y = torch.stack([data_tensor[i+1:i+config.block_size+1] for i in ix])
        x, y = x.to(config.device), y.to(config.device)

        # Forward Pass
        context_vectors = model(x)
        logits, loss = head(context_vectors, y)
        losses[k] = loss.item()
        
    model.train()
    head.train()
    return losses.mean().item()

def run_benchmark(model_class, model_name, data_tensor):
    """Initializes a pure model from scratch and benchmarks its training arc."""
    print(f"\n[{model_name.upper()}] Launching Arena...")
    
    # 1. Initialize Blank Brain & Output Head
    from shared.step1_tokenizer import encoder
    vocab_size = encoder.vocab_size
    
    model = model_class(vocab_size).to(config.device)
    head = OutputHead(vocab_size).to(config.device)
    
    # Track Parameters
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"[{model_name}] Parameter Count: {total_params / 1e6:.3f} M")

    # Optimizer
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=config.learning_rate)

    results = {
        "parameters": total_params,
        "history": []
    }

    print(f"[{model_name}] Commencing Training on Input.txt...")
    # Load the Shakespeare training dataset
    train_path = os.path.join(parent_dir, 'input.txt')
    from shared.step1_tokenizer import encoder
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = torch.tensor(encoder.encode(f.read()), dtype=torch.long)

    # 2. The Benchmark Training Loop
    start_time = time.time()
    steps_to_train = 500  # A quick test so we don't wait hours
    
    for step in range(steps_to_train):
        # Grab Shakespeare batch
        ix = torch.randint(len(train_data) - config.block_size, (config.batch_size,))
        xb = torch.stack([train_data[i:i+config.block_size] for i in ix])
        yb = torch.stack([train_data[i+1:i+config.block_size+1] for i in ix])
        xb, yb = xb.to(config.device), yb.to(config.device)

        # Forward Pass
        context_vectors = model(xb)
        logits, loss = head(context_vectors, yb)
        
        # Test on Alice In Wonderland every 100 steps
        if step % 100 == 0 or step == steps_to_train - 1:
            val_loss = evaluate_loss(model, head, data_tensor)
            elapsed = time.time() - start_time
            print(f"[{model_name}] Step {step} | Val Loss: {val_loss:.4f} | Time: {elapsed:.1f}s")
            results["history"].append({"step": step, "val_loss": val_loss, "time": elapsed})

        # Backward Pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return results

if __name__ == '__main__':
    # 1. Load the pristine un-memorized test dataset
    tensor_path = os.path.join(current_dir, 'alice.pt')
    if not os.path.exists(tensor_path):
        print("Test dataset missing! Run get_unseen_data.py first.")
        sys.exit(1)
    
    test_data = torch.load(tensor_path)
    print(f"Arena Loaded with {len(test_data)} unseen test tokens.")

    # 2. Run GPT-2 Benchmark
    from model_gpt2.step2_gpt import GPTLanguageModel as GPT2
    gpt2_results = run_benchmark(GPT2, "gpt2", test_data)

    # 3. Run LLaMA Benchmark
    from model_llama.step2_gpt import GPTLanguageModel as LLaMA
    llama_results = run_benchmark(LLaMA, "llama", test_data)

    # 4. Run Mistral Benchmark
    from model_mistral.step2_gpt import GPTLanguageModel as Mistral
    mistral_results = run_benchmark(Mistral, "mistral", test_data)

    # 5. Save Scorecard
    scorecard = {
        "gpt2": gpt2_results,
        "llama": llama_results,
        "mistral": mistral_results
    }
    
    save_path = os.path.join(current_dir, 'results', 'benchmark_results.json')
    with open(save_path, 'w') as f:
        json.dump(scorecard, f, indent=4)
        
    print(f"\nArena Complete! Scorecard saved to {save_path}")
