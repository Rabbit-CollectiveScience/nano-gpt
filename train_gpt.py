import torch
import config
from data.dataset import get_batch
from shared.step1_tokenizer import encoder
from shared.step3_output import OutputHead

if config.model_version == 'gpt2':
    from model_gpt2.step2_gpt import GPTLanguageModel
elif config.model_version == 'llama':
    from model_llama.step2_gpt import GPTLanguageModel
else:
    raise ValueError(f"Unknown model_version: {config.model_version}")

# Instantiating the components
model = GPTLanguageModel(encoder.vocab_size)
head = OutputHead(encoder.vocab_size)

m = model.to(config.device)
h = head.to(config.device)

# print the number of parameters in the model
total_params = sum(p.numel() for p in m.parameters()) + sum(p.numel() for p in h.parameters())
print(total_params/1e6, 'M parameters')

# Create a PyTorch optimizer (optimizing both model and head)
optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=config.learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    head.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split)
            x_out = model(X)           # Step 2: Get Contextual embeddings
            logits, loss = head(x_out, Y) # Step 3: Get Logits & Loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    head.train()
    return out

print("Starting training loop...")
# Training loop
for iter in range(config.max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    x_out = model(xb)             # Step 2: Get Contextual embeddings
    logits, loss = head(x_out, yb)   # Step 3: Get Logits & Loss
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training finished!")

# Save the model
print(f"Saving model to {config.checkpoint_path}")
checkpoint = {
    'model_state_dict': model.state_dict(),
    'head_state_dict': head.state_dict(),
}
torch.save(checkpoint, config.checkpoint_path)

# Generate from the model
print("\n--- Generating some text ---")
context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
print(encoder.decode(head.generate(model, context, max_new_tokens=500)[0].tolist()))
