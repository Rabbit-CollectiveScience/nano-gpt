import os
import sys
import json
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
results_file = os.path.join(current_dir, 'results', 'benchmark_results.json')

def plot_benchmark():
    if not os.path.exists(results_file):
        print(f"Cannot find {results_file}. Run run_arena.py first!")
        sys.exit(1)
        
    with open(results_file, 'r') as f:
        data = json.load(f)

    # 1. Setup Plot
    plt.figure(figsize=(10, 6))
    plt.title('GPT-2 vs LLaMA: Validation Loss on Unseen Text (Alice in Wonderland)')
    plt.xlabel('Training Steps')
    plt.ylabel('Validation Loss (Lower is Better)')
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. Extract Data
    for model_name, results in data.items():
        steps = [entry['step'] for entry in results['history']]
        losses = [entry['val_loss'] for entry in results['history']]
        
        # Color coding
        color = '#1f77b4' if model_name == 'gpt2' else '#d62728' 
        style = '-' if model_name == 'gpt2' else '--'
        
        # Add a rich label with parameter counts
        label = f"{model_name.upper()} ({results['parameters']/1e6:.1f}M params)"
        
        plt.plot(steps, losses, label=label, color=color, linestyle=style, linewidth=2)

    # 3. Format and Save
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(current_dir, 'results', 'loss_comparison.png')
    plt.savefig(save_path, dpi=300)
    print(f"Beautiful! Graph successfully drawn and saved to {save_path}")
    
if __name__ == '__main__':
    plot_benchmark()
