import React, { useState } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import Editor from './components/Editor';
import Visualizer from './components/Visualizer';
import './index.css';

// The centralized source of truth for the Interactive Textbook
// Now supporting multiple models, each with their own sequential steps!
const CURRICULUM = [
  {
    id: "gpt2",
    name: "GPT-2 (2019)",
    lessons: [
      {
        title: "Step 1: Token Embeddings",
        code: `import torch\n\n# Hyperparameters\nVocab_Size = 50257\nEmbed_Dim = 16\nT = 8\n\n# Simulate 8 input tokens (Integers representing words)\ntokens = torch.tensor([42, 105, 3, 99, 12, 4, 881, 7])\n\n# The Embedding Table (A giant lookup matrix)\ntorch.manual_seed(1337)\nwte = torch.randn(Vocab_Size, Embed_Dim)\n\n# Pull the 8 specific rows out of the giant table\ntoken_embeddings = wte[tokens]\n\n# We now have a mathematical 2D grid representing our sentence!\nvisualize_data = token_embeddings.detach().tolist()`,
        type: "matrix"
      },
      {
        title: "Step 2: Self-Attention",
        code: `import torch\nimport torch.nn.functional as F\nimport math\n\nn_heads = 4\nhead_dim = 16\nT = 8\n\ntorch.manual_seed(1337)\nq = torch.randn(1, n_heads, T, head_dim)\nk = torch.randn(1, n_heads, T, head_dim)\n\nscores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)\n\ntril = torch.tril(torch.ones(T, T))\nscores = scores.masked_fill(tril == 0, float('-inf'))\npercentages = F.softmax(scores, dim=-1)\n\n# Visualize the internal Attention matrices\nvisualize_data = percentages[0].detach().tolist()`,
        type: "attention"
      },
      {
        title: "Step 3: Feed-Forward (MLP)",
        code: `import torch\nimport torch.nn.functional as F\n\nEmbed_Dim = 16\nT = 8\n\n# Input to the MLP (from Attention)\ntorch.manual_seed(1337)\nx = torch.randn(T, Embed_Dim)\n\n# GPT-2 physically expands the dimension by 4x to "think" about the data\nlinear1 = torch.nn.Linear(Embed_Dim, 4 * Embed_Dim, bias=True)\nexpanded_x = linear1(x)\n\n# Apply the GELU mathematical activation function (creates non-linearity)\nactivated_x = F.gelu(expanded_x)\n\n# Compress it back down to the original dimension\nlinear2 = torch.nn.Linear(4 * Embed_Dim, Embed_Dim, bias=True)\nfinal_x = linear2(activated_x)\n\n# Let's visualize the massive expanded hidden state inside the MLP!\nvisualize_data = activated_x.detach().tolist()`,
        type: "matrix"
      },
      {
        title: "Step 4: Output Logits",
        code: `import torch\nimport torch.nn.functional as F\n\n# This is the final step of the entire neural network.\nEmbed_Dim = 16\nT = 8\nVocab_Size = 15 # Shrunk down from 50k so we can visualize it!\n\n# The final processed math from the 12th GPT-2 Block\ntorch.manual_seed(1337)\nfinal_hidden_states = torch.randn(T, Embed_Dim)\n\n# The Language Modeling Head (Maps the math back into Vocabulary probabilities)\nlm_head = torch.nn.Linear(Embed_Dim, Vocab_Size, bias=False)\n\nraw_logits = lm_head(final_hidden_states)\n\n# Convert absolute numbers to percentages (0.0 to 1.0) along the Vocab dimension\nprobabilities = F.softmax(raw_logits, dim=-1)\n\n# The grid shows the probability of the NEXT word for each of the 8 input tokens!\nvisualize_data = probabilities.detach().tolist()`,
        type: "matrix"
      }
    ]
  },
  {
    id: "llama",
    name: "LLaMA 2 (2023)",
    lessons: [
      {
        title: "Step 1: Token Embeddings",
        code: `import torch\n\n# Hyperparameters\nVocab_Size = 32000 # LLaMA has a smaller vocabulary than GPT-2\nEmbed_Dim = 16\nT = 8\n\n# Simulate 8 input tokens\ntokens = torch.tensor([42, 105, 3, 99, 12, 4, 881, 7])\n\n# The Embedding Table\ntorch.manual_seed(1337)\nwte = torch.randn(Vocab_Size, Embed_Dim)\n\ntoken_embeddings = wte[tokens]\n\nvisualize_data = token_embeddings.detach().tolist()`,
        type: "matrix"
      },
      {
        title: "Step 2: Rotary Embeddings",
        code: `import torch\n\n# LLaMA uses RoPE instead of Absolute Positions\nT = 8\nEmbed_Dim = 16\ntorch.manual_seed(1337)\n\nvisualize_data = torch.randn(T, Embed_Dim).tolist()`,
        type: "matrix"
      },
      {
        title: "Step 3: Self-Attention (RMSNorm)",
        code: `import torch\nimport torch.nn.functional as F\nimport math\n\nn_heads = 4\nhead_dim = 16\nT = 8\n\ntorch.manual_seed(1337)\nq = torch.randn(1, n_heads, T, head_dim)\nk = torch.randn(1, n_heads, T, head_dim)\n\nscores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)\n\ntril = torch.tril(torch.ones(T, T))\nscores = scores.masked_fill(tril == 0, float('-inf'))\npercentages = F.softmax(scores, dim=-1)\n\nvisualize_data = percentages[0].detach().tolist()`,
        type: "attention"
      },
      {
        title: "Step 4: SwiGLU Feed-Forward",
        code: `import torch\nimport torch.nn.functional as F\n\nEmbed_Dim = 16\nT = 8\n\ntorch.manual_seed(1337)\nx = torch.randn(T, Embed_Dim)\n\nw1 = torch.nn.Linear(Embed_Dim, 4 * Embed_Dim, bias=False)\nw2 = torch.nn.Linear(4 * Embed_Dim, Embed_Dim, bias=False)\nw3 = torch.nn.Linear(Embed_Dim, 4 * Embed_Dim, bias=False)\n\ngate = F.silu(w1(x)) * w3(x)\nfinal_x = w2(gate)\n\nvisualize_data = gate.detach().tolist()`,
        type: "matrix"
      },
      {
        title: "Step 5: Output Logits",
        code: `import torch\nimport torch.nn.functional as F\n\nEmbed_Dim = 16\nT = 8\nVocab_Size = 15\n\ntorch.manual_seed(1337)\nfinal_hidden_states = torch.randn(T, Embed_Dim)\n\nlm_head = torch.nn.Linear(Embed_Dim, Vocab_Size, bias=False)\nraw_logits = lm_head(final_hidden_states)\nprobabilities = F.softmax(raw_logits, dim=-1)\n\nvisualize_data = probabilities.detach().tolist()`,
        type: "matrix"
      }
    ]
  },
  {
    id: "mistral",
    name: "Mistral 7B (2023)",
    lessons: [
      {
        title: "Step 1: Token Embeddings",
        code: `import torch\n\n# Hyperparameters\nVocab_Size = 32000 # Mistral shares LLaMA's vocab size\nEmbed_Dim = 16\nT = 8\n\n# Simulate 8 input tokens\ntokens = torch.tensor([42, 105, 3, 99, 12, 4, 881, 7])\n\n# The Embedding Table\ntorch.manual_seed(1337)\nwte = torch.randn(Vocab_Size, Embed_Dim)\n\ntoken_embeddings = wte[tokens]\n\nvisualize_data = token_embeddings.detach().tolist()`,
        type: "matrix"
      },
      {
        title: "Step 2: Rotary Embeddings",
        code: `import torch\n\n# Mistral dropped Absolute positions for Rotary Position Embeddings (RoPE)\n# (We will visualize this complex math later!)\n\nT = 8\nEmbed_Dim = 16\ntorch.manual_seed(1337)\n\n# Just returning random data as a placeholder for now\nvisualize_data = torch.randn(T, Embed_Dim).tolist()`,
        type: "matrix"
      },
      {
        title: "Step 3: Grouped-Query Attention",
        code: `import torch\nimport torch.nn.functional as F\nimport math\n\nn_heads = 4\nn_kv_heads = 1\nhead_dim = 16\nT = 8 \n\ntorch.manual_seed(1337)\nq = torch.randn(1, n_heads, T, head_dim)\nk = torch.randn(1, n_kv_heads, T, head_dim)\n\n# ⚠️ We mathematically duplicate the 1 Key into 4 clones so the matrices match!\nn_rep = n_heads // n_kv_heads\nk_expanded = torch.repeat_interleave(k, n_rep, dim=1)\n\n# Now we can safely run the Attention Formula\nscores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)\n\ntril = torch.tril(torch.ones(T, T))\nscores = scores.masked_fill(tril == 0, float('-inf'))\npercentages = F.softmax(scores, dim=-1)\n\nvisualize_data = percentages[0].detach().tolist()`,
        type: "attention"
      },
      {
        title: "Step 4: SwiGLU Feed-Forward",
        code: `import torch\nimport torch.nn.functional as F\n\nEmbed_Dim = 16\nT = 8\n\n# Input to the MLP (from Attention)\ntorch.manual_seed(1337)\nx = torch.randn(T, Embed_Dim)\n\n# Mistral uses the SwiGLU 3-matrix architecture instead of the GPT-2 2-matrix architecture\nw1 = torch.nn.Linear(Embed_Dim, 4 * Embed_Dim, bias=False)\nw2 = torch.nn.Linear(4 * Embed_Dim, Embed_Dim, bias=False)\nw3 = torch.nn.Linear(Embed_Dim, 4 * Embed_Dim, bias=False)\n\n# The Swish/Silu gating mechanism\ngate = F.silu(w1(x)) * w3(x)\nfinal_x = w2(gate)\n\n# Visualizing the gating matrix before it compresses back down\nvisualize_data = gate.detach().tolist()`,
        type: "matrix"
      },
      {
        title: "Step 5: Output Logits",
        code: `import torch\nimport torch.nn.functional as F\n\nEmbed_Dim = 16\nT = 8\nVocab_Size = 15\n\ntorch.manual_seed(1337)\nfinal_hidden_states = torch.randn(T, Embed_Dim)\n\nlm_head = torch.nn.Linear(Embed_Dim, Vocab_Size, bias=False)\nraw_logits = lm_head(final_hidden_states)\nprobabilities = F.softmax(raw_logits, dim=-1)\n\nvisualize_data = probabilities.detach().tolist()`,
        type: "matrix"
      }
    ]
  },
  {
    id: "mixtral",
    name: "Mixtral 8x7B (2024)",
    lessons: [
      {
        title: "Step 1: Token Embeddings",
        code: `import torch\n\n# Hyperparameters\nVocab_Size = 32000 # Mixtral shares LLaMA's vocab size\nEmbed_Dim = 16\nT = 8\n\n# Simulate 8 input tokens\ntokens = torch.tensor([42, 105, 3, 99, 12, 4, 881, 7])\n\n# The Embedding Table\ntorch.manual_seed(1337)\nwte = torch.randn(Vocab_Size, Embed_Dim)\n\ntoken_embeddings = wte[tokens]\n\nvisualize_data = token_embeddings.detach().tolist()`,
        type: "matrix"
      },
      {
        title: "Step 2: Rotary Embeddings",
        code: `import torch\n\nT = 8\nEmbed_Dim = 16\ntorch.manual_seed(1337)\n\nvisualize_data = torch.randn(T, Embed_Dim).tolist()`,
        type: "matrix"
      },
      {
        title: "Step 3: Grouped-Query Attention",
        code: `import torch\nimport torch.nn.functional as F\nimport math\n\nn_heads = 4\nn_kv_heads = 1\nhead_dim = 16\nT = 8 \n\ntorch.manual_seed(1337)\nq = torch.randn(1, n_heads, T, head_dim)\nk = torch.randn(1, n_kv_heads, T, head_dim)\n\nn_rep = n_heads // n_kv_heads\nk_expanded = torch.repeat_interleave(k, n_rep, dim=1)\n\nscores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(head_dim)\n\ntril = torch.tril(torch.ones(T, T))\nscores = scores.masked_fill(tril == 0, float('-inf'))\npercentages = F.softmax(scores, dim=-1)\n\nvisualize_data = percentages[0].detach().tolist()`,
        type: "attention"
      },
      {
        title: "Step 4: Sparse MoE Router",
        code: `import torch\nimport torch.nn.functional as F\n\n# Hyperparameters\nnum_experts = 4\nnum_experts_per_tok = 2\nT = 8\nhead_dim = 16\n\ntorch.manual_seed(1337)\nhidden_states = torch.randn(T, head_dim)\n\ngate = torch.nn.Linear(head_dim, num_experts, bias=False)\nrouter_logits = gate(hidden_states)\n\nrouting_weights = F.softmax(router_logits, dim=1)\nrouting_weights, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)\nrouting_weights /= routing_weights.sum(dim=-1, keepdim=True)\n\nvisualize_data = {\n    "selected_experts": selected_experts.tolist(),\n    "routing_weights": routing_weights.tolist(),\n    "num_experts": num_experts,\n    "T": T\n}`,
        type: "router"
      },
      {
        title: "Step 5: Output Logits",
        code: `import torch\nimport torch.nn.functional as F\n\nEmbed_Dim = 16\nT = 8\nVocab_Size = 15\n\ntorch.manual_seed(1337)\nfinal_hidden_states = torch.randn(T, Embed_Dim)\n\nlm_head = torch.nn.Linear(Embed_Dim, Vocab_Size, bias=False)\nraw_logits = lm_head(final_hidden_states)\nprobabilities = F.softmax(raw_logits, dim=-1)\n\nvisualize_data = probabilities.detach().tolist()`,
        type: "matrix"
      }
    ]
  }
];

function App() {
  const [selectedModelId, setSelectedModelId] = useState(CURRICULUM[0].id);
  const [selectedLessonIdx, setSelectedLessonIdx] = useState(0);
  
  // Helper to get exactly which lesson we are currently looking at
  const currentModel = CURRICULUM.find(m => m.id === selectedModelId);
  const currentLesson = currentModel.lessons[selectedLessonIdx];

  const [code, setCode] = useState(currentLesson.code);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // When a user selects a DIFFERENT MODEL from the dropdown
  const handleSelectModel = (newModelId) => {
    const newModel = CURRICULUM.find(m => m.id === newModelId);
    setSelectedModelId(newModelId);
    setSelectedLessonIdx(0); // Reset to Step 1 of the new model
    setCode(newModel.lessons[0].code);
    setData(null);
    setError(null);
  };

  // When a user selects a DIFFERENT LESSON within the same model
  const handleSelectLesson = (idx) => {
    setSelectedLessonIdx(idx);
    setCode(currentModel.lessons[idx].code);
    setData(null);
    setError(null);
  };

  const runCode = async () => {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const response = await axios.post('http://localhost:8000/run_math', { code });
      if (response.data.status === 'success') {
        setData({
          payload: response.data.data.visualize_data,
          type: currentLesson.type
        });
      }
    } catch (err) {
      if (err.response && err.response.data && err.response.data.detail) {
        setError(err.response.data.detail);
      } else {
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Sidebar 
        curriculum={CURRICULUM}
        selectedModelId={selectedModelId}
        selectedLessonIdx={selectedLessonIdx}
        onSelectModel={handleSelectModel}
        onSelectLesson={handleSelectLesson} 
      />
      <Editor code={code} setCode={setCode} onRun={runCode} />
      <Visualizer data={data} loading={loading} error={error} />
    </div>
  );
}

export default App;
