import React, { useState } from 'react';
import axios from 'axios';
import Editor from './components/Editor';
import Visualizer from './components/Visualizer';
import './index.css';

const DEFAULT_CODE = `import torch
import torch.nn.functional as F
import math

# Try editing the hyperparameters below!
n_heads = 4
head_dim = 16
T = 8  # Sequence Length

# Generate dummy Queries and Keys
torch.manual_seed(1337)
q = torch.randn(1, n_heads, T, head_dim)
k = torch.randn(1, n_heads, T, head_dim)

# The Standard Attention Formula: Softmax(Q @ K.T / sqrt(d))
# Calculate the raw affinities
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

# Apply Causal Mask (can't look into the future)
tril = torch.tril(torch.ones(T, T))
scores = scores.masked_fill(tril == 0, float('-inf'))

# Convert raw scores to percentages (0.0 to 1.0)
percentages = F.softmax(scores, dim=-1)

# Pass the final tensor to the Web Visualizer!
# Since batch is 1, we extract [n_heads, T, T]
visualize_data = percentages[0].detach()
`;

function App() {
  const [code, setCode] = useState(DEFAULT_CODE);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const runCode = async () => {
    setLoading(true);
    setError(null);
    setData(null);
    try {
      const response = await axios.post('http://localhost:8000/run_math', { code });
      if (response.data.status === 'success') {
        setData(response.data.data.visualize_data);
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
      <Editor code={code} setCode={setCode} onRun={runCode} />
      <Visualizer data={data} loading={loading} error={error} />
    </div>
  );
}

export default App;
