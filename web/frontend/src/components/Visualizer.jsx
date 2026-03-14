import React, { useEffect, useRef } from 'react';
import { drawAttentionHeatmap } from '../d3/drawAttention';

export default function Visualizer({ data, loading, error }) {
  const svgRef = useRef(null);

  useEffect(() => {
    if (data && svgRef.current) {
      drawAttentionHeatmap(svgRef.current, data);
    }
  }, [data]);

  return (
    <div className="visualizer-pane">
      <div className="visualizer-header">
        <span style={{fontWeight: 'bold'}}>Live Mathematical Visualizer (D3.js)</span>
      </div>
      <div className="visualizer-content">
        {loading && <p style={{color: '#888'}}>Running PyTorch execution engine...</p>}
        {error && (
          <div style={{color: '#f48771', backgroundColor: '#3c0000', padding: 10, borderRadius: 5, whiteSpace: 'pre-wrap', fontFamily: 'monospace'}}>
            {error}
          </div>
        )}
        {!loading && !error && !data && (
          <p style={{color: '#888'}}>Write PyTorch math and click Run to visualize the resulting tensors in the browser!</p>
        )}
        
        {/* The D3 Canvas */}
        <div style={{width: '100%', height: '100%'}} ref={svgRef} />
      </div>
    </div>
  );
}
