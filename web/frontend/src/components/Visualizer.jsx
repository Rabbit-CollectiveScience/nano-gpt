import React, { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

function AttentionHead({ headMatrix, headIdx, T, totalHeads }) {
  // headMatrix is [T][T]
  const spacing = 1.2;
  
  const boxes = useMemo(() => {
    const b = [];
    for (let r = 0; r < T; r++) {
      for (let c = 0; c < T; c++) {
        const val = headMatrix[r][c];
        if (!Number.isNaN(val) && val >= 0) {
          b.push({ r, c, val });
        }
      }
    }
    return b;
  }, [headMatrix, T]);

  // Center the grid on X and Z
  const offsetX = -(T * spacing) / 2 + (spacing / 2);
  const offsetZ = -(T * spacing) / 2 + (spacing / 2);
  
  // Stack heads vertically on the Y axis
  // Add an offset so the total stack is somewhat centered
  const headSpacingY = 5;
  const offsetY = (headIdx - (totalHeads - 1) / 2) * headSpacingY; 

  return (
    <group position={[offsetX, offsetY, offsetZ]}>
      {boxes.map((b) => {
        // Height proportional to attention value (minimum height for visibility)
        const height = Math.max(0.05, b.val * 4);
        
        // Color mapping: 0 = blue (240 hue), 1 = red (0 hue)
        const hue = (1 - b.val) * 240; 
        
        return (
          <mesh 
            key={`${b.r}-${b.c}`} 
            position={[b.c * spacing, height / 2, b.r * spacing]}
          >
            <boxGeometry args={[1, height, 1]} />
            <meshStandardMaterial 
              color={`hsl(${Math.max(0, Math.floor(hue))}, 100%, 50%)`} 
              roughness={0.2}
              metalness={0.1}
            />
          </mesh>
        );
      })}
    </group>
  );
}

function MatrixVisualizer({ payload }) {
  // payload is arbitrary 2D matrix [Rows][Cols]
  const Rows = payload.length;
  const Cols = payload[0].length;
  const spacingX = 1.2;
  const spacingZ = 1.2;

  const offsetX = -(Cols * spacingX) / 2 + (spacingX / 2);
  const offsetZ = -(Rows * spacingZ) / 2 + (spacingZ / 2);

  // find max value for dynamic scaling
  let maxVal = 0.001;
  let minVal = 0;
  for (let r=0; r<Rows; r++){
    for (let c=0; c<Cols; c++) {
       if (payload[r][c] > maxVal) maxVal = payload[r][c];
       if (payload[r][c] < minVal) minVal = payload[r][c];
    }
  }
  const range = maxVal - minVal;

  const cubes = [];
  for (let r = 0; r < Rows; r++) {
    for (let c = 0; c < Cols; c++) {
      let val = payload[r][c];
      if (Number.isNaN(val)) val = 0;
      
      const normalized = range === 0 ? 0 : (val - minVal) / range;
      // Cube height based on intensity
      const height = Math.max(0.05, Math.abs(normalized) * 4);
      const hue = (1 - normalized) * 240; 
      
      cubes.push(
        <mesh 
          key={`${r}-${c}`} 
          position={[c * spacingX, height / 2, r * spacingZ]}
        >
          <boxGeometry args={[1, height, 1]} />
          <meshStandardMaterial 
            color={`hsl(${Math.max(0, Math.floor(hue))}, 100%, 50%)`} 
            roughness={0.2}
            metalness={0.1}
          />
        </mesh>
      );
    }
  }

  return (
    <group position={[offsetX, 0, offsetZ]}>
      {cubes}
    </group>
  );
}

export default function Visualizer({ data, loading, error }) {
  // data is now { payload: tensor/object, type: "attention" | "router" }
  return (
    <div className="visualizer-pane">
      <div className="visualizer-header">
        <span style={{fontWeight: 'bold'}}>Live 3D Mathematical Visualizer (React Three Fiber)</span>
      </div>
      <div className="visualizer-content" style={{ padding: 0, position: 'relative' }}>
        
        {/* Floating UI Overlays */}
        {loading && <div style={{ position: 'absolute', top: 20, left: 20, color: 'white', zIndex: 10 }}>Running PyTorch execution engine...</div>}
        {error && (
          <div style={{ position: 'absolute', top: 20, left: 20, color: '#f48771', backgroundColor: '#3c0000', padding: 10, borderRadius: 5, zIndex: 10, whiteSpace: 'pre-wrap', fontFamily: 'monospace', maxWidth: '80%'}}>
            {error}
          </div>
        )}
        {!loading && !error && !data && (
          <div style={{ position: 'absolute', top: 20, left: 20, color: '#888', zIndex: 10 }}>Select a lesson and click Run to visualize the architecture in 3D!</div>
        )}
        
        {/* The 3D WebGL Canvas */}
        <Canvas camera={{ position: [15, 12, 18], fov: 45 }}>
          <color attach="background" args={['#1e1e1e']} />
          <ambientLight intensity={0.6} />
          <pointLight position={[10, 20, 10]} intensity={1.5} />
          <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
          
          {/* Dynamic Component Rendering */}
          {data && data.type === 'attention' && data.payload.map((headMatrix, idx) => (
            <AttentionHead 
              key={idx} 
              headMatrix={headMatrix} 
              headIdx={idx} 
              T={headMatrix.length}
              totalHeads={data.payload.length}
            />
          ))}

          {data && data.type === 'matrix' && (
            <MatrixVisualizer payload={data.payload} />
          )}
          
          {/* Aesthetic grid floor */}
          <gridHelper 
            args={[50, 50, '#333333', '#222222']} 
            position={[0, Object.is(data?.type, 'attention') ? -(data.payload.length * 5)/2 - 1 : -1, 0]} 
          />
        </Canvas>
      </div>
    </div>
  );
}
