import React from 'react';
import MonacoEditor from '@monaco-editor/react';

export default function Editor({ code, setCode, onRun }) {
  return (
    <div className="editor-pane">
      <div className="editor-header">
        <span style={{fontWeight: 'bold'}}>PyTorch Mathematics Engine</span>
        <button className="run-btn" onClick={onRun}>Run Code</button>
      </div>
      <div className="editor-content">
        <MonacoEditor
          height="100%"
          language="python"
          theme="vs-dark"
          value={code}
          onChange={(val) => setCode(val)}
          options={{
            minimap: { enabled: false },
            fontSize: 14,
            wordWrap: "on",
          }}
        />
      </div>
    </div>
  );
}
