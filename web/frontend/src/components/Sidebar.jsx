import React from 'react';

export default function Sidebar({ curriculum, selectedModelId, selectedLessonIdx, onSelectModel, onSelectLesson }) {
  
  const currentModel = curriculum.find(m => m.id === selectedModelId);

  return (
    <div className="sidebar-pane">
      <div className="sidebar-header">
        <span style={{fontWeight: 'bold'}}>nano-GPT Textbooks</span>
      </div>
      
      {/* The Model Picker Dropdown */}
      <div style={{ padding: '15px 15px 5px 15px' }}>
        <label style={{ fontSize: '11px', textTransform: 'uppercase', color: '#888', fontWeight: 600, display: 'block', marginBottom: '8px' }}>
          Select Architecture
        </label>
        <select 
          value={selectedModelId}
          onChange={(e) => onSelectModel(e.target.value)}
          style={{ width: '100%', padding: '6px', backgroundColor: '#3c3c3c', color: 'white', border: '1px solid #555', borderRadius: '4px', outline: 'none' }}
        >
          {curriculum.map(model => (
            <option key={model.id} value={model.id}>
              {model.name}
            </option>
          ))}
        </select>
      </div>

      <div className="sidebar-content">
        <div className="sidebar-category">Forward Pass Steps</div>
        <ul className="lesson-list">
          {currentModel && currentModel.lessons.map((lesson, idx) => (
            <li 
              key={idx} 
              className={`lesson-item ${idx === selectedLessonIdx ? 'active' : ''}`}
              onClick={() => onSelectLesson(idx)}
            >
              {lesson.title}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
