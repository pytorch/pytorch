import React from 'react';
import { getColorForCallstack, getDarkerColor } from '../utils/colorGenerator';
import './EventCell.css';

function EventCell({ event, width, isSelected, onClick }) {
  const { callstack, recordIndex } = event;
  const lastLine = callstack[callstack.length - 1] || '';

  const color = getColorForCallstack(callstack);
  const backgroundColor = isSelected ? getDarkerColor(color) : color;

  const left = recordIndex * width;

  return (
    <div
      className={`event-cell ${isSelected ? 'selected' : ''}`}
      style={{
        left,
        width: width - 2,
        backgroundColor
      }}
      onClick={onClick}
      title={callstack.join('\n')}
    >
      <span className="event-text">{lastLine}</span>
    </div>
  );
}

export default EventCell;
