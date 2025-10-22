import React, { useState, useEffect, useRef, useCallback } from 'react';
import { processTraceData } from '../utils/dataProcessor';
import EventCell from './EventCell';
import IcicleModal from './IcicleModal';
import './EventGrid.css';

const ROW_HEIGHT = 40;
const VISIBLE_ROWS = 32;
const MIN_EVENT_WIDTH = 60;
const DEFAULT_EVENT_WIDTH = 120;

function EventGrid({ data }) {
  const [processedData, setProcessedData] = useState(null);
  const [selectedEvents, setSelectedEvents] = useState(new Set());
  const [showIcicle, setShowIcicle] = useState(false);
  const [collapsedGroups, setCollapsedGroups] = useState(new Set());
  const [zoom, setZoom] = useState(0.5); // Changed from 1 to 0.5 for 50% default zoom
  const [scrollTop, setScrollTop] = useState(0);
  const [scrollLeft, setScrollLeft] = useState(0);
  const gridRef = useRef(null);
  const contentRef = useRef(null);

  useEffect(() => {
    if (data) {
      const processed = processTraceData(data);
      setProcessedData(processed);
    }
  }, [data]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'T') {
        e.preventDefault();
        if (selectedEvents.size > 0) {
          setShowIcicle(true);
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedEvents]);

  const handleEventClick = useCallback((eventId) => {
    setSelectedEvents(prev => {
      const newSet = new Set(prev);
      if (newSet.has(eventId)) {
        newSet.delete(eventId);
      } else {
        newSet.add(eventId);
      }
      return newSet;
    });
  }, []);

  const toggleGroup = useCallback((pgName) => {
    setCollapsedGroups(prev => {
      const newSet = new Set(prev);
      if (newSet.has(pgName)) {
        newSet.delete(pgName);
      } else {
        newSet.add(pgName);
      }
      return newSet;
    });
  }, []);

  const handleZoom = useCallback((delta) => {
    setZoom(prev => Math.max(0.5, Math.min(3, prev + delta)));
  }, []);

  if (!processedData) {
    return <div>Processing data...</div>;
  }

  const { processGroupData, firstMismatchRecordId, maxRecordIndex } = processedData;

  // Build rows
  const rows = [];
  Object.entries(processGroupData).forEach(([pgName, ranks]) => {
    const isCollapsed = collapsedGroups.has(pgName);
    rows.push({ type: 'group', pgName, isCollapsed });

    if (!isCollapsed) {
      Object.entries(ranks).forEach(([rank, events]) => {
        rows.push({ type: 'rank', pgName, rank: parseInt(rank), events });
      });
    }
  });

  const eventWidth = DEFAULT_EVENT_WIDTH * zoom;
  const totalWidth = (maxRecordIndex + 1) * eventWidth;
  const totalHeight = rows.length * ROW_HEIGHT;
  const visibleHeight = VISIBLE_ROWS * ROW_HEIGHT;

  const startRow = Math.floor(scrollTop / ROW_HEIGHT);
  const endRow = Math.min(rows.length, startRow + VISIBLE_ROWS + 1);
  const visibleRows = rows.slice(startRow, endRow);

  // Position the mismatch line between (firstMismatchRecordId - 1) and firstMismatchRecordId
  // This is at the left edge of the event with recordIndex = firstMismatchRecordId
  const mismatchX = firstMismatchRecordId * eventWidth;

  return (
    <div className="event-grid-container">
      <div className="controls">
        <button onClick={() => handleZoom(0.1)}>Zoom In</button>
        <button onClick={() => handleZoom(-0.1)}>Zoom Out</button>
        <span className="zoom-level">Zoom: {(zoom * 100).toFixed(0)}%</span>
        <span className="selection-count">
          Selected: {selectedEvents.size}
        </span>
        {selectedEvents.size > 0 && (
          <button onClick={() => setShowIcicle(true)}>
            Show Icicle (Ctrl+Shift+T)
          </button>
        )}
      </div>

      <div
        className="event-grid"
        ref={gridRef}
        onScroll={(e) => {
          setScrollTop(e.target.scrollTop);
          setScrollLeft(e.target.scrollLeft);
        }}
        style={{ height: `calc(100vh - 140px)` }}
      >
        <div
          className="event-grid-content"
          ref={contentRef}
          style={{
            width: totalWidth,
            height: totalHeight,
            position: 'relative'
          }}
        >
          {/* Mismatch indicator - positioned between record (firstMismatchRecordId-1) and firstMismatchRecordId */}
          {firstMismatchRecordId !== undefined && (
            <>
              <div
                className="mismatch-label"
                style={{ left: mismatchX }}
              >
                Mismatches →
              </div>
              <div
                className="mismatch-line"
                style={{
                  left: mismatchX - 1, // Center the 2px line on the boundary
                  height: totalHeight
                }}
              />
            </>
          )}

          {/* Render visible rows */}
          {visibleRows.map((row, idx) => {
            const actualRowIndex = startRow + idx;
            const top = actualRowIndex * ROW_HEIGHT;

            if (row.type === 'group') {
              return (
                <div
                  key={`group-${row.pgName}`}
                  className="row-group-header"
                  style={{ top }}
                  onClick={() => toggleGroup(row.pgName)}
                >
                  <span className="collapse-icon">
                    {row.isCollapsed ? '▶' : '▼'}
                  </span>
                  {row.pgName}
                </div>
              );
            } else {
              // Rank row
              return (
                <div
                  key={`rank-${row.pgName}-${row.rank}`}
                  className="row-rank"
                  style={{ top }}
                >
                  <div className="row-label">
                    Rank {row.rank}
                  </div>
                  <div className="row-events">
                    {row.events.map(event => (
                      <EventCell
                        key={event.id}
                        event={event}
                        width={eventWidth}
                        isSelected={selectedEvents.has(event.id)}
                        onClick={() => handleEventClick(event.id)}
                      />
                    ))}
                  </div>
                </div>
              );
            }
          })}
        </div>
      </div>

      {showIcicle && (
        <IcicleModal
          events={Array.from(selectedEvents).map(id => {
            // Find event by id
            for (const ranks of Object.values(processGroupData)) {
              for (const events of Object.values(ranks)) {
                const event = events.find(e => e.id === id);
                if (event) return event;
              }
            }
            return null;
          }).filter(Boolean)}
          onClose={() => setShowIcicle(false)}
        />
      )}
    </div>
  );
}

export default EventGrid;
