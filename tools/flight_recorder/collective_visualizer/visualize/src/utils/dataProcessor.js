import React, { useState, useRef, useEffect } from 'react';
import '../components/IcicleModal.css';

export function processTraceData(rawData) {
  const { first_mismatch_record_id, traces } = rawData;

  // Structure: { pgName: { rank: [events] } }
  const processGroupData = {};

  // Track max record index
  let maxRecordIndex = 0;

  // Process traces
  Object.entries(traces).forEach(([rankStr, events]) => {
    const rank = parseInt(rankStr);

    events.forEach((eventData, recordIndex) => {
      const [pgName, callstack] = eventData;

      maxRecordIndex = Math.max(maxRecordIndex, recordIndex);

      if (!processGroupData[pgName]) {
        processGroupData[pgName] = {};
      }

      if (!processGroupData[pgName][rank]) {
        processGroupData[pgName][rank] = [];
      }

      processGroupData[pgName][rank].push({
        pgName,
        rank,
        recordIndex,
        callstack,
        id: `${pgName}-${rank}-${recordIndex}`
      });
    });
  });

  return {
    processGroupData,
    firstMismatchRecordId: first_mismatch_record_id,
    maxRecordIndex
  };
}

export function buildCallstackTree(events) {
  const root = {
    name: 'root',
    children: [],
    count: 0,
    depth: 0
  };

  events.forEach(event => {
    let currentNode = root;

    event.callstack.forEach((frame, depth) => {
      let childNode = currentNode.children.find(child => child.name === frame);

      if (!childNode) {
        childNode = {
          name: frame,
          children: [],
          count: 0,
          depth: depth + 1,
          parent: currentNode
        };
        currentNode.children.push(childNode);
      }

      childNode.count++;
      currentNode = childNode;
    });
  });

  return root;
}

const DEFAULT_VISIBLE_DEPTH = 15;

function IcicleModal({ events, onClose }) {
  const [tree, setTree] = useState(null);
  const [focusedNode, setFocusedNode] = useState(null);
  const [scrollOffset, setScrollOffset] = useState(0);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    const callstackTree = buildCallstackTree(events);
    setTree(callstackTree);
  }, [events]);

  useEffect(() => {
    if (!tree || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.clearRect(0, 0, width, height);

    const maxCount = Math.max(...getAllNodes(tree).map(n => n.count));
    const rootNode = focusedNode || tree;

    // Get max depth for focused subtree
    const maxDepth = getMaxDepth(rootNode);
    const visibleDepth = Math.min(maxDepth, DEFAULT_VISIBLE_DEPTH);
    const rowHeight = 30;

    function drawNode(node, x, nodeWidth, depth) {
      if (depth > visibleDepth + scrollOffset) return;
      if (depth < scrollOffset) {
        // Still need to traverse children
        node.children.forEach(child => {
          const childWidth = (child.count / node.count) * nodeWidth;
          drawNode(child, x, childWidth, depth + 1);
          x += childWidth;
        });
        return;
      }

      const displayDepth = depth - scrollOffset;
      const y = displayDepth * rowHeight;

      // Color based on depth
      const hue = (depth * 30) % 360;
      const saturation = 60 + (node.count / maxCount) * 20;
      const lightness = 70 - (node.count / maxCount) * 15;

      ctx.fillStyle = `hsl(${hue}, ${saturation}%, ${lightness}%)`;
      ctx.fillRect(x, y, nodeWidth, rowHeight - 1);

      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, nodeWidth, rowHeight - 1);

      // Draw text
      if (nodeWidth > 50) {
        ctx.fillStyle = '#000';
        ctx.font = '12px sans-serif';
        ctx.textBaseline = 'middle';

        const text = `${node.name} (${node.count})`;
        const textWidth = ctx.measureText(text).width;

        if (textWidth < nodeWidth - 10) {
          ctx.fillText(text, x + 5, y + rowHeight / 2);
        } else {
          // Truncate
          let truncated = node.name;
          while (ctx.measureText(truncated + '...').width > nodeWidth - 10 && truncated.length > 0) {
            truncated = truncated.slice(0, -1);
          }
          ctx.fillText(truncated + '...', x + 5, y + rowHeight / 2);
        }
      }

      // Draw children
      let childX = x;
      node.children.forEach(child => {
        const childWidth = (child.count / node.count) * nodeWidth;
        drawNode(child, childX, childWidth, depth + 1);
        childX += childWidth;
      });
    }

    drawNode(rootNode, 0, width, 0);
  }, [tree, focusedNode, scrollOffset]);

  const handleCanvasClick = (e) => {
    if (!tree) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const rowHeight = 30;
    const width = canvas.width;

    const rootNode = focusedNode || tree;

    function findNodeAt(node, nodeX, nodeWidth, depth, targetX, targetY) {
      const displayDepth = depth - scrollOffset;
      if (displayDepth < 0 || displayDepth > DEFAULT_VISIBLE_DEPTH) {
        // Check children anyway
        let childX = nodeX;
        for (const child of node.children) {
          const childWidth = (child.count / node.count) * nodeWidth;
          const found = findNodeAt(child, childX, childWidth, depth + 1, targetX, targetY);
          if (found) return found;
          childX += childWidth;
        }
        return null;
      }

      const nodeY = displayDepth * rowHeight;

      if (targetX >= nodeX && targetX < nodeX + nodeWidth &&
          targetY >= nodeY && targetY < nodeY + rowHeight) {
        return node;
      }

      let childX = nodeX;
      for (const child of node.children) {
        const childWidth = (child.count / node.count) * nodeWidth;
        const found = findNodeAt(child, childX, childWidth, depth + 1, targetX, targetY);
        if (found) return found;
        childX += childWidth;
      }

      return null;
    }

    const clickedNode = findNodeAt(rootNode, 0, width, 0, x, y);
    if (clickedNode && clickedNode !== tree) {
      setFocusedNode(clickedNode);
      setScrollOffset(0);
    }
  };

  const handleScroll = (e) => {
    const delta = e.deltaY > 0 ? 1 : -1;
    setScrollOffset(prev => Math.max(0, prev + delta));
  };

  const handleReset = () => {
    setFocusedNode(null);
    setScrollOffset(0);
  };

  if (!tree) return null;

  return (
    <div className="icicle-modal-overlay" onClick={onClose}>
      <div className="icicle-modal" onClick={e => e.stopPropagation()}>
        <div className="icicle-header">
          <h2>Callstack Icicle View</h2>
          <div className="icicle-controls">
            {focusedNode && (
              <button onClick={handleReset}>Reset Focus</button>
            )}
            <button onClick={onClose}>Close</button>
          </div>
        </div>
        <div className="icicle-info">
          {events.length} event{events.length !== 1 ? 's' : ''} selected
          {focusedNode && ` • Focused on: ${focusedNode.name}`}
        </div>
        <div
          className="icicle-canvas-container"
          ref={containerRef}
          onWheel={handleScroll}
        >
          <canvas
            ref={canvasRef}
            width={1200}
            height={450}
            onClick={handleCanvasClick}
          />
        </div>
        <div className="icicle-help">
          Click on a rectangle to focus • Scroll to navigate depth • Each row shows call stack depth
        </div>
      </div>
    </div>
  );
}

function getAllNodes(node) {
  const nodes = [node];
  node.children.forEach(child => {
    nodes.push(...getAllNodes(child));
  });
  return nodes;
}

function getMaxDepth(node, currentDepth = 0) {
  if (node.children.length === 0) return currentDepth;
  return Math.max(...node.children.map(child => getMaxDepth(child, currentDepth + 1)));
}

export default IcicleModal;
