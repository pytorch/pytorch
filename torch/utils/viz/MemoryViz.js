/**
 * ================================================================================
 * MemoryViz.js - PyTorch Memory Visualization Tool
 * ================================================================================
 *
 * OVERVIEW:
 * ---------
 * This file contains the core visualization logic for PyTorch's memory profiler.
 * It renders memory allocation timelines, stack traces, and provides interactive
 * exploration of memory snapshots captured during model execution.
 *
 * KEY FEATURES:
 * - Multiple visualization tabs/views for different memory analysis perspectives
 * - Interactive stack trace display (supports both click and hover modes)
 * - Zoom and brush controls for navigating large memory timelines
 * - Support for loading memory snapshot files (.pickle format)
 *
 * ================================================================================
 * TESTING INSTRUCTIONS FOR ENGINEERS & AGENTS
 * ================================================================================
 *
 * 1. LOCAL TESTING SETUP:
 *    - Create a simple HTML file that references this JS file:
 *
 *      <!DOCTYPE html>
 *      <html>
 *      <head><title>MemoryViz Test</title></head>
 *      <body>
 *        <script type="module" src="MemoryViz.js"></script>
 *      </body>
 *      </html>
 *
 *    - Serve locally using: python3 -m http.server 8888
 *    - Open http://localhost:8888 in your browser
 *
 * 2. WHAT TO TEST:
 *    - Ensure ALL tabs/views render correctly and switch properly
 *    - Verify BOTH interaction modes work:
 *      * Click mode: stack traces appear on click
 *      * Hover mode: stack traces appear on mouseover
 *    - Test zoom and brush controls for timeline navigation
 *    - Verify memory allocation blocks are rendered and interactive
 *
 * 3. TEST DATA REQUIREMENTS:
 *    - DO NOT just test with small dummy .pickle files
 *    - Use realistic, decent-sized .pickle files (10-100+ MB range)
 *    - Large files stress-test rendering performance and memory handling
 *    - Test with snapshots from real model training/inference runs
 *
 * 4. COMMON ISSUES TO WATCH FOR:
 *    - Performance degradation with large snapshots
 *    - Stack trace popups not appearing or positioning incorrectly
 *    - Tab switching not updating the visualization properly
 *    - Zoom/brush state not persisting across interactions
 *
 * ================================================================================
 */

'use strict';

import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7/+esm";
import {axisLeft} from "https://cdn.jsdelivr.net/npm/d3-axis@3/+esm";
import {scaleLinear} from "https://cdn.jsdelivr.net/npm/d3-scale@4/+esm";
import {zoom, zoomIdentity} from "https://cdn.jsdelivr.net/npm/d3-zoom@3/+esm";
import {brushX} from "https://cdn.jsdelivr.net/npm/d3-brush@3/+esm";
import {process_alloc_data, isPrivatePoolId, formatSize, formatAddr,
        elideRepeats, frameFilter, format_user_metadata,
        format_forward_frames, format_frames} from "./process_alloc_data.js";

// Global configuration for trace interaction mode
// 'hover' = show trace on hover (default)
// 'click' = show trace on click
let traceInteractionMode = 'hover';

function setTraceInteractionMode(mode) {
  if (mode === 'click' || mode === 'hover') {
    traceInteractionMode = mode;
  }
}

function getTraceInteractionMode() {
  return traceInteractionMode;
}

const schemeTableau10 = [
  '#4e79a7',
  '#f28e2c',
  '#e15759',
  '#76b7b2',
  '#59a14f',
  '#edc949',
  '#af7aa1',
  '#ff9da7',
  '#9c755f',
  '#bab0ab',
];

function version_space() {
  const version = {};
  return (addr, increment) => {
    if (!(addr in version)) {
      version[addr] = 0;
    }
    const r = version[addr];
    if (increment) {
      version[addr]++;
    }
    return r;
  };
}

function Segment(addr, size, stream, frames, version, user_metadata, segment_pool_id) {
  return {addr, size, stream, version, frames, user_metadata, segment_pool_id};
}

function Block(addr, size, requested_size, frames, free_requested, version, user_metadata) {
  return {addr, size, requested_size, frames, free_requested, version, user_metadata};
}

function EventSelector(outer, events, stack_info, memory_view) {
  const events_div = outer
    .append('div')
    .attr(
      'style',
      'grid-column: 1; grid-row: 1; overflow: auto; font-family: monospace',
    );

  const events_selection = events_div
    .selectAll('pre')
    .data(events)
    .enter()
    .append('pre')
    .text(e => formatEvent(e))
    .attr('style', '');

  let selected_event_idx = null;

  const es = {
    select(idx) {
      if (selected_event_idx !== null) {
        const selected_event = d3.select(
          events_div.node().children[selected_event_idx],
        );
        selected_event.attr('style', '');
      }
      if (idx !== null) {
        const div = d3.select(events_div.node().children[idx]);
        div.attr('style', `background-color: ${schemeTableau10[5]}`);
        const [reserved, allocated] = memory_view.draw(idx);
        const enter = () => eventStack(div.datum(), allocated, reserved);
        stack_info.highlight(enter);
        div.node().scrollIntoViewIfNeeded(false);
      } else {
        memory_view.draw(0);
      }
      selected_event_idx = idx;
    },
  };
  d3.select('body').on('keydown', event => {
    const key = event.key;
    const actions = {ArrowDown: 1, ArrowUp: -1};
    if (selected_event_idx !== null && key in actions) {
      const new_idx = selected_event_idx + actions[key];
      es.select(Math.max(0, Math.min(new_idx, events.length - 1)));
      event.preventDefault();
    }
  });

  stack_info.register(
    events_selection,
    t => eventStack(t.datum()),
    _t => {},
    d => es.select(d.datum().idx),
  );

  return es;
}

function formatEvent(event) {
  const stream =
    event.stream === null ? '' : `\n              (stream ${event.stream})`;
  switch (event.action) {
    case 'oom':
      return `OOM (requested ${formatSize(event.size)}, Device has ${formatSize(
        event.device_free,
      )} memory free)${stream}`;
    case 'snapshot':
      return 'snapshot';
    default:
      return `${event.action.padEnd(14)} ${formatAddr(event).padEnd(
        18,
      )} ${formatSize(event.size)}${stream}`;
  }
}

function eventStack(e, allocated, reserved) {
  let event = formatEvent(e);
  if (reserved !== undefined) {
    event = `(${formatSize(allocated)} allocated / ${formatSize(
      reserved,
    )} reserved)\n${event}`;
  }
  const user_metadata_str = format_user_metadata(e.user_metadata);
  const frames_str = format_frames(e.frames);
  const forward_frames_str = format_forward_frames(e.forward_frames);
  return event + '\n' + (user_metadata_str ? user_metadata_str + '\n' : '') + frames_str + forward_frames_str;
}

function hashCode(num) {
  const numStr = num.toString();
  let hash = 0;
  for (let i = 0; i < numStr.length; i++) {
    const charCode = numStr.charCodeAt(i);
    hash = (hash << 5) - hash + charCode;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash;
}

function addStroke(d) {
  d.attr('stroke', 'red')
    .attr('stroke-width', '2')
    .attr('vector-effect', 'non-scaling-stroke');
}

function removeStroke(d) {
  d.attr('stroke', '');
}

function calculate_fragmentation(blocks, sorted_segments) {
  const sorted_blocks = Object.values(blocks).sort((a, b) => {
    // See Note [Sort BigInt and Number Safely]
    if (a.addr === b.addr) return 0;
    return a.addr < b.addr ? -1 : 1;
  });
  let block_i = 0;
  let total_size = 0;
  let sum_squared_free = 0;
  for (const seg of sorted_segments) {
    let addr = seg.addr;
    total_size += seg.size;
    // See Note [BigInt and Number Safe Arithmetic]
    const seg_end =
      seg.addr + (typeof seg.addr === "bigint" ? BigInt(seg.size) : seg.size);
    while (
      block_i < sorted_blocks.length &&
      sorted_blocks[block_i].addr < seg_end
    ) {
      const block = sorted_blocks[block_i];
      if (block.addr > addr) {
        sum_squared_free += Number(block.addr - addr) ** 2;
      }
      addr =
        block.addr + (typeof block.addr === "bigint" ? BigInt(block.size) : block.size);
      block_i += 1;
    }
    if (addr < seg_end) {
      sum_squared_free += Number(seg_end - addr) ** 2;
    }
  }
  console.log(sum_squared_free / total_size ** 2);
}

function MemoryView(outer, stack_info, snapshot, device) {
  const svg = outer
    .append('svg')
    .attr('style', 'grid-column: 2; grid-row: 1; width: 100%; height: 100%;')
    .attr('viewBox', '0 0 200 100')
    .attr('preserveAspectRatio', 'xMinYMin meet');
  const g = svg.append('g');
  const seg_zoom = zoom();
  seg_zoom.on('zoom', (event) => {
    g.attr('transform', event.transform);
  });
  svg.call(seg_zoom);

  const sorted_segments = [];
  const block_map = {};
  for (const seg of snapshot.segments) {
    if (seg.device !== device) {
      continue;
    }
    sorted_segments.push(
      Segment(
        seg.address,
        seg.total_size,
        seg.stream,
        seg.frames || [],
        seg.version,
        seg.user_metadata,
        seg.segment_pool_id,
      ),
    );
    for (const b of seg.blocks) {
      if (b.state !== 'active_pending_free' && b.state !== 'active_allocated') {
        continue;
      }
      block_map[b.addr] = Block(
        b.addr,
        b.size,
        b.requested_size,
        b.frames,
        b.state === 'active_pending_free',
        b.version,
        b.user_metadata,
      );
    }
  }
  sorted_segments.sort((x, y) => {
    // Note [Sort BigInt and Number Safely]
    // x.addr and y.addr may be BigInt, so subtracting them directly can cause
    // errors. Use explicit comparison instead to safely handle both BigInt and
    // Number.
    if (x.addr === y.addr) return 0;
    return x.addr < y.addr ? -1 : 1;
  });

  function simulate_memory(idx) {
    // create a copy of segments because we edit size properties below
    const l_segments = sorted_segments.map(x => {
      return {...x};
    });
    const l_block_map = {...block_map};

    function map_segment(merge, seg) {
      let idx = l_segments.findIndex(e => e.addr > seg.addr);
      if (!merge) {
        l_segments.splice(idx, 0, seg);
        return;
      }
      if (idx === -1) {
        idx = l_segments.length;
      }
      l_segments.splice(idx, 0, seg);
      if (idx + 1 < l_segments.length) {
        const next = l_segments[idx + 1];
        // See Note [BigInt and Number Safe Arithmetic]
        const seg_end =
          seg.addr + (typeof seg.addr === "bigint" ? BigInt(seg.size) : seg.size);
        if (seg_end === next.addr && seg.stream === next.stream) {
          seg.size += next.size;
          l_segments.splice(idx + 1, 1);
        }
      }
      if (idx > 0) {
        const prev = l_segments[idx - 1];
        const prev_end =
          prev.addr + (typeof prev.addr === "bigint" ? BigInt(prev.size) : prev.size);
        if (prev_end === seg.addr && prev.stream === seg.stream) {
          prev.size += seg.size;
          l_segments.splice(idx, 1);
        }
      }
    }
    function unmap_segment(merge, seg) {
      if (!merge) {
        l_segments.splice(
          l_segments.findIndex(x => x.addr === seg.addr),
          1,
        );
        return;
      }
      // See Note [BigInt and Number Safe Arithmetic]
      const seg_end =
        seg.addr + (typeof seg.addr === "bigint" ? BigInt(seg.size) : seg.size);
      const idx = l_segments.findIndex( e => {
        const e_end =
          e.addr + (typeof e.addr === "bigint" ? BigInt(e.size) : e.size);
        return e.addr <= seg.addr && seg_end <= e_end;
      });
      const existing = l_segments[idx];
      const existing_end =
        existing.addr + (typeof existing.addr === "bigint" ? BigInt(existing.size) : existing.size);
      if (existing.addr === seg.addr) {
        existing.addr += typeof existing.addr === "bigint" ? BigInt(seg.size) : seg.size;
        existing.size -= seg.size;
        if (existing.size === 0) {
          l_segments.splice(idx, 1);
        }
      } else if (existing_end === seg_end) {
        existing.size -= seg.size;
      } else {
        existing.size = Number(seg.addr - existing.addr);
        seg.addr = seg_end;
        seg.size = Number(existing_end - seg_end);
        l_segments.splice(idx + 1, 0, seg);
      }
    }
    const events = snapshot.device_traces[device];
    for (let i = events.length - 1; i > idx; i--) {
      const event = events[i];
      switch (event.action) {
        case 'free':
          l_block_map[event.addr] = Block(
            event.addr,
            event.size,
            event.size,
            event.frames,
            false,
            event.version,
            event.user_metadata,
          );
          break;
        case 'free_requested':
          l_block_map[event.addr].free_requested = false;
          break;
        case 'free_completed':
          l_block_map[event.addr] = Block(
            event.addr,
            event.size,
            event.size,
            event.frames,
            true,
            event.version,
            event.user_metadata,
          );
          break;
        case 'alloc':
          delete l_block_map[event.addr];
          break;
        case 'segment_free':
        case 'segment_unmap':
          map_segment(
            event.action === 'segment_unmap',
            Segment(
              event.addr,
              event.size,
              event.stream,
              event.frames,
              event.version,
              event.user_metadata,
            ),
          );
          break;
        case 'segment_alloc':
        case 'segment_map':
          unmap_segment(
            event.action === 'segment_map',
            Segment(
              event.addr,
              event.size,
              event.stream,
              event.frames,
              event.version,
              event.user_metadata,
            ),
          );
          break;
        case 'oom':
          break;
        default:
          break;
      }
    }
    const new_blocks = Object.values(l_block_map);
    return [l_segments, new_blocks];
  }

  return {
    draw(idx) {
      const [segments_unsorted, blocks] = simulate_memory(idx);
      g.selectAll('g').remove();

      const segment_d = g.append('g');
      const block_g = g.append('g');
      const block_r = g.append('g');

      segment_d.selectAll('rect').remove();
      block_g.selectAll('rect').remove();
      block_r.selectAll('rect').remove();
      const segments = [...segments_unsorted].sort((x, y) => {
        // See Note [Sort BigInt and Number Safely].
        if (x.size > y.size) return 1;
        if (x.size < y.size) return -1;
        if (x.addr > y.addr) return 1;
        if (x.addr < y.addr) return -1;
        return 0;
      });

      const segments_by_addr = [...segments].sort((x, y) => {
        // See Note [Sort BigInt and Number Safely]
        if (x.addr === y.addr) return 0;
        return x.addr < y.addr ? -1 : 1;
      });

      const max_size = segments.length === 0 ? 0 : segments.at(-1).size;

      const xScale = scaleLinear().domain([0, max_size]).range([0, 200]);
      const padding = xScale.invert(1);

      let cur_row = 0;
      let cur_row_size = 0;
      for (const seg of segments) {
        seg.occupied = 0;
        seg.internal_free = 0;
        if (cur_row_size + seg.size > max_size) {
          cur_row_size = 0;
          cur_row += 1;
        }
        seg.offset = cur_row_size;
        seg.row = cur_row;
        cur_row_size += seg.size + padding;
      }

      const num_rows = cur_row + 1;

      const yScale = scaleLinear().domain([0, num_rows]).range([0, 100]);

      const segments_selection = segment_d
        .selectAll('rect')
        .data(segments)
        .enter()
        .append('rect')
        .attr('x', x => xScale(x.offset))
        .attr('y', x => yScale(x.row))
        .attr('width', x => xScale(x.size))
        .attr('height', yScale(4 / 5))
        .attr('stroke', 'black')
        .attr('stroke-width', '1')
        .attr('vector-effect', 'non-scaling-stroke')
        .attr('fill', 'white');

      stack_info.register(
        segments_selection,
        d => {
          addStroke(d);
          const t = d.datum();
          const free = t.size - t.occupied;
          let internal = '';
          if (t.internal_free > 0) {
            internal = ` (${(t.internal_free / free) * 100}% internal)`;
          }
          const user_metadata_str = format_user_metadata(t.user_metadata);
          const frames_str = format_frames(t.frames);
          const forward_frames_str = format_forward_frames(t.forward_frames);
          let pool_str = '';
          if (isPrivatePoolId(t.segment_pool_id)) {
            pool_str = `, pool_id (${t.segment_pool_id[0]}, ${t.segment_pool_id[1]})`;
          }
          return (
            `s${t.addr.toString(16)}_${t.version}: segment ${formatSize(
              t.size,
            )} allocated, ` +
            `${formatSize(free)} free${internal} (stream ${
              t.stream
            }${pool_str})\n` +
            (user_metadata_str ? user_metadata_str + '\n' : '') +
            frames_str +
            forward_frames_str
          );
        },
        d => {
          d.attr('stroke', 'black')
            .attr('stroke-width', '1')
            .attr('vector-effect', 'non-scaling-stroke');
        },
      );

      function find_segment(addr) {
        let left = 0;
        let right = segments_by_addr.length - 1;
        while (left <= right) {
          const mid = Math.floor((left + right) / 2);
          const seg = segments_by_addr[mid];
          // See Note [BigInt and Number Safe Arithmetic]
          const seg_end =
            seg.addr + (typeof seg.addr === "bigint" ? BigInt(seg.size) : seg.size);
          if (addr < seg.addr) {
            right = mid - 1;
          } else if (addr >= seg_end) {
            left = mid + 1;
          } else {
            return seg;
          }
        }
        return null;
      }

      for (const b of blocks) {
        b.segment = find_segment(b.addr);
        b.segment.occupied += b.requested_size;
        b.segment.internal_free += b.size - b.requested_size;
      }

      const block_selection = block_g
        .selectAll('rect')
        .data(blocks)
        .enter()
        .append('rect')
        .attr('x', x => xScale(x.segment.offset + Number(x.addr - x.segment.addr)))
        .attr('y', x => yScale(x.segment.row))
        .attr('width', x => xScale(x.requested_size))
        .attr('height', yScale(4 / 5))
        .attr('fill', (x, _i) =>
          x.free_requested
            ? 'red'
            : schemeTableau10[
                Math.abs(hashCode(x.addr)) % schemeTableau10.length
              ],
        );

      stack_info.register(
        block_selection,
        d => {
          addStroke(d);
          const t = d.datum();
          let requested = '';
          if (t.free_requested) {
            requested = ' (block freed but waiting due to record_stream)';
          }
          const user_metadata_str = format_user_metadata(t.user_metadata);
          const frames_str = format_frames(t.frames);
          const forward_frames_str = format_forward_frames(t.forward_frames);
          let pool_str = '';
          if (isPrivatePoolId(t.segment?.segment_pool_id)) {
            pool_str = `, pool_id (${t.segment.segment_pool_id[0]}, ${t.segment.segment_pool_id[1]})`;
          }
          return (
            `b${t.addr.toString(16)}_${t.version} ` +
            `${formatSize(t.requested_size)} allocation${requested} (stream ${
              t.segment.stream
            }${pool_str})\n` +
            (user_metadata_str ? user_metadata_str + '\n' : '') +
            frames_str +
            forward_frames_str
          );
        },
        removeStroke,
      );

      const free_selection = block_r
        .selectAll('rect')
        .data(blocks)
        .enter()
        .append('rect')
        .attr('x', x =>
          xScale(
            x.segment.offset + Number(x.addr - x.segment.addr) + x.requested_size,
          ),
        )
        .attr('y', x => yScale(x.segment.row))
        .attr('width', x => xScale(x.size - x.requested_size))
        .attr('height', yScale(4 / 5))
        .attr('fill', (_x, _i) => 'red');

      stack_info.register(
        free_selection,
        d => {
          addStroke(d);
          const t = d.datum();
          const user_metadata_str = format_user_metadata(t.user_metadata);
          const frames_str = format_frames(t.frames);
          const forward_frames_str = format_forward_frames(t.forward_frames);
          return (
            `Free space lost due to rounding ${formatSize(
              t.size - t.requested_size,
            )}` +
            ` (stream ${t.segment.stream})\n` +
            (user_metadata_str ? user_metadata_str + '\n' : '') +
            frames_str +
            forward_frames_str
          );
        },
        removeStroke,
      );

      const reserved = segments.reduce((x, y) => x + y.size, 0);
      const allocated = blocks.reduce((x, y) => x + y.requested_size, 0);
      return [reserved, allocated];
    },
  };
}

function StackInfo(outer) {
  const stack_trace = outer
    .append('pre')
    .attr('style', 'grid-column: 1 / 3; grid-row: 2; overflow: auto');
  let selected = {
    enter: () => {
      stack_trace.text('');
    },
    leave: () => {},
  };
  return {
    register(dom, enter, leave = _e => {}, select = _e => {}) {
      if (getTraceInteractionMode() === 'hover') {
        // Hover mode: show on mouseover, pin on click
        dom
          .on('mouseover', function (event) {
            selected.leave();
            stack_trace.text(enter(d3.select(event.target)));
          })
          .on('mousedown', function (event) {
            const obj = d3.select(event.target);
            selected = {
              enter: () => stack_trace.text(enter(obj)),
              leave: () => leave(obj),
            };
            select(obj);
          })
          .on('mouseleave', function (event) {
            leave(d3.select(event.target));
            selected.enter();
          });
      } else {
        // Click mode: show only on click
        dom
          .on('click', function (event) {
            selected.leave();
            const obj = d3.select(event.target);
            selected = {
              enter: () => stack_trace.text(enter(obj)),
              leave: () => leave(obj),
            };
            stack_trace.text(enter(obj));
            select(obj);
          });
      }
    },
    highlight(enter, leave = () => {}) {
      selected = {enter: () => stack_trace.text(enter()), leave};
      selected.enter();
    },
  };
}

function create_segment_view(dst, snapshot, device) {
  const outer = dst
    .append('div')
    .attr(
      'style',
      'display: grid; grid-template-columns: 1fr 2fr; grid-template-rows: 2fr 1fr; height: 100%; gap: 10px',
    );

  const events = snapshot.device_traces[device];
  const stack_info = StackInfo(outer);
  const memory_view = MemoryView(outer, stack_info, snapshot, device);
  const event_selector = EventSelector(outer, events, stack_info, memory_view);

  window.requestAnimationFrame(function () {
    event_selector.select(events.length > 0 ? events.length - 1 : null);
  });
}

function annotate_snapshot(snapshot) {
  snapshot.segment_version = version_space();
  snapshot.block_version = version_space();
  snapshot.categories = [];
  const empty_list = [];
  let next_stream = 1;
  const stream_names = {0: 0};
  function stream_name(s) {
    if (!(s in stream_names)) {
      stream_names[s] = next_stream++;
    }
    return stream_names[s];
  }
  const new_traces = [];
  for (const device_trace of snapshot.device_traces) {
    const new_trace = [];
    new_traces.push(new_trace);
    for (const t of device_trace) {
      if (!('frames' in t)) {
        t.frames = empty_list;
      }
      // set unique version for each time an address is used
      // so that ctrl-f can be used to search for the beginning
      // and end of allocations and segments
      t.stream = stream_name(t.stream);
      switch (t.action) {
        case 'free_completed':
          t.version = snapshot.block_version(t.addr, true);
          if (new_trace.length > 0) {
            // elide free_requested/free_completed into a single event
            const prev = new_trace.at(-1);
            if (prev.action === 'free_requested' && prev.addr === t.addr) {
              prev.action = 'free';
              continue;
            }
          }
          break;
        case 'free_requested':
        case 'alloc':
          t.version = snapshot.block_version(t.addr, false);
          break;
        case 'segment_free':
        case 'segment_unmap':
          t.version = snapshot.segment_version(t.addr, true);
          break;
        case 'segment_alloc':
        case 'segment_map':
          t.version = snapshot.segment_version(t.addr, false);
          break;
        default:
          break;
      }
      if ('category' in t && !snapshot.categories.includes(t.category)) {
        snapshot.categories.push(t.category);
      }
      t.idx = new_trace.length;
      new_trace.push(t);
    }
  }
  snapshot.device_traces = new_traces;

  for (const seg of snapshot.segments) {
    seg.stream = stream_name(seg.stream);
    seg.version = snapshot.segment_version(seg.address, false);
    let addr = seg.address;
    for (const b of seg.blocks) {
      b.addr = addr;
      if (!('frames' in b)) {
        // legacy format where 'requested_size' may be missing
        // and frames might be in history rather than directly on block
        if ('history' in b) {
          b.frames = b.history[0].frames || empty_list;
          b.requested_size = b.requested_size || b.history[0].real_size;
        } else {
          b.frames = empty_list;
          b.requested_size = b.requested_size || b.size;
        }
      }
      b.version = snapshot.block_version(b.addr, false);
      b.segment_pool_id = seg.segment_pool_id;
      // Note [BigInt and Number Safe Arithmetic]
      // Device pointer addresses may be represented as either Number or BigInt.
      // Use explicit conversions to perform arithmetic safely and avoid mixing
      // BigInt and Number types, which would otherwise trigger JS type errors.
      addr += typeof addr === "bigint" ? BigInt(b.size) : b.size;
    }
  }

  if (
    snapshot.categories.length > 0 &&
    !snapshot.categories.includes('unknown')
  ) {
    snapshot.categores.push('unknown');
  }
}

function MemoryPlot(
  svg,
  data,
  left_pad,
  width,
  height,
  colors = schemeTableau10,
) {
  function format_points(d) {
    const size = d.size;
    const xs = d.timesteps.map(t => xscale(t));
    const bottom = d.offsets.map(t => yscale(t));
    const m = Array.isArray(size)
      ? (t, i) => yscale(t + size[i])
      : t => yscale(t + size);
    const top = d.offsets.map(m);
    const p0 = xs.map((x, i) => `${x},${bottom[i]}`);
    const p1 = xs.map((x, i) => `${x},${top[i]}`).reverse();
    return `${p0.join(' ')} ${p1.join(' ')}`;
  }

  const max_timestep = data.max_at_time.length;
  const max_size = data.max_size;

  const plot_width = width - left_pad;
  const plot_height = height;

  const yscale = scaleLinear().domain([0, max_size]).range([plot_height, 0]);
  // Use formatSize with showBytes=false for clean axis labels
  const yaxis = axisLeft(yscale).tickFormat(d => formatSize(d, false));
  const xscale = scaleLinear().domain([0, max_timestep]).range([0, plot_width]);
  const plot_coordinate_space = svg
    .append('g')
    .attr('transform', `translate(${left_pad}, ${0})`);
  const plot_outer = plot_coordinate_space.append('g');

  function view_rect(a) {
    return a
      .append('rect')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', plot_width)
      .attr('height', plot_height)
      .attr('fill', 'white');
  }

  view_rect(plot_outer);

  const cp = svg.append('clipPath').attr('id', 'clip');
  view_rect(cp);
  plot_outer.attr('clip-path', 'url(#clip)');

  const zoom_group = plot_outer.append('g');
  const scrub_group = zoom_group.append('g');

  const plot = scrub_group
    .selectAll('polygon')
    .data(data.allocations_over_time)
    .enter()
    .append('polygon')
    .attr('points', format_points)
    .attr('fill', d => colors[d.color % colors.length])
    .attr('opacity', d => d.opacity ?? 1)
    .attr('stroke', d => typeof d.elem === 'string' && d.elem.startsWith('pool:') ? 'black' : null)
    .attr('stroke-width', d => typeof d.elem === 'string' && d.elem.startsWith('pool:') ? 3 : null)
    .attr('vector-effect', d => typeof d.elem === 'string' && d.elem.startsWith('pool:') ? 'non-scaling-stroke' : null);

  const axis = plot_coordinate_space.append('g').call(yaxis);

  function handleZoom(event) {
    const t = event.transform;
    zoom_group.attr('transform', t);
    axis.call(yaxis.scale(event.transform.rescaleY(yscale)));
  }

  const thezoom = zoom().on('zoom', handleZoom);
  plot_outer.call(thezoom);

  return {
    select_window: (stepbegin, stepend, max) => {
      const begin = xscale(stepbegin);
      const size = xscale(stepend) - xscale(stepbegin);
      const scale = plot_width / size;
      const translate = -begin;
      const yscale = max_size / max;
      scrub_group.attr(
        'transform',
        `scale(${scale / yscale}, 1) translate(${translate}, 0)`,
      );
      plot_outer.call(
        thezoom.transform,
        zoomIdentity
          .scale(yscale)
          .translate(0, -(plot_height - plot_height / yscale)),
      );
    },
    set_delegate: delegate => {
      if (getTraceInteractionMode() === 'hover') {
        // Hover mode: show on mouseover, pin on click
        plot
          .on('mouseover', function (_e, _d) {
            delegate.set_selected(d3.select(this));
          })
          .on('mousedown', function (_e, _d) {
            delegate.default_selected = d3.select(this);
          })
          .on('mouseleave', function (_e, _d) {
            delegate.set_selected(delegate.default_selected);
          });
      } else {
        // Click mode: show only on click
        plot
          .on('click', function (_e, _d) {
            delegate.default_selected = d3.select(this);
            delegate.set_selected(d3.select(this));
          });
      }
    },
  };
}

function ContextViewer(text, data) {
  let current_selected = null;

  function restore_search_highlight(d) {
    if (!d) return;
    const addr = d.attr('data-search-match') === 'true';
    const frame = d.attr('data-frame-match') === 'true';
    if (addr && frame) {
      d.attr('stroke', '#ff00ff')
        .attr('stroke-width', 3)
        .attr('stroke-dasharray', '6,3')
        .attr('vector-effect', 'non-scaling-stroke');
    } else if (addr) {
      d.attr('stroke', 'red')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', null)
        .attr('vector-effect', 'non-scaling-stroke');
    } else if (frame) {
      d.attr('stroke', '#2196F3')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', null)
        .attr('vector-effect', 'non-scaling-stroke');
    }
  }

  return {
    default_selected: null,
    set_selected: d => {
      if (current_selected !== null) {
        const prev = current_selected.datum();
        const is_pool = prev && typeof prev.elem === 'string' && prev.elem.startsWith('pool:');
        current_selected
          .attr('stroke', is_pool ? 'black' : null)
          .attr('stroke-width', is_pool ? 3 : null)
          .attr('stroke-dasharray', null)
          .attr('vector-effect', is_pool ? 'non-scaling-stroke' : null);
        restore_search_highlight(current_selected);
      }
      if (d === null) {
        text.text('');
      } else {
        const dd = d.datum();
        if (dd.elem === 'summarized') {
          text.html(
            'Small tensors that were not plotted to cutdown on render time.\n' +
              'Use detail slider to see smaller allocations.',
          );
        } else if (typeof dd.elem === 'string' && dd.elem.startsWith('pool:')) {
          const pool_key = dd.elem.slice(5);
          const capacity = Array.isArray(dd.size) ? dd.size.at(-1) : dd.size;
          text.text(`Private Pool (${pool_key}): capacity ${formatSize(capacity)}`);
        } else {
          text.text(`${dd.elem} ${data.context_for_id(dd.elem)}`);
        }
        const is_pool_sel = typeof dd.elem === 'string' && dd.elem.startsWith('pool:');
        d.attr('stroke', 'black')
          .attr('stroke-width', is_pool_sel ? 5 : 1)
          .attr('vector-effect', 'non-scaling-stroke');
      }
      current_selected = d;
    },
  };
}

function MiniMap(mini_svg, plot, data, left_pad, width, height = 70) {
  const max_at_time = data.max_at_time;
  const plot_width = width - left_pad;
  const yscale = scaleLinear().domain([0, data.max_size]).range([height, 0]);
  const minixscale = scaleLinear()
    .domain([0, max_at_time.length])
    .range([left_pad, width]);

  const mini_points = [
    [max_at_time.length, 0],
    [0, 0],
  ];

  for (const [i, m] of max_at_time.entries()) {
    const [_lastx, lasty] = mini_points[mini_points.length - 1];
    if (m !== lasty) {
      mini_points.push([i, lasty]);
      mini_points.push([i, m]);
    } else if (i === max_at_time.length - 1) {
      mini_points.push([i, m]);
    }
  }

  let points = mini_points.map(([t, o]) => `${minixscale(t)}, ${yscale(o)}`);
  points = points.join(' ');
  mini_svg
    .append('polygon')
    .attr('points', points)
    .attr('fill', schemeTableau10[0]);

  const xscale = scaleLinear()
    .domain([0, max_at_time.length])
    .range([0, plot_width]);

  const brush = brushX();
  brush.extent([
    [left_pad, 0],
    [width, height],
  ]);
  brush.on('brush', function (event) {
    const [begin, end] = event.selection.map(x => x - left_pad);

    const stepbegin = Math.floor(xscale.invert(begin));
    const stepend = Math.floor(xscale.invert(end));
    let max = 0;
    for (let i = stepbegin; i < stepend; i++) {
      max = Math.max(max, max_at_time[i]);
    }
    plot.select_window(stepbegin, stepend, max);
  });
  mini_svg.call(brush);
  return {};
}

function Legend(plot_svg, categories) {
  const xstart = 100;
  const ystart = 5;
  plot_svg
    .append('g')
    .selectAll('rect')
    .data(categories)
    .enter()
    .append('rect')
    .attr('x', (c, i) => xstart)
    .attr('y', (c, i) => ystart + i * 15)
    .attr('width', 10)
    .attr('height', 10)
    .attr('fill', (c, i) => schemeTableau10[i % schemeTableau10.length]);
  plot_svg
    .append('g')
    .selectAll('text')
    .data(categories)
    .enter()
    .append('text')
    .attr('x', (c, i) => xstart + 20)
    .attr('y', (c, i) => ystart + i * 15 + 8)
    .attr('font-family', 'helvetica')
    .attr('font-size', 10)
    .text(c => c);
  return {};
}

function create_trace_view(
  dst,
  snapshot,
  device,
  plot_segments = false,
  max_entries = 15000,
  include_private_inactive = false,
) {
  const left_pad = 70;
  const data = process_alloc_data(snapshot, device, plot_segments, max_entries, include_private_inactive);
  dst.selectAll('svg').remove();
  dst.selectAll('div').remove();

  max_entries = Math.min(max_entries, data.elements_length);
  if (include_private_inactive) {
    dst.append('div')
      .attr('style', 'padding: 4px 8px; background: #fff3cd; border: 1px solid #ffc107; font-size: 13px; margin-bottom: 4px;')
      .text('Note: Private pool memory (the gray bar) is shown as allocated until the pool\'s segment is freed. '
          + 'This view requires that MemPools are not deleted before torch.cuda.memory._snapshot() is called.');
  }
  const d = dst.append('div');
  d.append('input')
    .attr('type', 'range')
    .attr('min', 0)
    .attr('max', data.elements_length)
    .attr('value', max_entries)
    .on('change', function () {
      create_trace_view(dst, snapshot, device, plot_segments, this.value, include_private_inactive);
    });
  d.append('label').text(
    `Detail: ${max_entries} of ${data.elements_length} entries`,
  );

  d.append('span').text('  |  ');
  const search_input = d.append('input')
    .attr('type', 'text')
    .attr('placeholder', 'Search address (hex)...')
    .attr('style', 'width: 180px; margin-left: 4px; font-family: monospace;');
  const search_label = d.append('label')
    .attr('style', 'margin-left: 4px;');

  d.append('span').text('  |  ');
  const frame_input = d.append('input')
    .attr('type', 'text')
    .attr('placeholder', 'Search stack frame...')
    .attr('style', 'width: 200px; margin-left: 4px; font-family: monospace;');
  const frame_label = d.append('label')
    .attr('style', 'margin-left: 4px;');

  const grid_container = dst
    .append('div')
    .attr(
      'style',
      'display: grid; grid-template-columns: 1fr; grid-template-rows: 10fr 1fr 8fr; flex: 1; min-height: 0; gap: 10px',
    );

  const plot_svg = grid_container
    .append('svg')
    .attr('display', 'block')
    .attr('viewBox', '0 0 1024 576')
    .attr('preserveAspectRatio', 'none')
    .attr('style', 'grid-column: 1; grid-row: 1; width: 100%; height: 100%;');

  const plot = MemoryPlot(plot_svg, data, left_pad, 1024, 576);

  if (snapshot.categories.length !== 0) {
    Legend(plot_svg.append('g'), snapshot.categories);
  }

  const mini_svg = grid_container
    .append('svg')
    .attr('display', 'block')
    .attr('viewBox', '0 0 1024 60')
    .attr('preserveAspectRatio', 'none')
    .attr('style', 'grid-column: 1; grid-row: 2; width: 100%; height: 100%;');

  MiniMap(mini_svg, plot, data, left_pad, 1024);
  const context_div = grid_container
    .append('div')
    .attr(
      'style',
      'grid-column: 1; grid-row: 3; width: 100%; height: 100%; min-height: 0; overflow: auto;',
    );
  const delegate = ContextViewer(context_div.append('pre').text('none'), data);
  plot.set_delegate(delegate);

  function apply_search_highlights() {
    const addr_query = search_input.node().value.toLowerCase().trim();
    const frame_query = frame_input.node().value.toLowerCase().trim();
    const polygons = plot_svg.selectAll('polygon');
    let addr_matches = 0;
    let frame_matches = 0;
    polygons.each(function () {
      const dd = d3.select(this).datum();
      if (!dd || typeof dd.elem !== 'number') {
        d3.select(this)
          .attr('data-search-match', null)
          .attr('data-frame-match', null);
        return;
      }
      const ctx = data.context_for_id(dd.elem);
      const ctx_lower = ctx.toLowerCase();
      const addr_hit = addr_query && ctx_lower.includes(addr_query);
      const frame_hit = frame_query && ctx_lower.includes(frame_query);
      d3.select(this)
        .attr('data-search-match', addr_hit ? 'true' : null)
        .attr('data-frame-match', frame_hit ? 'true' : null);
      if (addr_hit && frame_hit) {
        d3.select(this)
          .attr('stroke', '#ff00ff')
          .attr('stroke-width', 3)
          .attr('stroke-dasharray', '6,3')
          .attr('vector-effect', 'non-scaling-stroke');
      } else if (addr_hit) {
        d3.select(this)
          .attr('stroke', 'red')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', null)
          .attr('vector-effect', 'non-scaling-stroke');
      } else if (frame_hit) {
        d3.select(this)
          .attr('stroke', '#2196F3')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', null)
          .attr('vector-effect', 'non-scaling-stroke');
      } else {
        d3.select(this)
          .attr('stroke', null)
          .attr('stroke-width', null)
          .attr('stroke-dasharray', null);
      }
      if (addr_hit) addr_matches++;
      if (frame_hit) frame_matches++;
    });
    search_label.text(addr_query ? `${addr_matches} match${addr_matches !== 1 ? 'es' : ''}` : '');
    frame_label.text(frame_query ? `${frame_matches} match${frame_matches !== 1 ? 'es' : ''}` : '');
  }

  search_input.on('input', apply_search_highlights);
  frame_input.on('input', apply_search_highlights);
}

function create_settings_view(dst, snapshot, device) {
  dst.selectAll('svg').remove();
  dst.selectAll('div').remove();
  const settings_div = dst.append('div');
  settings_div.append('p').text('Caching Allocator Settings:');

  // Check if allocator_settings exists in snapshot
  if ('allocator_settings' in snapshot) {
    settings_div
      .append('pre')
      .text(JSON.stringify(snapshot.allocator_settings, null, 2));
  } else {
    settings_div.append('p').text('No allocator settings found.');
  }
}

function unpickle(buffer) {
  try {
    const decoder = new TextDecoder();
    const jsonString = decoder.decode(new Uint8Array(buffer));
    const data = JSON.parse(jsonString);

    return data;
  } catch (e) {
    console.log('Failed to decode the data as JSON, fall back to pickle', e);
  }
  return unpickleData(buffer);
}

function unpickleData(buffer) {
  const bytebuffer = new Uint8Array(buffer);
  const decoder = new TextDecoder();

  const stack = [];
  const marks = [];
  const memo = [];
  let offset = 0;
  let memo_id = 0;

  const APPENDS = 'e'.charCodeAt(0);
  const BINGET = 'h'.charCodeAt(0);
  const BININT = 'J'.charCodeAt(0);
  const BININT1 = 'K'.charCodeAt(0);
  const BININT2 = 'M'.charCodeAt(0);
  const EMPTY_DICT = '}'.charCodeAt(0);
  const EMPTY_LIST = ']'.charCodeAt(0);
  const FRAME = 0x95;
  const LONG1 = 0x8a;
  const LONG_BINGET = 'j'.charCodeAt(0);
  const MARK = '('.charCodeAt(0);
  const MEMOIZE = 0x94;
  const PROTO = 0x80;
  const SETITEMS = 'u'.charCodeAt(0);
  const SHORT_BINUNICODE = 0x8c;
  const STOP = '.'.charCodeAt(0);
  const TUPLE2 = 0x86;
  const APPEND = 'a'.charCodeAt(0);
  const NEWFALSE = 0x89;
  const BINPUT = 'q'.charCodeAt(0);
  const BINUNICODE = 'X'.charCodeAt(0);
  const EMPTY_TUPLE = ')'.charCodeAt(0);
  const NEWTRUE = 0x88;
  const NONE = 'N'.charCodeAt(0);
  const BINFLOAT = 'G'.charCodeAt(0);
  const TUPLE = 't'.charCodeAt(0);
  const TUPLE1 = 0x85;
  const TUPLE3 = 0x87;
  // untested
  const LONG_BINPUT = 'r'.charCodeAt(0);
  const LIST = 'l'.charCodeAt(0);
  const DICT = 'd'.charCodeAt(0);
  const SETITEM = 's'.charCodeAt(0);

  const scratch_buffer = new ArrayBuffer(8);
  const scratch_bytes = new Uint8Array(scratch_buffer);
  const big = new BigInt64Array(scratch_buffer);
  const float64 = new Float64Array(scratch_buffer);

  function read_uint4() {
    const n =
      bytebuffer[offset] +
      bytebuffer[offset + 1] * 256 +
      bytebuffer[offset + 2] * 65536 +
      bytebuffer[offset + 3] * 16777216;
    offset += 4;
    return n;
  }
  function setitems(d, mark) {
    for (let i = mark; i < stack.length; i += 2) {
      d[stack[i]] = stack[i + 1];
    }
    stack.splice(mark, Infinity);
  }

  while (true) {
    const opcode = bytebuffer[offset++];
    switch (opcode) {
      case PROTO:
        {
          const version = bytebuffer[offset++];
          if (version < 2 || version > 4) {
            throw new Error(`Unhandled version ${version}`);
          }
        }
        break;
      case APPEND:
        {
          const v = stack.pop();
          stack.at(-1).push(v);
        }
        break;
      case APPENDS:
        {
          const mark = marks.pop();
          const arr = stack[mark - 1];
          arr.push(...stack.splice(mark, Infinity));
        }
        break;
      case LIST:
      case TUPLE:
        {
          const mark = marks.pop();
          stack.push([...stack.splice(mark, Infinity)]);
        }
        break;
      case NEWFALSE:
        stack.push(false);
        break;
      case NEWTRUE:
        stack.push(true);
        break;
      case NONE:
        stack.push(null);
        break;
      case BINGET:
        stack.push(memo[bytebuffer[offset++]]);
        break;
      case BININT:
        {
          let i32 = read_uint4();
          if (i32 > 0x7fffffff) {
            i32 -= 0x100000000;
          }
          stack.push(i32);
        }
        break;
      case BININT1:
        stack.push(bytebuffer[offset++]);
        break;
      case BININT2:
        {
          const v = bytebuffer[offset] + bytebuffer[offset + 1] * 256;
          stack.push(v);
          offset += 2;
        }
        break;
      case EMPTY_DICT:
        stack.push({});
        break;
      case EMPTY_LIST:
        stack.push([]);
        break;
      case FRAME:
        offset += 8;
        break;
      case LONG1:
        {
          const s = bytebuffer[offset++];
          if (s <= 8) {
            for (let i = 0; i < s; i++) {
              scratch_bytes[i] = bytebuffer[offset++];
            }
            const fill = scratch_bytes[s - 1] >= 128 ? 0xff : 0x0;
            for (let i = s; i < 8; i++) {
              scratch_bytes[i] = fill;
            }
            stack.push(Number(big[0]));
          } else { // BigInt
            let scratch_bytes_unbounded = [];
            for (let i = 0; i < s; i++) {
              scratch_bytes_unbounded.push(bytebuffer[offset++]);
            }

            // BigInt can only convert from unsigned hex, thus we need to
            // convert from twos-complement if negative
            const negative = scratch_bytes_unbounded[s - 1] >= 128;
            if (negative) {
              // implements scratch_bytes_unbounded = ~scratch_bytes_unbounded + 1
              // byte-by-byte.
              let carry = 1;
              for (let i = 0; i < s; i++) {
                const twos_complement = (0xff ^ scratch_bytes_unbounded[i]) + carry;
                carry = twos_complement > 0xff ? 1 : 0;
                scratch_bytes_unbounded[i] = 0xff & twos_complement;
              }
            }

            const hex_str = Array.from(scratch_bytes_unbounded.reverse(), byte => {
              return byte.toString(16).padStart(2, '0');
            }).join('');

            const big_int = negative ? -BigInt(`0x${hex_str}`) : BigInt(`0x${hex_str}`);
            stack.push(big_int);
          }
        }
        break;
      case LONG_BINGET:
        {
          const idx = read_uint4();
          stack.push(memo[idx]);
        }
        break;
      case MARK:
        marks.push(stack.length);
        break;
      case MEMOIZE:
        memo[memo_id++] = stack.at(-1);
        break;
      case BINPUT:
        memo[bytebuffer[offset++]] = stack.at(-1);
        break;
      case LONG_BINPUT:
        memo[read_uint4()] = stack.at(-1);
        break;
      case SETITEMS:
        {
          const mark = marks.pop();
          const d = stack[mark - 1];
          setitems(d, mark);
        }
        break;
      case SETITEM: {
        const v = stack.pop();
        const k = stack.pop();
        stack.at(-1)[k] = v;
        break;
      }
      case DICT:
        {
          const mark = marks.pop();
          const d = {};
          setitems(d, mark);
          stack.push(d);
        }
        break;
      case SHORT_BINUNICODE:
        {
          const n = bytebuffer[offset++];
          stack.push(decoder.decode(new Uint8Array(buffer, offset, n)));
          offset += n;
        }
        break;
      case BINUNICODE:
        {
          const n = read_uint4();
          stack.push(decoder.decode(new Uint8Array(buffer, offset, n)));
          offset += n;
        }
        break;
      case STOP:
        return stack.pop();
      case EMPTY_TUPLE:
        stack.push([]);
        break;
      case TUPLE1:
        stack.push([stack.pop()]);
        break;
      case TUPLE2:
        stack.push(stack.splice(-2, Infinity));
        break;
      case TUPLE3:
        stack.push(stack.splice(-3, Infinity));
        break;
      case BINFLOAT:
        for (let i = 7; i >= 0; i--) {
          // stored in big-endian order
          scratch_bytes[i] = bytebuffer[offset++];
        }
        stack.push(float64[0]);
        break;
      default:
        throw new Error(`UNKNOWN OPCODE: ${opcode}`);
    }
  }
}

function decode_base64(input) {
  function decode_char(i, shift) {
    const nChr = input.charCodeAt(i);
    const r =
      nChr > 64 && nChr < 91
        ? nChr - 65
        : nChr > 96 && nChr < 123
        ? nChr - 71
        : nChr > 47 && nChr < 58
        ? nChr + 4
        : nChr === 43
        ? 62
        : nChr === 47
        ? 63
        : 0;
    return r << shift;
  }
  const output = new Uint8Array((input.length / 4) * 3);
  for (let i = 0, j = 0; i < input.length; i += 4, j += 3) {
    const u24 =
      decode_char(i, 18) +
      decode_char(i + 1, 12) +
      decode_char(i + 2, 6) +
      decode_char(i + 3);
    output[j] = u24 >> 16;
    output[j + 1] = (u24 >> 8) & 0xff;
    output[j + 2] = u24 & 0xff;
  }
  return output.buffer;
}

const kinds = {
  'Active Memory Timeline': create_trace_view,
  'Allocated Memory (incl. Private Pools)': (dst, snapshot, device) =>
    create_trace_view(dst, snapshot, device, false, 15000, true),
  'Allocator State History': create_segment_view,
  'Active Cached Segment Timeline': (dst, snapshot, device) =>
    create_trace_view(dst, snapshot, device, true),
  'Allocator Settings': create_settings_view,
};

const snapshot_cache = {};
const snapshot_to_loader = {};
const snapshot_to_url = {};
const selection_to_div = {};

const style = `
pre {
  margin: 0px;
}
html, body {
  height: 100%;
  margin: 0;
  overflow: clip;
}
body {
  display: flex;
  flex-direction: column;
}`;

const head = d3.select('head');
head.append('style').text(style);
const body = d3.select('body');
const controls = body.append('div');
const snapshot_select = controls.append('select');
const view = controls.append('select');
for (const x in kinds) {
  view.append('option').text(x);
}
const gpu = controls.append('select');

// Add interaction mode toggle (hover vs click)
const interactionLabel = body.append('label')
  .attr('style', 'margin-left: 15px; cursor: pointer;');
const interactionCheckbox = interactionLabel.append('input')
  .attr('type', 'checkbox')
  .attr('id', 'interaction-mode-toggle')
  .attr('style', 'cursor: pointer; margin-right: 5px;');
interactionLabel.append('span').text('Require click to show trace (applies on file load)');

interactionCheckbox.on('change', function() {
  const mode = this.checked ? 'click' : 'hover';
  setTraceInteractionMode(mode);
  // Only refresh the view if a snapshot is already loaded
  if (snapshot_select.node().value) {
    selected_change();
  }
});

function unpickle_and_annotate(data) {
  data = unpickle(data);
  console.log(data);
  annotate_snapshot(data);
  return data;
}

function snapshot_change(f) {
  const view_value = view.node().value;
  let no_starting_gpu = gpu.node().value == '';
  let device = Number(gpu.node().value);
  const snapshot = snapshot_cache[f];
  gpu.selectAll('option').remove();
  const has_segments = {};
  for (const s of snapshot.segments) {
    has_segments[s.device] = true;
  }
  let device_valid = false;
  let maxTraceLength = -1;
  let defaultDevice = null;
  for (const [i, trace] of snapshot.device_traces.entries()) {
    if (trace.length > 0 || i in has_segments) {
      gpu.append('option').text(i);
      if (trace.length > maxTraceLength) {
        maxTraceLength = trace.length;
        defaultDevice = i;
      }
      if (i === device) {
        device_valid = true;
        gpu.node().selectedIndex = gpu.node().children.length - 1;
      }
    }
  }
  if (!device_valid) {
    device = Number(gpu.node().value);
  }

  if (no_starting_gpu) {
    device = defaultDevice;
    gpu.node().value = device;
  }

  const key = [f, view_value, device];
  if (!(key in selection_to_div)) {
    selection_to_div[key] = d3.select('body').append('div');
    kinds[view_value](selection_to_div[key], snapshot, device);
  }
  const selected_div = selection_to_div[key];

  selected_div.attr('style', 'display: flex; flex-direction: column; flex: 1; min-height: 0');
}

function selected_change() {
  for (const d of Object.values(selection_to_div)) {
    d.attr('style', 'display: none; flex-direction: column; flex: 1; min-height: 0');
  }
  const f = snapshot_select.node().value;
  if (f === '') {
    return;
  }
  if (!(f in snapshot_cache)) {
    snapshot_to_loader[f](f);
  } else {
    snapshot_change(f);
  }
}

snapshot_select.on('change', selected_change);
view.on('change', selected_change);
gpu.on('change', selected_change);

body.on('dragover', e => {
  event.preventDefault();
});

body.on('drop', () => {
  console.log(event.dataTransfer.files);
  Array.from(event.dataTransfer.files).forEach(file => {
    add_snapshot(file.name, unique_name => {
      const reader = new FileReader();
      reader.onload = e => {
        finished_loading(unique_name, e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
  });
  event.preventDefault();
  snapshot_select.node().selectedIndex =
    snapshot_select.node().options.length - 1;
  selected_change();
});

selection_to_div[''] = body
  .append('div')
  .text(
    'Drag and drop or select a file to load a local snapshot. No data from the snapshot is uploaded.',
  );

const fileInput = body.append('input')
  .attr('type', 'file')
  .attr('multiple', true)    // allow several snapshots at once
  .style('margin-left', '8px')
  .on('change', function () {
    Array.from(this.files).forEach(file => {
      add_snapshot(file.name, unique_name => {
        const reader = new FileReader();
        reader.onload = e =>
          finished_loading(unique_name, e.target.result);
        reader.readAsArrayBuffer(file);
      });
    });
    this.value = null;                       // reset so the same file can be picked again
    snapshot_select.node().selectedIndex =
      snapshot_select.node().options.length - 1;
    selected_change();                       // refresh the UI
  });

let next_unique_n = 1;
function add_snapshot(name, loader) {
  if (name in snapshot_to_loader) {
    name = `${name} (${next_unique_n++})`;
  }
  snapshot_select.append('option').text(name);
  snapshot_to_loader[name] = loader;
}

function finished_loading(name, data) {
  snapshot_cache[name] = unpickle_and_annotate(data);
  snapshot_change(name);
}

export function add_remote_files(files) {
  files.forEach(f =>
    add_snapshot(f.name, unique_name => {
      console.log('fetching', f.url);
      fetch(f.url)
        .then(x => x.arrayBuffer())
        .then(data => finished_loading(unique_name, data));
    }),
  );
  if (files.length > 0) {
    selected_change();
  }
}

export function add_local_files(files, view_value) {
  view.node().value = view_value;
  files.forEach(f =>
    add_snapshot(f.name, unique_name => {
      finished_loading(unique_name, decode_base64(f.base64));
    }),
  );
  if (files.length > 0) {
    selected_change();
  }
}

// Export configuration functions for external use
export { setTraceInteractionMode, getTraceInteractionMode };
