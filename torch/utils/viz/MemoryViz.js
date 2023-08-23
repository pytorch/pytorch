'use strict';

import * as d3 from "https://cdn.skypack.dev/d3@5";
import {axisLeft} from "https://cdn.skypack.dev/d3-axis@1";
import {scaleLinear} from "https://cdn.skypack.dev/d3-scale@1";
import {zoom, zoomIdentity} from "https://cdn.skypack.dev/d3-zoom@1";
import {brushX} from "https://cdn.skypack.dev/d3-brush@1";

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

function Segment(addr, size, stream, frames, version) {
  return {addr, size, stream, version, frames};
}

function Block(addr, size, real_size, frames, free_requested, version) {
  return {addr, size, real_size, frames, free_requested, version};
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
  d3.select('body').on('keydown', _e => {
    const key = d3.event.key;
    const actions = {ArrowDown: 1, ArrowUp: -1};
    if (selected_event_idx !== null && key in actions) {
      const new_idx = selected_event_idx + actions[key];
      es.select(Math.max(0, Math.min(new_idx, events.length - 1)));
      d3.event.preventDefault();
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

function formatSize(num) {
  const orig = num;
  // https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
  const units = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'];
  for (const unit of units) {
    if (Math.abs(num) < 1024.0) {
      return `${num.toFixed(1)}${unit}B (${orig} bytes)`;
    }
    num /= 1024.0;
  }
  return `${num.toFixed(1)}YiB`;
}
function formatAddr(event) {
  const prefix = event.action.startsWith('segment') ? 's' : 'b';
  return `${prefix}${event.addr.toString(16)}_${event.version}`;
}
function formatEvent(event) {
  const stream =
    event.stream === 0 ? '' : `\n              (stream ${event.stream})`;
  switch (event.action) {
    case 'oom':
      return `OOM (requested ${formatSize(event.size)}, CUDA has ${formatSize(
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
  return event + '\n' + format_frames(e.frames);
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
  const sorted_blocks = Object.values(blocks).sort((a, b) => a.addr - b.addr);
  let block_i = 0;
  let total_size = 0;
  let sum_squared_free = 0;
  for (const seg of sorted_segments) {
    let addr = seg.addr;
    total_size += seg.size;
    while (
      block_i < sorted_blocks.length &&
      sorted_blocks[block_i].addr < seg.addr + seg.size
    ) {
      const block = sorted_blocks[block_i];
      if (block.addr > addr) {
        sum_squared_free += (block.addr - addr) ** 2;
      }
      addr = block.addr + block.size;
      block_i += 1;
    }
    if (addr < seg.addr + seg.size) {
      sum_squared_free += (seg.addr + seg.size - addr) ** 2;
    }
  }
  console.log(sum_squared_free / (total_size**2))
}

function MemoryView(outer, stack_info, snapshot, device) {
  const svg = outer
    .append('svg')
    .attr('style', 'grid-column: 2; grid-row: 1; width: 100%; height: 100%;')
    .attr('viewBox', '0 0 200 100')
    .attr('preserveAspectRatio', 'xMinYMin meet');
  const g = svg.append('g');
  const seg_zoom = zoom();
  seg_zoom.on('zoom', () => {
    g.attr('transform', d3.event.transform);
  });
  svg.call(seg_zoom);

  const sorted_segments = [];
  const block_map = {};
  for (const seg of snapshot.segments) {
    if (seg.device !== device) {
      continue;
    }
    sorted_segments.push(
      Segment(seg.address, seg.total_size, seg.stream, [], seg.version),
    );
    for (const b of seg.blocks) {
      if (b.state !== 'active_pending_free' && b.state !== 'active_allocated') {
        continue;
      }
      block_map[b.addr] = Block(
        b.addr,
        b.size,
        b.real_size,
        b.frames,
        b.state === 'active_pending_free',
        b.version,
      );
    }
  }
  sorted_segments.sort((x, y) => x.addr - y.addr);

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
        if (seg.addr + seg.size === next.addr && seg.stream === next.stream) {
          seg.size += next.size;
          l_segments.splice(idx + 1, 1);
        }
      }
      if (idx > 0) {
        const prev = l_segments[idx - 1];
        if (prev.addr + prev.size === seg.addr && prev.stream === seg.stream) {
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
      const seg_end = seg.addr + seg.size;
      const idx = l_segments.findIndex(
        e => e.addr <= seg.addr && seg_end <= e.addr + e.size,
      );
      const existing = l_segments[idx];
      const existing_end = existing.addr + existing.size;
      if (existing.addr === seg.addr) {
        existing.addr += seg.size;
        existing.size -= seg.size;
        if (existing.size === 0) {
          l_segments.splice(idx, 1);
        }
      } else if (existing_end === seg_end) {
        existing.size -= seg.size;
      } else {
        existing.size = seg.addr - existing.addr;
        seg.addr = seg_end;
        seg.size = existing_end - seg_end;
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
      const segments = [...segments_unsorted].sort((x, y) =>
        x.size === y.size ? x.addr - y.addr : x.size - y.size,
      );

      const segments_by_addr = [...segments].sort((x, y) => x.addr - y.addr);

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
          return (
            `s${t.addr.toString(16)}_${t.version}: segment ${formatSize(
              t.size,
            )} allocated, ` +
            `${formatSize(free)} free${internal} (stream ${
              t.stream
            })\n${format_frames(t.frames)}`
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
          if (addr < segments_by_addr[mid].addr) {
            right = mid - 1;
          } else if (
            addr >=
            segments_by_addr[mid].addr + segments_by_addr[mid].size
          ) {
            left = mid + 1;
          } else {
            return segments_by_addr[mid];
          }
        }
        return null;
      }

      for (const b of blocks) {
        b.segment = find_segment(b.addr);
        b.segment.occupied += b.real_size;
        b.segment.internal_free += b.size - b.real_size;
      }

      const block_selection = block_g
        .selectAll('rect')
        .data(blocks)
        .enter()
        .append('rect')
        .attr('x', x => xScale(x.segment.offset + (x.addr - x.segment.addr)))
        .attr('y', x => yScale(x.segment.row))
        .attr('width', x => xScale(x.real_size))
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
          return (
            `b${t.addr.toString(16)}_${t.version} ` +
            `${formatSize(t.real_size)} allocation${requested} (stream ${
              t.segment.stream
            })\n` +
            format_frames(t.frames)
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
          xScale(x.segment.offset + (x.addr - x.segment.addr) + x.real_size),
        )
        .attr('y', x => yScale(x.segment.row))
        .attr('width', x => xScale(x.size - x.real_size))
        .attr('height', yScale(4 / 5))
        .attr('fill', (_x, _i) => 'red');

      stack_info.register(
        free_selection,
        d => {
          addStroke(d);
          const t = d.datum();
          return (
            `Free space lost due to rounding ${formatSize(
              t.size - t.real_size,
            )}` +
            ` (stream ${t.segment.stream})\n` +
            format_frames(t.frames)
          );
        },
        removeStroke,
      );

      const reserved = segments.reduce((x, y) => x + y.size, 0);
      const allocated = blocks.reduce((x, y) => x + y.real_size, 0);
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
      dom
        .on('mouseover', _e => {
          selected.leave();
          stack_trace.text(enter(d3.select(d3.event.target)));
        })
        .on('mousedown', _e => {
          const obj = d3.select(d3.event.target);
          selected = {
            enter: () => stack_trace.text(enter(obj)),
            leave: () => leave(obj),
          };
          select(obj);
        })
        .on('mouseleave', _e => {
          leave(d3.select(d3.event.target));
          selected.enter();
        });
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
        snapshot.categories.push(t.category)
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
      if ('history' in b) {
        b.frames = b.history[0].frames || empty_list;
        b.real_size = b.history[0].real_size;
      } else {
        b.frames = empty_list;
        b.real_size = b.requested_size || b.size;
      }
      b.version = snapshot.block_version(b.addr, false);
      addr += b.size;
    }
  }

  if (snapshot.categories.length > 0 && !snapshot.categories.includes('unknown')) {
    snapshot.categores.push('unknown')
  }
}

function elideRepeats(frames) {
  const result = [];
  const length = frames.length;
  for (let i = 0; i < length; ) {
    let j = i + 1;
    const f = frames[i];
    while (j < length && f === frames[j]) {
      j++;
    }
    switch (j - i) {
      case 1:
        result.push(f);
        break;
      case 2:
        result.push(f, f);
        break;
      default:
        result.push(f, `<repeats ${j - i - 1} times>`);
        break;
    }
    i = j;
  }
  return result;
}
function frameFilter({name, filename}) {
  const omitFunctions = [
    'unwind::unwind',
    'CapturedTraceback::gather',
    'gather_with_cpp',
    '_start',
    '__libc_start_main',
    'PyEval_',
    'PyObject_',
    'PyFunction_',
  ];

  const omitFilenames = [
    'core/boxing',
    '/Register',
    '/Redispatch',
    'pythonrun.c',
    'Modules/main.c',
    'Objects/call.c',
    'Objects/methodobject.c',
    'pycore_ceval.h',
    'ceval.c',
    'cpython/abstract.h',
  ];

  for (const of of omitFunctions) {
    if (name.includes(of)) {
      return false;
    }
  }

  for (const of of omitFilenames) {
    if (filename.includes(of)) {
      return false;
    }
  }

  return true;
}

function format_frames(frames) {
  if (frames.length === 0) {
    return `<block was allocated before _record_history was enabled>`;
  }
  const frame_strings = frames
    .filter(frameFilter)
    .map(f => `${f.filename}:${f.line}:${f.name}`);
  return elideRepeats(frame_strings).join('\n');
}

function process_alloc_data(snapshot, device, plot_segments, max_entries) {
  const elements = [];
  const initially_allocated = [];
  const actions = [];
  const addr_to_alloc = {};

  const alloc = plot_segments ? 'segment_alloc' : 'alloc';
  const [free, free_completed] = plot_segments
    ? ['segment_free', 'segment_free']
    : ['free', 'free_completed'];
  for (const e of snapshot.device_traces[device]) {
    switch (e.action) {
      case alloc:
        elements.push(e);
        addr_to_alloc[e.addr] = elements.length - 1;
        actions.push(elements.length - 1);
        break;
      case free:
      case free_completed:
        if (e.addr in addr_to_alloc) {
          actions.push(addr_to_alloc[e.addr]);
          delete addr_to_alloc[e.addr];
        } else {
          elements.push(e);
          initially_allocated.push(elements.length - 1);
          actions.push(elements.length - 1);
        }
        break;
      default:
        break;
    }
  }
  for (const seg of snapshot.segments) {
    if (seg.device !== device) {
      continue;
    }
    if (plot_segments) {
      if (!(seg.address in addr_to_alloc)) {
        const element = {
          action: 'alloc',
          addr: seg.address,
          size: seg.total_size,
          frames: [],
          stream: seg.stream,
          version: seg.version,
        };
        elements.push(element);
        initially_allocated.push(elements.length - 1);
      }
    } else {
      for (const b of seg.blocks) {
        if (b.state === 'active_allocated' && !(b.addr in addr_to_alloc)) {
          const element = {
            action: 'alloc',
            addr: b.addr,
            size: b.real_size,
            frames: b.frames,
            stream: seg.stream,
            version: b.version,
          };
          elements.push(element);
          initially_allocated.push(elements.length - 1);
        }
      }
    }
  }
  initially_allocated.reverse();
  // if there are no actions, the graph will be blank,
  // but if there are existing allocations we do not want to hide them
  // by having just one allocate action it will show a flat graph with all segments
  if (actions.length === 0 && initially_allocated.length > 0) {
    actions.push(initially_allocated.pop());
  }

  const current = [];
  const current_data = [];
  const data = [];
  let max_size = 0;

  let total_mem = 0;
  let total_summarized_mem = 0;
  let timestep = 0;

  const max_at_time = [];

  const summarized_mem = {
    elem: 'summarized',
    timesteps: [],
    offsets: [total_mem],
    size: [],
    color: 0,
  };
  const summarized_elems = {};

  function advance(n) {
    summarized_mem.timesteps.push(timestep);
    summarized_mem.offsets.push(total_mem);
    summarized_mem.size.push(total_summarized_mem);
    timestep += n;
    for (let i = 0; i < n; i++) {
      max_at_time.push(total_mem + total_summarized_mem);
    }
  }

  const sizes = elements
    .map((x, i) => [x.size, i])
    .sort(([x, _xi], [y, _yi]) => y - x);

  const draw_elem = {};
  for (const [_s, e] of sizes.slice(0, max_entries)) {
    draw_elem[e] = true;
  }

  function add_allocation(elem) {
    let element_obj = elements[elem]
    const size = element_obj.size;
    current.push(elem);
    let color = elem
    if (snapshot.categories.length > 0) {
      color = snapshot.categories.indexOf(element_obj.category || 'unknown')
    }
    const e = {
      elem,
      timesteps: [timestep],
      offsets: [total_mem],
      size,
      color: color,
    };
    current_data.push(e);
    data.push(e);
    total_mem += size;
  }

  for (const elem of initially_allocated) {
    if (elem in draw_elem) {
      add_allocation(elem);
    } else {
      total_summarized_mem += elements[elem].size;
      summarized_elems[elem] = true;
    }
  }

  for (const elem of actions) {
    const size = elements[elem].size;
    if (!(elem in draw_elem)) {
      if (elem in summarized_elems) {
        advance(1);
        total_summarized_mem -= size;
        summarized_elems[elem] = null;
      } else {
        total_summarized_mem += size;
        summarized_elems[elem] = true;
        advance(1);
      }
      continue;
    }
    const idx = current.findLastIndex(x => x === elem);
    // first time we see an action we add it
    // second time we remove it
    if (idx === -1) {
      add_allocation(elem);
      advance(1);
    } else {
      advance(1);
      const removed = current_data[idx];
      removed.timesteps.push(timestep);
      removed.offsets.push(removed.offsets.at(-1));
      current.splice(idx, 1);
      current_data.splice(idx, 1);

      if (idx < current.length) {
        for (let j = idx; j < current.length; j++) {
          const e = current_data[j];
          e.timesteps.push(timestep);
          e.offsets.push(e.offsets.at(-1));
          e.timesteps.push(timestep + 3);
          e.offsets.push(e.offsets.at(-1) - size);
        }
        advance(3);
      }
      total_mem -= size;
    }
    max_size = Math.max(total_mem + total_summarized_mem, max_size);
  }

  for (const elem of current_data) {
    elem.timesteps.push(timestep);
    elem.offsets.push(elem.offsets.at(-1));
  }
  data.push(summarized_mem);

  return {
    max_size,
    allocations_over_time: data,
    max_at_time,
    summarized_mem,
    elements_length: elements.length,
    context_for_id: id => {
      const elem = elements[id];
      let text = `${formatAddr(elem)} ${formatSize(elem.size)} allocation (${
        elem.size
      } bytes)`;
      if (elem.stream !== 0) {
        text = `${text}, stream ${elem.stream}`;
      }
      if (!elem.action.includes('alloc')) {
        text = `${text}\nalloc not recorded, stack trace for free:`;
      }
      text = `${text}\n${format_frames(elem.frames)}`;
      return text;
    },
  };
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
  const yaxis = axisLeft(yscale).tickFormat(d3.format('.3s'));
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
    .attr('fill', d => colors[d.color % colors.length]);

  const axis = plot_coordinate_space.append('g').call(yaxis);

  function handleZoom() {
    const t = d3.event.transform;
    zoom_group.attr('transform', t);
    axis.call(yaxis.scale(d3.event.transform.rescaleY(yscale)));
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
    },
  };
}

function ContextViewer(text, data) {
  let current_selected = null;

  return {
    default_selected: null,
    set_selected: d => {
      if (current_selected !== null) {
        current_selected.attr('stroke', null).attr('stroke-width', null);
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
        } else {
          text.text(`${dd.elem} ${data.context_for_id(dd.elem)}`);
        }
        d.attr('stroke', 'black')
          .attr('stroke-width', 1)
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
  brush.on('brush', function () {
    const [begin, end] = d3.event.selection.map(x => x - left_pad);

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
    let xstart = 100
    let ystart = 5
    plot_svg.append('g').selectAll('rect')
    .data(categories)
    .enter()
    .append('rect')
    .attr('x', (c, i) => xstart)
    .attr('y', (c, i) => ystart + i*15)
    .attr('width', 10)
    .attr('height', 10)
    .attr('fill', (c, i) => schemeTableau10[i % schemeTableau10.length])
    plot_svg.append('g').selectAll('text')
    .data(categories)
    .enter()
    .append('text')
    .attr('x', (c, i) => xstart + 20)
    .attr('y', (c, i) => ystart + i*15 + 8)
    .attr("font-family", "helvetica")
    .attr('font-size', 10)
    .text((c) => c)
    return {}
}

function create_trace_view(
  dst,
  snapshot,
  device,
  plot_segments = false,
  max_entries = 15000,
) {
  const left_pad = 70;
  const data = process_alloc_data(snapshot, device, plot_segments, max_entries);
  dst.selectAll('svg').remove();
  dst.selectAll('div').remove();

  const d = dst.append('div');
  d.append('input')
    .attr('type', 'range')
    .attr('min', 0)
    .attr('max', data.elements_length)
    .attr('value', max_entries)
    .on('change', function () {
      create_trace_view(dst, snapshot, device, plot_segments, this.value);
    });
  d.append('label').text('Detail');

  const grid_container = dst
    .append('div')
    .attr(
      'style',
      'display: grid; grid-template-columns: 1fr; grid-template-rows: 10fr 1fr 8fr; height: 100%; gap: 10px',
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
      'grid-column: 1; grid-row: 3; width: 100%; height: 100%; overflow: auto;',
    );
  const delegate = ContextViewer(context_div.append('pre').text('none'), data);
  plot.set_delegate(delegate);
}

function unpickle(buffer) {
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
          if (s > 8) {
            throw new Error(`Unsupported number bigger than 8 bytes ${s}`);
          }
          for (let i = 0; i < s; i++) {
            scratch_bytes[i] = bytebuffer[offset++];
          }
          const fill = scratch_bytes[s - 1] >= 128 ? 0xff : 0x0;
          for (let i = s; i < 8; i++) {
            scratch_bytes[i] = fill;
          }
          stack.push(Number(big[0]));
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
    const nChr = input.charCodeAt(i)
    const r = nChr > 64 && nChr < 91
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
    return r << shift
  }
  let output = new Uint8Array(input.length / 4 * 3)
  for (let i = 0, j = 0; i < input.length; i += 4, j += 3) {
      let u24 = decode_char(i, 18) + decode_char(i + 1, 12) + decode_char(i + 2, 6) + decode_char(i + 3)
    output[j] = u24 >> 16
    output[j+1] = (u24 >> 8) & 0xFF
    output[j+2] = u24 & 0xFF;
  }
  return output.buffer
}

const kinds = {
  'Active Memory Timeline': create_trace_view,
  'Allocator State History': create_segment_view,
  'Active Cached Segment Timeline': (dst, snapshot, device) =>
    create_trace_view(dst, snapshot, device, true),
};

const snapshot_cache = {};
const snapshot_to_loader = {};
const snapshot_to_url = {};
const selection_to_div = {};

const body = d3.select('body');
const snapshot_select = body.append('select');
const view = body.append('select');
for (const x in kinds) {
  view.append('option').text(x);
}
const gpu = body.append('select');

function unpickle_and_annotate(data) {
  data = unpickle(data);
  console.log(data);
  annotate_snapshot(data);
  return data;
}

function snapshot_change(f) {
  const view_value = view.node().value;
  let device = Number(gpu.node().value);
  const snapshot = snapshot_cache[f];
  gpu.selectAll('option').remove();
  const has_segments = {};
  for (const s of snapshot.segments) {
    has_segments[s.device] = true;
  }
  let device_valid = false;
  for (const [i, trace] of snapshot.device_traces.entries()) {
    if (trace.length > 0 || i in has_segments) {
      gpu.append('option').text(i);
      if (i === device) {
        device_valid = true;
        gpu.node().selectedIndex = gpu.node().children.length - 1;
      }
    }
  }
  if (!device_valid) {
    device = Number(gpu.node().value);
  }
  const key = [f, view_value, device];
  if (!(key in selection_to_div)) {
    selection_to_div[key] = d3.select('body').append('div');
    kinds[view_value](selection_to_div[key], snapshot, device);
  }
  const selected_div = selection_to_div[key];

  selected_div.attr('style', 'display: float; height: 100%');
}

function selected_change() {
  for (const d of Object.values(selection_to_div)) {
    d.attr('style', 'display: none; height: 100%');
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
    add_snapshot(file.name, (unique_name) => {
      const reader = new FileReader();
      reader.onload = e => {
        finished_loading(unique_name, e.target.result);
      };
      reader.readAsArrayBuffer(file);
    });
  });
  event.preventDefault();
  snapshot_select.node().selectedIndex = snapshot_select.node().options.length - 1;
  selected_change();
});

selection_to_div[''] = body
  .append('div')
  .text('Drag and drop a file to load a local snapshot. No data from the snapshot is uploaded.');

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
    add_snapshot(f.name, (unique_name) => {
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
  view.node().value = view_value
  files.forEach(f =>
    add_snapshot(f.name, (unique_name) => {
      finished_loading(unique_name, decode_base64(f.base64))
    }),
  );
  if (files.length > 0) {
    selected_change();
  }
}
