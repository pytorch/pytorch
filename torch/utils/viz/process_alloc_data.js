// Pure data-processing functions for PyTorch memory visualization.
// Extracted from MemoryViz.js so they can be tested independently (no d3/DOM deps).
//
// This file is the single source of truth for these functions:
//   - MemoryViz.js imports them via ESM: import {...} from "./process_alloc_data.js"
//   - Node.js tests load this file by stripping the export line and eval-ing
//
// TRACE EVENT ACTIONS (from c10/core/CachingDeviceAllocator.h TraceEntry::Action):
//
//   "alloc"          - Sub-allocation returned to user from the caching allocator.
//                      Recorded in alloc_found_block() (CUDACachingAllocator.cpp:1834).
//
//   "free_requested" - User code called free (tensor out of scope). The block may not
//                      be immediately returned to the free pool if it's in use on
//                      another stream via record_stream.
//                      Recorded in free() (CUDACachingAllocator.cpp:2123).
//
//   "free_completed" - Block actually returned to the allocator's free pool. For simple
//                      cases this fires immediately after free_requested. For cross-stream
//                      blocks, deferred until CUDA events confirm all streams are done.
//                      Recorded in free_block() (CUDACachingAllocator.cpp:3148).
//
//   "segment_alloc"  - New segment allocated from OS via cudaMalloc (or cuMemCreate for
//                      expandable segments).
//                      Recorded in alloc_from_expandable_segment() (CUDACachingAllocator.cpp:3548).
//
//   "segment_free"   - Segment returned to OS via cudaFree. Happens during empty_cache()
//                      or defragmentation. Only for non-expandable segments.
//                      Recorded in release_block() (CUDACachingAllocator.cpp:3686).
//
//   "segment_map"    - Physical pages mapped into an expandable segment via cuMemMap.
//                      The segment grows. Only with expandable segments enabled.
//                      Recorded in alloc_from_expandable_segment() (CUDACachingAllocator.cpp:3092).
//
//   "segment_unmap"  - Physical pages unmapped from an expandable segment via cuMemUnmap.
//                      Virtual address range retained, physical memory returned to OS.
//                      Only with expandable segments. Causes "pool_id unknown" for any
//                      trace events whose addresses fall in the unmapped range, since
//                      the segment no longer exists at snapshot time.
//                      Recorded in unmap_block() (CUDACachingAllocator.cpp:3790).
//
//   "snapshot"       - A call to torch.cuda.memory._snapshot(). Timestamp marker to
//                      correlate trace events with snapshot state. addr=0.
//                      Recorded in snapshot() (CUDACachingAllocator.cpp:2689).
//
//   "oom"            - Allocator failed to satisfy an allocation after all retries.
//                      addr=device_free (bytes free on GPU), size=requested allocation.
//                      Recorded in malloc() (CUDACachingAllocator.cpp:1629).
//
// HOW SEGMENT EVENTS ARE USED IN VISUALIZATION:
//
//   The snapshot pickle contains two separate data sources:
//     1. device_traces  - Ring buffer of TraceEntry actions (alloc, free, segment_map, etc.)
//     2. segments       - Point-in-time dump of all segments/blocks at _snapshot() time
//
//   Block-level views ("Active Memory Timeline", "Allocated Memory (incl. Private Pools)"):
//     - process_alloc_data matches "alloc" and "free_completed" from device_traces.
//     - segment_alloc/segment_free/segment_map/segment_unmap are skipped in the main
//       alloc/free switch, but when include_private_inactive=true, segment events for
//       private pools are captured separately (pool_segment_events) and used to drive
//       pool envelope sizing based on reserved memory rather than active allocations.
//     - The segments snapshot is used to resolve pool_id via find_pool_id() and to
//       compute initial reserved memory per pool for envelope sizing.
//
//   Segment-level view ("Active Cached Segment Timeline"):
//     - process_alloc_data is called with plot_segments=true.
//     - Matches "segment_alloc" and "segment_free" instead of alloc/free.
//     - segment_map/segment_unmap are NOT matched (they don't appear in the switch).
//     - Segments from the snapshot that weren't seen in the trace are added as
//       initially_allocated (Phase 2).
//
//   Allocator State History ("Allocator State History"):
//     - EventSelector lists ALL trace events including segment_map/segment_unmap.
//     - MemoryView renders the segment/block layout from the segments snapshot.
//     - Clicking an event in the list redraws the layout at that point in time.
//
//   Ring buffer overflow:
//     - All trace event types share the same ring buffer. When it overflows, older
//       events are overwritten. The allocator_settings.trace_alloc_overflowed flag
//       indicates this happened, and trace_alloc_max_entries gives the buffer size.
//     - Segment snapshot data (segments array) is NOT affected by ring buffer overflow.
//     - The segment snapshot is always complete regardless of overflow.

/**
 * Returns true if pool_id represents a private (user-created) memory pool,
 * as opposed to the default pool [0, 0].
 *
 * @param {number[]|null} pool_id - Two-element array [owner_id, pool_id] from
 *   the CUDA caching allocator. The default pool is [0, 0]; any other non-null
 *   value is a private pool (e.g. FSDP's MemPool).
 * @returns {boolean}
 */
function isPrivatePoolId(pool_id) {
  return pool_id && !(pool_id[0] === 0 && pool_id[1] === 0);
}

/**
 * Formats a byte count as a human-readable string (e.g. "1.5GiB (1610612736 bytes)").
 *
 * @param {number} num - Size in bytes.
 * @param {boolean} [showBytes=true] - Whether to include the raw byte count in parentheses.
 * @returns {string}
 */
function formatSize(num, showBytes = true) {
  const orig = num;
  // https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
  const units = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'];
  for (const unit of units) {
    if (Math.abs(num) < 1024.0) {
      if (showBytes) {
        return `${num.toFixed(1)}${unit}B (${orig} bytes)`;
      }
      return `${num.toFixed(1)}${unit}B`;
    }
    num /= 1024.0;
  }
  return `${num.toFixed(1)}YiB`;
}

/**
 * Formats a trace event's address as a display string like "b'7f4c00000_3".
 * Segment-level events get an "s'" prefix, block-level events get "b'".
 *
 * @param {{action: string, addr: number|BigInt, version: number}} event
 * @returns {string}
 */
function formatAddr(event) {
  const prefix = event.action.startsWith('segment') ? 's\'' : 'b\'';
  return `${prefix}${event.addr.toString(16)}_${event.version}`;
}

/**
 * Collapses consecutive duplicate strings in an array. If a string appears
 * N > 2 times in a row, it's replaced with [str, "<repeats N-1 times>"].
 * Used to compress repetitive stack frames in the display.
 *
 * @param {string[]} frames
 * @returns {string[]}
 */
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

/**
 * Returns false for stack frames that are internal runtime noise
 * (e.g. Python interpreter internals, C++ dispatch machinery).
 * Used as a filter predicate on frame arrays.
 *
 * @param {{name: string, filename: string}} frame
 * @returns {boolean}
 */
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

/**
 * Formats user-attached metadata (from torch.cuda.memory._record_memory_history)
 * as a display string. Returns '' if no metadata is present.
 *
 * @param {string|Object|null|undefined} user_metadata
 * @returns {string}
 */
function format_user_metadata(user_metadata) {
  if (!user_metadata) {
    return '';
  }
  if (typeof user_metadata === 'string') {
    return `User Metadata:\n  ${user_metadata}`;
  }
  if (typeof user_metadata === 'object' && Object.keys(user_metadata).length === 0) {
    return '';
  }
  const metadata_lines = Object.entries(user_metadata)
    .map(([key, value]) => `  ${key}: ${value}`);
  return 'User Metadata:\n' + metadata_lines.join('\n');
}

/**
 * Formats the forward-pass stack trace (captured via torch.autograd) as a
 * display string showing where a tensor was originally created.
 *
 * @param {string[]|null|undefined} forward_frames
 * @returns {string}
 */
function format_forward_frames(forward_frames) {
  if (!forward_frames || forward_frames.length === 0) {
    return '';
  }
  let frames_str = forward_frames.join('');
  frames_str = frames_str.trimEnd();
  return `\n\n=== Forward Pass Stack Trace (where this tensor was created) ===\n${frames_str}`;
}

/**
 * Formats an array of stack frames into a human-readable string.
 * Filters out runtime noise via frameFilter, annotates FX graph debug info
 * when available, and collapses consecutive duplicate frames.
 *
 * @param {{filename: string, line: number, name: string,
 *          fx_node_op?: string, fx_node_name?: string,
 *          fx_node_target?: string, fx_original_trace?: string}[]} frames
 * @returns {string}
 */
function format_frames(frames) {
  if (frames.length === 0) {
    return (
      `This block has no frames. Potential causes:\n` +
      `1) This block was allocated before _record_memory_history was enabled.\n` +
      `2) The context or stacks passed to _record_memory_history does not include this block. Consider changing context to 'state', 'alloc', or 'all', or changing stacks to 'all'.\n` +
      `3) This event occurred during backward, which has no python frames, and memory history did not include C++ frames. Use stacks='all' to record both C++ and python frames.\n` +
      `4) This block was reconstructed from the allocator's segment snapshot (not from a trace event). The snapshot records which blocks exist at the moment _snapshot() is called, but does not carry stack frames. This typically happens for blocks that were allocated before tracing started and never freed, or for inactive blocks in private memory pools.\n` +
      `5) The original alloc event was evicted from the trace ring buffer (older entries are overwritten when the buffer is full). Increase the max_entries argument to _record_memory_history to retain more events.`
    );
  }
  const frame_strings = frames
    .filter(frameFilter)
    .map(f => {
      let frame_str = `${f.filename}:${f.line}:${f.name}`;

      if (f.fx_node_op || f.fx_node_name || f.fx_node_target) {
        const fx_parts = [];
        if (f.fx_node_name) fx_parts.push(`node=${f.fx_node_name}`);
        if (f.fx_node_op) fx_parts.push(`op=${f.fx_node_op}`);
        if (f.fx_node_target) fx_parts.push(`target=${f.fx_node_target}`);
        frame_str += `\n    >> FX: ${fx_parts.join(', ')}`;
      }

      if (f.fx_original_trace) {
        frame_str += `\n    >> Original Model Code:`;
        const original_lines = f.fx_original_trace.trim().split('\n');
        for (const line of original_lines) {
          frame_str += `\n       ${line}`;
        }
      }

      return frame_str;
    });
  return elideRepeats(frame_strings).join('\n');
}

/**
 * Transforms a memory snapshot into a stacked-area timeline suitable for
 * rendering by MemoryPlot. This is the core data-processing function behind
 * the "Active Memory Timeline" and "Allocated Memory (incl. Private Pools)"
 * visualization tabs.
 *
 * HIGH-LEVEL ALGORITHM:
 *
 * 1. TRACE EVENT MATCHING: Scans device_traces to pair alloc events with their
 *    corresponding free_completed events (by address). Events whose matching
 *    alloc was lost (e.g. ring buffer wrap) become "initially_allocated" —
 *    blocks assumed to exist at the start of the trace.
 *
 * 2. SEGMENT SNAPSHOT: Supplements trace data with the current segment state.
 *    Blocks marked active_allocated (or inactive in private pools, when
 *    include_private_inactive=true) that weren't seen in the trace are also
 *    added as initially_allocated.
 *
 * 3. DETAIL LIMITING: Only the largest max_entries elements get individual
 *    rectangles in the plot. Smaller elements are aggregated into a single
 *    "summarized" band to keep rendering fast.
 *
 * 4. STACKED AREA CONSTRUCTION: Replays alloc/free events in order, building
 *    a stacked-area dataset where each element has timesteps, y-offsets, and
 *    a size. Elements are stacked bottom-to-top; frees remove from the stack
 *    and shift elements above downward.
 *
 * 5. PRIVATE POOL ENVELOPES (include_private_inactive=true): Each private pool
 *    (e.g. FSDP's MemPool) gets a single gray "envelope" rectangle whose
 *    height is the pool's reserved memory (from segment_map/segment_unmap
 *    events and the segment snapshot). Active blocks within the pool are
 *    rendered as colored stripes inside the envelope. The envelope only grows
 *    (never shrinks), representing the pool's actual GPU memory footprint.
 *    This correctly handles fragmentation: when a large alloc triggers a
 *    segment_map because existing free blocks aren't contiguous, the envelope
 *    grows by the reserved amount, not just the active allocation.
 *
 *    Initially-allocated private pool blocks are PRE-LOADED into pool state
 *    so that when their free event appears in the trace, they are correctly
 *    recognized as frees (not misinterpreted as new allocations).
 *
 * NOTE ON FREE EVENT MATCHING: The C++ allocator emits 'free_requested' and
 * 'free_completed' for each deallocation. This function matches against 'free'
 * (which no longer appears in modern traces — effectively dead code) and
 * 'free_completed'. Only free_completed does the actual matching. Matching
 * both 'free_requested' AND 'free_completed' would cause double-processing
 * since they share the same address.
 *
 * @param {Object} snapshot - Memory snapshot from torch.cuda.memory._snapshot().
 * @param {Object[]} snapshot.segments - Current allocator segment state.
 * @param {Object[][]} snapshot.device_traces - Per-device arrays of trace events.
 *   Each event has {action, addr, size, frames, stream, segment_pool_id?, ...}.
 * @param {string[]} snapshot.categories - Category names for color-coding.
 * @param {number} device - Device index into snapshot.device_traces.
 * @param {boolean} plot_segments - If true, plot segment-level (cudaMalloc)
 *   events instead of sub-allocation events.
 * @param {number} max_entries - Maximum number of elements to render individually.
 *   Elements beyond this limit are aggregated into the "summarized" band.
 * @param {boolean} [include_private_inactive=false] - If true, include inactive
 *   blocks from private pools and render pool envelopes. Used by the
 *   "Allocated Memory (incl. Private Pools)" tab.
 *
 * @returns {{
 *   max_size: number,
 *   allocations_over_time: Object[],
 *   max_at_time: number[],
 *   summarized_mem: Object,
 *   elements_length: number,
 *   context_for_id: function(number): string
 * }}
 *   - max_size: peak total memory observed during the action replay (used for
 *     y-axis scaling). Note: this is only updated inside the action loop, so
 *     the initial state from initially_allocated may not be reflected here
 *     (use max_at_time for the true peak).
 *   - allocations_over_time: array of stacked-area data objects, each with
 *     {elem, timesteps[], offsets[], size, color}.
 *   - max_at_time: total memory at each timestep (for minimap rendering).
 *   - summarized_mem: the aggregated band for small elements.
 *   - elements_length: total number of unique allocation elements.
 *   - context_for_id: function that returns a human-readable description
 *     string for a given element index (address, size, stack trace, etc.).
 */
function process_alloc_data(snapshot, device, plot_segments, max_entries, include_private_inactive = false) {
  const elements = [];
  // Contains two types of blocks
  // 1. free without alloc in trace
  // 2. actively allocated in segments, but no matching alloc in trace
  const initially_allocated = [];
  const actions = [];
  const addr_to_alloc = {};

  const device_segments = snapshot.segments
    .filter(s => s.device === device)
    .sort((a, b) => {
      if (a.address === b.address) return 0;
      return a.address < b.address ? -1 : 1;
    });

  // Binary search to find which segment contains a given address.
  function find_pool_id(addr) {
    let left = 0;
    let right = device_segments.length - 1;
    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      const seg = device_segments[mid];
      const seg_end = seg.address + (typeof seg.address === "bigint" ? BigInt(seg.total_size) : seg.total_size);
      if (addr < seg.address) {
        right = mid - 1;
      } else if (addr >= seg_end) {
        left = mid + 1;
      } else {
        return seg.segment_pool_id;
      }
    }
    return null;
  }

  const alloc = plot_segments ? 'segment_alloc' : 'alloc';
  const [free, free_completed] = plot_segments
    ? ['segment_free', 'segment_free']
    : ['free', 'free_completed'];
  // pool_segment_events tracks segment_map/segment_unmap events for private
  // pools, recording their position relative to the actions list. This lets
  // the Phase 3 replay grow pool envelopes based on actual reserved memory
  // (segments) rather than just active allocations.
  const pool_segment_events = [];
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
          // Matched: reuse the element from the alloc event
          actions.push(addr_to_alloc[e.addr]);
          delete addr_to_alloc[e.addr];
        } else {
          // Unmatched free: alloc happened before recording (or was evicted
          // from the ring buffer). Create a new element from the free event;
          // its stack trace will show the free site, not the alloc site.
          elements.push(e);
          initially_allocated.push(elements.length - 1);
          actions.push(elements.length - 1);
        }
        break;
      default:
        break;
    }
    if (include_private_inactive &&
        (e.action === 'segment_alloc' || e.action === 'segment_free' ||
         e.action === 'segment_map' || e.action === 'segment_unmap')) {
      const pid = find_pool_id(e.addr);
      if (isPrivatePoolId(pid)) {
        const is_add = e.action === 'segment_alloc' || e.action === 'segment_map';
        pool_segment_events.push({
          position: actions.length,
          delta: is_add ? e.size : -e.size,
          pool_key: format_pool_key(pid, e.stream ?? 0),
        });
      }
    }
  }

  // --- Phase 2: Add elements from the snapshot ---
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
        const addr = b.addr ?? b.address;
        if (b.state === 'active_allocated' && !(addr in addr_to_alloc)) {
          const element = {
            action: 'alloc',
            addr,
            size: b.requested_size,
            frames: b.frames,
            stream: seg.stream,
            version: b.version,
            segment_pool_id: seg.segment_pool_id,
            ghost: true,
          };
          elements.push(element);
          initially_allocated.push(elements.length - 1);
        }
      }
    }
  }

  // Resolve pool IDs for trace elements by looking up which segment they fall in
  for (const elem of elements) {
    if (!elem.segment_pool_id) {
      elem.segment_pool_id = find_pool_id(elem.addr);
    }
  }

  initially_allocated.reverse();
  // If there are no trace actions but there are existing allocations,
  // show a flat graph with the initial state
  if (actions.length === 0 && initially_allocated.length > 0) {
    actions.push(initially_allocated.pop());
  }

  // --- Phase 3: Build the stacked-area timeline ---
  const current = [];      // stack of element indices (bottom to top)
  const current_data = []; // parallel array of visualization data objects
  const data = [];         // all data objects (including completed ones)
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

  // Record the current memory state and advance time by n steps
  function advance(n) {
    summarized_mem.timesteps.push(timestep);
    summarized_mem.offsets.push(total_mem);
    summarized_mem.size.push(total_summarized_mem);
    timestep += n;
    for (let i = 0; i < n; i++) {
      max_at_time.push(total_mem + total_summarized_mem);
    }
  }

  // Only render the largest max_entries elements individually (across all
  // pools). Pools with larger allocations naturally get more of the budget.
  // Remaining pool elements go into per-pool summarized stripes; remaining
  // non-pool elements go into the global summarized band.
  const sizes = elements
    .map((x, i) => [x.size, i])
    .sort(([x, _xi], [y, _yi]) => y - x);

  const draw_elem = {};
  for (const [_s, e] of sizes.slice(0, max_entries)) {
    draw_elem[e] = true;
  }

  // Push an element onto the memory stack
  function add_allocation(elem) {
    const element_obj = elements[elem];
    const size = element_obj.size;
    current.push(elem);
    let color = elem;
    if (snapshot.categories.length > 0) {
      color = snapshot.categories.indexOf(element_obj.category || 'unknown');
    }
    const e = {
      elem,
      timesteps: [timestep],
      offsets: [total_mem],
      size,
      color,
    };
    if (element_obj.ghost) e.ghost = true;
    current_data.push(e);
    data.push(e);
    total_mem += size;
    element_obj.max_allocated_mem = total_mem + total_summarized_mem;
  }

  // --- Pool envelope tracking (only when include_private_inactive=true) ---
  // Each private pool gets a gray envelope whose height = high-water mark.
  // Active blocks are rendered as colored stripes within the envelope.
  const pools = {};
  const pool_active_elems = {};

  function format_pool_key(pid, stream) {
    return `${pid[0]},${pid[1]},s${stream}`;
  }

  function get_pool_key(elem_idx) {
    const pid = elements[elem_idx].segment_pool_id;
    if (!isPrivatePoolId(pid)) return null;
    return format_pool_key(pid, elements[elem_idx].stream);
  }

  function get_or_create_pool(pool_key) {
    if (!(pool_key in pools)) {
      pools[pool_key] = {
        max: 0, active: 0, reserved: 0,
        drawn_active: 0, summarized_active: 0,
        envelope_data: null, summarized_data: null,
        block_stack: [],  // [{elem, size, inner_offset, stripe_data}]
      };
    }
    return pools[pool_key];
  }

  function elem_color(elem_idx) {
    if (snapshot.categories.length > 0) {
      return snapshot.categories.indexOf(elements[elem_idx].category || 'unknown');
    }
    return elem_idx;
  }

  function shift_pool_stripes(pool, delta) {
    for (const block of pool.block_stack) {
      const s = block.stripe_data;
      s.timesteps.push(timestep);
      s.offsets.push(s.offsets.at(-1));
      s.timesteps.push(timestep + 3);
      s.offsets.push(s.offsets.at(-1) + delta);
    }
    if (pool.summarized_data) {
      const sd = pool.summarized_data;
      sd.timesteps.push(timestep);
      sd.offsets.push(sd.offsets.at(-1));
      sd.size.push(sd.size.at(-1));
      sd.timesteps.push(timestep + 3);
      sd.offsets.push(sd.offsets.at(-1) + delta);
      sd.size.push(sd.size.at(-1));
    }
  }

  // Update or create the per-pool summarized stripe. Sits on top of drawn
  // stripes (offset = envelope base + drawn_active), size = summarized_active.
  function update_pool_summary(pool, ts) {
    if (!pool.envelope_data) return;
    const base = pool.envelope_data.offsets.at(-1) + pool.drawn_active;
    if (pool.summarized_data === null) {
      pool.summarized_data = {
        elem: 'summarized',
        timesteps: [ts],
        offsets: [base],
        size: [pool.summarized_active],
        color: 0,
        opacity: 0.3,
      };
      data.push(pool.summarized_data);
    } else {
      const sd = pool.summarized_data;
      sd.timesteps.push(ts);
      sd.offsets.push(base);
      sd.size.push(pool.summarized_active);
    }
  }

  // Animate shifting all elements above idx by delta (used when an element
  // is inserted or removed from the middle of the stack)
  function shift_elements_above(idx, delta) {
    for (let j = idx; j < current.length; j++) {
      const e = current_data[j];
      e.timesteps.push(timestep);
      e.offsets.push(e.offsets.at(-1));
      e.timesteps.push(timestep + 3);
      e.offsets.push(e.offsets.at(-1) + delta);
      if (Array.isArray(e.size)) {
        e.size.push(e.size.at(-1));
        e.size.push(e.size.at(-1));
      }
      const pk = typeof current[j] === 'string' && current[j].startsWith('pool:')
        ? current[j].slice(5) : null;
      if (pk && pk in pools) {
        shift_pool_stripes(pools[pk], delta);
      }
    }
  }

  // Shift all elements stacked above a pool envelope by delta (no animation).
  // Used during timestep-0 initialization when there are no transition frames.
  function shift_above_pool_no_anim(pool_key, delta) {
    const pidx = current.indexOf(`pool:${pool_key}`);
    if (pidx >= 0) {
      for (let j = pidx + 1; j < current.length; j++) {
        const e = current_data[j];
        e.offsets[e.offsets.length - 1] += delta;
      }
    }
  }

  // Grow a pool envelope to accommodate new_size bytes (the larger of active
  // allocations and reserved segment memory). The envelope only grows (never
  // shrinks) — it represents the pool's actual GPU memory footprint.
  function grow_pool_envelope(pool, pool_key, new_size) {
    if (new_size <= pool.max) return;
    const delta = new_size - pool.max;
    pool.max = new_size;
    const env = pool.envelope_data;
    env.timesteps.push(timestep);
    env.offsets.push(env.offsets.at(-1));
    env.size.push(env.size.at(-1));
    env.timesteps.push(timestep + 3);
    env.offsets.push(env.offsets.at(-1));
    env.size.push(pool.max);
    const pidx = current.indexOf(`pool:${pool_key}`);
    if (pidx >= 0) {
      shift_elements_above(pidx + 1, delta);
    }
    total_mem += delta;
    advance(3);
  }

  // --- Process initially_allocated elements ---
  // These are blocks that existed before the trace window started. They come
  // from two sources:
  //   1. Unmatched free events (free_completed without a prior alloc in trace)
  //   2. active_allocated blocks in the segment snapshot with no trace event
  //
  // For private pool blocks: pre-load into pool state at timestep 0 (no
  // animation). This serves two purposes:
  //   - The envelope starts at the correct initial size
  //   - When the free event fires during replay, it's recognized as a free
  //     (not misinterpreted as a new allocation)
  //
  // For non-pool blocks: added to the global stack (draw_elem) or global
  // summarized band.
  for (const elem of initially_allocated) {
    if (include_private_inactive && get_pool_key(elem)) {
      const pk = get_pool_key(elem);
      const size = elements[elem].size;
      const pool = get_or_create_pool(pk);
      // Mark as active so the replay loop recognizes the free event
      pool_active_elems[elem] = pk;

      // Create pool envelope on first encounter
      if (pool.envelope_data === null) {
        const env = {
          elem: `pool:${pk}`,
          timesteps: [0],
          offsets: [total_mem],
          size: [0],
          color: 9,
        };
        pool.envelope_data = env;
        // Add to the global stack so elements above it shift when it grows
        current.push(`pool:${pk}`);
        current_data.push(env);
        data.push(env);
      }

      pool.active += size;

      // Grow envelope to fit: use max(active, reserved) because active can
      // exceed reserved when block sizes are stale (e.g. segment shrank via
      // unmap after the block was allocated).
      const init_target = Math.max(pool.active, pool.reserved);
      if (init_target > pool.max) {
        const delta = init_target - pool.max;
        pool.max = init_target;
        const env = pool.envelope_data;
        env.size[env.size.length - 1] = pool.max;
        total_mem += delta;
        // Shift all elements stacked above this pool's envelope up by delta
        shift_above_pool_no_anim(pk, delta);
      }

      if (elem in draw_elem) {
        const inner_offset = pool.drawn_active;
        pool.drawn_active += size;
        const stripe = {
          elem,
          timesteps: [0],
          offsets: [pool.envelope_data.offsets.at(-1) + inner_offset],
          size,
          color: elem_color(elem),
          opacity: 0.5,
          ghost: elements[elem].ghost || false,
        };
        pool.block_stack.push({elem, size, inner_offset, stripe_data: stripe});
        data.push(stripe);
      } else {
        pool.summarized_active += size;
      }
      continue;
    }
    // Non-pool element: render individually or add to global summarized band
    if (elem in draw_elem) {
      add_allocation(elem);
    } else {
      total_summarized_mem += elements[elem].size;
      summarized_elems[elem] = true;
    }
  }

  // Fix up pool stripe offsets — stripes are not in current_data so they
  // don't get shifted when other pools grow during initially_allocated
  // processing. Recompute from the envelope's final offset.
  // Also create per-pool summarized data for initial non-drawn elements.
  for (const pk in pools) {
    const p = pools[pk];
    if (!p.envelope_data) continue;
    const env_offset = p.envelope_data.offsets.at(-1);
    for (const block of p.block_stack) {
      const s = block.stripe_data;
      for (let i = 0; i < s.offsets.length; i++) {
        s.offsets[i] = env_offset + block.inner_offset;
      }
    }
    if (p.summarized_active > 0) {
      p.summarized_data = {
        elem: 'summarized',
        timesteps: [0],
        offsets: [env_offset + p.drawn_active],
        size: [p.summarized_active],
        color: 0,
        opacity: 0.3,
      };
      data.push(p.summarized_data);
    }
  }

  // --- Initialize pool reserved memory from snapshot ---
  // The envelope height for each private pool should reflect its reserved
  // (segment) memory, not just active allocations. We compute the initial
  // reserved value so that replaying segment_map/segment_unmap events from
  // the trace arrives at the correct final value (the snapshot total).
  //
  // Formula: initial = snapshot_total - net_trace_delta
  //   - snapshot_total: sum of segment total_size for this pool (ground truth
  //     at snapshot time)
  //   - net_trace_delta: sum of segment_map sizes minus segment_unmap sizes
  //     for this pool in the trace
  //
  // This works regardless of trace truncation (ring buffer overflow): the
  // initial value represents the reserved memory at the start of the trace
  // window, not at program start. If there are no segment events in the
  // trace for a pool, net_trace_delta is 0 and initial = snapshot_total.
  if (include_private_inactive) {
    const snapshot_reserved = {};
    for (const seg of device_segments) {
      const pid = seg.segment_pool_id;
      if (isPrivatePoolId(pid)) {
        const pk = format_pool_key(pid, seg.stream ?? 0);
        snapshot_reserved[pk] = (snapshot_reserved[pk] || 0) + seg.total_size;
      }
    }
    const net_from_trace = {};
    for (const se of pool_segment_events) {
      net_from_trace[se.pool_key] = (net_from_trace[se.pool_key] || 0) + se.delta;
    }
    for (const pk in snapshot_reserved) {
      const pool = get_or_create_pool(pk);
      pool.reserved = snapshot_reserved[pk] - (net_from_trace[pk] || 0);
      // Grow envelope to initial reserved (no animation — pre-existing)
      if (pool.reserved > pool.max && pool.envelope_data) {
        const delta = pool.reserved - pool.max;
        pool.max = pool.reserved;
        const env = pool.envelope_data;
        env.size[env.size.length - 1] = pool.max;
        total_mem += delta;
        shift_above_pool_no_anim(pk, delta);
      }
    }
    // Fix up pool stripe offsets again after reserved-based envelope growth
    for (const pk in pools) {
      const p = pools[pk];
      if (!p.envelope_data) continue;
      const env_offset = p.envelope_data.offsets.at(-1);
      for (const block of p.block_stack) {
        const s = block.stripe_data;
        for (let i = 0; i < s.offsets.length; i++) {
          s.offsets[i] = env_offset + block.inner_offset;
        }
      }
    }
  }

  // --- Replay alloc/free actions to build the timeline ---
  let seg_event_idx = 0;
  for (let action_i = 0; action_i < actions.length; action_i++) {
    // Process segment events that occurred at or before this action position.
    // These grow pool envelopes based on actual reserved memory changes.
    while (seg_event_idx < pool_segment_events.length &&
           pool_segment_events[seg_event_idx].position <= action_i) {
      const se = pool_segment_events[seg_event_idx];
      const pool = get_or_create_pool(se.pool_key);
      pool.reserved += se.delta;
      if (pool.reserved > pool.max && pool.envelope_data) {
        grow_pool_envelope(pool, se.pool_key, pool.reserved);
      }
      seg_event_idx++;
    }

    const elem = actions[action_i];
    const size = elements[elem].size;
    const pool_key = include_private_inactive ? get_pool_key(elem) : null;

    if (pool_key) {
      // --- Private pool element ---
      if (!(elem in pool_active_elems)) {
        // Pool alloc: add to pool, grow envelope if needed
        pool_active_elems[elem] = pool_key;
        const pool = get_or_create_pool(pool_key);

        if (pool.envelope_data === null) {
          const env = {
            elem: `pool:${pool_key}`,
            timesteps: [timestep],
            offsets: [total_mem],
            size: [0],
            color: 9,
          };
          pool.envelope_data = env;
          current.push(`pool:${pool_key}`);
          current_data.push(env);
          data.push(env);
        }

        pool.active += size;

        const envelope_target = Math.max(pool.active, pool.reserved);
        if (envelope_target > pool.max) {
          grow_pool_envelope(pool, pool_key, envelope_target);
        }

        if (elem in draw_elem) {
          const inner_offset = pool.drawn_active;
          pool.drawn_active += size;
          const stripe = {
            elem,
            timesteps: [timestep],
            offsets: [pool.envelope_data.offsets.at(-1) + inner_offset],
            size,
            color: elem_color(elem),
            opacity: 0.5,
          };
          pool.block_stack.push({elem, size, inner_offset, stripe_data: stripe});
          data.push(stripe);
          // Shift summarized stripe up (it sits on top of drawn stripes)
          if (pool.summarized_data) {
            update_pool_summary(pool, timestep);
          }
        } else {
          pool.summarized_active += size;
          update_pool_summary(pool, timestep);
        }
        advance(1);
        elements[elem].max_allocated_mem = total_mem + total_summarized_mem;
      } else {
        // Pool free: end stripe, shift stripes above down within the pool.
        // The envelope stays at its high-water mark (never shrinks).
        const pool = pools[pool_key];
        const block_idx = pool.block_stack.findIndex(b => b.elem === elem);
        if (block_idx >= 0) {
          // Drawn stripe freed
          advance(1);
          const block = pool.block_stack[block_idx];
          block.stripe_data.timesteps.push(timestep);
          block.stripe_data.offsets.push(block.stripe_data.offsets.at(-1));

          pool.block_stack.splice(block_idx, 1);
          pool.active -= size;
          pool.drawn_active -= size;

          // Shift drawn stripes above and the summarized stripe down
          const need_shift = block_idx < pool.block_stack.length || pool.summarized_data;
          if (need_shift) {
            for (let j = block_idx; j < pool.block_stack.length; j++) {
              const b = pool.block_stack[j];
              b.inner_offset -= size;
              const s = b.stripe_data;
              s.timesteps.push(timestep);
              s.offsets.push(s.offsets.at(-1));
              s.timesteps.push(timestep + 3);
              s.offsets.push(pool.envelope_data.offsets.at(-1) + b.inner_offset);
            }
            if (pool.summarized_data) {
              update_pool_summary(pool, timestep);
            }
            advance(3);
          }
        } else {
          // Non-drawn element freed — summarized stripe shrinks on top
          pool.active -= size;
          pool.summarized_active -= size;
          update_pool_summary(pool, timestep);
          advance(1);
        }
        delete pool_active_elems[elem];
      }
      max_size = Math.max(total_mem + total_summarized_mem, max_size);
      continue;
    }

    // --- Non-pool element ---
    if (!(elem in draw_elem)) {
      // Too small to render individually — goes into the summarized band
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
    if (idx === -1) {
      // First appearance → alloc
      add_allocation(elem);
      advance(1);
    } else {
      // Second appearance → free: remove from stack, shift elements above down
      advance(1);
      const removed = current_data[idx];
      removed.timesteps.push(timestep);
      removed.offsets.push(removed.offsets.at(-1));
      current.splice(idx, 1);
      current_data.splice(idx, 1);

      if (idx < current.length) {
        shift_elements_above(idx, -size);
        advance(3);
      }
      total_mem -= size;
    }
    max_size = Math.max(total_mem + total_summarized_mem, max_size);
  }

  // Process any remaining segment events after the last action
  while (seg_event_idx < pool_segment_events.length) {
    const se = pool_segment_events[seg_event_idx];
    const pool = get_or_create_pool(se.pool_key);
    pool.reserved += se.delta;
    if (pool.reserved > pool.max && pool.envelope_data) {
      grow_pool_envelope(pool, se.pool_key, pool.reserved);
    }
    max_size = Math.max(total_mem + total_summarized_mem, max_size);
    seg_event_idx++;
  }

  // --- Finalize: close all still-active elements ---
  for (const elem of current_data) {
    elem.timesteps.push(timestep);
    elem.offsets.push(elem.offsets.at(-1));
    if (Array.isArray(elem.size)) {
      elem.size.push(elem.size.at(-1));
    }
  }
  for (const pk in pools) {
    for (const block of pools[pk].block_stack) {
      const s = block.stripe_data;
      s.timesteps.push(timestep);
      s.offsets.push(s.offsets.at(-1));
    }
    if (pools[pk].summarized_data) {
      const sd = pools[pk].summarized_data;
      sd.timesteps.push(timestep);
      sd.offsets.push(sd.offsets.at(-1));
      sd.size.push(sd.size.at(-1));
    }
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
      let text = `Addr: ${formatAddr(elem)}`;
      text = `${text}, Size: ${formatSize(elem.size)} allocation`;
      text = `${text}, Total memory used after allocation: ${formatSize(
        elem.max_allocated_mem,
      )}`;
      const context = elem?.compile_context ?? 'None';
      text = `${text}, Compile context: ${context}`;
      if (elem.stream !== null) {
        text = `${text}, stream ${elem.stream}`;
      }
      if (elem.segment_pool_id) {
        text = `${text}, pool_id (${elem.segment_pool_id[0]}, ${elem.segment_pool_id[1]})`;
      } else {
        text = `${text}, pool_id unknown`;
      }
      if (elem.timestamp !== null) {
        var d = new Date(elem.time_us / 1000);
        text = `${text}, timestamp ${d}`;
      }
      if (!elem.action.includes('alloc')) {
        text = `${text}\nalloc not recorded, stack trace for free:`;
      }
      if (elem.ghost) {
        text = `${text}\n[Ghost block] This block exists in the segment snapshot but has no alloc trace events. ` +
          `It was allocated before _record_memory_history() was called, or its alloc event was evicted ` +
          `from the trace ring buffer. The block is still active (not freed) at snapshot time.`;
      }
      const user_metadata_str = format_user_metadata(elem.user_metadata);
      if (user_metadata_str) {
        text = `${text}\n${user_metadata_str}`;
      }
      text = `${text}\n${format_frames(elem.frames)}`;
      text = `${text}${format_forward_frames(elem.forward_frames)}`;
      return text;
    },
  };
}

export { process_alloc_data, isPrivatePoolId, formatSize, formatAddr,
         elideRepeats, frameFilter, format_user_metadata,
         format_forward_frames, format_frames };
