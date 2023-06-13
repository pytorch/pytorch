import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.7.0/+esm";
import {schemeTableau10} from "https://cdn.skypack.dev/d3-scale-chromatic@3";
import {axisLeft} from "https://cdn.skypack.dev/d3-axis@3";
import {scaleLinear} from "https://cdn.skypack.dev/d3-scale@4";
import {zoom, zoomIdentity} from "https://cdn.skypack.dev/d3-zoom@3";
import {brushX} from "https://cdn.skypack.dev/d3-brush@3";

let trace_data = null

function frames_for(id) {
    let strings = []
    while (id !== null) {
        const [sid, next_id] = trace_data.suffix_table[id]
        strings.push(trace_data.string_table[sid])
        id = next_id
    }
    return `${strings.join('\n')}\n`
}

class Event {
    constructor(id) {
        this.idx = id
    }
    get size() { return trace_data.events.size[this.idx] }
    get action() { return trace_data.string_table[trace_data.events.action[this.idx]] }
    get addr() { return trace_data.events.addr[this.idx] }
    get frames() { return frames_for(this.frames_idx) }
    get frames_idx() { return trace_data.events.frames[this.idx]}
    get stream() { return trace_data.events.stream[this.idx] }
    get device_free() { return trace_data.events.addr[this.idx] }
}

function createEvents() {
    let events = trace_data.events.action.map( (_, i) => new Event(i))

    function version_space() {
        let version = {}
        return (addr, increment) => {
            if (!(addr in version)) {
                version[addr] = 0
            }
            let r = version[addr]
            if (increment) {
                version[addr]++
            }
            return r
        }
    }

    let segment_version = version_space()
    let block_version = version_space()
    for (let t of events) {
        // set unique version for each time an address is used
        // so that ctrl-f can be used to search for the beginning
        // and end of allocations and segments
        switch (t.action) {
            case 'free':
            case 'free_completed':
                t.version = block_version(t.addr, true)
                break
            case 'free_requested':
            case 'alloc':
                t.version = block_version(t.addr, false)
                break
            case 'segment_free':
            case 'segment_unmap':
                t.version = segment_version(t.addr, true)
                break;
            case 'segment_alloc':
            case 'segment_map':
                t.version = segment_version(t.addr, false)
                break
            default:
                break
        }
    }
    trace_data.segment_version = segment_version
    trace_data.block_version = block_version
    return events
}

function Segment(addr, size, stream, frame_idx, version) {
    return {addr, size, stream, version, get frames() { return frames_for(frame_idx)}}
}
function Block(addr, size, real_size, frame_idx, free_requested, version) {
    console.assert(frame_idx !== undefined)
    return {addr, size, real_size, get frames() { return frames_for(frame_idx)}, free_requested, version}
}

function EventSelector(body, outer, events, stack_info, memory_view) {
    let events_div = outer
    .append("div")
    .attr('style', 'grid-column: 1; grid-row: 1; overflow: auto; font-family: monospace')

    let events_selection = events_div
    .selectAll("pre")
    .data(events)
    .enter()
    .append("pre")
    .text((e) => formatEvent(e))
    .attr('style', '')

    let selected_event_idx = null

    let es = {
        select(idx) {
            if (selected_event_idx !== null) {
                let selected_event = d3.select(events_div.node().children[selected_event_idx])
                selected_event.attr('style', '')
            }
            if (idx !== null) {
                let div = d3.select(events_div.node().children[idx])
                div.attr('style', `background-color: ${schemeTableau10[5]}`)
                let [reserved, allocated] = memory_view.draw(idx)
                let enter = () => eventStack(div.datum(), allocated, reserved)
                stack_info.highlight(enter)
                div.node().scrollIntoViewIfNeeded(false)
            } else {
                memory_view.draw(0)
            }
            selected_event_idx = idx
        }
    }
    body.on('keydown', (e) => {
        let actions = {ArrowDown: 1, ArrowUp: -1}
        if (selected_event_idx !== null && e.key in actions) {
            let new_idx = selected_event_idx + actions[e.key]
            es.select(Math.max(0, Math.min(new_idx, events.length - 1)))
            e.preventDefault()
        }
    })

    stack_info.register(events_selection, (t) => eventStack(t.datum()), (t) => {}, (d) => es.select(d.datum().idx))

    return es
}

function formatSize(num) {
    let orig = num
    // https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    const units = ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"];
    for (const unit of units) {
        if (Math.abs(num) < 1024.0) {
        return `${num.toFixed(1)}${unit}B (${orig} bytes)`;
        }
        num /= 1024.0;
    }
    return `${num.toFixed(1)}YiB`;
}

function formatEvent(event) {
    function formatAddr(event) {
        let version = event.version == 0 ? "" : `_${event.version}`
        let prefix = event.action.startsWith("segment") ? "s" : "b"
        return `${prefix}${event.addr.toString(16)}_${event.version}`
    }
    let stream = event.stream == 0 ? "" : `\n              (stream ${event.stream})`
    switch (event.action) {
        case 'oom':
            return `OOM (requested ${formatSize(event.size)}, CUDA has ${formatSize(event.device_free)} memory free)${stream}`
        case 'snapshot':
            return "snapshot"
        default:
            return `${event.action.padEnd(14)} ${formatAddr(event).padEnd(18)} ${formatSize(event.size)}${stream}`
    }
}

function eventStack(e, allocated, reserved) {
    let event = formatEvent(e)
    if (reserved !== undefined) {
        event = `(${formatSize(allocated)} allocated / ${formatSize(reserved)} reserved)\n${event}`
    }
    return event + "\n" + e.frames
}

function hashCode(num) {
  const numStr = num.toString();
  let hash = 0;
  for (let i = 0; i < numStr.length; i++) {
    const charCode = numStr.charCodeAt(i);
    hash = ((hash << 5) - hash) + charCode;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return hash;
}

function addStroke(d) {
    d.attr('stroke', 'red')
    .attr('stroke-width', '2')
    .attr('vector-effect', 'non-scaling-stroke')
}

function removeStroke(d) {
    d.attr('stroke', '')
}

function MemoryView(outer, stack_info, trace_data, events) {
    let svg = outer.append('svg')
    .attr('style', 'grid-column: 2; grid-row: 1; width: 100%; height: 100%;')
    .attr('viewBox', '0 0 200 100')
    .attr('preserveAspectRatio', 'xMinYMin meet')
    let g = svg.append('g')
    let seg_zoom = zoom()
    seg_zoom.on('zoom', ({transform}) => {
        g.attr('transform', transform)
    })
    svg.call(seg_zoom)

    let sorted_segments = []
    let block_map = {}

    let segments_data = trace_data.segments
    for (let [i, addr] of trace_data.segments.addr.entries()) {
        sorted_segments.push(Segment(addr, segments_data.size[i], segments_data.stream[i],
                                     null, trace_data.segment_version(addr, false)))
    }
    sorted_segments.sort((x, y) => x.addr - y.addr)

    let blocks_data = trace_data.blocks
    for (let [i, addr] of trace_data.blocks.addr.entries()) {
        block_map[addr] = Block(addr, blocks_data.size[i], blocks_data.real_size[i],
                                blocks_data.frames[i], blocks_data.pending_free[i],
                                trace_data.block_version(addr, false))
    }

    function simulate_memory(idx) {
        // create a copy of segments because we edit size properties below
        let l_segments = sorted_segments.map((x) => { return {...x} })
        let l_block_map = {...block_map}

        function map_segment(merge, seg) {
            let idx = l_segments.findIndex(e => e.addr > seg.addr)
            if (!merge) {
                l_segments.splice(idx, 0, seg)
                return
            }
            if (idx == -1) {
                idx = l_segments.length
            }
            l_segments.splice(idx, 0, seg)
            if (idx + 1 < l_segments.length) {
                let next = l_segments[idx + 1]
                if (seg.addr + seg.size == next.addr && seg.stream == next.stream) {
                    seg.size += next.size
                    l_segments.splice(idx + 1, 1)
                }
            }
            if (idx > 0) {
                let prev = l_segments[idx - 1]
                if (prev.addr + prev.size == seg.addr && prev.stream == seg.stream) {
                    prev.size += seg.size
                    l_segments.splice(idx, 1)
                }
            }
        }
        function unmap_segment(merge, seg) {
            if (!merge) {
                l_segments.splice(l_segments.findIndex(x => x.addr == seg.addr), 1)
                return
            }
            let seg_end = seg.addr + seg.size
            let idx = l_segments.findIndex(e => e.addr <= seg.addr && seg_end <= e.addr + e.size)
            let existing = l_segments[idx]
            let existing_end = existing.addr + existing.size
            if (existing.addr == seg.addr) {
                existing.addr += seg.size
                existing.size -= seg.size
                if (existing.size == 0) {
                    l_segments.splice(idx, 1)
                }
            } else if (existing_end == seg_end) {
                existing.size -= seg.size
            } else {
                existing.size = seg.addr - existing.addr
                seg.addr = seg_end
                seg.size = existing_end - seg_end
                l_segments.splice(idx + 1, 0, seg)
            }
        }

        for (let i = events.length - 1; i > idx; i--) {
            let event = events[i]
            switch (event.action) {
                case 'free':
                    l_block_map[event.addr] = Block(event.addr, event.size, event.size, event.frames_idx, false, event.version)
                    break
                case 'free_requested':
                    l_block_map[event.addr].free_requested = false
                    break
                case 'free_completed':
                    l_block_map[event.addr] = Block(event.addr, event.size, event.size, event.frames_idx, true, event.version)
                    break
                case 'alloc':
                    delete l_block_map[event.addr]
                    break
                case 'segment_free':
                case 'segment_unmap':
                    map_segment(event.action == 'segment_unmap',
                                Segment(event.addr, event.size, event.stream, event.frames_idx, event.version))
                    break
                case 'segment_alloc':
                case 'segment_map':
                    unmap_segment(event.action == 'segment_map',
                                  Segment(event.addr, event.size, event.stream, event.frames_idx, event.version))
                    break
                case 'oom':
                    break
                default:
                    console.log(`unknown event: ${event.action}`)
                    break
            }
        }
        let new_blocks = Object.values(l_block_map)
        return [l_segments, new_blocks]
    }

    return {
        draw(idx) {
            let [segments_unsorted, blocks] = simulate_memory(idx)
            g.selectAll('g').remove()

            let segment_d = g.append('g')
            let block_g = g.append('g')
            let block_r = g.append('g')

            segment_d.selectAll('rect').remove()
            block_g.selectAll('rect').remove()
            block_r.selectAll('rect').remove()
            let segments = [...segments_unsorted].sort((x, y) => x.size == y.size ? (x.addr - y.addr) : (x.size - y.size))

            let segments_by_addr = [...segments].sort((x, y) => x.addr - y.addr)

            let max_size = segments.length == 0 ? 0 : segments.at(-1).size

            let xScale = scaleLinear([0, max_size], [0, 200])
            let padding = xScale.invert(1)

            let cur_row = 0
            let cur_row_size = 0
            for (let seg of segments) {
                seg.occupied = 0
                seg.internal_free = 0
                if (cur_row_size + seg.size > max_size) {
                    cur_row_size = 0
                    cur_row += 1
                }
                seg.offset = cur_row_size
                seg.row = cur_row
                cur_row_size += seg.size + padding
            }

            let num_rows = cur_row + 1

            let yScale = scaleLinear([0, num_rows], [0, 100])

            let segments_selection = segment_d.selectAll('rect').data(segments).enter()
            .append('rect')
            .attr('x', (x) => xScale(x.offset))
            .attr('y', (x) => yScale(x.row))
            .attr('width', (x) => xScale(x.size))
            .attr('height', yScale(4/5))
            .attr('stroke', 'black')
            .attr('stroke-width', '1')
            .attr('vector-effect', 'non-scaling-stroke')
            .attr('fill', 'white')

            stack_info.register(segments_selection,
                (d) => {
                    addStroke(d)
                    let t = d.datum()
                    let free = t.size - t.occupied
                    let internal = ""
                    if (t.internal_free > 0) {
                        internal = ` (${t.internal_free/free*100}% internal)`
                    }
                    return `s${t.addr.toString(16)}_${t.version}: segment ${formatSize(t.size)} allocated, ` +
                           `${formatSize(free)} free${internal} (stream ${t.stream})\n${t.frames}`
                },
                (d) => {
                    d.attr('stroke', 'black')
                    .attr('stroke-width', '1')
                    .attr('vector-effect', 'non-scaling-stroke')
                }
            )

            function find_segment(addr) {
                let left = 0;
                let right = segments_by_addr.length - 1;
                while (left <= right) {
                    let mid = Math.floor((left + right) / 2);
                    if (addr < segments_by_addr[mid].addr) {
                        right = mid - 1;
                    } else if (addr >= segments_by_addr[mid].addr + segments_by_addr[mid].size) {
                        left = mid + 1;
                    } else {
                        return segments_by_addr[mid];
                    }
                }
                return null;
            }

            for (let b of blocks) {
                b.segment = find_segment(b.addr)
                b.segment.occupied += b.real_size
                b.segment.internal_free += (b.size - b.real_size)
            }

            let block_selection = block_g.selectAll('rect').data(blocks).enter()
            .append('rect')
            .attr('x', (x) => xScale(x.segment.offset + (x.addr - x.segment.addr)))
            .attr('y', (x) => yScale(x.segment.row))
            .attr('width', (x) => xScale(x.real_size))
            .attr('height', yScale(4/5))
            .attr('fill', (x, i) => x.free_requested ? 'red' : schemeTableau10[Math.abs(hashCode(x.addr)) % schemeTableau10.length])

            stack_info.register(block_selection, (d) => {
                addStroke(d)
                let t = d.datum()
                let requested = ""
                if (t.free_requested) {
                    requested = " (block freed but waiting due to record_stream)"
                }
                return `b${t.addr.toString(16)}_${t.version} ` +
                       `${formatSize(t.real_size)} allocation${requested} (stream ${t.segment.stream})\n` + t.frames
            }, removeStroke)

            let free_selection = block_r.selectAll('rect').data(blocks).enter()
            .append('rect')
            .attr('x', (x) => xScale(x.segment.offset + (x.addr - x.segment.addr) + x.real_size))
            .attr('y', (x) => yScale(x.segment.row))
            .attr('width', (x) => xScale(x.size - x.real_size))
            .attr('height', yScale(4/5))
            .attr('fill', (x, i) => 'red')

            stack_info.register(free_selection, (d) => {
                addStroke(d)
                let t = d.datum()
                return `Free space lost due to rounding ${formatSize(t.size - t.real_size)}` +
                       ` (stream ${t.segment.stream})\n` + t.frames
            }, removeStroke)

            let reserved = segments.reduce((x, y) => x + y.size, 0)
            let allocated = blocks.reduce((x, y) => x + y.real_size, 0)
            return [reserved, allocated]
        }
    }
}

function StackInfo(outer) {
    let stack_trace = outer
    .append('pre')
    .attr("style", "grid-column: 1 / 3; grid-row: 2; overflow: auto")
    let selected = {
        enter: () => { stack_trace.text("") },
        leave: () => {},
    }
    return {
        register(dom, enter, leave = (e) => {}, select = (e) => {}) {
            dom.on('mouseover', (e) => {
                selected.leave()
                stack_trace.text(enter(d3.select(e.target)))
            })
            .on('mousedown', (e) => {
                let obj = d3.select(e.target)
                selected = {enter: () => stack_trace.text(enter(obj)), leave: () => leave(obj)}
                select(obj)
            })
            .on('mouseleave', (e) =>  {
                leave(d3.select(e.target))
                selected.enter()
            })
        },
        highlight(enter, leave = () => {}) {
            selected = {enter: () => stack_trace.text(enter()), leave: leave}
            selected.enter()
        }
    }
}

export function main(data) {
    trace_data = data;
    // add some supplement information to trace_data
    let events = createEvents()

    let body = d3.select("body")

    let outer = body.append("div")
    .attr('style', "display: grid; grid-template-columns: 1fr 2fr; grid-template-rows: 2fr 1fr; height: 100%; gap: 10px")

    let stack_info = StackInfo(outer)
    let memory_view = MemoryView(outer, stack_info, trace_data, events)
    let event_selector = EventSelector(body, outer, events, stack_info, memory_view)

    window.addEventListener('load', function() {
        event_selector.select(events.length > 0 ? events.length - 1 : null)
    });
}
