import pickle
import sys
import os
import io
import subprocess
import json
from functools import lru_cache
from typing import List, Tuple

cache = lru_cache(None)

__all__ = ["format_flamegraph", "segments", "memory", "compare"]

def _frame_fmt(f, full_filename=False):
    i = f['line']
    fname = f['filename']
    if not full_filename:
        fname = fname.split('/')[-1]
    func = f['name']
    return f'{fname}:{i}:{func}'

def format_flamegraph(flamegraph_lines, flamegraph_script=None):
    if flamegraph_script is None:
        flamegraph_script = f'/tmp/{os.getuid()}_flamegraph.pl'
    if not os.path.exists(flamegraph_script):
        import urllib.request
        print(f"Downloading flamegraph.pl to: {flamegraph_script}")
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl', flamegraph_script)
        subprocess.run(['chmod', '+x', flamegraph_script])
    args = [flamegraph_script, '--countname', 'bytes']
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf-8')
    assert p.stdin is not None
    assert p.stdout is not None
    p.stdin.write(flamegraph_lines)
    p.stdin.close()
    result = p.stdout.read()
    p.stdout.close()
    p.wait()
    assert p.wait() == 0
    return result

def _write_blocks(f, prefix, blocks):
    for b in blocks:
        if 'history' not in b:
            f.write(f'{prefix};{b["state"]} {b["size"]}\n')
            continue
        accounted_for_size = 0
        for h in b['history']:
            sz = h['real_size']
            accounted_for_size += sz
            if 'frames' in h:
                frames = h['frames']
                if frames:
                    frame_s = ';'.join([_frame_fmt(f) for f in reversed(frames)])
                else:
                    frame_s = "<non-python>"
                f.write(f'{prefix};{b["state"]};{frame_s} {sz}\n')
            else:
                f.write(f'{prefix};{b["state"]};<no-context> {sz}\n')
        gaps = b['size'] - accounted_for_size
        if gaps:
            f.write(f'{prefix};{b["state"]};<gaps> {gaps}\n')

def segments(snapshot, format_flamegraph=format_flamegraph):
    f = io.StringIO()
    for seg in snapshot['segments']:
        prefix = f'stream_{seg["stream"]};seg_{seg["address"]}'
        _write_blocks(f, prefix, seg['blocks'])
    return format_flamegraph(f.getvalue())

def memory(snapshot, format_flamegraph=format_flamegraph):
    f = io.StringIO()
    for seg in snapshot['segments']:
        prefix = f'stream_{seg["stream"]}'
        _write_blocks(f, prefix, seg['blocks'])
    return format_flamegraph(f.getvalue())

def compare(before, after, format_flamegraph=format_flamegraph):
    def _seg_key(seg):
        return (seg['address'], seg['total_size'])

    def _seg_info(seg):
        return f'stream_{seg["stream"]};seg_{seg["address"]}'

    f = io.StringIO()

    before_segs = {_seg_key(seg) for seg in before}
    after_segs = {_seg_key(seg) for seg in after}

    print(f'only_before = {[a for a,_ in (before_segs - after_segs)]}')
    print(f'only_after = {[a for a,_ in (after_segs - before_segs)]}')

    for seg in before:
        if _seg_key(seg) not in after_segs:
            _write_blocks(f, f'only_before;{_seg_info(seg)}', seg['blocks'])

    for seg in after:
        if _seg_key(seg) not in before_segs:
            _write_blocks(f, f'only_after;{_seg_info(seg)}', seg['blocks'])

    return format_flamegraph(f.getvalue())

def _format_size(num):
    # https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}B"
        num /= 1024.0
    return f"{num:.1f}YiB"

class Bytes:
    def __init__(self, value):
        self.value = value

    def __add__(self, rhs):
        return Bytes(self.value + rhs)

    def __repr__(self):
        return _format_size(self.value)

def calc_active(seg):
    return sum(b['size'] for b in seg['blocks'] if b['state'] == 'active_allocated')

def _report_free(free_external, free_internal):
    total = free_external + free_internal
    pct = (free_internal / total) * 100
    suffix = f' ({pct:.1f}% internal)'
    return f'{Bytes(total)}{suffix}'

PAGE_SIZE = 1024 * 1024 * 20
legend = f"""\

Legend:
    [a     ] - a segment in the allocator
     ^-- a page {Bytes(PAGE_SIZE)} of memory in the segment
    a-z: pages filled with a single block's content
    ' ': page is completely free
    *: page if completely full with multiple blocks
    0-9: page is partially full with tensors of multiple blocks (9 == 90% full)
    (X% internal) - of the free memory, X% is free because we rounded the size of the allocation.
"""

def segsum(data):
    """" Visually reports how the allocator has filled its segments. This printout can help debug fragmentation issues
    since free fragments will appear as gaps in this printout.  The amount of free space is reported for each segment.
    We distinguish between internal free memory which occurs because the allocator rounds the allocation size, and
    external free memory, which are the gaps between allocations in a segment.
    Args:
        data: snapshot dictionary created from _snapshot()

    """
    segments = []
    out = io.StringIO()
    out.write(f"Summary of segments >= {Bytes(PAGE_SIZE)} in size\n")
    total_reserved = 0
    total_allocated = 0
    free_external = 0
    free_internal = 0
    for seg in sorted(data['segments'], key=lambda x: (x['total_size'], calc_active(x))):
        total_reserved += seg['total_size']

        seg_free_external = 0
        seg_free_internal = 0
        seg_allocated = 0
        all_ranges = []
        boffset = 0
        for b in seg['blocks']:
            active = b['state'] == 'active_allocated'
            if 'history' in b:
                # use the more accureate real_size to account for internal fragmenetation if we have it
                for h in b['history']:
                    if active:
                        all_ranges.append((h['addr'] - seg['address'], h['real_size'], active))
                        seg_allocated += h['real_size']
                        assert len(b['history']) == 1
                        seg_free_internal += b['size'] - h['real_size']
            else:
                if active:
                    all_ranges.append((boffset, b['size'], True))
                    seg_allocated += b['size']
            if not active:
                seg_free_external += b['size']

            boffset += b['size']

        total_allocated += seg_allocated
        free_external += seg_free_external
        free_internal += seg_free_internal

        nseg = (seg['total_size'] - 1) // PAGE_SIZE + 1
        occupied = [' ' for _ in range(nseg)]
        frac = [0.0 for _ in range(nseg)]
        active_size = 0
        for i, (start_, size, active) in enumerate(all_ranges):
            active_size += size
            finish_ = (start_ + size)
            start = start_ // PAGE_SIZE
            finish = (finish_ - 1) // PAGE_SIZE + 1
            m = chr((ord('a' if active else 'A') + (i % 26)))
            for j in range(start, finish):
                s = max(start_, j * PAGE_SIZE)
                e = min(finish_, (j + 1) * PAGE_SIZE)
                frac[j] += (e - s) / PAGE_SIZE
                if occupied[j] != ' ':
                    occupied[j] = '0123456789*'[int(frac[j] * 10)]
                else:
                    occupied[j] = m
        stream = '' if seg['stream'] == 0 else f', stream_{seg["stream"]}'
        body = ''.join(occupied)
        assert seg_free_external + seg_free_internal + seg_allocated == seg['total_size']
        stream = f' stream_{seg["stream"]}' if seg['stream'] != 0 else ''
        if seg['total_size'] >= PAGE_SIZE:
            out.write(f'[{body}] {Bytes(seg["total_size"])} allocated, '
                      f'{_report_free(seg_free_external, seg_free_internal)} free{stream}\n')
    out.write(f'segments: {len(data["segments"])}\n')
    out.write(f'total_reserved: {Bytes(total_reserved)}\n')
    out.write(f'total_allocated: {Bytes(total_allocated)}\n')
    internal_external = f' ({Bytes(free_internal)} internal + {Bytes(free_external)} external)' if free_internal else ''
    out.write(f'total_free: {_report_free(free_external, free_internal)}\n')
    out.write(legend)
    assert free_internal + free_external + total_allocated == total_reserved
    return out.getvalue()

def trace(data):
    out = io.StringIO()

    def format(entries):
        segment_intervals : list = []
        segment_addr_to_name = {}
        allocation_addr_to_name = {}

        free_names : list = []
        next_name = 0

        def _name():
            nonlocal next_name
            if free_names:
                return free_names.pop()
            r, m = next_name // 26, next_name % 26
            next_name += 1
            return f'{chr(ord("a") + m)}{"" if r == 0 else r}'

        def find_segment(addr):
            for name, saddr, size in segment_intervals:
                if addr >= saddr and addr < saddr + size:
                    return name, saddr
            for i, seg in enumerate(data['segments']):
                saddr = seg['address']
                size = seg['allocated_size']
                if addr >= saddr and addr < saddr + size:
                    return f'seg_{i}', saddr
            return None, None
        count = 0
        out.write(f'{len(entries)} entries\n')


        total_reserved = 0
        for seg in data['segments']:
            total_reserved += seg['total_size']

        for count, e in enumerate(entries):
            if e['action'] == 'alloc':
                addr, size = e['addr'], e['size']
                n = _name()
                seg_name, seg_addr = find_segment(addr)
                if seg_name is None:
                    seg_name = "MEM"
                    offset = addr
                else:
                    offset = addr - seg_addr
                out.write(f'{n} = {seg_name}[{offset}:{Bytes(size)}]\n')
                allocation_addr_to_name[addr] = (n, size, count)
                count += size
            elif e['action'] == 'free_requested':
                addr, size = e['addr'], e['size']
                name, _, _ = allocation_addr_to_name.get(addr, (addr, None, None))
                out.write(f'del {name} # {Bytes(size)}\n')
            elif e['action'] == 'free_completed':
                addr, size = e['addr'], e['size']
                count -= size
                name, _, _ = allocation_addr_to_name.get(addr, (addr, None, None))
                out.write(f'# free completed for {name} {Bytes(size)}\n')
                if name in allocation_addr_to_name:
                    free_names.append(name)
                    del allocation_addr_to_name[name]
            elif e['action'] == 'segment_alloc':
                addr, size = e['addr'], e['size']
                name = _name()
                out.write(f'{name} = cudaMalloc({addr}, {Bytes(size)})\n')
                segment_intervals.append((name, addr, size))
                segment_addr_to_name[addr] = name
            elif e['action'] == 'segment_free':
                addr, size = e['addr'], e['size']
                name = segment_addr_to_name.get(addr, addr)
                out.write(f'cudaFree({name}) # {Bytes(size)}\n')
                if name in segment_addr_to_name:
                    free_names.append(name)
                    del segment_addr_to_name[name]
            elif e['action'] == 'oom':
                size = e['size']
                free = e['device_free']
                out.write(f'raise OutOfMemoryError() # {Bytes(size)} requested, {Bytes(free)} free in CUDA\n')
            else:
                out.write(f'{e}\n')
        out.write(f"TOTAL MEM: {Bytes(count)}")
    for i, d in enumerate(data['device_traces']):
        if d:
            out.write(f'Device {i} ----------------\n')
            format(d)
    return out.getvalue()

class PlotWriter:
    def __init__(self):
        string_table: List[str] = []
        suffix_table: List[Tuple[int, int]] = []

        elements = []
        actions: List[int] = []

        initially_allocated: List[int] = []

        @cache
        def intern_str(s):
            string_table.append(s)
            return len(string_table) - 1

        @cache
        def intern_suffix(sid, restid):
            suffix_table.append((sid, restid))
            return len(suffix_table) - 1


        def intern_stack(frames):
            sids = [intern_str(f) for f in frames]
            next_id = None
            for sid in reversed(sids):
                next_id = intern_suffix(sid, next_id)
            return next_id

        def add_element(size, lines):
            elements.append({'size': size, 'info': intern_stack(lines)})
            return len(elements) - 1

        def to_html():
            r = {
                'actions': actions,
                'elements': elements,
                'suffix_table': suffix_table,
                'string_table': string_table,
                'initially_allocated': list(reversed(initially_allocated)),
            }
            plot_data = json.dumps(r)
            return _memory_over_time_template.replace('$PLOT_DATA', plot_data)

        self.add_element = add_element
        self.allocate = actions.append
        self.free = actions.append
        self.initially_allocated = initially_allocated.append
        self.to_html = to_html

def trace_plot(data, device=None, plot_segments=False):
    w = PlotWriter()
    addr_to_alloc = {}

    if device is None:
        for i, t in enumerate(data['device_traces']):
            if len(t) > 0:
                if device is not None:
                    raise ValueError(f'Both device {device} and {i} have traces, use --device to specify which trace.')
                device = i
        if device is None:
            raise ValueError('No trace information was recorded.')

    trace = data['device_traces'][device]

    if plot_segments:
        alloc = 'segment_alloc'
        free = 'segment_free'
    else:
        alloc = 'alloc'
        free = 'free_completed'

    def add_element(size, frames, extra=()):
        frames = [f"{_format_size(size)} allocation", *extra, *(_frame_fmt(f, full_filename=True) for f in frames)]
        return w.add_element(size, frames)

    for i, e in enumerate(trace):
        if e['action'] == alloc:
            elemid = add_element(e['size'], e['frames'])
            addr_to_alloc[e['addr']] = elemid
            w.allocate(elemid)
        elif e['action'] == free:
            idx = addr_to_alloc.pop(e['addr'], None)
            if idx is None:
                idx = add_element(e['size'], e['frames'], extra=('alloc not recorded, stack trace for free:',))
                w.initially_allocated(idx)
            w.free(idx)
    return w.to_html()

def profile_plot(memory_profile, device=None):
    import torch
    from torch.profiler._memory_profiler import Action, TensorKey
    from torch._C._profiler import _EventType

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda', torch.cuda.current_device())
        else:
            device = torch.device('cpu')
    w = PlotWriter()


    allocation_stacks = {}
    for event in memory_profile._op_tree.sorted_nodes:
        if event.tag == _EventType.Allocation:
            parent = event.parent
            python_parents = []
            while parent:
                if parent.tag in (_EventType.PyCall, _EventType.PyCCall):
                    python_parents.append(parent)
                parent = parent.parent
            key = TensorKey.from_allocation(event.extra_fields)

            # Corner case: If allocation doesn't have an ID (can't prove it was used as a Tensor)
            #              key will be None. I should add some way to identify these, I just haven't yet.
            if key and event.extra_fields.alloc_size > 0:
                allocation_stacks[key] = python_parents

    def add_element(size, tensor_key, version):
        category = memory_profile._categories.get(tensor_key, version)
        if category is None:
            category = 'unknown'
        else:
            category = category.name.lower()
        stack = allocation_stacks.get(tensor_key, ())
        return w.add_element(size, [f"{_format_size(size)} allocation ({category})", *(p.name for p in stack)])

    kv_to_elem = {}
    for time, action, (tensor_key, version), size in memory_profile.timeline:
        if tensor_key.device != device:
            continue
        if action == Action.CREATE:
            kv_to_elem[(tensor_key, version)] = elemid = add_element(size, tensor_key, version)
            w.allocate(elemid)
        elif action == Action.DESTROY:
            w.free(kv_to_elem.pop((tensor_key, version)))
        elif action == Action.INCREMENT_VERSION:
            w.free(kv_to_elem.pop((tensor_key, version)))
            kv_to_elem[(tensor_key, version + 1)] = elemid = add_element(size, tensor_key, version + 1)
            w.allocate(elemid)
        elif action == Action.PREEXISTING:
            kv_to_elem[(tensor_key, version)] = elemid = add_element(size, tensor_key, version)
            w.initially_allocated(elemid)
    return w.to_html()

# note: this template should eventually move to its own file,
# however, we first need to package _memory_viz.py so that it can be
# pip-installed separately from pytorch so it is easy to run e.g.
# on a laptop with downloaded snapshots. Currently this is
# accomplished by downloading _memory_viz.py so the template
# needs to be included
_memory_over_time_template = r"""
<!DOCTYPE html>
<html>
<head></head>
<body>
<script type="module">
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.7.0/+esm";
import {schemeTableau10} from "https://cdn.skypack.dev/d3-scale-chromatic@3";
import {axisLeft} from "https://cdn.skypack.dev/d3-axis@3";
import {scaleLinear} from "https://cdn.skypack.dev/d3-scale@4";
import {zoom, zoomIdentity} from "https://cdn.skypack.dev/d3-zoom@3";
import {brushX} from "https://cdn.skypack.dev/d3-brush@3";

let alloc_data = $PLOT_DATA

function process_alloc_data(fraction_of_memory_reported=1) {
    let current = []
    let current_data = []
    let data = []
    let max_size = 0

    let total_mem = 0
    let timestep = 0

    let max_at_time = []
    function advance(n, max) {
        timestep += n
        for (let i = 0; i < n; i++) {
            max_at_time.push(max)
        }
    }

    let mini_points = []

    let sizes = alloc_data.elements.map(x => x.size).sort((x, y) => y - x)
    let total_size = sizes.reduce((x, y) => x + y)
    const memory_threshold = fraction_of_memory_reported * total_size
    let total_seen = 0
    let memory_threshold_size = 0

    for (const [i, size] of sizes.entries()) {
        total_seen += size
        if (total_seen > memory_threshold) {
            memory_threshold_size = size
            break
        }
    }

    function add_allocation(elem) {
        let size = alloc_data.elements[elem].size
        current.push(elem)
        let e = {elem: elem, timesteps: [timestep], offsets: [total_mem], size: alloc_data.elements[elem].size}
        current_data.push(e)
        data.push(e)
        total_mem += size
    }

    for (const elem of alloc_data.initially_allocated) {
        add_allocation(elem)
    }

    for (const action of alloc_data.actions) {
        const elem = action
        const idx = current.findIndex(x => x === elem)
        const size = alloc_data.elements[elem].size
        if (size < memory_threshold_size) {
            continue
        }
        // first time we see an action we add it
        // second time we remove it
        if (idx == -1) {
            add_allocation(elem)
            advance(1, total_mem)
        } else {
            advance(1, total_mem)
            const removed = current_data[idx]
            removed.timesteps.push(timestep)
            removed.offsets.push(removed.offsets.at(-1))
            current.splice(idx, 1)
            current_data.splice(idx, 1)

            if (idx < current.length) {
                for (let j = idx; j < current.length; j++) {
                    const e = current_data[j]
                    e.timesteps.push(timestep)
                    e.offsets.push(e.offsets.at(-1))
                    e.timesteps.push(timestep + 3)
                    e.offsets.push(e.offsets.at(-1) - size)
                }
                advance(3, total_mem)
            }
            total_mem -= size
        }
        max_size = Math.max(total_mem, max_size)
    }

    for (const elem of current_data) {
        elem.timesteps.push(timestep)
        elem.offsets.push(elem.offsets.at(-1))
    }
    return {
        max_size: max_size,
        allocations_over_time: data,
        max_at_time: max_at_time,
        context_for_id:  (elem) => {
            let strings = []
            let id = alloc_data.elements[elem].info
            while (id !== null) {
                const [sid, next_id] = alloc_data.suffix_table[id]
                strings.push(alloc_data.string_table[sid])
                id = next_id
            }
            return `${strings.join('\n')}\n`
        }
    }
}

function MemoryPlot(svg, data, left_pad, colors=schemeTableau10) {
    function format_points(d) {
        const size = d.size
        const xs = d.timesteps.map(t => xscale(t))
        const bottom = d.offsets.map(t => yscale(t))
        const top = d.offsets.map(t => yscale(t + size))

        const p0 = xs.map((x, i) => `${x},${bottom[i]}`)
        const p1 = xs.map((x, i) => `${x},${top[i]}`).reverse()

        return `${p0.join(' ')} ${p1.join(' ')}`
    }

    let max_timestep = data.max_at_time.length
    let max_size = data.max_size

    let width = svg.attr('width')
    let height = svg.attr('height')
    let plot_width = width - left_pad
    let plot_height = height

    let yscale = scaleLinear().domain([0, max_size]).range([plot_height, 0]);
    let heightscale = scaleLinear().domain([0, max_size]).range([0, plot_height]);
    let yaxis = axisLeft(yscale).tickFormat(d3.format("~s"))
    let xscale = scaleLinear().domain([0, max_timestep]).range([0, plot_width])
    let plot_coordinate_space = svg.append("g").attr("transform", `translate(${left_pad}, ${0})`)
    let plot_outer = plot_coordinate_space.append('g')

    function view_rect(a) {
        return a.append('rect').attr('x', 0).attr('y', 0)
                .attr('width', plot_width).attr('height', plot_height)
                .attr('fill', 'white')
    }

    view_rect(plot_outer)

    let cp = svg.append("clipPath").attr("id", "clip")
    view_rect(cp)
    plot_outer.attr('clip-path', "url(#clip)")


    let zoom_group = plot_outer.append("g")
    let scrub_group = zoom_group.append('g')

    let plot = scrub_group.selectAll("polygon")
    .data(data.allocations_over_time)
    .enter()
    .append("polygon")
    .attr('points', format_points)
    .attr('fill', d => colors[d.elem % colors.length])

    let axis = plot_coordinate_space.append('g').call(yaxis)


    let scale_mini = 0
    let translate_mini = 0
    function handleZoom(e) {
        const t = e.transform
        zoom_group.attr("transform", t)
        axis.call(yaxis.scale(e.transform.rescaleY(yscale)))
    }

    const thezoom = zoom().on('zoom', handleZoom)
    plot_outer.call(thezoom)

    return {
        select_window: (stepbegin, stepend, max) => {
            let begin = xscale(stepbegin)
            let size = xscale(stepend) - xscale(stepbegin);
            let scale = plot_width / size
            let translate = -begin
            let yscale =  max_size/max
            scrub_group.attr("transform", `scale(${scale/yscale}, 1) translate(${translate}, 0)`)
            plot_outer.call(thezoom.transform, zoomIdentity.scale(yscale).translate(0, -(plot_height - plot_height/yscale)))
        },
        set_delegate: (delegate) => {
            plot.on('mouseover', function (e, d) { delegate.set_selected(d3.select(this)) } )
            .on('mousedown', function(e, d) { delegate.default_selected = d3.select(this)})
            .on('mouseleave', function (e, d) { delegate.set_selected(delegate.default_selected) } )
        }
    }
}

function ContextViewer(text, data) {
    let current_selected = null

    return {
        default_selected: null,
        set_selected: (d) => {
            if (current_selected !== null) {
                current_selected.attr('stroke', null).attr('stroke-width', null);
            }
            if (d === null) {
                text.text("")
            } else {
                const dd = d.datum()
                text.text(`${dd.elem} ${data.context_for_id(dd.elem)}`)
                d.attr('stroke', 'black').attr('stroke-width', 1).attr('vector-effect', 'non-scaling-stroke')
            }
            current_selected = d
        }
    }
}


function MiniMap(mini_svg, plot, data, left_pad, height=70) {
    let max_at_time = data.max_at_time
    let width = mini_svg.attr('width')
    let plot_width = width - left_pad
    let yscale = scaleLinear().domain([0, data.max_size]).range([height, 0]);
    let minixscale = scaleLinear().domain([0, max_at_time.length]).range([left_pad, width])

    let mini_points = [[max_at_time.length, 0], [0, 0]]

    for (const [i, m] of max_at_time.entries()) {
        let [lastx, lasty] = mini_points[mini_points.length - 1]
        if (m !== lasty) {
            mini_points.push([i, lasty])
            mini_points.push([i, m])
        } else if (i === max_at_time.length - 1) {
            mini_points.push([i, m])
        }
    }


    let points = mini_points.map(([t, o]) => `${minixscale(t)}, ${yscale(o)}`)
    points = points.join(' ')
    mini_svg.append('polygon').attr('points', points).attr('fill', schemeTableau10[0])

    let xscale = scaleLinear().domain([0, max_at_time.length]).range([0, plot_width])


    const brush = brushX()
    brush.extent([[left_pad, 0], [width, height]])
    brush.on('brush', function({selection}) {
        let [begin, end] = selection.map(x => x - left_pad)

        let stepbegin = Math.floor(xscale.invert(begin))
        let stepend = Math.floor(xscale.invert(end))
        let max = 0
        for (let i = stepbegin; i < stepend; i++) {
            max = Math.max(max, max_at_time[i])
        }
        plot.select_window(stepbegin, stepend, max)
    })
    mini_svg.call(brush)
    return {}
}

let left_pad = 70
let width = 1024
let height = 768
let data = process_alloc_data()
let body = d3.select("body")

let plot = MemoryPlot(body.append("svg").attr('width', width).attr('height', height).attr('display', 'block'), data, left_pad)

MiniMap(body.append("svg").attr('width', width).attr('height', 80).attr('display', 'block'), plot, data, left_pad)
let delegate = ContextViewer(body.append("div").append("pre").text('none'), data)
plot.set_delegate(delegate)

</script>
</body>
</html>
"""

if __name__ == "__main__":
    import os.path
    thedir = os.path.realpath(os.path.dirname(__file__))
    if thedir in sys.path:
        # otherwise we find cuda/random.py as random...
        sys.path.remove(thedir)
    import argparse

    fn_name = 'torch.cuda.memory._snapshot()'
    pickled = f'pickled memory statistics from {fn_name}'
    parser = argparse.ArgumentParser(description=f'Visualize memory dumps produced by {fn_name}')

    subparsers = parser.add_subparsers(dest='action')

    def _output(p):
        p.add_argument('-o', '--output', default='output.svg', help='flamegraph svg (default: output.svg)')

    description = 'Prints overall allocation statistics and a visualization of how the allocators segments are currently filled.'
    stats_a = subparsers.add_parser('stats', description=description)
    stats_a.add_argument('input', help=pickled)

    description = 'Prints buffer of the most recent allocation events embedded in the snapshot in a Pythonic style.'
    trace_a = subparsers.add_parser('trace', description=description)
    trace_a.add_argument('input', help=pickled)

    description = 'Generate a flamegraph that visualizes what memory is stored in each allocator segment (aka block)'
    segments_a = subparsers.add_parser('segments', description=description)
    segments_a.add_argument('input', help=pickled)
    _output(segments_a)

    description = "Generate a flamegraph the program locations contributing to CUDA memory usage."
    memory_a = subparsers.add_parser('memory', description=description)
    memory_a.add_argument('input', help=pickled)
    _output(memory_a)

    description = 'Generate a flamegraph that shows segments (aka blocks) that have been added ' \
        'or removed between two different memorys snapshots.'
    compare_a = subparsers.add_parser('compare', description=description)
    compare_a.add_argument('before', help=pickled)
    compare_a.add_argument('after', help=pickled)
    _output(compare_a)

    description = "Generate a visualization over time of the memory usage recorded by the trace as an html file."
    trace_plot_a = subparsers.add_parser('trace_plot', description=description)
    trace_plot_a.add_argument('input', help=pickled)
    help = 'visualize trace from this device (default: chooses the only device with trace info or errors)'
    trace_plot_a.add_argument('-d', '--device', type=int, default=None, help=help)
    help = 'path to save the visualization(default: output.html)'
    trace_plot_a.add_argument('-o', '--output', default='output.html', help=help)
    help = 'visualize change to segments rather than individual allocations'
    trace_plot_a.add_argument('-s', '--segments', action='store_true', help=help)


    args = parser.parse_args()

    def _read(name):
        if name == '-':
            f = sys.stdin.buffer
        else:
            f = open(name, 'rb')
        data = pickle.load(f)
        if isinstance(data, list):  # segments only...
            data = {'segments': data, 'traces': []}
        return data

    def _write(name, data):
        with open(name, 'w') as f:
            f.write(data)

    if args.action == 'segments':
        data = _read(args.input)
        _write(args.output, segments(data))
    elif args.action == 'memory':
        data = _read(args.input)
        _write(args.output, memory(data))
    elif args.action == 'stats':
        data = _read(args.input)
        print(segsum(data))
    elif args.action == 'trace':
        data = _read(args.input)
        print(trace(data))
    elif args.action == 'compare':
        before = _read(args.before)
        after = _read(args.after)
        _write(args.output, compare(before, after))
    elif args.action == 'trace_plot':
        data = _read(args.input)
        _write(args.output, trace_plot(data, device=args.device, plot_segments=args.segments))
