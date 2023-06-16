import pickle
import sys
import os
import io
import subprocess
import json
from functools import lru_cache
from typing import List, Tuple, Optional, Any, Dict

cache = lru_cache(None)

__all__ = ["format_flamegraph", "segments", "memory", "compare"]

def _frame_fmt(f, full_filename=False):
    i = f['line']
    fname = f['filename']
    if not full_filename:
        fname = fname.split('/')[-1]
    func = f['name']
    return f'{fname}:{i}:{func}'

@cache
def _frame_filter(name, filename):
    omit_functions = [
        "unwind::unwind",
        "CapturedTraceback::gather",
        "gather_with_cpp",
        "_start",
        "__libc_start_main",
        "PyEval_",
        "PyObject_",
        "PyFunction_",
    ]
    omit_filenames = [
        "core/boxing",
        "/Register",
        "/Redispatch",
        "pythonrun.c",
        "Modules/main.c",
        "Objects/call.c",
        "Objects/methodobject.c",
        "pycore_ceval.h",
        "ceval.c",
        "cpython/abstract.h",
    ]
    for of in omit_functions:
        if of in name:
            return False
    for of in omit_filenames:
        if of in filename:
            return False
    return True

def _frames_fmt(frames, full_filename=False, reverse=False):
    if reverse:
        frames = reversed(frames)
    return [_frame_fmt(f, full_filename) for f in frames if _frame_filter(f['name'], f['filename'])]

def _block_extra(b):
    if 'history' in b:
        frames = b['history'][0].get('frames', [])
        real_size = b['history'][0]['real_size']
    else:
        real_size = b.get('requested_size', b['size'])
        frames = []
    return frames, real_size

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
                    frame_s = ';'.join(_frames_fmt(frames, reverse=True))
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
    suffix = ''
    if total != 0:
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
                # use the more accurate real_size to account for internal fragmentation if we have it
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
    def __init__(self, categories: List[str] = None):
        string_table: List[str] = []

        # compresses lists of strings that have common suffixes
        # such as stack traces with the most recent frame first.
        # (reference to string table, another suffix table entry)
        suffix_table: List[Tuple[int, Optional[int]]] = []

        elements_size = []

        # indexes into the suffix_table
        elements_info = []

        # indexes into category table
        elements_category: Optional[List[int]] = None if categories is None else []

        # indexes into the elements_ tables
        actions: List[int] = []

        # indexes into the elements_ tables
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
            next_id = None
            for f in reversed(frames):
                next_id = intern_suffix(intern_str(f), next_id)
            return next_id

        def add_element(size, lines, category=None):
            nonlocal elements_category
            # note: struct of arrays format to info about elements
            # avoids a lot of repeated string keys when serialized
            elements_size.append(size)
            elements_info.append(intern_stack(lines))

            # lazily create since we will not always have categories
            if categories is not None:
                assert category >= 0 and category < len(categories)
                assert elements_category is not None
                elements_category.append(category)

            return len(elements_size) - 1

        def to_html():
            r = {
                'actions': actions,
                'elements_size': elements_size,
                'elements_info': elements_info,
                'elements_category': elements_category,
                'suffix_table': suffix_table,
                'string_table': string_table,
                'initially_allocated': list(reversed(initially_allocated)),
                'categories': categories,
            }
            plot_data = json.dumps(r)
            return _memory_over_time_template.replace('$PLOT_DATA', plot_data)

        self.add_element = add_element
        self.allocate = actions.append
        self.free = actions.append
        self.initially_allocated = initially_allocated.append
        self.to_html = to_html
        self.categories = categories

def _choose_device(data, device):
    if device is None:
        for i, t in enumerate(data['device_traces']):
            if len(t) > 0:
                if device is not None:
                    raise ValueError(f'Both device {device} and {i} have traces, use --device to specify which trace.')
                device = i
    return device

def trace_plot(data, device=None, plot_segments=False):
    """Generate a visualization over time of the memory usage recorded by the trace as an html file.

    Args:
        data: Memory snapshot as generated from torch.cuda.memory._snapshot()
        device (torch.device, optional): Generate the trace for this device, needed if multiple devices have allocations.
        plot_segments (bool, optional): Plots memory returned from cudaMalloc, rather than individual allocations.
                                        Defaults to False.

    Returns:
        str: HTML of visualization
    """
    w = PlotWriter()
    addr_to_alloc = {}
    device = _choose_device(data, device)
    if device is None:
        raise ValueError('No trace information was recorded.')

    trace = data['device_traces'][device]

    if plot_segments:
        addr_prefix = 's'
        alloc = 'segment_alloc'
        free = 'segment_free'
    else:
        addr_prefix = 'b'
        alloc = 'alloc'
        free = 'free_completed'

    addr_versions: Dict[int, int] = {}

    def add_element(addr, size, frames, extra=()):
        next_version = addr_versions[addr] = addr_versions.get(addr, 0) + 1
        frames = [f"{addr_prefix}{addr:x}_{next_version - 1} {_format_size(size)} allocation ({size} bytes)",
                  *extra,
                  *_frames_fmt(frames, full_filename=True)]
        return w.add_element(size, frames)

    for i, e in enumerate(trace):
        if e['action'] == alloc:
            elemid = add_element(e['addr'], e['size'], e.get('frames', []))
            addr_to_alloc[e['addr']] = elemid
            w.allocate(elemid)
        elif e['action'] == free:
            idx = addr_to_alloc.pop(e['addr'], None)
            if idx is None:
                idx = add_element(e['addr'], e['size'], e.get('frames', []), extra=('alloc not recorded, stack trace for free:',))
                w.initially_allocated(idx)
            w.free(idx)

    for seg in data['segments']:
        if seg['device'] != device:
            continue
        addr = seg['address']
        for b in seg['blocks']:
            if b['state'] == 'active_allocated' and addr not in addr_to_alloc:
                frames, real_size = _block_extra(b)
                extra = () if frames else ('<block was allocated before _record_history was enabled>',)
                elemid = add_element(addr, real_size, frames, extra=extra)
                w.initially_allocated(elemid)
            addr += b['size']

    return w.to_html()

def profile_plot(profile, device=None):
    """Generate a visualization over time of the memory usage recorded by kineto memory profiling as an html file.

    Args:
        profile: profile as generated by `torch.profiler.profile(profile_memory=True)`
        device (torch.device, optional): Generate the trace for this device, needed if multiple devices have allocations.

    Returns:
        str: HTML of visualization
    """
    import torch
    from torch.profiler._memory_profiler import Action, TensorKey, Category
    from torch._C._profiler import _EventType
    memory_profile = profile._memory_profile()

    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda', torch.cuda.current_device())
        else:
            device = torch.device('cpu')

    w = PlotWriter(categories=["unknown", *(c.name.lower() for c in Category)])


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
        category = category.value if category is not None else 0
        stack = allocation_stacks.get(tensor_key, ())
        assert w.categories is not None
        return w.add_element(size,
                             [f"{_format_size(size)} ({size} bytes) allocation ({w.categories[category]})",
                              *(p.name for p in stack)],
                             category)

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

function process_alloc_data(max_entries) {
    let current = []
    let current_data = []
    let data = []
    let max_size = 0

    let total_mem = 0
    let total_summarized_mem = 0
    let timestep = 0

    let max_at_time = []


    let summarized_mem = {elem: 'summarized', timesteps: [], offsets: [total_mem], size: [], color: 0}
    let summarized_elems = {}

    function advance(n) {
        summarized_mem.timesteps.push(timestep)
        summarized_mem.offsets.push(total_mem)
        summarized_mem.size.push(total_summarized_mem)
        timestep += n
        for (let i = 0; i < n; i++) {
            max_at_time.push(total_mem + total_summarized_mem)
        }
    }

    let sizes = alloc_data.elements_size.map((x, i) => [x, i]).sort(([x, xi], [y, yi]) => y - x)

    let draw_elem = {}
    for (const [s, e] of sizes.slice(0, max_entries)) {
        draw_elem[e] = true
    }

    function add_allocation(elem) {
        let size = alloc_data.elements_size[elem]
        current.push(elem)
        let color = elem
        if (alloc_data.elements_category !== null) {
            color = alloc_data.elements_category[elem]
        }
        let e = {elem: elem, timesteps: [timestep], offsets: [total_mem], size: size, color: color}
        current_data.push(e)
        data.push(e)
        total_mem += size
    }

    for (const elem of alloc_data.initially_allocated) {
        if (elem in draw_elem) {
            add_allocation(elem)
        } else {
            total_summarized_mem += alloc_data.elements_size[elem]
            summarized_elems[elem] = true
        }
    }

    for (const action of alloc_data.actions) {
        const elem = action
        const size = alloc_data.elements_size[elem]
        if ( !(elem in draw_elem)) {
            if (elem in summarized_elems) {
                advance(1)
                total_summarized_mem -= size
                summarized_elems[elem] = null
            } else {
                total_summarized_mem += size
                summarized_elems[elem] = true
                advance(1)
            }
            continue
        }
        const idx = current.findLastIndex(x => x === elem)
        // first time we see an action we add it
        // second time we remove it
        if (idx == -1) {
            add_allocation(elem)
            advance(1)
        } else {
            advance(1)
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
                advance(3)
            }
            total_mem -= size
        }
        max_size = Math.max(total_mem + total_summarized_mem, max_size)
    }

    for (const elem of current_data) {
        elem.timesteps.push(timestep)
        elem.offsets.push(elem.offsets.at(-1))
    }
    data.push(summarized_mem)

    return {
        max_size: max_size,
        allocations_over_time: data,
        max_at_time: max_at_time,
        summarized_mem: summarized_mem,
        context_for_id:  (elem) => {
            let strings = []
            let id = alloc_data.elements_info[elem]
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
        const m = Array.isArray(size) ? ((t, i) => yscale(t + size[i]))
                                      :  (t => yscale(t + size))
        const top = d.offsets.map(m)
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
    .attr('fill', d => colors[d.color % colors.length])

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
                if (dd.elem === 'summarized') {
                    text.html(
                        "Small tensors that were not plotted to cutdown on render time.\n" +
                        "Use detail slider to see smaller allocations.")
                } else {
                    text.text(`${dd.elem} ${data.context_for_id(dd.elem)}`)
                }
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

function Legend(plot_svg, categories, width) {
    let xstart = width - 100
    let ystart = 30
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


function create(max_entries) {
    let left_pad = 70
    let width = 1024
    let height = 768
    let data = process_alloc_data(max_entries)
    let body = d3.select("body")
    body.selectAll('svg').remove()
    body.selectAll('div').remove()

    if (alloc_data.elements_info.length > max_entries) {
         let d = body.append('div')
         d.append('input')
         .attr("type", "range")
         .attr('min', 0)
         .attr('max', alloc_data.elements_info.length)
         .attr("value", max_entries)
         .on('change', function() {
            create(this.value)
         })
         d.append('label').text('Detail')
    }

    let plot_svg = body.append("svg").attr('width', width).attr('height', height).attr('display', 'block')
    let plot = MemoryPlot(plot_svg, data, left_pad)

    if (alloc_data.categories !== null) {
        Legend(plot_svg.append('g'), alloc_data.categories, width)
    }

    MiniMap(body.append("svg").attr('width', width).attr('height', 80).attr('display', 'block'), plot, data, left_pad)
    let delegate = ContextViewer(body.append("div").append("pre").text('none'), data)
    plot.set_delegate(delegate)
}

create(15000)

</script>
</body>
</html>
"""

def segment_plot(data: Any, device=None):
    device = _choose_device(data, device)
    if device is None:
        trace = []
    else:
        trace = data['device_traces'][device]

    string_table: List[str] = []
    suffix_table: List[Tuple[int, Optional[int]]] = []

    @cache
    def intern_str(s):
        string_table.append(s)
        return len(string_table) - 1

    @cache
    def intern_suffix(sid, restid):
        suffix_table.append((sid, restid))
        return len(suffix_table) - 1

    def intern_stack(frames):
        next_id = None
        for f in reversed(frames):
            next_id = intern_suffix(intern_str(f), next_id)
        return next_id

    def format_frames(frames):
        return intern_stack(_frames_fmt(frames, full_filename=True))

    result: Any = {
        'string_table': string_table,
        'suffix_table': suffix_table,
        'events': {
            'action': [],  # reference to string table
            'addr': [],  # for OOM, this will hold device_free value
            'size': [],
            'stream': [],
            'frames': []  # reference to suffix_table
        },
        'segments': {
            'addr': [],
            'size': [],
            'stream': []
        },
        'blocks': {
            'addr': [],
            'size': [],
            'real_size': [],
            'frames': [],  # reference to string table
            'pending_free': [],
        }
    }

    def fold_free(ts):
        # turn a free_requested/free_completed pair into a single free event
        i = 0
        while i < len(ts):
            t = ts[i]
            if i + 1 < len(ts):
                tnext = ts[i + 1]
                if t['action'] == 'free_requested' and tnext['action'] == 'free_completed' and t['addr'] == tnext['addr']:
                    yield {**t, 'action': 'free'}
                    i += 2
                    continue
            if t['action'] == 'oom':
                yield {**t, 'addr': t['device_free']}
            else:
                yield t
            i += 1

    preproc: Any = {
        'action': intern_str,
        'frames': format_frames,
    }

    events: Any = result['events']
    for event in fold_free(trace):
        for k in events.keys():
            # stack frames not recorded on event
            # happens for snapshot even when
            # frames are recorded for other things.
            if k == 'frames' and k not in event:
                events[k].append(None)
                continue
            events[k].append(preproc.get(k, lambda x: x)(event[k]))

    segments = result['segments']
    blocks = result['blocks']

    segment_names = {
        'addr': 'address',
        'size': 'total_size',
    }

    for seg in data['segments']:
        if seg['device'] != device:
            continue
        for k in segments.keys():
            sk = segment_names.get(k, k)
            segments[k].append(preproc.get(k, lambda x: x)(seg[sk]))
        addr = seg['address']
        for b in seg['blocks']:
            if b['state'] in ('active_pending_free', 'active_allocated'):
                frames, real_size = _block_extra(b)
                blocks['addr'].append(addr)
                blocks['size'].append(b['size'])
                blocks['real_size'].append(real_size)
                blocks['frames'].append(format_frames(frames))
                blocks['pending_free'].append(1 if b['state'] == 'active_pending_free' else 0)
            addr += b['size']

    plot_data = json.dumps(result)
    return _events_template.replace('$PLOT_DATA', plot_data)

_events_template = r"""
<!DOCTYPE html>
<html>
<head>
<style>
pre {
    margin: 0px;
}
html, body {
    height: 100%;
}
</style>
</head>
<body>
<script type="module">
import * as d3 from "https://cdn.jsdelivr.net/npm/d3@7.7.0/+esm";
import {schemeTableau10} from "https://cdn.skypack.dev/d3-scale-chromatic@3";
import {axisLeft} from "https://cdn.skypack.dev/d3-axis@3";
import {scaleLinear} from "https://cdn.skypack.dev/d3-scale@4";
import {zoom, zoomIdentity} from "https://cdn.skypack.dev/d3-zoom@3";
import {brushX} from "https://cdn.skypack.dev/d3-brush@3";

let trace_data = $PLOT_DATA

function frames_for(id) {
    let strings = []
    while (id !== null) {
        const [sid, next_id] = trace_data.suffix_table[id]
        strings.push(trace_data.string_table[sid])
        id = next_id
    }
    return `${strings.join('\n')}\n`
}

let events_data = trace_data.events
let string_table = trace_data.string_table

class Event {
    constructor(id) {
        this.idx = id
    }
    get size() { return events_data.size[this.idx] }
    get action() { return string_table[events_data.action[this.idx]] }
    get addr() { return events_data.addr[this.idx] }
    get frames() { return frames_for(this.frames_idx) }
    get frames_idx() { return events_data.frames[this.idx]}
    get stream() { return events_data.stream[this.idx] }
    get device_free() { return events_data.addr[this.idx] }
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

function main() {
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
main()

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

    plots = (
        ("trace_plot", "Generate a visualization over time of the memory usage recorded by the trace as an html file."),
        ("segment_plot", "Visualize how allocations are packed into allocator segments at each point in a trace as an html file.")
    )
    for cmd, description in plots:
        trace_plot_a = subparsers.add_parser(cmd, description=description)
        trace_plot_a.add_argument('input', help=pickled)
        help = 'visualize trace from this device (default: chooses the only device with trace info or errors)'
        trace_plot_a.add_argument('-d', '--device', type=int, default=None, help=help)
        help = 'path to save the visualization(default: output.html)'
        trace_plot_a.add_argument('-o', '--output', default='output.html', help=help)
        if cmd == "trace_plot":
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
    elif args.action == 'segment_plot':
        data = _read(args.input)
        _write(args.output, segment_plot(data, device=args.device))
