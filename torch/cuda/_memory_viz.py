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
import {main} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch/torch/utils/viz/MemoryPlot.js"
let alloc_data = $PLOT_DATA
main(alloc_data)
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
import {main} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch/torch/utils/viz/StatePlot.js"
let trace_data = $PLOT_DATA
main(trace_data)
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
