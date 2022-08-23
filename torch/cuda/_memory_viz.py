import pickle
import sys
import os
import io
import subprocess
from typing import Dict, Any
__all__ = ["format_flamegraph", "segments", "memory", "compare", "stats", "Bytes"]

def _frame_fmt(f):
    i = f['line']
    fname = f['filename'].split('/')[-1]
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
            frames = h['frames']
            if frames:
                frame_s = ';'.join([_frame_fmt(f) for f in reversed(frames)])
            else:
                frame_s = "<non-python>"
            f.write(f'{prefix};{b["state"]};{frame_s} {sz}\n')
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

    before_segs = set(_seg_key(seg) for seg in before)
    after_segs = set(_seg_key(seg) for seg in after)

    print(f'only_before = {list(a for a,_ in (before_segs - after_segs))}')
    print(f'only_after = {list(a for a,_ in (after_segs - before_segs))}')

    for seg in before:
        if _seg_key(seg) not in after_segs:
            _write_blocks(f, f'only_before;{_seg_info(seg)}', seg['blocks'])

    for seg in after:
        if _seg_key(seg) not in before_segs:
            _write_blocks(f, f'only_after;{_seg_info(seg)}', seg['blocks'])

    return format_flamegraph(f.getvalue())

class Bytes:
    def __init__(self, value):
        self.value = value

    def __add__(self, rhs):
        return Bytes(self.value + rhs)

    def __repr__(self):
        num = self.value
        # https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
        for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
            if abs(num) < 1024.0:
                return f"{num:3.1f}{unit}B"
            num /= 1024.0
        return f"{num:.1f}YiB"

def stats(snapshot):
    result : Dict[str, Any] = {}
    result['segments'] = len(snapshot['segments'])
    result['total_size'] = Bytes(0)
    for seg in snapshot['segments']:
        total_size = 0
        for b in seg['blocks']:
            if b['state'] not in result:
                result[b['state']] = Bytes(0)
            total_size += b['size']
            result[b['state']] += b['size']
        assert seg['total_size'] == total_size
        result['total_size'] += total_size
    return result


def segsum(data):
    PAGE_SIZE = 1024*1024*20
    segments = []
    out = io.StringIO()
    out.write("LARGE POOL SEGMENT SUMMARY\n")
    out.write("(each line is a segment, each letter is 20MB, different letters for different tensors in the segment, * means multiple tensors in the 20MBs, ' ' means free)\n")
    total_estimated = 0
    all_size = 0
    for seg in data['segments']:
        all_size += seg['total_size']

        all_ranges = []
        boffset = 0
        #active_size = sum(b['size'] for b in seg['blocks'] if b['state'] == 'active_allocated')
        for b in seg['blocks']:
            active =  b['state'] == 'active_allocated'
            if 'history' in b:
                # use the more accureate real_size to account for internal fragmenetation if we have it
                for h in b['history']:
                    if active or len(h['frames']) == 0:
                        all_ranges.append((h['addr'] - seg['address'], h['real_size'], active))
            else:
                if active:
                    all_ranges.append((boffset, b['size'], True))

            boffset += b['size']
        nseg = (seg['total_size'] - 1) // PAGE_SIZE + 1
        occupied = [' ' for _ in range(nseg)]
        frac = [0.0  for _ in range(nseg)]
        active_size = 0
        for i, (start_, size, active) in enumerate(all_ranges):
            active_size += size
            total_estimated += size
            finish_ = (start_ + size)
            start = start_ // PAGE_SIZE
            finish = (finish_ - 1) // PAGE_SIZE + 1
            m = chr( (ord('a' if active else 'A') + (i % 26)))
            for j in range(start, finish):
                s = max(start_, j*PAGE_SIZE)
                e = min(finish_, (j+1)*PAGE_SIZE)
                frac[j] += (e - s) / PAGE_SIZE
                if occupied[j] != ' ':
                    occupied[j] = '0123456789*'[int(frac[j]*10)]
                else:
                    occupied[j] = m
        body = ''.join(occupied)
        if seg['total_size'] > PAGE_SIZE:
            out.write(f'[{body}] {Bytes(seg["total_size"])}, {Bytes(seg["total_size"] - active_size)} free\n')
    out.write(f'total_estimated_active: {Bytes(total_estimated)}\n')
    out.write(f'total_estimated_free: {Bytes(all_size - total_estimated)}\n')
    return out.getvalue()

if __name__ == "__main__":
    import os.path
    thedir = os.path.realpath(os.path.dirname(__file__))
    if thedir in sys.path:
        # otherwise we find cuda/random.py as random...
        sys.path.remove(thedir)
    from pprint import pprint
    import argparse

    fn_name = 'torch.cuda.memory_dbg.snapshot()'
    pickled = f'pickled memory statistics from {fn_name}'
    parser = argparse.ArgumentParser(description=f'Visualize memory dumps produced by {fn_name}')
    subparsers = parser.add_subparsers(dest='action')

    def _output(p):
        p.add_argument('-o', '--output', default='output.svg', help='flamegraph svg (default: output.svg)')

    stats_a = subparsers.add_parser('stats', description='Prints overall allocation statistics')
    stats_a.add_argument('input', help=pickled)

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


    args = parser.parse_args()

    def _read(name):
        if name == '-':
            f = sys.stdin.buffer
        else:
            f = open(name, 'rb')
        data = pickle.load(f)
        if isinstance(data, list): # segements only...
            data = { 'segments': data, 'traces': [] }
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
        pprint(stats(data))
        print(segsum(data))
    elif args.action == 'compare':
        before = _read(args.before)
        after = _read(args.after)
        _write(args.output, compare(before, after))
