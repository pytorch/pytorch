import pickle
import sys
import os
import io
import subprocess
import bisect
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
    if 'oom' in snapshot:
        result['last_alloc'] = Bytes(snapshot['oom']['alloc'])
        result['device_free'] = Bytes(snapshot['oom']['device_free'])
        result['device_allocated'] = Bytes(snapshot['oom']['device_allocated'])
    result['segments'] = len(snapshot['segments'])
    result['allocator_reserved'] = Bytes(0)
    for seg in snapshot['segments']:
        total_size = 0
        for b in seg['blocks']:
            if b['state'] not in result:
                result[b['state']] = Bytes(0)
            total_size += b['size']
            result[b['state']] += b['size']
        assert seg['total_size'] == total_size
        result['allocator_reserved'] += total_size
    return result

def calc_active(seg):
    return sum(b['size'] for b in seg['blocks'] if b['state'] == 'active_allocated')

def _report_free(free_external, free_internal):
    total = free_external + free_internal
    suffix = ''
    if free_internal:
        pct = (free_internal / total)*100
        if pct >= 0.1:
            suffix = f'({pct:.1f}% internal)'
    return f'{Bytes(total)}{suffix}'

def segsum(data):
    PAGE_SIZE = 1024*1024*20
    segments = []
    out = io.StringIO()
    out.write("LARGE POOL SEGMENT SUMMARY\n")
    out.write("(each line is a segment, each letter is 20MB, different letters for different tensors in the segment, * means multiple tensors in the 20MBs, ' ' means free)\n")
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
        #active_size = sum(b['size'] for b in seg['blocks'] if b['state'] == 'active_allocated')
        for b in seg['blocks']:
            active =  b['state'] == 'active_allocated'
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
        frac = [0.0  for _ in range(nseg)]
        active_size = 0
        for i, (start_, size, active) in enumerate(all_ranges):
            active_size += size
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
        stream = '' if seg['stream'] == 0 else f', stream_{seg["stream"]}'
        body = ''.join(occupied)
        assert seg_free_external + seg_free_internal + seg_allocated == seg['total_size']
        stream = f' stream_{seg["stream"]}' if seg['stream'] != 0 else ''
        if True or seg['total_size'] > PAGE_SIZE:
            out.write(f'[{body}] {len(seg["blocks"])} {Bytes(seg["total_size"])}, {_report_free(seg_free_external, seg_free_internal)} free{stream}\n')
    out.write(f'total_reserved: {Bytes(total_reserved)}\n')
    out.write(f'total_allocated: {Bytes(total_allocated)} {total_allocated}\n')
    internal_external = f' ({Bytes(free_internal)} internal + {Bytes(free_external)} external)' if free_internal else ''
    out.write(f'total_free: {_report_free(free_external, free_internal)}\n')
    assert free_internal + free_external + total_allocated == total_reserved
    return out.getvalue()

def trace(data):
    out = io.StringIO()
    def format(entries):
        segment_intervals = []
        segment_addr_to_name = {}
        allocation_addr_to_name = {}

        free_names = []
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

        def draw_memory():
            entries = list((sorted((creation, size, name) for addr, (name, size, creation) in allocation_addr_to_name.items())))
            total_size = sum(s for _, s, n in entries)
            PAGE_SIZE = 32*1024*1024 #total_reserved // 120
            total_pages = ((total_size - 1) // PAGE_SIZE + 1)
            out.write(f'{" "*40}[')
            N_ENTRIES = len(entries)
            if N_ENTRIES:
                _, size, name = entries[0]
                it = 1
                for i in range(total_pages):
                    consumed = size
                    to_print = name[0]
                    while consumed < PAGE_SIZE and it < N_ENTRIES:
                        to_print = '*'
                        _, size, name = entries[it]
                        it += 1
                        consumed += size
                    size = (consumed - PAGE_SIZE)
                    out.write(to_print)
            out.write(']\n')

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
                draw_memory()
            elif e['action'] == 'free':
                addr, size = e['addr'], e['size']
                count -= size
                _, name, _ = allocation_addr_to_name.get(addr, (addr, None, None))
                out.write(f'del {name} # {Bytes(size)}\n')
                if name in allocation_addr_to_name:
                    free_names.append(name)
                    del allocation_addr_to_name[name]
                draw_memory()
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
                # for i, (_, saddr, _2) in enumerate(segment_intervals):
                #     if saddr == addr:
                #         del segment_intervals[i]
                #         break
                if name in segment_addr_to_name:
                    free_names.append(name)
                    del segment_addr_to_name[name]
            else:
                out.write(f'{e}\n')
        out.write(f"TOTAL MEM: {Bytes(count)}")
    for i, d in enumerate(data['device_traces']):
        if d:
            out.write(f'Device {i} ----------------\n')
            format(d)
    return out.getvalue()


class KeyifyList(object):
    def __init__(self, inner, key):
        self.inner = inner
        self.key = key

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, k):
        return self.key(self.inner[k])


def bisect_left(a, x, key):
    return bisect.bisect_left(KeyifyList(a, key), x)

def sim(data, total_mem):

    class OutOfMemory(RuntimeError):
        pass

    def run(entries):
        free_device_memory = total_mem
        print(f'{Bytes(free_device_memory)} of GPU memory avail')

        class Block:
            def __init__(self, segment, size):
                self.size = size
                self.segment = segment
                self.allocated = True

        class Segment:
            def __init__(self, size, pool, stream):
                self.size = size
                self.pool = pool
                self.stream = stream
                self.blocks = [Block(self, size)]

        small_pool = []
        large_pool = []
        segments = []

        def round_size(sz):
            return ((sz - 1) // 512 + 1) * 512

        # def round_size(sz):
        #     if sz <= 512:
        #         return 512
        #     f = 512
        #     while 2*f < sz:
        #         f *= 2
        #     DIVISIONS = 4
        #     div_size = (f*2 - f) // DIVISIONS
        #     r = ((sz - 1) // div_size + 1) * div_size
        #     # print(Bytes(sz), Bytes(r))
        #     return r

        def seg_round(sz):
            if sz <= 1024*1024:
                return 2*1024*1024
            elif sz < 10*1024*1024:
                return 20*1024*1024
            else:
                tmb = 1024*1024*2
                return ((sz - 1) // tmb + 1)*tmb

        def find_block(size, pool, stream):
            pt = bisect_left(pool, size, key=lambda b: b.size)
            for i in range(pt, len(pool)):
                blk = pool[pt]
                assert not blk.allocated
                if blk.segment.stream == stream:
                    blk.allocated = True
                    del pool[pt]
                    return blk
            return None

        def insert_block(blk, pool):
            pt = bisect_left(pool, blk.size, key=lambda b: b.size)
            assert blk.allocated
            blk.allocated = False
            pool.insert(pt, blk)

        def return_empty_segments():
            nonlocal free_device_memory
            to_remove = [s for s in segments if len(s.blocks) == 1 and not s.blocks[0].allocated]
            for s in to_remove:
                s.pool.remove(s.blocks[0])
                segments.remove(s)
                free_device_memory += s.size
        next_should_be = None

        def alloc_segment(size, stream):
            nonlocal free_device_memory
            nonlocal next_should_be
            seg_size = seg_round(size)
            if free_device_memory < seg_size:
                return_empty_segments()
            if free_device_memory < seg_size:
                raise OutOfMemory(size)
            free_device_memory -= seg_size
            # print(f"SIM : {Bytes(seg_size)}\n")
            # assertSync(next_should_be == seg_size)
            seg = Segment(seg_size, small_pool if size <= 1024*1024 else large_pool, stream)
            segments.append(seg)
            next_should_be = None
            return seg

        def maybe_split_block(blk, size):
            remaining = blk.size - size
            should_split = (size <= 1024*1024 and remaining >= 512) or (size > 1024*1024 and remaining > 1024*1024)
            if not should_split:
                return blk, None
            seg = blk.segment
            loc = seg.blocks.index(blk)
            rest = Block(seg, remaining)
            blk.size = size
            seg.blocks.insert(loc + 1, rest)
            assert seg.size == sum(blk.size for blk in seg.blocks)
            return blk, rest

        def alloc(size_: int, stream: int) -> Block:
            nonlocal free_device_memory
            size = round_size(size_)
            pool = small_pool if size <= 1024*1024 else large_pool

            blk = find_block(size, pool, stream)
            if blk is None:
                blk = alloc_segment(size, stream).blocks[0]

            blk, rest = maybe_split_block(blk, size)
            if rest is not None:
                insert_block(rest, pool)
            blk.real_size = size_
            return blk

        def free(blk: Block):
            seg = blk.segment
            pool = seg.pool
            pt = seg.blocks.index(blk)

            assert blk.allocated
            if pt > 0 and not seg.blocks[pt - 1].allocated:
                prev = seg.blocks[pt - 1]
                pool.remove(prev)
                blk.size += prev.size
                del seg.blocks[pt - 1]
                pt = pt - 1
            if pt + 1 < len(seg.blocks) and not seg.blocks[pt + 1].allocated:
                next_ = seg.blocks[pt + 1]
                pool.remove(next_)
                blk.size += next_.size
                del seg.blocks[pt + 1]
            assert sum(blk.size for blk in seg.blocks) == seg.size
            insert_block(blk, pool)



        def export_block(blk, boff):
            r = {'size': blk.size, 'state': 'active_allocated' if blk.allocated else 'inactive' }
            if blk.allocated:
                r['history'] = [{'real_size': blk.real_size, 'addr': boff}]
            return r

        def export_segment(seg):
            blocks = []
            boff = 0
            for blk in seg.blocks:
                blocks.append(export_block(blk, boff))
                boff += blk.size
            return {'blocks': blocks, 'address': 0, 'total_size': seg.size, 'stream': seg.stream}


        def export_snapshot():
            return {'segments': [export_segment(seg) for seg in segments], 'device_traces': [[]] }

        def assertSync(cond):
            if not cond:
                print("SIMULATION DESYNCED!")
                print(segsum(export_snapshot()))
            assert cond

        total_allocated = 0
        addr_to_id = {}
        addr_to_size = {}
        for count, e in enumerate(entries):
            if e['action'] == 'alloc':
                try:
                    addr, size, stream = e['addr'], e['size'], e['stream']
                    # print(f'{count}/{len(entries)} alloc', Bytes(size), stream)
                    n = alloc(size, stream)
                    assert n.allocated
                    assert addr not in addr_to_id
                    addr_to_id[addr] = n
                    addr_to_size[addr] = size
                    # assertSync(next_should_be is None)
                    assert n.real_size == size
                    total_allocated += size
                except OutOfMemory:
                    print(f"{count}/{len(entries)} SHOULD NOT HAVE OOMED BUT DID, EXITING... {Bytes(size)}")
                    break
            elif e['action'] == 'free':
                addr, size, stream = e['addr'], e['size'], e['stream']
                # print(f'{count}/{len(entries)} free', Bytes(size), stream)
                total_allocated -= addr_to_size[addr]
                assert addr_to_size[addr] == addr_to_id[addr].real_size
                free(addr_to_id[addr])
                del addr_to_id[addr]
            elif e['action'] == 'oom':
                addr, size = e['addr'], e['size']
                try:
                    n = alloc(size)
                    print("SHOULD HAVE OOMED BUT DIDN'T, WE WILL DESYNC NOW ", Bytes(size))
                    free(n) # try to stay close to what happened in the trace where this allocation didnt succeed
                except OutOfMemory:
                    print("EXPECTED OOM")
            elif e['action'] == 'segment_alloc':
                addr, size = e['addr'], e['size']
                # print(f"REAL: {Bytes(size)}")
                next_should_be = size
            else:
                pass

        return segsum(data) + f'\nsimulated with {Bytes(total_mem)}:\n' + segsum(export_snapshot())

    to_run = None
    for i, d in enumerate(data['device_traces']):
        if d:
            to_run = d

    return run(to_run)

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

    trace_a = subparsers.add_parser('trace', description='Prints ring buffer of most recent allocation events')
    trace_a.add_argument('input', help=pickled)

    sim_a = subparsers.add_parser('sim', description='Simulates most recent events, prints summary of simulation')
    sim_a.add_argument('input', help=pickled)
    sim_a.add_argument('-m' , '--mode', default='sim', help='which algorithm to use')


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
    elif args.action == 'trace':
        data = _read(args.input)
        print(trace(data))
    elif args.action == 'sim':
        data = _read(args.input)
        the_sim = globals()[args.mode]
        if 'oom' in data:
            total_free = data['oom']['device_allocated']
        else:
            total_free = 0
            for seg in data['segments']:
                total_free += seg['total_size']
            # total_free = 50*1024*1024*1024
        print(the_sim(data, total_free))

    elif args.action == 'compare':
        before = _read(args.before)
        after = _read(args.after)
        _write(args.output, compare(before, after))
