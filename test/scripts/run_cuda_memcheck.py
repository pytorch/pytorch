#!/usr/bin/env python

"""This script runs cuda-memcheck on the specified unit test. Each test case
is run in its isolated process with a timeout so that:
1) different test cases won't influence each other, and
2) in case of hang, the script would still finish in a finite amount of time.
The output will be written to a log file result.log

Example usage:
    python run_cuda_memcheck.py ../test_torch.py 600

Note that running cuda-memcheck could be very slow.
"""

import asyncio
import torch
import multiprocessing
import argparse
import subprocess
import tqdm
import re

ALL_TESTS = []
NUM_PROCESSES = multiprocessing.cpu_count()
GPUS = torch.cuda.device_count()

# parse arguments
parser = argparse.ArgumentParser(description="Run isolated cuda-memcheck on unit tests")
parser.add_argument('filename', help="the python file for a test, such as test_torch.py")
parser.add_argument('timeout', type=int, help='kill the test if it does not terminate in a certain amount of seconds')
parser.add_argument('--ignore', nargs='+', default=['cudaErrorInvalidDeviceFunction'],
                    help='list of regex of the failures not interested, default to ["cudaErrorInvalidDeviceFunction"], '
                         'because cublas does not run error-free under cuda-memcheck')
args = parser.parse_args()

# Discover tests:
# To get a list of tests, run:
# pytest --setup-only test/test_torch.py
# and then parse the output
proc = subprocess.Popen(['pytest', '--setup-only', args.filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = proc.communicate()
lines = stdout.decode().strip().splitlines()
for line in lines:
    if '(fixtures used:' in line:
        line = line.strip().split()[0]
        line = line[line.find('::') + 2:]
        line = line.replace('::', '.')
        ALL_TESTS.append(line)

# Run tests:
# Since running cuda-memcheck on PyTorch unit tests is very slow, these tests must be run in parallel.
# This is done by using the coroutine feature in new Python versions.  A number of coroutines are created;
# they create subprocesses and awaiting them to finish. The number of running subprocesses is the same as
# the number of CPUs in the machine. These subprocesses are balanced across different GPUs on the system.
progress = 0
logfile = open('result.log', 'w')
progressbar = tqdm.tqdm(total=len(ALL_TESTS))

async def run1():
    global progress
    while progress < len(ALL_TESTS):
        test = ALL_TESTS[progress]
        progress += 1
        cmd = f'CUDA_VISIBLE_DEVICES={progress % GPUS} cuda-memcheck --error-exitcode 1 python {args.filename} {test}'
        proc = await asyncio.create_subprocess_shell(cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), args.timeout)
        except asyncio.TimeoutError:
            print('Timeout:', test, file=logfile)
            proc.kill()
        else:
            if proc.returncode == 0:
                print('Success:', test, file=logfile)
            else:
                stdout = stdout.decode()
                stderr = stderr.decode()
                should_ignore = False
                for pattern in args.ignore:
                    pattern = re.compile(pattern)
                    found = (pattern.search(stdout) is not None) or \
                            (pattern.search(stderr) is not None)
                    if found:
                        should_ignore = True
                        break
                if should_ignore:
                    print('Ignored:', test, file=logfile)
                else:
                    print('Fail:', test, file=logfile)
                    print(stdout, file=logfile)
                    print(stderr, file=logfile)
        del proc
        progressbar.update(1)

async def main():
    tasks = [asyncio.create_task(run1()) for _ in range(NUM_PROCESSES)]
    for t in tasks:
        await t

if __name__ == '__main__':
    asyncio.run(main())
