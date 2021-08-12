#!/usr/bin/env python3

from __future__ import print_function

import argparse
import collections
import re
import sys


def parse_stack_name(line):
    # --- Thread 7f9fe9720340 (name: main/1) stack: ---
    m = re.match(r'\s*---\s*(.*)\s+stack:', line)
    if m:
        return m.group(1)
    # Thread 170 (Thread 0x7ffe7affb500 (LWP 38000)):
    m = re.match(r'Thread (\d+) \(Thread (0x[^\s]+) \(LWP (\d+)\)\):', line)
    if m:
        return m.group(2)
    m = re.match(r'Thread (\d+) \(LWP (\d+)\):', line)
    if m:
        return m.group(2)


def parse_stack_location(line):
    # PC:  0x7f9fe9759623: epoll_wait
    m = re.match(r'\s+PC:\s+0x[0-9a-fA-F]+', line)
    if m:
        return line
    # 0x5603eadc8ce1: Thread::ThreadBody(void*)
    m = re.match(r'\s+0x[0-9a-fA-F]+', line)
    if m:
        return line
    # #0  pthread_cond_wait@@GLIBC_2.3.2 () at ../sysdeps/unix/sysv/linux/x86_64/pthread_cond_wait.S:185
    m = re.match(r'#\d+\s+', line)
    if m:
        return re.sub(r'([a-zA-Z_]+)=(0x[a-fA-F0-9]+|[+\-]?\d+(\.\d*)?)', '\\1=X',
                      line)


def is_same_stack(line):
    # [same as previous thread]
    return re.search(r'\[same as previous thread\]', line)


def parse_stacks(lines):
    stacks = collections.defaultdict(list)
    name, stack, last_stack = None, '', ''
    for line in lines:
        if name is not None:
            location = parse_stack_location(line)
            if location:
                stack += location
                continue
            if not stack and is_same_stack(line) and last_stack:
                stacks[last_stack].append(name)
                name, stack = None, ''
                continue
            if stack:
                stacks[stack].append(name)
                last_stack = stack
                stack = ''
                name = parse_stack_name(line)
                continue
        else:
            name = parse_stack_name(line)
    return stacks


def create_report(args, stacks):
    mrkeys = sorted(stacks.keys(), key=lambda x: len(stacks[x]), reverse=True)
    for key in mrkeys:
        print('STACK (count={}):'.format(len(stacks[key])))
        print('{}'.format(key))
        print('LOCATIONS:')
        for name in stacks[key]:
            print('  {}'.format(name))
        print('')


def process_stack_lines(ln_iter, args):
    stacks = parse_stacks(ln_iter)
    create_report(args, stacks)


def process_stacks(args):
    fd = sys.stdin if args.input is None else open(args.input, 'r')
    process_stack_lines(fd, args)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--input', type=str)
    args, files = arg_parser.parse_known_args()
    args.files = files
    process_stacks(args)
