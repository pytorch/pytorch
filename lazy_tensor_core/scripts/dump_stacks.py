#!/usr/bin/env python3
#
# The following command is needed (as root) in order to enable GDB to attach
# existing user processes:
#
#  echo 0 > /proc/sys/kernel/yama/ptrace_scope
#

from __future__ import print_function

import argparse
import stack_trace_parse as stp
import subprocess


def get_stacks(pid):
    return subprocess.check_output([
        'gdb', '-p',
        str(pid), '-batch', '-ex', 'thread apply all bt', '-ex', 'quit'
    ]).decode('utf-8')


def dump_stacks(args):
    stacks = get_stacks(args.pid)
    stp.process_stack_lines(stacks.splitlines(True), args)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'pid',
        type=int,
        metavar='PID',
        help='The process ID whose stacks need to be dumped')
    args, files = arg_parser.parse_known_args()
    dump_stacks(args)
