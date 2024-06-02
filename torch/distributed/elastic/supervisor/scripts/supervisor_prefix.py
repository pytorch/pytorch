# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import select
import sys

from typing import BinaryIO

parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    description="add prefix each line of stderr/stdout while running a command",
)
parser.add_argument("-p", type=str, help="prefix to add to each line of stdout/stderr")
parser.add_argument("cmd", type=str, help="command to run")
parser.add_argument("args", nargs=argparse.REMAINDER, help="command arguments")
args = parser.parse_args()
if args.p is not None:
    args.p = args.p.format(**os.environ)


class Stream:
    def __init__(self, prefix: str, stream: BinaryIO):
        self.stream = stream
        self.prefix = prefix.encode()
        if prefix:
            self.nprefix = f"\n{prefix}".encode()
        self.linestart = True

    def write(self, fd: int) -> bool:
        # read at most 64k at once
        msg = os.read(fd, 64 * 1024)
        if len(msg) == 0:
            return True
        if self.prefix:
            output = [self.prefix] if self.linestart else []
            self.linestart = msg[-1:] == b"\n"
            if self.linestart:
                output.append(msg[:-1].replace(b"\n", self.nprefix))
                output.append(b"\n")
            else:
                output.append(msg.replace(b"\n", self.nprefix))
            msg = b"".join(output)
        try:
            self.stream.write(msg)
            self.stream.flush()
        except BrokenPipeError:
            pass
        return False


if __name__ == "__main__":
    out_r, out_w = os.pipe()
    err_r, err_w = os.pipe()
    # the parent process becomes the wrapped command
    # and the line-prefixing process becomes its child
    # to reduce the chance errors with this process
    # make the exit of the command get stuck.
    # This process exits when all open connections to
    # its input pipes are closed, so it will continue to
    # prefix any output from potential children of the command
    # as well.
    if os.fork() != 0:
        os.close(out_r)
        os.close(err_r)
        os.dup2(out_w, 1)
        os.dup2(err_w, 2)
        # either never returns or raises OSError
        try:
            os.execvp(args.cmd, [args.cmd, *args.args])
        except FileNotFoundError:
            raise Exception(  # noqa: TRY002
                "Could not find executable '%s'", args.cmd
            ) from None
    os.close(out_w)
    os.close(err_w)
    # don't let this process get killed when the parent dies
    # if there are still open file descriptors that can be written to.
    os.setsid()
    fd_map = {
        out_r: Stream(args.p, sys.stdout.buffer),
        err_r: Stream(args.p, sys.stderr.buffer),
    }
    poller = select.poll()
    for k in fd_map.keys():
        poller.register(k, select.POLLIN)
    while fd_map:
        for fd, _ in poller.poll():
            if fd_map[fd].write(fd):
                poller.unregister(fd)
                del fd_map[fd]
