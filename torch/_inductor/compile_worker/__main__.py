# mypy: allow-untyped-defs
import argparse
import os
import sys
import typing

from torch._inductor.async_compile import pre_fork_setup
from torch._inductor.compile_worker.subproc_pool import Pipe, SubprocMain
from torch._inductor.compile_worker.watchdog import _async_compile_initializer
from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path

_set_triton_ptxas_path()

try:
    import triton

    assert triton is not None  # preload in parent
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int)
    parser.add_argument("--parent", type=int)
    args = parser.parse_args()
    if os.getppid() != args.parent:
        sys.exit(0)
    write_fd = typing.cast(Pipe, os.fdopen(os.dup(sys.stdout.fileno()), "wb"))
    read_fd = typing.cast(Pipe, os.fdopen(os.dup(sys.stdin.fileno()), "rb"))

    # nobody else should read stdin
    sys.stdin.close()

    # redirect output of workers to stderr
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    pre_fork_setup()

    _async_compile_initializer(args.parent)
    SubprocMain(args.workers, read_fd, write_fd).main()


if __name__ == "__main__":
    main()
