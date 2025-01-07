# mypy: allow-untyped-defs
import argparse
import functools
import importlib
import logging
import os
import sys
from typing import Type, TypeVar

from torch._inductor.async_compile import pre_fork_setup
from torch._inductor.compile_worker.subproc_pool import (
    SubprocKind,
    SubprocMain,
    SubprocPickler,
)
from torch._inductor.compile_worker.watchdog import _async_compile_initializer
from torch._inductor.runtime.compile_tasks import _set_triton_ptxas_path


_T = TypeVar("_T")


log = logging.getLogger(__name__)

_set_triton_ptxas_path()

try:
    import triton

    assert triton is not None  # preload in parent
except ImportError:
    pass


def _lookup_and_create_type(base: Type[_T], qname: str) -> _T:
    """
    Given a base type and qualified name: import & lookup that name, check
    that it's of the given type and then instantiate it.
    """
    pkg, name = qname.rsplit(".", 1)
    mod = importlib.import_module(pkg)
    ty = getattr(mod, name)
    if not issubclass(ty, base):
        raise TypeError(f"Type {ty} is not a subtype of {base}")
    return ty()


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--pickler", type=functools.partial(_lookup_and_create_type, SubprocPickler)
        )
        parser.add_argument("--kind", type=SubprocKind)
        parser.add_argument("--workers", type=int)
        parser.add_argument("--parent", type=int)
        parser.add_argument("--read-fd", type=int)
        parser.add_argument("--write-fd", type=int)
        args = parser.parse_args()
        if os.getppid() != args.parent:
            sys.exit(0)
        read_fd = os.fdopen(args.read_fd, "rb")
        write_fd = os.fdopen(args.write_fd, "wb")

        pre_fork_setup()

        _async_compile_initializer(args.parent)

        SubprocMain(args.pickler, args.kind, args.workers, read_fd, write_fd).main()
    except Exception:
        log.exception("Uncaught exception in compile_worker subprocess")


if __name__ == "__main__":
    main()
