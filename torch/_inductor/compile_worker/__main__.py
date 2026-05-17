# mypy: allow-untyped-defs
import argparse
import base64
import functools
import importlib
import importlib.util
import logging
import os
import signal
import sys
from typing import TypeVar


_T = TypeVar("_T")


log = logging.getLogger(__name__)


def _load_subproc_pool_worker():
    # Keep this entrypoint torch-free until after SubprocMain forks/spawns the
    # actual compile workers. Importing through the torch package here can start
    # native threads in the sidecar process, recreating fork-after-thread-start
    # hazards for the default fork worker pool.
    module_name = "torch._inductor.compile_worker.subproc_pool_worker"
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = os.path.join(os.path.dirname(__file__), "subproc_pool_worker.py")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_subproc_pool_worker = _load_subproc_pool_worker()
SubprocKind = _subproc_pool_worker.SubprocKind
SubprocMain = _subproc_pool_worker.SubprocMain
SubprocPickler = _subproc_pool_worker.SubprocPickler


def _lookup_and_create_type(base: type[_T], qname: str) -> _T:
    """
    Given a base type and qualified name: import & lookup that name, check
    that it's of the given type and then instantiate it.
    """
    pkg, name = qname.rsplit(".", 1)
    mod = sys.modules.get(pkg)
    if mod is None:
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
        parser.add_argument("--torch-key", type=str)
        args = parser.parse_args()
        if os.getppid() != args.parent:
            sys.exit(0)
        read_fd = os.fdopen(args.read_fd, "rb")
        write_fd = os.fdopen(args.write_fd, "wb")
        torch_key_data = base64.b64decode(args.torch_key.encode("utf-8"))

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        SubprocMain(
            args.pickler, args.kind, args.workers, read_fd, write_fd, torch_key_data
        ).main()
    except Exception:
        log.exception("Uncaught exception in compile_worker subprocess")


if __name__ == "__main__":
    main()
