# Copyright (c) 2025, Tri Dao.
# Persistent subprocess worker for parallel autotuning pre-compilation.
# Receives length-prefixed pickled tasks on stdin, creates FakeTensors
# matching the parent's tensor metadata, and compiles with COMPILE_ONLY=True.
# Stays alive to process multiple configs (amortizes import overhead).

import importlib
import os
import pickle
import signal
import struct
import sys
import threading
import time

import torch

from . import cache
from .cache import CompileOnlyFakeTensorMode


# Watchdog poll interval. 60 s matches PyTorch Inductor's
# ``_async_compile_initializer``; short enough that orphan workers don't
# linger long, but long enough that the syscall cost is negligible.
_WATCHDOG_POLL_SECS = 60.0


def _install_parent_watchdog() -> None:
    """Self-terminate if the spawning parent process dies.

    Without this, a worker whose parent died (segfault, OOM-kill, the
    cute.compile MLIR retention leak that ``conftest.pytest_handlecrashitem``
    works around) gets reparented to init (PID 1) and lingers — consuming
    CPU/memory until something else reaps it. Long-lived self-hosted CI
    runners accumulate orphans across runs.

    The parent's PID is passed via the ``QUACK_COMPILE_WORKER_PARENT_PID``
    env var set by ``quack.autotuner._precompile`` before ``subprocess.Popen``.
    A daemon thread polls ``os.getppid()`` every ``_WATCHDOG_POLL_SECS`` and
    ``os.kill(self, SIGKILL)`` if the observed ppid no longer matches.

    Also installs ``SIGINT → SIG_IGN`` so Ctrl-C in the parent doesn't spam
    worker logs (the parent's pipe-close handles the orderly shutdown path).
    """
    raw = os.environ.get("QUACK_COMPILE_WORKER_PARENT_PID")
    if raw is None:
        # No env var — worker was launched outside of _precompile (e.g. by
        # an external driver script). Skip silently; orphan risk is on the
        # external caller.
        return
    try:
        orig_ppid = int(raw)
    except ValueError:
        return

    # Ignore SIGINT regardless of whether the watchdog poll is meaningful;
    # this keeps worker logs clean on Ctrl-C in the parent shell.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _poll():
        while True:
            time.sleep(_WATCHDOG_POLL_SECS)
            # ``os.getppid()`` returns 1 (init) once the original parent
            # exits. Comparing against the recorded ``orig_ppid`` catches
            # both the reparented-to-init case and the unusual case where
            # ppid changes to some other PID.
            if os.getppid() != orig_ppid:
                os.kill(os.getpid(), signal.SIGKILL)

    t = threading.Thread(target=_poll, name="quack-compile-worker-watchdog", daemon=True)
    t.start()


# This subprocess lives in compile-only mode for its entire lifetime; push
# the depth counter once at module load and let process exit pop it. We
# deliberately reach into the underscore-prefixed ``_COMPILE_ONLY_DEPTH``
# ContextVar instead of using ``compile_only_mode()`` because:
#
#   1. ``compile_only_mode()`` is a context manager that also enters a
#      ``CompileOnlyFakeTensorMode``. The worker enters its own per-task
#      ``CompileOnlyFakeTensorMode`` (see ``main()`` below) and doesn't want
#      a long-lived outer one shadowing it.
#   2. The depth is permanent for this process; there is nothing to ``.reset()``
#      — the process terminates instead.
#
# This is an intentional internal exception to the rule documented in
# ``quack/cache/__init__.py`` ("only :func:`compile_only_mode` mutates the
# depth"). External callers should still use ``compile_only_mode()``.
cache._COMPILE_ONLY_DEPTH.set(cache._COMPILE_ONLY_DEPTH.get() + 1)

_TENSOR_META_TAG = "__quack_tensor_meta__"


_dtype_map = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "torch.int8": torch.int8,
    "torch.uint8": torch.uint8,
    "torch.bool": torch.bool,
}


def _make_fake_tensor(meta):
    shape = meta["shape"]
    stride = meta["stride"]
    dtype = _dtype_map[meta["dtype"]]
    return torch.empty_strided(shape, stride, dtype=dtype, device="cuda")


def _deserialize_precompile_value(value):
    if isinstance(value, dict):
        if value.get(_TENSOR_META_TAG):
            return _make_fake_tensor(value)
        return {k: _deserialize_precompile_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_deserialize_precompile_value(v) for v in value)
    if isinstance(value, list):
        return [_deserialize_precompile_value(v) for v in value]
    return value


def _recv(stream):
    """Read a length-prefixed pickled message. Returns None on EOF."""
    header = stream.read(4)
    if len(header) < 4:
        return None
    length = struct.unpack("<I", header)[0]
    if length == 0:
        return None
    data = stream.read(length)
    return pickle.loads(data)


def _send(stream, msg):
    """Write a length-prefixed pickled message."""
    data = pickle.dumps(msg)
    stream.write(struct.pack("<I", len(data)))
    stream.write(data)
    stream.flush()


def main():
    # Install the watchdog before doing any work so an orphaned worker that
    # gets stuck in ``cute.compile`` self-terminates within the poll window.
    _install_parent_watchdog()

    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    # Signal ready
    _send(stdout, "READY")

    fn_cache = {}
    while True:
        payload = _recv(stdin)
        if payload is None:
            break

        fn_module = payload["fn_module"]
        fn_qualname = payload["fn_qualname"]
        fn_key = (fn_module, fn_qualname)
        if fn_key not in fn_cache:
            mod = importlib.import_module(fn_module)
            obj = mod
            for part in fn_qualname.split("."):
                obj = getattr(obj, part)
            fn_cache[fn_key] = getattr(obj, "fn", obj)
        fn = fn_cache[fn_key]

        serialized_args = payload["args"]
        kwargs = payload["kwargs"]
        config_kwargs = payload["config_kwargs"]

        with CompileOnlyFakeTensorMode():
            fake_args = _deserialize_precompile_value(serialized_args)
            fake_kwargs = _deserialize_precompile_value(kwargs)
            try:
                fn(*fake_args, **fake_kwargs, **config_kwargs)
                _send(stdout, "OK")
            except Exception as e:
                _send(stdout, f"ERR:{e}")


if __name__ == "__main__":
    main()
