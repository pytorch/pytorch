"""Thread-pool sizing for ONNX Runtime sessions."""

from __future__ import annotations

import os


def intra_op_num_threads() -> int:
    """The number of CPUs this process may actually run on.

    ONNX Runtime sizes its intra-op thread pool from
    ``std::thread::hardware_concurrency()``, which reads ``_SC_NPROCESSORS_ONLN``
    and therefore counts every CPU on the machine. Under a container runtime
    that hands the process a cpuset -- a Kubernetes pod with a CPU limit, a
    ``docker run --cpuset-cpus``, a ``taskset`` -- that is more threads than
    there are CPUs to run them on, and ORT then fails to pin each surplus
    thread::

        [E:onnxruntime:Default, env.cc:226 ThreadMain] pthread_setaffinity_np
        failed for thread: 110251, index: 46, mask: {47, }, error code: 22
        error msg: Invalid argument. Specify the number of threads explicitly
        so the affinity is not set.

    ``sched_getaffinity`` is cpuset-aware, so it reports the number ORT should
    have used. Where no cpuset is in force the two agree and this changes
    nothing. It is Linux-only, which is also the only place the cpuset case
    arises; elsewhere fall back to the machine count, matching ORT's own
    default.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1
