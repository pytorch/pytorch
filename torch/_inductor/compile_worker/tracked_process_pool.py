# _thread_safe_fork is needed because the subprocesses in the pool can read
# justknobs, e.g., in the Triton compiler. For internal, the import installs
# functionality to destroy singletons before forking and re-enable them after.
import torch._thread_safe_fork  # noqa: F401
from torch._inductor.compile_worker.subproc_pool_worker import (
    TrackedProcessPoolExecutor as _TorchFreeTrackedProcessPoolExecutor,
)


class TrackedProcessPoolExecutor(_TorchFreeTrackedProcessPoolExecutor):
    """Tracked executor for parents that already import torch."""
