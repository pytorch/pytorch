# mypy: allow-untyped-defs
from typing_extensions import deprecated as _deprecated


@_deprecated(
    "Usage of `backward_compatibility.worker_init_fn` is deprecated "
    "as `DataLoader` automatically applies sharding in every worker",
    category=FutureWarning,
)
def worker_init_fn(worker_id):
    pass
