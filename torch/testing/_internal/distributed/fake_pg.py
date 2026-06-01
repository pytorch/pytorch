# mypy: allow-untyped-defs

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import FakeProcessGroup, ProcessGroupGloo


class FakeStore(dist.Store):
    """
    A fake store is a fake Key-Value store simply for initialization usage
    the of fake process group, one can either use FakeStore or HashStore.
    """


def _create_fake_pg(common_opts, backend_opts):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convenient tool when playing
    with distributed but don't care about the actual data.
    """
    return FakeProcessGroup._create_internal(
        common_opts.group_rank, common_opts.group_size, backend_opts
    )


def set_fake_pg_delegate(store, pg=None, backend="gloo"):
    """
    Attach a real communication backend to a FakeProcessGroup so that
    collectives produce numerically correct results.

    This is useful for multi-process numerics testing: each process
    initializes with FakePG (lightweight, no NCCL dependency), then
    attaches a gloo delegate for deterministic CPU-based collectives.
    PREMUL_SUM operations are automatically decomposed into SUM +
    post-multiply at the C++ level.

    Args:
        store: A ``dist.Store`` for the delegate backend's rendezvous.
            When using torchrun, use ``dist.distributed_c10d._default_pg_init_method``
            or create a new ``TCPStore``.
        pg: The FakeProcessGroup to modify. Defaults to the world group.
        backend: Delegate backend name. Currently only ``"gloo"`` is supported.

    Example::

        dist.init_process_group("fake", rank=rank, world_size=world_size)
        store = dist.TCPStore(master_addr, port, world_size, is_master=(rank == 0))
        set_fake_pg_delegate(store)
        # Now dist.all_reduce etc. use real gloo collectives
    """
    if pg is None:
        pg = dist.distributed_c10d._get_default_group()

    # The world PG wraps the actual backend; extract FakeProcessGroup from it.
    backend_obj = pg._get_backend(torch.device("cpu"))
    if not isinstance(backend_obj, FakeProcessGroup):
        raise TypeError(
            f"Expected FakeProcessGroup, got {type(backend_obj).__name__}. "
            "set_fake_pg_delegate only works with the 'fake' backend."
        )

    if backend != "gloo":
        raise ValueError(f"Unsupported delegate backend: {backend!r}")

    rank = backend_obj.rank()
    size = backend_obj.size()
    delegate_pg = ProcessGroupGloo(
        dist.PrefixStore("fake_delegate/", store), rank, size
    )
    backend_obj.set_delegate(delegate_pg)


dist.Backend.register_backend(
    dist.Backend.FAKE,
    _create_fake_pg,
    extended_api=True,
    devices=["cpu", "cuda", "hpu", "xpu"],
)
