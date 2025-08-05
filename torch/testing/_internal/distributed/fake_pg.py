# mypy: allow-untyped-defs

import torch.distributed as dist


try:
    from torch.distributed._distributed_c10d import FakeProcessGroup
except ImportError:
    # Fallback for non-distributed builds
    class FakeProcessGroup:
        def __init__(self, rank: int, world_size: int):
            self._rank = rank
            self._world_size = world_size

        def rank(self):
            return self._rank

        def size(self):
            return self._world_size


class FakeStore(dist.Store):
    """
    A fake store is a fake Key-Value store simply for initialization usage
    the of fake process group, one can either use FakeStore or HashStore.
    """


def _create_fake_pg(prefix_store, rank, world_size, timeout):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convenient tool when playing
    with distributed but don't care about the actual data.
    """
    return FakeProcessGroup(rank, world_size)


dist.Backend.register_backend("fake", _create_fake_pg, devices=["cpu", "cuda", "hpu"])
