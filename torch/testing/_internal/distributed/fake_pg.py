import torch.distributed as dist

# A fake process group (not related to FakeTensor) is a process group which
# doesn't actually do any communication, it just hallucinates some
# communication.  You can run a single rank with a fake process group
# without needing multiple processes.

class FakeProcessGroup(dist.ProcessGroup):
    pass

class FakeStore(dist.Store):
    pass

def _create_fake_pg(prefix_store, rank, world_size, timeout):
    return FakeProcessGroup(rank, world_size)

dist.Backend.register_backend("fake", _create_fake_pg)
