# mypy: allow-untyped-defs

import sys
from functools import partial, wraps

import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    TEST_SKIPS,
    tp_transports,
)


TEST_GPU_NUM = 4


class ShardedTensorTestBase(MultiProcessTestCase):
    @property
    def world_size(self):
        return TEST_GPU_NUM

    @property
    def backend(self) -> str:
        device_type = getattr(self, "device_type", None)
        if device_type is None:
            return "nccl"
        return dist.get_default_backend_for_device(device_type)

    @property
    def current_device(self) -> torch.device:
        if self.rank < 0:
            return torch.device(self.device_type)
        return torch.device(self.device_type, self.rank)

    def init_pg(self, backend=None):
        backend = backend or self.backend
        # Ask the distributed framework rather than matching a fixed list, so a
        # backend registered by an out-of-tree device is accepted the same way
        # the built-in ones are.
        if not dist.is_backend_available(backend):
            raise RuntimeError(f"Backend {backend} not supported!")

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

        # Bind the per-rank device for accelerator backends. gloo is excluded: it
        # is the default backend for cpu and mps, which have no per-rank
        # accelerator device index.
        if torch.accelerator.is_available() and backend != dist.Backend.GLOO:
            torch.accelerator.set_device_index(self.rank)

    def init_rpc(self):
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
            _transports=tp_transports()
        )
        rpc_backend_options.init_method = f"file://{self.file_name}"
        for rank in range(self.world_size):
            rpc_backend_options.set_device_map(
                f"worker{rank}", {rank: self.rank, self.rank: rank}
            )

        rpc.init_rpc(
            name=f"worker{self.rank:d}",
            rank=self.rank,
            world_size=self.world_size,
            rpc_backend_options=rpc_backend_options,
        )

    def init_comms(self, init_rpc=True, backend=None):
        if init_rpc:
            self.init_rpc()
        self.init_pg(backend=backend)

    def destroy_comms(self, destroy_rpc=True):
        # Wait for all ranks to reach here before starting shutdown.
        dist.barrier()

        if destroy_rpc:
            rpc.shutdown()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def assert_sharded_tensor_equal(self, st1, st2):
        st1_local_shards = st1.local_shards()
        st2_local_shards = st2.local_shards()
        self.assertEqual(len(st1_local_shards), len(st2_local_shards))
        for i, st1_local_shard in enumerate(st1_local_shards):
            self.assertEqual(st1_local_shard.tensor, st2_local_shards[i].tensor)
            self.assertEqual(st1_local_shard.metadata, st2_local_shards[i].metadata)

        self.assertEqual(st1.metadata(), st2.metadata())
        self.assertEqual(st1.sharding_spec(), st2.sharding_spec())
        self.assertEqual(len(st1.remote_shards()), len(st2.remote_shards()))


# wrapper to initialize comms (processgroup + rpc)
def with_comms(func=None, init_rpc=True, backend=None):
    if func is None:
        return partial(
            with_comms,
            init_rpc=init_rpc,
            backend=backend,
        )

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Skip test if backend requires accelerator but not enough devices available
        if (backend or self.backend) != dist.Backend.GLOO:
            if torch.accelerator.device_count() < self.world_size:
                sys.exit(TEST_SKIPS[f"multi-device-{self.world_size}"].exit_code)
        self.init_comms(init_rpc=init_rpc, backend=backend)
        func(self, *args, **kwargs)
        self.destroy_comms(destroy_rpc=init_rpc)

    return wrapper
