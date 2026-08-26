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
        return dist.get_default_backend_for_device(self.device_type)

    def init_pg(self, backend=None):
        if backend is None:
            backend = self.backend

        # set device for accelerator pg for collectives
        # must be called BEFORE init_process_group, otherwise HCCL
        # communicator creation fails with "same physical device ID" error
        acc = torch.accelerator.current_accelerator()
        if acc is not None:
            curr_backend = dist.get_default_backend_for_device(acc)
            if backend == curr_backend:
                torch.accelerator.set_device_index(self.rank)

        dist.init_process_group(
            backend=backend,
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )

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
        # Resolve backend: use self.backend if not explicitly provided
        if backend is None:
            resolved_backend = self.backend
        else:
            resolved_backend = backend

        # Skip test if backend requires accelerator but not enough devices available
        acc = torch.accelerator.current_accelerator()
        if acc is not None:
            curr_backend = dist.get_default_backend_for_device(acc)
            if resolved_backend == curr_backend:
                if torch.accelerator.device_count() < self.world_size:
                    sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)
        else:
            # No accelerator available, only allow gloo
            if resolved_backend != "gloo":
                sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        self.init_comms(init_rpc=init_rpc, backend=resolved_backend)
        func(self, *args, **kwargs)
        self.destroy_comms(destroy_rpc=init_rpc)

    return wrapper
