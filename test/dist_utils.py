from __future__ import absolute_import, division, print_function, unicode_literals

from functools import wraps
from os import getenv
from datetime import timedelta

import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import RpcBackend


if not dist.is_available():
    print("c10d not available, skipping tests")
    sys.exit(0)


class TestConfig:
    __slots__ = ['rpc_backend']

    def __init__(self, *args, **kwargs):
        assert len(args) == 0, "TestConfig only takes kwargs."
        for k, v in kwargs.items():
            setattr(self, k, v)


TEST_CONFIG = TestConfig(rpc_backend=getenv("RPC_BACKEND", RpcBackend.PROCESS_GROUP))
INIT_METHOD_TEMPLATE = "file://{file_name}?rank={rank}&world_size={world_size}"


def dist_init(sync_at_shutdown=True):
    """
    We use this decorator for setting up and tearing down state since
    MultiProcessTestCase runs each `test*` method in a separate process and
    each process just runs the `test*` method without actually calling
    'setUp' and 'tearDown' methods of unittest.
    """

    def decorator(test_method):
        def wrapper(self, *arg, **kwargs):
            self.worker_id = self.rank
            dist.init_process_group(backend="gloo", init_method=self.init_method, timeout=timedelta(seconds=5))
            # Use enough 'num_send_recv_threads' until we fix https://github.com/pytorch/pytorch/issues/26359
            rpc.init_model_parallel(
                self_name="worker%d" % self.rank,
                backend=TEST_CONFIG.rpc_backend,
                self_rank=self.rank,
                init_method=self.init_method,
                num_send_recv_threads=16
            )
            test_method(self, *arg, **kwargs)
            if sync_at_shutdown:
                dist.barrier()
        return wrapper

    return decorator
