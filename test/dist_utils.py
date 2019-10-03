from __future__ import absolute_import, division, print_function, unicode_literals

from os import getenv
from functools import wraps
import torch.distributed as dist
import torch.distributed.rpc as rpc

if not dist.is_available():
    print("c10d not available, skipping tests")
    sys.exit(0)


BACKEND = getenv('RPC_BACKEND', rpc.RpcBackend.PROCESS_GROUP)
RPC_INIT_URL = getenv('RPC_INIT_URL', '')

def dist_init(func):
    """
    We use this decorator for setting up and tearing down state since
    MultiProcessTestCase runs each `test*` method in a separate process and
    each process just runs the `test*` method without actually calling
    'setUp' and 'tearDown' methods of unittest.
    """
    @wraps(func)
    def wrapper(self):
        self.worker_id = self.rank
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(backend='gloo', rank=self.rank,
                                world_size=self.world_size, store=store)
        rpc.init_model_parallel(self_name='worker%d' % self.rank,
                                backend=BACKEND,
                                self_rank=self.rank,
                                init_method=RPC_INIT_URL)
        func(self)
        rpc.join_rpc()

    return wrapper
