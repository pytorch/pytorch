from __future__ import absolute_import, division, print_function, unicode_literals

import time
from functools import partial, wraps

import torch.distributed as dist
import torch.distributed.rpc as rpc


if not dist.is_available():
    print("c10d not available, skipping tests")
    sys.exit(0)


class TestConfig:
    __slots__ = ["rpc_backend_name", "build_rpc_backend_options"]

    def __init__(self, *args, **kwargs):
        assert len(args) == 0, "TestConfig only takes kwargs."
        for k, v in kwargs.items():
            setattr(self, k, v)


TEST_CONFIG = TestConfig()
INIT_METHOD_TEMPLATE = "file://{file_name}"


def dist_init(old_test_method=None, setup_rpc=True, clean_shutdown=True):
    """
    We use this decorator for setting up and tearing down state since
    MultiProcessTestCase runs each `test*` method in a separate process and
    each process just runs the `test*` method without actually calling
    'setUp' and 'tearDown' methods of unittest.
    """

    # If we use dist_init without arguments (ex: @dist_init), old_test_method is
    # appropriately set and we return the wrapper appropriately. On the other
    # hand if dist_init has arguments (ex: @dist_init(clean_shutdown=False)),
    # old_test_method is None and we return a functools.partial which is the real
    # decorator that is used and as a result we recursively call dist_init with
    # old_test_method and the rest of the arguments appropriately set.
    if old_test_method is None:
        return partial(
            dist_init,
            setup_rpc=setup_rpc,
            clean_shutdown=clean_shutdown,
        )

    @wraps(old_test_method)
    def new_test_method(self, *arg, **kwargs):
        # Setting _ignore_rref_leak to make sure OwnerRRefs are properly deleted
        # in tests.
        import torch.distributed.rpc.api as api
        api._ignore_rref_leak = False

        self.worker_id = self.rank

        if setup_rpc:
            global _ALL_NODE_NAMES
            _ALL_NODE_NAMES = {
                "worker{}".format(rank) for rank in range(self.world_size)
            }

            rpc.init_rpc(
                name="worker%d" % self.rank,
                backend=self.rpc_backend,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )

        return_value = old_test_method(self, *arg, **kwargs)

        if setup_rpc:
            rpc.shutdown(graceful=clean_shutdown)

        return return_value

    return new_test_method


# Set PROCESS_GROUP as the default RPC backend.
TEST_CONFIG.rpc_backend_name = "PROCESS_GROUP"
TEST_CONFIG.build_rpc_backend_options = lambda test_object: rpc.backend_registry.construct_rpc_backend_options(
    test_object.rpc_backend,
    init_method=test_object.init_method,
    # Some tests need additional threads (ex: test_trainer_ps)
    num_send_recv_threads=8,
)

def noop():
    pass

def wait_until_node_failure(rank):
    '''
    Loops until an RPC to the given rank fails. This is used to
    indicate that the node has failed in unit tests.
    '''
    while True:
        try:
            rpc.rpc_sync("worker{}".format(rank), noop, args=())
            time.sleep(0.5)
        except Exception:
            break

def initialize_pg(init_method, rank, world_size):
    # This is for tests using `dist.barrier`.
    # For `RpcAgent` other than `ProcessGroupAgent`,
    # no `_default_pg` is initialized.
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
