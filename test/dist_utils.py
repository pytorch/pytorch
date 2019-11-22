from __future__ import absolute_import, division, print_function, unicode_literals

import threading
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


MASTER_RANK = 0
_ALL_NODE_NAMES = set()
_DONE_NODE_NAMES = set()
_TERMINATION_SIGNAL = threading.Event()


def on_master_follower_report_done(worker_name):
    assert (
        worker_name in _ALL_NODE_NAMES
    ), "{worker_name} is not expected by master.".format(worker_name=worker_name)
    assert (
        worker_name not in _DONE_NODE_NAMES
    ), "{worker_name} report done twice.".format(worker_name=worker_name)
    _DONE_NODE_NAMES.add(worker_name)
    if _ALL_NODE_NAMES != _DONE_NODE_NAMES:
        return
    set_termination_signal()


def set_termination_signal():
    assert not _TERMINATION_SIGNAL.is_set(), "Termination signal got set twice."
    _TERMINATION_SIGNAL.set()


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
                init_method=self.init_method,
                rank=self.rank,
                world_size=self.world_size,
                rpc_backend_options=self.rpc_backend_options,
            )

        return_value = old_test_method(self, *arg, **kwargs)

        if setup_rpc:
            if clean_shutdown:
                # Follower reports done.
                if self.rank == MASTER_RANK:
                    on_master_follower_report_done("worker{}".format(MASTER_RANK))
                else:
                    rpc.rpc_async(
                        "worker{}".format(MASTER_RANK),
                        on_master_follower_report_done,
                        args=("worker{}".format(self.rank),),
                    )

                # Master waits for followers to report done.
                # Follower waits for master's termination command.
                _TERMINATION_SIGNAL.wait()
                if self.rank == MASTER_RANK:
                    # Master sends termination command.
                    futs = []
                    for dst_rank in range(self.world_size):
                        # torch.distributed.rpc module does not support sending to self.
                        if dst_rank == MASTER_RANK:
                            continue
                        dst_name = "worker{}".format(dst_rank)
                        fut = rpc.rpc_async(dst_name, set_termination_signal, args=())
                        futs.append(fut)
                    for fut in futs:
                        assert fut.wait() is None, "Sending termination signal failed."

            # Close RPC. Need to do this even if we don't have a clean shutdown
            # since we need to shutdown the RPC agent. If we don't shutdown the
            # RPC agent, tests would fail since RPC agent threads, locks and
            # condition variables are not properly terminated.
            rpc.wait_all_workers()
            rpc.shutdown()

        return return_value

    return new_test_method


# Set PROCESS_GROUP as the default RPC backend.
TEST_CONFIG.rpc_backend_name = "PROCESS_GROUP"
TEST_CONFIG.build_rpc_backend_options = lambda test_object: rpc.backend_registry.construct_rpc_backend_options(
    test_object.rpc_backend,
    # Use enough 'num_send_recv_threads' until we fix https://github.com/pytorch/pytorch/issues/26359
    num_send_recv_threads=16,
)
