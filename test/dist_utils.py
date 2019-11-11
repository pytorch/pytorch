from __future__ import absolute_import, division, print_function, unicode_literals

import threading
from functools import partial, wraps
from os import getenv

import torch.distributed as dist
import torch.distributed.rpc as rpc


if not dist.is_available():
    print("c10d not available, skipping tests")
    sys.exit(0)


class TestConfig:
    __slots__ = ["rpc_backend_name"]

    def __init__(self, *args, **kwargs):
        assert len(args) == 0, "TestConfig only takes kwargs."
        for k, v in kwargs.items():
            setattr(self, k, v)


TEST_CONFIG = TestConfig(rpc_backend_name=getenv("RPC_BACKEND_NAME", "PROCESS_GROUP"))
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


def dist_init(old_test_method=None, setup_model_parallel=True, clean_shutdown=True):
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
            setup_model_parallel=setup_model_parallel,
            clean_shutdown=clean_shutdown,
        )

    @wraps(old_test_method)
    def new_test_method(self, *arg, **kwargs):
        self.worker_id = self.rank
        self.worker_name_to_id = {
            "worker{}".format(rank): rank for rank in range(self.world_size)
        }

        if setup_model_parallel:
            global _ALL_NODE_NAMES
            _ALL_NODE_NAMES = self.worker_name_to_id.keys()

            # Use enough 'num_send_recv_threads' until we fix https://github.com/pytorch/pytorch/issues/26359
            rpc.init_model_parallel(
                self_name="worker%d" % self.rank,
                backend=rpc.backend_registry.BackendType[TEST_CONFIG.rpc_backend_name],
                init_method=self.init_method,
                self_rank=self.rank,
                worker_name_to_id=self.worker_name_to_id,
                num_send_recv_threads=16,
            )

        return_value = old_test_method(self, *arg, **kwargs)

        if setup_model_parallel:
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
            rpc.join_rpc()

        return return_value

    return new_test_method
