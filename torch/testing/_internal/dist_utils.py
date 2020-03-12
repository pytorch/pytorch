from __future__ import absolute_import, division, print_function, unicode_literals

import time
from functools import partial, wraps
import re

import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import _rref_context_get_debug_info


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

def wait_until_node_failure(rank, expected_error_regex=".*"):
    '''
    Loops until an RPC to the given rank fails. This is used to
    indicate that the node has failed in unit tests.
    Args:
    rank (int): Rank of the node expected to fail
    expected_error_regex (optional, str): Regex of exception message expected. Useful to ensure a specific failure
    occurs, not just any.
    '''
    while True:
        try:
            rpc.rpc_sync("worker{}".format(rank), noop, args=())
            time.sleep(0.1)
        except Exception as e:
            if re.match(pattern=expected_error_regex, string=str(e)):
                return str(e)

# Shutdown sequence is not well defined, so we may see any of the following errors
# When running tests that simulate errors via a shutdown on the remote end.
def get_shutdown_error_regex(rpc_backend):
    """
    Return various error message we may see from RPC agents while running tests that check for failures. This function
    is used to match against possible errors to ensure failures were raised properly.
    """
    if rpc_backend == "PROCESS_GROUP":
        error_regexes = ["Encountered exception in ProcessGroupAgent::enqueueSend"]
    else:
        error_regexes = [
            "Request aborted during client shutdown",
            "worker.: Error in reponse from worker.: server shutting down",
            "worker.: Error in response from worker.: Failed to write to remote endpoint",
            "worker.: Error in response from worker.: AsyncSocketException: recv() failed",
        ]
    error_regex = "".join(["({})|".format(error_str) for error_str in error_regexes])
    # Strip out the last | or else it will match anything
    error_regex = error_regex[:-1]
    return error_regex

def wait_until_pending_users_flushed():
    '''
    The RRef protocol holds forkIds of rrefs in a map until those forks are
    confirmed by the owner. The message confirming the fork may arrive after
    our tests check whether this map is empty, which leads to failures and
    flaky tests. to_here also does not guarantee that we have finished
    processind the owner's confirmation message for the RRef. This function
    loops until the map is empty, which means the messages have been received
    as processed. Call this function before asserting the map returned by
    _get_debug_info is empty.
    '''
    num_pending_users = int(_rref_context_get_debug_info()["num_pending_users"])
    while num_pending_users != 0:
        time.sleep(0.1)
        num_pending_users = int(_rref_context_get_debug_info()["num_pending_users"])
    return

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

def worker_name(rank):
    return "worker{}".format(rank)
