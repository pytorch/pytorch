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


def dist_init(old_test_method=None, setup_rpc=True, clean_shutdown=True,
              faulty_messages=None, messages_to_delay=None):
    """
    We use this decorator for setting up and tearing down state since
    MultiProcessTestCase runs each `test*` method in a separate process and
    each process just runs the `test*` method without actually calling
    'setUp' and 'tearDown' methods of unittest.

    Note: pass the string representation of MessageTypes that should be used
    with the faulty agent's send function. By default, all retriable messages
    ("RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT", "RREF_USER_DELETE",
    "CLEANUP_AUTOGRAD_CONTEXT_REQ") will use the faulty send (this default is
    set from faulty_rpc_agent_test_fixture.py).
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
            faulty_messages=faulty_messages,
            messages_to_delay=messages_to_delay,
        )

    @wraps(old_test_method)
    def new_test_method(self, *arg, **kwargs):
        # Setting _ignore_rref_leak to make sure OwnerRRefs are properly deleted
        # in tests.
        import torch.distributed.rpc.api as api
        api._ignore_rref_leak = False

        self.worker_id = self.rank

        if (
            rpc.backend_registry.backend_registered("FAULTY_PROCESS_GROUP")
            and self.rpc_backend
            == rpc.backend_registry.BackendType.FAULTY_PROCESS_GROUP
        ):
            _build_faulty_backend_options(self, faulty_messages, messages_to_delay)

        if (
            rpc.backend_registry.backend_registered("TENSORPIPE")
            and self.rpc_backend
            == rpc.backend_registry.BackendType.TENSORPIPE
        ):
            TEST_CONFIG.rpc_backend_name = "TENSORPIPE"
            _build_tensorpipe_backend_options()

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

def _build_faulty_backend_options(faulty_agent_fixture, faulty_messages, messages_to_delay):
    '''
    Constructs the backend options object for the faulty process group agent
    based on the faulty_messages input to dist_init.
    '''
    messages_to_fail = (
        faulty_messages
        if faulty_messages is not None
        else faulty_agent_fixture.retryable_message_types
    )
    messages_to_delay = (
        messages_to_delay
        if messages_to_delay is not None
        else faulty_agent_fixture.default_messages_to_delay
    )
    TEST_CONFIG.build_rpc_backend_options = lambda test_object: rpc.backend_registry.construct_rpc_backend_options(
        test_object.rpc_backend,
        init_method=test_object.init_method,
        num_send_recv_threads=8,
        num_fail_sends=faulty_agent_fixture.num_fail_sends,
        messages_to_fail=messages_to_fail,
        messages_to_delay=messages_to_delay,
    )

def _build_tensorpipe_backend_options():
    TEST_CONFIG.build_rpc_backend_options = lambda test_object: rpc.backend_registry.construct_rpc_backend_options(
        test_object.rpc_backend,
        init_method=test_object.init_method,
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
            if re.search(pattern=expected_error_regex, string=str(e)):
                return str(e)

# Shutdown sequence is not well defined, so we may see any of the following errors
# When running tests that simulate errors via a shutdown on the remote end.
def get_shutdown_error_regex(rpc_backend):
    """
    Return various error message we may see from RPC agents while running tests that check for failures. This function
    is used to match against possible errors to ensure failures were raised properly.
    """
    if rpc_backend == "PROCESS_GROUP":
        error_regexes = [
            "Encountered exception in ProcessGroupAgent::enqueueSend",
            "Encountered exception in ProcessGroupAgent::listenLoop()",
            "Exception in thread pool task",
            "Connection reset by peer",
            "Connection closed by peer"
        ]
    elif rpc_backend == "TENSORPIPE":
        # FIXME Once we consolidate the error messages returned by the
        # TensorPipe agent put some more specific regex here.
        error_regexes = [".*"]
    else:
        error_regexes = [
            "Request aborted during client shutdown",
            "worker.: Error in reponse from worker.: server shutting down",
            "worker.: Error in response from worker.: Failed to write to remote endpoint",
            "worker.: Error in response from worker.: AsyncSocketException: recv() failed",
            "worker.: Error in response from worker.: Dropping unsent request"
        ]
    error_regex = "".join(["({})|".format(error_str) for error_str in error_regexes])
    # Strip out the last | or else it will match anything
    error_regex = error_regex[:-1]
    return error_regex

def get_timeout_error_regex(rpc_backend_name):
    """
    Given an RPC backend name, returns a partial string indicating the error we
    should receive when an RPC has timed out. Useful for use with
    assertRaisesRegex() to ensure we have the right errors during timeout.
    """
    if rpc_backend_name in ["PROCESS_GROUP", "FAULTY_PROCESS_GROUP", "TENSORPIPE"]:
        return "RPC ran for more than"
    else:
        return "(Timed out)|(Task expired)"


def wait_until_pending_futures_and_users_flushed(timeout=20):
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
    start = time.time()
    while True:
        debug_info = _rref_context_get_debug_info()
        num_pending_futures = int(debug_info["num_pending_futures"])
        num_pending_users = int(debug_info["num_pending_users"])
        if num_pending_futures == 0 and num_pending_users == 0:
            break
        time.sleep(0.1)
        if time.time() - start > timeout:
            raise ValueError(
                "Timed out waiting to flush pending futures and users, had {} pending futures and {} pending users".format(
                    num_pending_futures, num_pending_users
                )
            )


def get_num_owners_and_forks():
    rref_dbg_info = _rref_context_get_debug_info()
    num_owners = rref_dbg_info["num_owner_rrefs"]
    num_forks = rref_dbg_info["num_forks"]
    return num_owners, num_forks


def wait_until_owners_and_forks_on_rank(num_owners, num_forks, rank, timeout=20):
    """
    Waits until timeout for num_forks and num_owners to exist on the rank. Used
    to ensure proper deletion of RRefs in tests.
    """
    start = time.time()
    while True:
        num_owners_on_rank, num_forks_on_rank = rpc.rpc_sync(
            worker_name(rank), get_num_owners_and_forks, args=(), timeout=5
        )
        num_owners_on_rank = int(num_owners_on_rank)
        num_forks_on_rank = int(num_forks_on_rank)
        if num_owners_on_rank == num_owners and num_forks_on_rank == num_forks:
            return
        time.sleep(1)
        if time.time() - start > timeout:
            raise ValueError(
                "Timed out waiting for {} owners and {} forks on rank, had {} owners and {} forks".format(
                    num_owners, num_forks, num_owners_on_rank, num_forks_on_rank
                )
            )


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

def get_function_event(function_events, partial_event_name):
    """
    Returns the first event that matches partial_event_name in the provided
    function_events. These function_events should be the output of
    torch.autograd.profiler.function_events().

    Args:
    function_events: function_events returned by the profiler.
    event_name (str): partial key that the event was profiled with.
    """
    event = [event for event in function_events if partial_event_name in event.name][0]
    return event
