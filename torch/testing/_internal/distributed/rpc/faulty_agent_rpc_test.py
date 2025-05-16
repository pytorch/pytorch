# mypy: allow-untyped-defs

import time

import torch
import torch.distributed.rpc as rpc
from torch.distributed.rpc.api import _delete_all_user_and_unforked_owner_rrefs
from torch.testing._internal.dist_utils import (
    dist_init,
    wait_until_owners_and_forks_on_rank,
    wait_until_pending_futures_and_users_flushed,
    worker_name,
)
from torch.testing._internal.distributed.rpc.rpc_agent_test_fixture import (
    RpcAgentTestFixture,
)


def my_sleep_func(seconds=1):
    time.sleep(seconds)
    return torch.mul(torch.tensor(1), torch.tensor(1))


@torch.jit.script
def my_script_func(tensor):
    return torch.add(tensor, tensor)


def add_rref_to_value(rref, value):
    return rref.to_here() + value


class FaultyAgentRpcTest(RpcAgentTestFixture):
    # no faulty_messages defined so this fails all retryable messages - see
    # faulty_rpc_agent_test_fixture.py for the list of retryable messages.
    @dist_init(messages_to_delay={})
    def test_check_failed_messages(self):
        if self.rank == 0:
            dst_worker_b = worker_name((self.rank + 1) % self.world_size)
            dst_worker_c = worker_name((self.rank + 2) % self.world_size)

            # Worker0 sends RPC to Worker1 and creates an RRef there
            rref = rpc.remote(
                dst_worker_b, torch.add, args=(torch.ones(2, 2), torch.ones(2, 2))
            )
            # Worker0 sends an RPC to Worker2 with the RRef as an arg
            rpc.remote(dst_worker_c, add_rref_to_value, args=(rref, torch.ones(2, 2)))
            # check if the output is as expected
            self.assertEqual(
                rref.to_here(), torch.add(torch.ones(2, 2), torch.ones(2, 2))
            )
        # explicitly delete all User RRefs
        _delete_all_user_and_unforked_owner_rrefs()

    @dist_init
    def test_verify_backend_options(self):
        self.assertEqual(
            self.rpc_backend, rpc.backend_registry.BackendType.FAULTY_TENSORPIPE
        )
        self.assertEqual(self.rpc_backend_options.num_worker_threads, 8)
        self.assertEqual(self.rpc_backend_options.num_fail_sends, 3)
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 4)
        self.assertEqual(len(self.rpc_backend_options.messages_to_delay), 2)
        self.assertEqual(
            self.rpc_backend_options.rpc_timeout, rpc.constants.DEFAULT_RPC_TIMEOUT_SEC
        )

    @dist_init(faulty_messages=["RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT"])
    def test_custom_faulty_messages(self):
        self.assertEqual(
            {"RREF_FORK_REQUEST", "RREF_CHILD_ACCEPT"},
            set(self.rpc_backend_options.messages_to_fail),
        )

    @dist_init(faulty_messages=[])
    def test_no_faulty_messages(self):
        self.assertEqual(len(self.rpc_backend_options.messages_to_fail), 0)

    @dist_init(messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_custom_messages_to_delay(self):
        self.assertEqual(
            self.rpc_backend_options.messages_to_delay, {"SCRIPT_CALL": 1.5}
        )

    def _test_remote_message_dropped_pickle(self, dst=None):
        if self.rank != 0:
            return
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # Since we fail python_remote_call messages synchronously, the future
        # corresponding to this remote call will be marked with an error when
        # this function returns.
        rref = rpc.remote(dst_worker, my_sleep_func, args=(1,))
        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # Attempt to fork the RRef should raise an error indicating the rpc.remote timeout.
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref._serialize()
        # Test that using RRef as arg over RPC (which forks) results in the same
        # error
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rpc.rpc_async(dst_worker, add_rref_to_value, args=(rref, 1))

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_remote_message_dropped_pickle(self):
        self._test_remote_message_dropped_pickle()

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_remote_message_dropped_pickle_to_self(self):
        self._test_remote_message_dropped_pickle(self.rank)

    def _test_remote_message_dropped_timeout(self, func, args, dst=None):
        if self.rank != 0:
            return

        # test the case where rpc.remote() message creation is completely dropped.
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # Since we fail python_remote_call messages synchronously, the future
        # corresponding to this remote call will be marked with an error when
        # this function returns.
        rref = rpc.remote(dst_worker, func, args=args)
        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()
        # Note: during shutdown, logs will indicate "Could not find OwnerRRef..."
        # on the owning nodes, this is expected because the OwnerRRef was never
        # successfully created. Therefore, delAllUsers will work as expected.

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_builtin_remote_message_dropped_timeout(self):
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_dropped_timeout(func, args)

    @dist_init(faulty_messages=["SCRIPT_REMOTE_CALL"])
    def test_builtin_remote_message_dropped_timeout_to_self(self):
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_dropped_timeout(func, args, dst=0)

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_udf_remote_message_dropped_timeout(self):
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_dropped_timeout(func, args)

    @dist_init(faulty_messages=["PYTHON_REMOTE_CALL"])
    def test_udf_remote_message_dropped_timeout_to_self(self):
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_dropped_timeout(func, args, dst=0)

    def _test_remote_message_delay_timeout(self, func, args, dst=None):
        if self.rank != 0:
            return
        # Test the case where remote message is eventually processed on the owner,
        # but the future on the creator times out before the response comes back.
        dst_rank = dst if dst is not None else (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        # 10 ms timeout
        rref = rpc.remote(dst_worker, func, args=args, timeout=0.001)
        # Future corresponding to the remote creation should time out.
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref._get_future().wait()

        # Call to ensure pending callbacks are run.
        wait_until_pending_futures_and_users_flushed()
        # to_here() should now pick up that rpc.remote() creation has failed.
        with self.assertRaisesRegex(RuntimeError, "RRef creation"):
            rref.to_here()

        # Test the case where rpc.remote() times out, but to_here() has already
        # started blocking before.
        # NOTE: we only test this when not sending to self, as to_here() calls
        # calls localValue(), which does not send an RPC and thus does not have
        # a timeout. This can be supported by allowing future.wait() to
        # take in an optional timeout (https://github.com/pytorch/pytorch/issues/39280)
        if dst_rank != self.rank:
            slow_rref = rpc.remote(dst_worker, func, args=args, timeout=2)

            with self.assertRaisesRegex(RuntimeError, expected_error):
                # to_here() should raise timeout error, since it does not know about the
                # status of rpc.remote().
                slow_rref.to_here(0.001)
        # Note: If we proceed with shutdown, UserRRef will send out a RRefUserDelete
        # but this can be a noop since it may not exist on the owner yet. Later,
        # the owner can process the RRef creation and wait for the delete message,
        # thus leading to a timeout.
        # Therefore, we wait until we get notification that pending owners have
        # been confirmed before sending out RRefUserDeletes.
        if dst_rank != self.rank:
            wait_until_owners_and_forks_on_rank(2, 2, rank=dst_rank)

    @dist_init(faulty_messages=[], messages_to_delay={"PYTHON_REMOTE_CALL": 2})
    def test_udf_remote_message_delay_timeout(self):
        func = my_sleep_func
        args = (2,)
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(faulty_messages=[], messages_to_delay={"PYTHON_REMOTE_CALL": 2})
    def test_udf_remote_message_delay_timeout_to_self(self):
        func = my_sleep_func
        args = (1,)
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_builtin_delay_timeout(self):
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_builtin_delay_timeout_to_self(self):
        func = torch.add
        args = (torch.tensor(1), torch.tensor(1))
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_script_delay_timeout(self):
        func = my_script_func
        args = (torch.tensor(1),)
        self._test_remote_message_delay_timeout(func, args)

    @dist_init(
        faulty_messages=[],
        messages_to_delay={"SCRIPT_REMOTE_CALL": 2, "SCRIPT_RREF_FETCH_CALL": 1},
    )
    def test_remote_message_script_delay_timeout_to_self(self):
        func = my_script_func
        args = (torch.tensor(1),)
        self._test_remote_message_delay_timeout(func, args, dst=0)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_RREF_FETCH_CALL": 1})
    def test_rref_to_here_timeout(self):
        if self.rank != 0:
            return

        dst_rank = (self.rank + 1) % self.world_size
        dst_worker = f"worker{dst_rank}"
        rref = rpc.remote(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rref.to_here(0.01)

        rref.to_here()

    @dist_init(faulty_messages=[])
    def test_rpc_builtin_timeout(self):
        next_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(next_rank)
        expected_error = self.get_timeout_error_regex()
        # PYTHON_CALL message types which correspond to Python UDF over RPC
        # by default get a delay (see faulty_rpc_agent_test_fixture)
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(
                dst_worker,
                torch.add,
                args=(torch.tensor(1), torch.tensor(1)),
                timeout=1,
            )

        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=1
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure that the currently set default timeout is large enough such
        # that RPCs with delays still complete.
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        fut.wait()

        # Ensure timeout if we set a new default and don't override
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1))
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure run to completion if we specify timeout of 0
        fut = rpc.rpc_async(
            dst_worker, torch.add, args=(torch.tensor(1), torch.tensor(1)), timeout=0
        )
        fut.wait()
        # Reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)

    @dist_init(faulty_messages=[], messages_to_delay={"SCRIPT_CALL": 1.5})
    def test_rpc_script_timeout(self):
        next_rank = (self.rank + 1) % self.world_size
        dst_worker = worker_name(next_rank)
        expected_error = self.get_timeout_error_regex()
        with self.assertRaisesRegex(RuntimeError, expected_error):
            rpc.rpc_sync(dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1)

        fut = rpc.rpc_async(
            dst_worker, my_script_func, args=(torch.tensor(1),), timeout=1
        )
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure that the currently set default timeout is large enough such
        # that RPCs with delays still complete.
        fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),))
        fut.wait()

        # Ensure timeout if we set a new default and don't override
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(dst_worker, my_script_func, args=(torch.tensor(1),))
        with self.assertRaisesRegex(RuntimeError, expected_error):
            fut.wait()

        # Ensure run to completion if we specify timeout of 0
        rpc._set_rpc_timeout(0.001)
        fut = rpc.rpc_async(
            dst_worker, my_script_func, args=(torch.tensor(1),), timeout=0
        )
        fut.wait()
        # Reset for clean shutdown
        rpc._set_rpc_timeout(rpc.constants.DEFAULT_RPC_TIMEOUT_SEC)
