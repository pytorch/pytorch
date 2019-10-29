from __future__ import absolute_import, division, print_function, unicode_literals

import time
import unittest

import torch
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from dist_utils import INIT_METHOD_TEMPLATE, dist_init, TEST_CONFIG
from torch.distributed.rpc import RpcBackend

import threading

# Right now we test up to 3-layer nested rpc calls.
# rpc_done[1] and ctx_ids[1] represent rpc is done in prev rank, and context id
# sent from prev rank respectively.
# rpc_done[2] and ctx_ids[2] represents for prev of prev rank.
# rpc_done[3] and ctx_ids[3] represents for prev of prev of prev rank.
# rpc_done[0] and ctx_ids[0] represents for current rank, but mostly not used.
rpc_done = [False, False, False, False]
ctx_ids = [-1, -1, -1, -1]

known_context_ids = []


# Send rpc done info and context_id to
# dst_rank = (self.rank + rank_distance) % self.world_size
# we don't need a lock here since the GIL is held while executing remote
# python UDFs, so access is serialized across several workers.
def _set_rpc_done(ctx_id, rank_distance):
    global rpc_done
    global ctx_ids
    global known_context_ids
    rpc_done[rank_distance] = True
    ctx_ids[rank_distance] = ctx_id
    known_context_ids.append(ctx_id)


def my_py_add(t1, t2):
    return torch.add(t1, t2)


def my_py_nested_call(t1, t2, dst, world_size, hops):
    next_dst = (dst + 1) % world_size
    if hops > 0:
        return rpc.rpc_sync("worker{}".format(next_dst), my_py_nested_call,
                            args=(t1, t2, next_dst, world_size, hops - 1))
    else:
        return rpc.rpc_sync("worker{}".format(next_dst), torch.add, args=(t1, t2))

# after dist autograd context is cleaned up, it should be cleaned up on other
# nodes. This helper allows timeout_seconds for those RPCs to be completed, and
# ensures that all the contexts have been cleaned up in that timeframe.any
def _all_contexts_cleaned_up(timeout_seconds=10):
    global known_context_ids
    start = time.time()
    context_id_to_raised = {}
    while time.time() - start < timeout_seconds:
        for context_id in known_context_ids:
            try:
                dist_autograd._retrieve_context(context_id)
            except RuntimeError:
                context_id_to_raised[context_id] = True
        if len(context_id_to_raised) == len(known_context_ids):
            break
    # all contexts have been cleaned up if trying to retrieve any context resulted in a RuntimeError.
    success = len(context_id_to_raised) == len(known_context_ids) and all(context_id_to_raised.values())
    return success


from torch.autograd import Function
from torch.autograd.function import once_differentiable

class SimulateBackwardError(Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    @once_differentiable
    def backward(ctx, input):
        raise Exception('Simulate error on backward pass')

from enum import Enum

class ExecMode(Enum):
    LOCAL = 1  # Run the operation locally.
    REMOTE = 2  # Run the operation using RPC.


@unittest.skipIf(
    not torch._six.PY3, "Pytorch distributed autograd package " "does not support python2"
)
class DistAutogradTest(object):

    def _exec_func(self, exec_mode, method, *args):
        if ExecMode.LOCAL == exec_mode:
            if len(args) == 1 and isinstance(args[0], list):
                return method(*args[0])
            return method(*args)
        else:
            return rpc.rpc_sync('worker{}'.format(self._next_rank()), method,
                                args=(args))

    def _next_rank(self):
        if hasattr(self, 'dst_rank'):
            self.dst_rank = (self.dst_rank + 1) % self.world_size
            if self.dst_rank == self.rank:
                self._next_rank()
        else:
            self.dst_rank = (self.rank + 1) % self.world_size
        return self.dst_rank

    def _check_rpc_done(self, rank_distance):
        while not rpc_done[rank_distance]:
            time.sleep(0.1)
            pass

    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        return INIT_METHOD_TEMPLATE.format(file_name=self.file_name)

    @dist_init(setup_model_parallel=True)
    def test_autograd_context(self):
        # Verify max possible id.
        max_auto_increment = 281474976710655
        self.assertEqual(
            max_auto_increment + (self.worker_id << 48), dist_autograd._get_max_id()
        )

        context_ids = []
        for i in range(1000):
            with dist_autograd.context() as context_id:
                self.assertEqual(
                    context_id,
                    dist_autograd._retrieve_context(context_id)._context_id(),
                )
                # First 16 bits should be worker_id.
                self.assertEqual(self.worker_id, context_id >> 48)
                context_ids.append(context_id)

        for context_id in context_ids:
            with self.assertRaisesRegex(
                RuntimeError,
                "Could not find autograd context with id: {}".format(context_id),
            ):
                dist_autograd._retrieve_context(context_id)

    @dist_init(setup_model_parallel=True)
    def test_nested_context(self):
        with dist_autograd.context() as context_id:
            # Nested contexts not supported.
            with self.assertRaisesRegex(RuntimeError, "Already have an autograd context id for this thread"):
                with dist_autograd.context() as context_id:
                    pass

    # For current context, this rank sends t1 and t2 tensors to dst_rank,
    # then get t3 = torch.add(t1, t2) result tensor.
    # For the current context in this rank, it expects graph like this:
    #  send function:
    #              rpcSendBackward
    #                  /          \
    #  t1.AccumulateGrad         t2.AccumulateGrad
    #
    #  recv function:
    #
    #            |
    #          t3.rpcRecvBackward
    #
    def _verify_graph_for_first_rpc_call(self, send_function, recv_function, t1, t2, ret):
        # Retrieve the next functions in the graph.
        next_funcs = send_function.next_functions
        self.assertEqual(2, len(next_funcs))

        # We should now hit t1 and t2 in the autograd graph.
        self.assertEqual("torch::autograd::AccumulateGrad", next_funcs[0][0].name())
        self.assertEqual(t1, next_funcs[0][0].variable)
        self.assertEqual(0, next_funcs[0][1])
        self.assertEqual("torch::autograd::AccumulateGrad", next_funcs[1][0].name())
        self.assertEqual(t2, next_funcs[1][0].variable)
        self.assertEqual(0, next_funcs[1][1])

        # Test recv functions.
        self.assertEqual(ret.grad_fn, recv_function)

    # For a context passed from previous nested chain calls, this rank
    # receives two tensors t1 and t2, executes torch.add(t1, t2) and sends
    # result tensor t3 back.
    # For this context in this rank, it expects graph like this:
    #  send and recv functions:
    #       rpcSendBackward
    #           |
    #          t3.AddBackward0
    #          /             \
    # t1.recvRpcBackward    t2.recvRpcBackward
    def _verify_graph_for_rpc_call_exec(self, send_function):
        # Verify next function is AddBackward0
        next_funcs = send_function.next_functions
        self.assertEqual(1, len(next_funcs))
        add_backward_fn = next_funcs[0][0]
        self.assertEqual("AddBackward0", add_backward_fn.name())

        # Verify the next two functions are the same recv backward function.
        next_funcs = add_backward_fn.next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
        )
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[1][0].name()
        )
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])

    # For a context passed from previous nested chain calls, this rank
    # receives two tensors t1 and t2, forwards t1 and t2 tensors using
    # nested rpc call to next dst. In return route, receive result tensor t3
    # from next dst and forwarding t3 back to previous calls.
    # For this context in this rank, it expects graph like this:
    #  send and recv functions for receving and forwarding t1 and t2:
    #       rpcSendBackward
    #          /          \
    # t1.recvRpcBackward    t2.recvRpcBackward
    #  send and recv functions for receiving and forwarding t3:
    #       rpcSendBackward
    #             |
    #           t3.recvRpcBackward
    def _verify_graph_for_nested_rpc_call(self, ctx):
        send_functions = ctx._send_functions()
        self.assertEqual(2, len(send_functions))

        # For send function when making nest rpc call,
        # next functions of the send function are two recv functions
        # for recevied two tensors from previous call
        next_funcs = list(send_functions.values())[0].next_functions
        self.assertEqual(2, len(next_funcs))
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
        )
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[1][0].name()
        )
        self.assertEqual(next_funcs[0][0], next_funcs[1][0])


        # For send function when returning resonpose to previous call
        # next function of the send function is the recv function
        # for received tensor result returned from nested call
        next_funcs = list(send_functions.values())[1].next_functions
        self.assertEqual(1, len(next_funcs))
        self.assertEqual(
            "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
        )

    def _test_graph(self, fn):
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            ret = rpc.rpc_sync("worker{}".format(dst_rank), fn, args=(t1, t2))
            rpc.rpc_sync("worker{}".format(dst_rank),
                         _set_rpc_done, args=(context_id, 1))

            # Verify graph for current context id.
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self._verify_graph_for_first_rpc_call(list(send_functions.values())[0],
                                                  list(recv_functions.values())[0],
                                                  t1, t2, ret)

            # Wait for the prev rank to be done with rpc.
            self._check_rpc_done(1)
            # Verify graph for previous context id.
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            self._verify_graph_for_rpc_call_exec(list(send_functions.values())[0])
            # this barrier is needed so one worker does not clean up their
            # autograd context before another worker tries to access it.
            dist.barrier()

        # autograd context should be cleaned up by now.
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._retrieve_context(context_id)

        # No autograd context available.
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._current_context()

    @dist_init(setup_model_parallel=True)
    def test_graph_for_builtin_call(self):
        self._test_graph(torch.add)

    @dist_init(setup_model_parallel=True)
    def test_graph_for_python_call(self):
        self._test_graph(my_py_add)

    # 3-layer nested calls
    @dist_init(setup_model_parallel=True)
    def test_graph_for_py_nested_call(self):
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            nest_dst_rank = (dst_rank + 1) % self.world_size
            ret = rpc.rpc_sync("worker{}".format(dst_rank),
                               my_py_nested_call, args=(t1, t2, dst_rank, self.world_size, 1))
            for rd in [1, 2, 3]:
                rpc.rpc_sync("worker{}".format((self.rank + rd) % self.world_size),
                             _set_rpc_done, args=(context_id, rd))

            # For self.rank, it has 4 graphs to verify
            # One is for current context id when this rank send first rpc call.
            # Second one is for prev context id when this rank make 1st nested
            # call.
            # Third one is for prev prev context id when this rank make
            # 2nd nested call.
            # Last one is for prev prev prev context id when this rank
            # execute the torch.add() operator.

            # Verify first graph for current context id.
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self._verify_graph_for_first_rpc_call(list(send_functions.values())[0],
                                                  list(recv_functions.values())[0],
                                                  t1, t2, ret)

            # Verify second graph for 1st nested call.
            self._check_rpc_done(1)
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            self._verify_graph_for_nested_rpc_call(ctx)

            # Verify third graph for 2nd nested call.
            self._check_rpc_done(2)
            ctx = dist_autograd._retrieve_context(ctx_ids[2])
            self._verify_graph_for_nested_rpc_call(ctx)

            # verify last graph for rpc call execution.
            self._check_rpc_done(3)
            ctx = dist_autograd._retrieve_context(ctx_ids[3])
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))
            self._verify_graph_for_rpc_call_exec(list(send_functions.values())[0])
            # this barrier is needed so one worker does not clean up their
            # autograd context before another worker tries to access it.
            dist.barrier()

    # Rank0->Rank1->Rank0
    @dist_init(setup_model_parallel=True)
    def test_graph_for_py_nested_call_itself(self):
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            ret = rpc.rpc_sync("worker{}".format(dst_rank),
                               my_py_nested_call,
                               args=(t1, t2, (self.rank - 1 + self.world_size) % self.world_size, self.world_size, 0))
            rpc.rpc_sync("worker{}".format((self.rank + 1) % self.world_size),
                         _set_rpc_done, args=(context_id, 1))

            # For self.rank, it has 2 graphs to verify.
            # One is for current context id when this rank send first rpc
            # call and execute the torch.add() operator.
            # Another one is for prev context id when this rank make
            # nested call.
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(2, len(send_functions))
            recv_functions = ctx._recv_functions()
            self.assertEqual(2, len(recv_functions))
            self._verify_graph_for_first_rpc_call(list(send_functions.values())[0],
                                                  list(recv_functions.values())[1],
                                                  t1, t2, ret)
            self._verify_graph_for_rpc_call_exec(list(send_functions.values())[1])

            # Verify two pairs of send and recv functions for nested
            # call
            self._check_rpc_done(1)
            ctx = dist_autograd._retrieve_context(ctx_ids[1])
            self._verify_graph_for_nested_rpc_call(ctx)
            # this barrier is needed so one worker does not clean up their
            # autograd context before another worker tries to access it.
            dist.barrier()

    @dist_init(setup_model_parallel=True)
    def test_no_graph_with_tensors_not_require_grad(self):
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=False)
            t2 = torch.zeros(3, 3, requires_grad=False)
            ret = rpc.rpc_sync("worker{}".format(dst_rank), torch.add, args=(t1, t2))
            rpc.rpc_sync("worker{}".format(dst_rank),
                         _set_rpc_done, args=(context_id, 1))

            ctx = dist_autograd._current_context()
            send_functions = ctx._send_functions()
            self.assertEqual(len(send_functions), 0)
            recv_functions = ctx._recv_functions()
            self.assertEqual(len(recv_functions), 0)

            # Wait for the prev rank to be done with rpc.
            self._check_rpc_done(1)
            # prev context id is not passed over as tensors do not require grads
            with self.assertRaises(RuntimeError):
                ctx = dist_autograd._retrieve_context(ctx_ids[1])

    @dist_init(setup_model_parallel=True)
    def test_rpc_complex_args(self):
        with dist_autograd.context() as context_id:
            num_tensors = 10
            tensors = []
            for i in range(num_tensors):
                tensors.append(torch.ones(3, 3, requires_grad=(i % 2 == 0)))
            ret = rpc.rpc_sync(
                "worker{}".format(self._next_rank()), torch.stack, args=(tensors,)
            )
            self.assertEqual(torch.stack(tensors), ret)

            # Verify appropriate tensors have been attached the autograd graph.
            next_funcs = list(
                dist_autograd._current_context()._send_functions().values()
            )[0].next_functions
            idx = 0
            for i in range(num_tensors):
                if i % 2 == 0:
                    self.assertEqual(
                        "torch::autograd::AccumulateGrad", next_funcs[i][0].name()
                    )
                    self.assertEqual(tensors[i], next_funcs[i][0].variable)
                else:
                    self.assertIsNone(next_funcs[i][0])

            # Verify that the worker id has been recorded in the context
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(len(worker_ids), 1)
            dst_rank = (self.rank + 1) % self.world_size
            self.assertEqual(worker_ids[0], dst_rank)


    @dist_init(setup_model_parallel=True)
    def test_context_cleanup_many_workers(self):
        dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            for dst_rank in dst_ranks:
                ret = rpc.rpc_sync("worker{}".format(dst_rank), torch.add, args=(t1, t2))
                rpc.rpc_sync("worker{}".format(dst_rank), _set_rpc_done, args=(context_id, 1))
        # the thread's context id should be cleaned up
        with self.assertRaises(RuntimeError):
            dist_autograd._retrieve_context(context_id)
        # check that all contexts have been cleaned up.
        success = _all_contexts_cleaned_up()
        self.assertTrue(success)

    @dist_init(setup_model_parallel=True)
    def test_context_cleanup_nested_rpc(self):
        dst_rank = (self.rank + 1) % self.world_size
        nested_dst_rank = (dst_rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            rpc.rpc_sync("worker{}".format(dst_rank),
                         my_py_nested_call, args=(t1, t2, dst_rank, self.world_size, 0))
            # tell next worker and nested next worker to store this context id
            # so we can verify that it has been cleaned up
            rpc.rpc_sync("worker{}".format(dst_rank), _set_rpc_done, args=(context_id, 1))
            rpc.rpc_sync("worker{}".format(nested_dst_rank), _set_rpc_done, args=(context_id, 2))
        dist.barrier()  # let all nodes finish sending their RPCs
        success = _all_contexts_cleaned_up()
        self.assertTrue(success)

    @dist_init(setup_model_parallel=True)
    def test_worker_ids_recorded(self):
        dst_ranks = {rank for rank in range(self.world_size) if rank != self.rank}
        with dist_autograd.context() as context_id:
            # if no tensors require grad, we do not add the send functions, so
            # no worker ids should be recorded.
            t1 = torch.ones(3, 3, requires_grad=False)
            t2 = torch.zeros(3, 3, requires_grad=False)
            for dst_rank in dst_ranks:
                rpc.rpc_sync("worker{}".format(dst_rank), torch.add, args=(t1, t2))
                rpc.rpc_sync(
                    "worker{}".format(dst_rank), _set_rpc_done, args=(context_id, 1)
                )
            # no worker ids should be recorded.
            ctx = dist_autograd._current_context()
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(len(worker_ids), 0)

            # worker_ids should be recorded when tensors do require grad
            t1.requires_grad = True
            t2.requires_grad = True
            for dst_rank in dst_ranks:
                ret = rpc.rpc_sync("worker{}".format(dst_rank), torch.add, args=(t1, t2))
                rpc.rpc_sync(
                    "worker{}".format(dst_rank), _set_rpc_done, args=(context_id, 1)
                )
            # all worker_ids in dst_ranks should be recorded.
            worker_ids = ctx._known_worker_ids()
            self.assertEqual(len(worker_ids), len(dst_ranks))
            self.assertEqual(set(worker_ids), dst_ranks)

    @dist_init(setup_model_parallel=True)
    def test_error_in_context(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(6, 6, requires_grad=True)


            with self.assertRaises(RuntimeError):
                # This should throw an error since matrix sizes don't match.
                rpc.rpc_sync('worker{}'.format(self._next_rank()), torch.matmul,
                             args=(t1, t2))

    def _verify_backwards(self, exec_mode, tensors, context_id, local_grads, *args):
        if exec_mode == ExecMode.REMOTE:
            self._verify_backwards_remote(tensors, context_id, local_grads, *args)
        else:
            torch.autograd.backward(tensors)
            return [arg.grad for arg in args]

    def _verify_backwards_remote(self, tensors, context_id, local_grads, *args):
        dist_autograd.backward(tensors)

        # Verify grads were accumulated appropriately.
        grads = dist_autograd.get_gradients(context_id)
        nargs = len(args)
        ngrads = 0
        for i in range(0, nargs):
            if local_grads[i] is not None:
                self.assertIn(args[i], grads)
                self.assertEqual(local_grads[i], grads[args[i]])
                ngrads += 1
            else:
                self.assertNotIn(args[i], grads)

        self.assertEqual(ngrads, len(grads))


    @dist_init(setup_model_parallel=True)
    def test_backward_simple(self):
        # Run the same code locally and with dist autograd and verify gradients
        # are same.
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                ret = self._exec_func(exec_mode, torch.add, t1, t2)
                loss = ret.sum()
                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)

    @dist_init(setup_model_parallel=True)
    def test_backward_multiple_round_trips(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3))
        t3 = torch.rand((3, 3), requires_grad=True)
        t4 = torch.rand((3, 3))
        t5 = torch.rand((3, 3), requires_grad=True)

        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                # Multiple RPCs between different nodes.
                val = self._exec_func(exec_mode, torch.add, t1, t2)
                val = self._exec_func(exec_mode, torch.mul, t3, val)
                s1 = self._exec_func(exec_mode, torch.stack, (t4, val))
                s2 = self._exec_func(exec_mode, torch.stack, (t5, val))
                val = self._exec_func(exec_mode, torch.bmm, s1, s2)
                val = self._exec_func(exec_mode, torch.matmul, val, val)
                loss = val.sum()

                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t3, t4, t5)

    @dist_init(setup_model_parallel=True)
    def test_backward_different_tensor_dims(self):
        local_grads = None
        t1 = torch.rand((4, 6), requires_grad=True)
        t2 = torch.rand((6, 5))
        t3 = torch.rand((5, 7), requires_grad=True)
        t4 = torch.rand((7, 9))

        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                val = self._exec_func(exec_mode, torch.matmul, t1, t2)
                val = self._exec_func(exec_mode, torch.chain_matmul, [val, t3, t4])
                loss = val.sum()

                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t2, t3, t4)

    @dist_init(setup_model_parallel=True)
    def test_backward_unused_tensors(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        t3 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                s = self._exec_func(exec_mode, torch.stack, (t1, t2, t3))
                val = self._exec_func(exec_mode, torch.matmul, torch.narrow(s, 0, 0, 1), torch.narrow(s, 0, 2, 1))

                loss = val.sum()
                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2, t3)

    @dist_init(setup_model_parallel=True)
    def test_backward_multiple_output_tensors(self):
        local_grads = None
        t = torch.rand((10, 2), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                tensor_list = self._exec_func(exec_mode, torch.split, t, 2)
                t1 = tensor_list[0]
                t2 = tensor_list[2]
                t3 = tensor_list[4]

                val = self._exec_func(exec_mode, torch.chain_matmul, [t1, t2, t3])

                loss = val.sum()
                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t)

    def _run_test_backward_unused_send_function_in_thread(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)

            # We don't use the result of an RPC function, as a result the
            # backward pass would hang in the "FAST" mode.
            res = rpc.rpc_sync('worker{}'.format(self._next_rank()), torch.add,
                               args=(t1, t2))

            val = torch.mul(t1, t2)

            # Run backward, this would hang forever.
            dist_autograd.backward([val.sum()])


    @dist_init(setup_model_parallel=True)
    def test_backward_unused_send_function(self):
        # Run the test in a thread which would never finish.
        t = threading.Thread(target=self._run_test_backward_unused_send_function_in_thread)
        t.daemon = True
        t.start()
        t.join(10)  # Wait for 10s.

        # Verify thread is still alive (indicating backward hasn't completed yet).
        self.assertTrue(t.is_alive())

    @dist_init(setup_model_parallel=True)
    def test_backward_autograd_engine_error(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)

            # Perform some ops before error simulation.
            tmp = (t1 + t2) * (t1 + t2)
            t3 = SimulateBackwardError.apply(tmp)

            # Run multiple round trips across different nodes and verify the
            # original node receives an error thrown on a node deep in the chain.
            val = rpc.rpc_sync('worker{}'.format(self._next_rank()), torch.add,
                               args=(t2, t3))
            val = rpc.rpc_sync('worker{}'.format(self._next_rank()), torch.mul,
                               args=(val, t2))
            val = rpc.rpc_sync('worker{}'.format(self._next_rank()), torch.matmul,
                               args=(val, t2))
            val = rpc.rpc_sync('worker{}'.format(self._next_rank()), torch.div,
                               args=(val, t2))

            with self.assertRaises(RuntimeError):
                # Run backwards, and validate we receive an error.
                dist_autograd.backward([val.sum()])

    @unittest.skipIf(TEST_CONFIG.rpc_backend == RpcBackend.PROCESS_GROUP,
                     "Skipping this test temporarily since ProcessGroupAgent does not report errors on node failures")
    @dist_init(clean_shutdown=False)
    def test_backward_node_failure(self):
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)

            res = rpc.rpc_sync('worker{}'.format(self._next_rank()), torch.add,
                               args=(t1, t2))

            # Wait for all RPCs to be done.
            dist.barrier()

            # Kill all odd rank nodes.
            if self.rank % 2 == 0:
                # Wait a bit for all other nodes to die.
                time.sleep(5)
                with self.assertRaisesRegex(RuntimeError, "Request aborted during client shutdown"):
                    # Run backwards, and validate we receive an error since all
                    # other nodes are dead.
                    dist_autograd.backward([res.sum()])
            else:
                # Exit all other nodes.
                pass

    @dist_init(setup_model_parallel=True)
    def test_backward_without_context(self):
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        with self.assertRaisesRegex(RuntimeError, "Current thread doesn't have a valid autograd context"):
            res = rpc.rpc_sync('worker{}'.format(self._next_rank()), torch.add,
                               args=(t1, t2))
            dist_autograd.backward([res.sum()])

    @dist_init(setup_model_parallel=True)
    def test_backward_without_rpc(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            t3 = torch.add(t1, t2)

            dist_autograd.backward([t3.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(torch.ones(3, 3), grads[t1])
            self.assertEqual(torch.ones(3, 3), grads[t2])

    @dist_init(setup_model_parallel=True)
    def test_backward_invalid_args(self):
        with dist_autograd.context() as context_id:

            with self.assertRaisesRegex(TypeError, "incompatible function arguments"):
                dist_autograd.backward(None)

            with self.assertRaisesRegex(RuntimeError, "No tensors provided for gradient computation"):
                dist_autograd.backward([])

            with self.assertRaisesRegex(RuntimeError, "requires_grad not set on"):
                t = torch.rand(3, 3)
                dist_autograd.backward([t])

            with self.assertRaisesRegex(RuntimeError, "is not a scalar, all roots need to be scalar"):
                t = torch.rand(3, 3, requires_grad=True)
                dist_autograd.backward([t])

            with self.assertRaisesRegex(RuntimeError, "does not have a valid gradient function"):
                t = torch.rand(1, requires_grad=True)
                dist_autograd.backward([t])

    @dist_init(setup_model_parallel=True)
    def test_backward_multiple_roots(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                r1 = self._exec_func(exec_mode, torch.add, t1, t2).sum()
                r2 = self._exec_func(exec_mode, torch.mul, t1, t2).sum()
                r3 = self._exec_func(exec_mode, torch.cos, t1).sum()
                r4 = self._exec_func(exec_mode, torch.div, t1, t2).sum()

                local_grads = self._verify_backwards(exec_mode, [r1, r2, r3, r4], context_id, local_grads, t1, t2)

    @dist_init
    def test_backward_different_dtypes(self):
        local_grads = None
        t1 = torch.rand((3, 3), requires_grad=True, dtype=torch.float32)
        t2 = torch.rand((3, 3), requires_grad=True, dtype=torch.float64)
        for exec_mode in [ExecMode.LOCAL, ExecMode.REMOTE]:
            with dist_autograd.context() as context_id:
                loss = self._exec_func(exec_mode, torch.add, t1, t2).sum()

                local_grads = self._verify_backwards(exec_mode, [loss], context_id, local_grads, t1, t2)


if __name__ == '__main__':
    unittest.main()
