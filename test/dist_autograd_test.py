from __future__ import absolute_import, division, print_function, unicode_literals

import time
import unittest

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from dist_utils import INIT_METHOD_TEMPLATE, dist_init


prev_rank_rpc_done = False
prev_rank_context_id = -1
prev_prev_rank_rpc_done = False
prev_prev_rank_context_id = -1


def _set_rpc_done_from_prev_rank(context_id):
    global prev_rank_rpc_done
    global prev_rank_context_id
    prev_rank_rpc_done = True
    prev_rank_context_id = context_id


def _set_rpc_done_from_prev_prev_rank(context_id):
    global prev_prev_rank_rpc_done
    global prev_prev_rank_context_id
    prev_prev_rank_rpc_done = True
    prev_prev_rank_context_id = context_id


def my_py_add(t1, t2):
    return torch.add(t1, t2)


def my_py_nested_call(t1, t2, nest_dst_rank):
    return rpc.rpc_sync("worker{}".format(nest_dst_rank),
                        torch.add, args=(t1, t2))


@unittest.skipIf(
    not torch._six.PY3, "Pytorch distributed autograd package " "does not support python2"
)
class DistAutogradTest(object):
    @property
    def world_size(self):
        return 4

    @property
    def init_method(self):
        return INIT_METHOD_TEMPLATE.format(
            file_name=self.file_name, rank=self.rank, world_size=self.world_size
        )

    @dist_init
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

    # Client sends tensors and then receives tensor result
    def _verify_send_recv_functions_in_client(self, context_id, t1, t2, ret):
        # Get send function.
        ctx = dist_autograd._current_context()
        self.assertEqual(context_id, ctx._context_id())
        send_functions = ctx._send_functions()
        self.assertEqual(1, len(send_functions))

        # Retrieve the next functions in the graph.
        next_funcs = list(send_functions.values())[0].next_functions
        self.assertEqual(2, len(next_funcs))

        # We should now hit t1 and t2 in the autograd graph.
        self.assertEqual("torch::autograd::AccumulateGrad", next_funcs[0][0].name())
        self.assertEqual(t1, next_funcs[0][0].variable)
        self.assertEqual(0, next_funcs[0][1])
        self.assertEqual("torch::autograd::AccumulateGrad", next_funcs[1][0].name())
        self.assertEqual(t2, next_funcs[1][0].variable)
        self.assertEqual(0, next_funcs[1][1])

        # Test recv functions.
        recv_functions = ctx._recv_functions()
        self.assertEqual(1, len(recv_functions))
        self.assertEqual(ret.grad_fn, list(recv_functions.values())[0])

    # Host receives tensors and actually runs tensor operations, return tensor
    # result
    def _verify_send_recv_functions_in_tensor_run(self, ctx):
        # Get the send function.
        send_functions = ctx._send_functions()
        self.assertEqual(1, len(send_functions))

        # Verify next function is AddBackward0
        next_funcs = list(send_functions.values())[0].next_functions
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

    def _test_autograd_functions(self, fn):
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            ret = rpc.rpc_sync("worker{}".format(dst_rank), fn, args=(t1, t2))
            rpc.rpc_sync("worker{}".format(dst_rank),
                         _set_rpc_done_from_prev_rank, args=(context_id,))

            self._verify_send_recv_functions_in_client(context_id, t1, t2, ret)

            # We should have send/recv functions from the previous rank, get all
            # contexts in this node to find them.

            # Wait for the prev rank to be done with rpc.
            while not prev_rank_rpc_done:
                time.sleep(0.1)
                pass

            # Now verify the autograd graph.
            ctx = dist_autograd._retrieve_context(prev_rank_context_id)

            self._verify_send_recv_functions_in_tensor_run(ctx)

        # autograd context should be cleaned up by now.
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._retrieve_context(context_id)

        # No autograd context available.
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._current_context()

    @dist_init
    def test_autograd_functions_for_builtin_call(self):
        self._test_autograd_functions(torch.add)

    @dist_init
    def test_autograd_functions_for_python_call(self):
        self._test_autograd_functions(my_py_add)

    @dist_init
    def test_autograd_functions_for_python_nested_call(self):
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            nest_dst_rank = (dst_rank + 1) % self.world_size
            ret = rpc.rpc_sync("worker{}".format(dst_rank),
                               my_py_nested_call, args=(t1, t2, nest_dst_rank))
            rpc.rpc_sync("worker{}".format(dst_rank),
                         _set_rpc_done_from_prev_rank, args=(context_id,))
            rpc.rpc_sync("worker{}".format(nest_dst_rank),
                         _set_rpc_done_from_prev_prev_rank, args=(context_id,))

            # For self.rank, it has four pairs of send and recv funcitons
            # One pair is for current context id when this rank is worked as
            # client
            # Another two pairs are for prev context id when this rank is worked
            # as server
            # Last pair is for prev prev context id when this rank is worked as
            # nested dst rank host to run the nested rpc call inside
            # my_py_nested_call

            # verify first pair of send and recv functions for context_id
            # Get send function.
            self._verify_send_recv_functions_in_client(context_id, t1, t2, ret)

            # verify two pairs of send and recv functions for
            # prev_rank_context_id
            # We should have send/recv functions from the previous rank, get all
            # contexts in this node to find them.
            # Wait for the prev rank to be done with rpc.
            while not prev_rank_rpc_done:
                time.sleep(0.1)
                pass
            ctx = dist_autograd._retrieve_context(prev_rank_context_id)

            # There two send functions, one is send function for nest rpc call,
            # another one is for returning response to client
            send_functions = ctx._send_functions()
            self.assertEqual(2, len(send_functions))

            # For send function when making nest rpc call,
            # next functions of the send function are two recv functions
            # for recevied two tensors from client
            next_funcs = list(send_functions.values())[0].next_functions
            self.assertEqual(2, len(next_funcs))
            self.assertEqual(
                "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
            )
            self.assertEqual(
                "torch::distributed::autograd::RecvRpcBackward", next_funcs[1][0].name()
            )
            self.assertEqual(next_funcs[0][0], next_funcs[1][0])

            # For send function when returning resonpose to client
            # next function of the send function is the recv function
            # for received tensor result returned from nested call
            next_funcs = list(send_functions.values())[1].next_functions
            self.assertEqual(1, len(next_funcs))
            self.assertEqual(
                "torch::distributed::autograd::RecvRpcBackward", next_funcs[0][0].name()
            )

            # verify third pair of send and recv functions for
            # prev_prev_rank_context_id
            # We should have send/recv functions from the previous of previous
            # rank, get all contexts in this node to find them.
            # Wait for the prev prev rank to be done with rpc.
            while not prev_prev_rank_rpc_done:
                time.sleep(0.1)
                pass
            ctx = dist_autograd._retrieve_context(prev_prev_rank_context_id)
            self._verify_send_recv_functions_in_tensor_run(ctx)

    @dist_init
    def test_rpc_complex_args(self):
        dst_rank = (self.rank + 1) % self.world_size
        with dist_autograd.context() as context_id:
            num_tensors = 10
            tensors = []
            for i in range(num_tensors):
                tensors.append(torch.ones(3, 3, requires_grad=(i % 2 == 0)))
            ret = rpc.rpc_sync(
                "worker{}".format(dst_rank), torch.stack, args=(tensors,)
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

    @dist_init
    def test_nested_contex(self):
        with self.assertRaises(RuntimeError):
            with dist_autograd.context() as context_id_1:
                with dist_autograd.context() as context_id_2:
                    a = 1
                b = 1
