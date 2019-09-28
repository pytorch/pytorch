from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from common_distributed import MultiProcessTestCase
from functools import wraps
import six
import unittest
import torch
import time
import threading

if not dist.is_available():
    print("c10d not available, skipping tests")
    sys.exit(0)

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
        store = dist.FileStore(self.file.name, self.world_size)
        dist.init_process_group(backend='gloo', rank=self.rank,
                                world_size=self.world_size, store=store)
        # Use enough 'num_send_recv_threads' until we fix https://github.com/pytorch/pytorch/issues/26359
        dist.init_model_parallel('worker%d' % self.rank, num_send_recv_threads=16)
        func(self)
        dist.join_rpc()

    return wrapper

prev_rank_rpc_done = False
prev_rank_context_id = 0
def _set_rpc_done(context_id):
    global prev_rank_rpc_done
    global prev_rank_context_id
    prev_rank_rpc_done = True
    prev_rank_context_id = context_id

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

@unittest.skipIf(not six.PY3, "Pytorch distributed autograd package "
                 "does not support python2")
class TestDistAutograd(MultiProcessTestCase):

    def _next_rank(self, rank):
        return (rank + 1) % self.world_size

    @property
    def world_size(self):
        return 4

    @dist_init
    def test_autograd_context(self):
        # Verify max possible id.
        max_auto_increment = 281474976710655
        self.assertEqual(max_auto_increment + (self.worker_id << 48), dist_autograd._get_max_id())

        context_ids = []
        for i in range(1000):
            with dist_autograd.context() as context_id:
                self.assertEqual(context_id, dist_autograd._retrieve_context(context_id)._context_id())
                # First 16 bits should be worker_id.
                self.assertEqual(self.worker_id, context_id >> 48)
                context_ids.append(context_id)

        for context_id in context_ids:
            with self.assertRaisesRegex(RuntimeError, 'Could not find autograd context with id: {}'.format(context_id)):
                dist_autograd._retrieve_context(context_id)

    @dist_init
    def test_autograd_functions(self):
        dst_rank = self._next_rank(self.rank)
        with dist_autograd.context() as context_id:
            t1 = torch.ones(3, 3, requires_grad=True)
            t2 = torch.zeros(3, 3, requires_grad=True)
            ret = dist.rpc('worker{}'.format(dst_rank), torch.add,
                           args=(t1, t2))
            # Notify the next rank that we're done with the RPC.
            dist.rpc('worker{}'.format(dst_rank), _set_rpc_done, args=(context_id,))

            # Get send function.
            ctx = dist_autograd._current_context()
            self.assertEqual(context_id, ctx._context_id())
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))

            # Retrieve the next functions in the graph.
            next_funcs = list(send_functions.values())[0].next_functions
            self.assertEqual(2, len(next_funcs))

            # We should now hit t1 and t2 in the autograd graph.
            self.assertEqual('torch::autograd::AccumulateGrad', next_funcs[0][0].name())
            self.assertEqual(t1, next_funcs[0][0].variable)
            self.assertEqual(0, next_funcs[0][1])
            self.assertEqual('torch::autograd::AccumulateGrad', next_funcs[1][0].name())
            self.assertEqual(t2, next_funcs[1][0].variable)
            self.assertEqual(0, next_funcs[1][1])

            # Test recv functions.
            recv_functions = ctx._recv_functions()
            self.assertEqual(1, len(recv_functions))
            self.assertEqual(ret.grad_fn, list(recv_functions.values())[0])

            # We should have send/recv functions from the previous rank, get all
            # contexts in this node to find them.

            # Wait for the prev rank to be done with rpc.
            while not prev_rank_rpc_done:
                time.sleep(0.1)
                pass

            # Now verify the autograd graph.
            ctx = dist_autograd._retrieve_context(prev_rank_context_id)

            # Get the send function.
            send_functions = ctx._send_functions()
            self.assertEqual(1, len(send_functions))

            # Verify next function is AddBackward0
            next_funcs = list(send_functions.values())[0].next_functions
            self.assertEqual(1, len(next_funcs))
            add_backward_fn = next_funcs[0][0]
            self.assertEqual('AddBackward0', add_backward_fn.name())

            # Verify the next two functions are the same recv backward function.
            next_funcs = add_backward_fn.next_functions
            self.assertEqual(2, len(next_funcs))
            self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[0][0].name())
            self.assertEqual('torch::distributed::autograd::RecvRpcBackward', next_funcs[1][0].name())
            self.assertEqual(next_funcs[0][0], next_funcs[1][0])

        # autograd context should be cleaned up by now.
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._retrieve_context(context_id)

        # No autograd context available.
        with self.assertRaises(RuntimeError):
            ctx = dist_autograd._current_context()

    @dist_init
    def test_rpc_complex_args(self):
        dst_rank = self._next_rank(self.rank)
        with dist_autograd.context() as context_id:
            num_tensors = 10
            tensors = []
            for i in range(num_tensors):
                tensors.append(torch.ones(3, 3, requires_grad=(i % 2 == 0)))
            ret = dist.rpc('worker{}'.format(dst_rank), torch.stack,
                           args=(tensors,))
            self.assertEqual(torch.stack(tensors), ret)

            # Verify appropriate tensors have been attached the autograd graph.
            next_funcs = list(dist_autograd._current_context()._send_functions().values())[0].next_functions
            idx = 0
            for i in range(num_tensors):
                if i % 2 == 0:
                    self.assertEqual('torch::autograd::AccumulateGrad', next_funcs[i][0].name())
                    self.assertEqual(tensors[i], next_funcs[i][0].variable)
                else:
                    self.assertIsNone(next_funcs[i][0])

    @dist_init
    def test_error_in_context(self):
        dst_rank = self._next_rank(self.rank)
        with dist_autograd.context() as context_id:
            t1 = torch.rand(3, 3, requires_grad=True)
            t2 = torch.rand(6, 6, requires_grad=True)


            with self.assertRaises(RuntimeError):
                # This should throw an error.
                dist.rpc('worker{}'.format(dst_rank), torch.matmul,
                         args=(t1, t2))


    @dist_init
    def test_backward_simple(self):
        dst_rank = self._next_rank(self.rank)
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            ret = dist.rpc('worker{}'.format(dst_rank), torch.add,
                           args=(t1, t2))
            self.assertEqual(torch.add(t1, t2), ret)
            dist_autograd.backward([ret.sum()])

            # Verify grads were accumulated appropriately.
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)
            self.assertEqual(torch.ones(3, 3), grads[t1])
            self.assertEqual(torch.ones(3, 3), grads[t2])

    @dist_init
    def test_backward_multiple_round_trips(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3))
            t3 = torch.rand((3, 3), requires_grad=True)
            t4 = torch.rand((3, 3))
            t5 = torch.rand((3, 3), requires_grad=True)

            # Multiple RPCs between different nodes.
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.add,
                           args=(t1, t2))
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.mul,
                           args=(t3, val))
            s1 = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.stack,
                          args=([t4, val],))
            s2 = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.stack,
                          args=([t5, val],))
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.bmm,
                           args=(s1, s2))
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.matmul,
                           args=(val, val))

            # Now run backwards.
            dist_autograd.backward([val.sum()])

            # Verify grads were accumulated appropriately.
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(3, len(grads))
            self.assertIn(t1, grads)
            self.assertNotEqual(torch.zeros_like(t1), grads[t1])
            self.assertNotIn(t2, grads)
            self.assertIn(t3, grads)
            self.assertNotEqual(torch.zeros_like(t3), grads[t3])
            self.assertNotIn(t4, grads)
            self.assertIn(t5, grads)
            self.assertNotEqual(torch.zeros_like(t5), grads[t5])

    @dist_init
    def test_backward_different_tensor_dims(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t1 = torch.rand((4, 6), requires_grad=True)
            t2 = torch.rand((6, 5))
            t3 = torch.rand((5, 7), requires_grad=True)
            t4 = torch.rand((7, 9))
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.matmul,
                           args=(t1, t2))
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.chain_matmul,
                           args=([val, t3, t4],))

            # Now run backwards.
            dist_autograd.backward([val.sum()])

            # Verify grads were accumulated appropriately.
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertNotEqual(torch.zeros_like(t1), grads[t1])
            self.assertNotIn(t2, grads)
            self.assertIn(t3, grads)
            self.assertNotEqual(torch.zeros_like(t3), grads[t3])
            self.assertNotIn(t4, grads)

    @dist_init
    def test_backward_unused_tensors(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            t3 = torch.rand((3, 3), requires_grad=True)
            s = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.stack,
                         args=([t1, t2, t3],))
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.matmul,
                           args=(torch.narrow(s, 0, 0, 1), torch.narrow(s, 0, 2, 1)))

            # Now run backwards.
            dist_autograd.backward([val.sum()])

            # Verify grads were accumulated appropriately.
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(3, len(grads))
            self.assertIn(t1, grads)
            self.assertNotEqual(torch.zeros_like(t1), grads[t1])

            # t2 should have 0 grad (since it is not used).
            self.assertIn(t2, grads)
            self.assertEqual(torch.zeros_like(t2), grads[t2])

            self.assertIn(t3, grads)
            self.assertNotEqual(torch.zeros_like(t3), grads[t3])

    @dist_init
    def test_backward_multiple_output_tensors(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t = torch.rand((10, 2), requires_grad=True)

            tensor_list = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.split,
                                   args=(t, 2))
            self.assertEqual(5, len(tensor_list))
            t1 = tensor_list[0]
            t2 = tensor_list[2]
            t3 = tensor_list[4]

            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.chain_matmul,
                           args=([t1, t2, t3],))

            # Now run backwards.
            dist_autograd.backward([val.sum()])

            # Verify grads were accumulated appropriately.
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(1, len(grads))
            self.assertIn(t, grads)

            self.assertNotEqual(torch.zeros_like(t1), grads[t][0:2])
            self.assertEqual(torch.zeros_like(t1), grads[t][2:4])
            self.assertNotEqual(torch.zeros_like(t1), grads[t][4:6])
            self.assertEqual(torch.zeros_like(t1), grads[t][6:8])
            self.assertNotEqual(torch.zeros_like(t1), grads[t][8:10])

    def _run_test_backward_unused_send_function_in_thread(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)

            # We don't use the result of an RPC function, as a result the
            # backward pass would hang in the "FAST" mode.
            res = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.add,
                           args=(t1, t2))

            val = torch.mul(t1, t2)

            # Run backward, this would hang forever.
            dist_autograd.backward([val.sum()])


    @dist_init
    def test_backward_unused_send_function(self):
        # Run the test in a thread which would never finish.
        t = threading.Thread(target=self._run_test_backward_unused_send_function_in_thread)
        t.daemon = True
        t.start()
        t.join(10)  # Wait for 10s.

        # Verify thread is still alive (indicating backward hasn't completed yet).
        self.assertTrue(t.is_alive())

    @dist_init
    def test_backward_autograd_engine_error(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)
            t3 = SimulateBackwardError.apply(t1)

            # Run multiple round trips across different nodes and verify the
            # original node receives an error thrown on a node deep in the chain.
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.add,
                           args=(t2, t3))
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.mul,
                           args=(val, t2))
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.matmul,
                           args=(val, t2))
            val = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.div,
                           args=(val, t2))

            with self.assertRaises(RuntimeError):
                # Run backwards, and validate we receive an error.
                dist_autograd.backward([val.sum()])

    @dist_init
    @unittest.skip("Skipping this test temporarily since ProcessGroupAgent does not report errors on node failures")
    def test_backward_node_failure(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)

            res = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.add,
                           args=(t1, t2))

            if self.rank == 0:
                # Wait a bit for all other nodes to die.
                time.sleep(3)
                with self.assertRaises(RuntimeError):
                    # Run backwards, and validate we receive an error since all
                    # other nodes are dead.
                    dist_autograd.backward([res.sum()])
            else:
                # Kill all other nodes.
                sys.exit(0)

    @dist_init
    def test_backward_without_context(self):
        dst_rank = self.rank
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        with self.assertRaisesRegex(RuntimeError, "Current thread doesn't have a valid autograd context"):
            res = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.add,
                           args=(t1, t2))
            dist_autograd.backward([res.sum()])

    @dist_init
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

    @dist_init
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

    @dist_init
    def test_backward_multiple_roots(self):
        dst_rank = self.rank
        with dist_autograd.context() as context_id:
            t1 = torch.rand((3, 3), requires_grad=True)
            t2 = torch.rand((3, 3), requires_grad=True)

            r1 = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.add,
                          args=(t1, t2))
            r2 = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.mul,
                          args=(t1, t2))
            r3 = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.cos,
                          args=(t1,))
            r4 = dist.rpc('worker{}'.format(self._next_rank(dst_rank)), torch.div,
                          args=(t1, t2))

            dist_autograd.backward([r1.sum(), r2.sum(), r3.sum(), r4.sum()])
            grads = dist_autograd.get_gradients(context_id)
            self.assertEqual(2, len(grads))
            self.assertIn(t1, grads)
            self.assertIn(t2, grads)


if __name__ == '__main__':
    unittest.main()
