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
        dist.init_model_parallel('worker%d' % self.rank)
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

@unittest.skipIf(not six.PY3, "Pytorch distributed autograd package "
                 "does not support python2")
class TestDistAutograd(MultiProcessTestCase):

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
        dst_rank = (self.rank + 1) % self.world_size
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
        dst_rank = (self.rank + 1) % self.world_size
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


if __name__ == '__main__':
    unittest.main()
