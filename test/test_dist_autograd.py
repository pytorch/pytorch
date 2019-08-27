from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import torch.distributed.autograd as dist_autograd
from common_distributed import MultiProcessTestCase
from functools import wraps
import unittest

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

@unittest.skipIf(sys.version_info < (3, 0), "Pytorch distributed autograd package "
                 "does not support python2")
class TestDistAutograd(MultiProcessTestCase):

    @dist_init
    def test_autograd_context(self):
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

if __name__ == '__main__':
    unittest.main()
