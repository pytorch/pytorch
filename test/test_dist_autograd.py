from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import torch.distributed.autograd as dist_autograd
import unittest

@unittest.skipIf(sys.version_info < (3, 0), "Pytorch distributed autograd package "
                 "does not support python2")
class TestDistAutograd(unittest.TestCase):

    def setUp(self):
        self.worker_id = 16
        # Using private init method here since 'init_model_parallel' currently
        # requires ProcessGroupGloo which in turns requires at least two
        # processes. We'd like to avoid multiprocessing in this unit test to
        # keep it simple and hence we use the private _init method.
        dist_autograd._init(self.worker_id)

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
