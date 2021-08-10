




import numpy as np

import caffe2.python.models.resnet as resnet
import hypothesis.strategies as st
from hypothesis import given, settings
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.models.imagenet_trainer_test_utils as utils

class ResnetMemongerTest(hu.HypothesisTestCase):

    @given(with_shapes=st.booleans(), **hu.gcs_cpu_only)
    @settings(max_examples=2, deadline=None)
    def test_resnet_shared_grads(self, with_shapes, gc, dc):
        results = utils.test_shared_grads(
            with_shapes,
            resnet.create_resnet50,
            'gpu_0/conv1_w',
            'gpu_0/last_out_L1000'
        )
        self.assertTrue(results[0][0] < results[0][1])
        np.testing.assert_almost_equal(results[1][0], results[1][1])
        np.testing.assert_almost_equal(results[2][0], results[2][1])

    def test_resnet_forward_only(self):
        results = utils.test_forward_only(
            resnet.create_resnet50,
            'gpu_0/last_out_L1000'
        )
        self.assertTrue(results[0][0] < results[0][1])
        self.assertTrue(results[1] < 7 and results[1] > 0)
        np.testing.assert_almost_equal(results[2][0], results[2][1])

    def test_resnet_forward_only_fast_simplenet(self):
        '''
        Test C++ memonger that is only for simple nets
        '''
        results = utils.test_forward_only_fast_simplenet(
            resnet.create_resnet50,
            'gpu_0/last_out_L1000'
        )

        self.assertTrue(results[0][0] < results[0][1])
        self.assertTrue(results[1] < 4 and results[1] > 0)
        np.testing.assert_almost_equal(results[2][0], results[2][1])


if __name__ == "__main__":
    import unittest
    import random
    random.seed(2603)
    # pyre-fixme[10]: Name `workspace` is used but not defined in the current scope
    workspace.GlobalInit([
        'caffe2',
        '--caffe2_log_level=0',
        '--caffe2_print_blob_sizes_at_exit=0',
        '--caffe2_gpu_memory_tracking=1'])
    unittest.main()
