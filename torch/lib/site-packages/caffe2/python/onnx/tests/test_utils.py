## @package onnx
# Module caffe2.python.onnx.tests.test_utils

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import unittest

import numpy as np
from caffe2.python.models.download import downloadFromURLToFile, getURLFromName, deleteDirectory


class TestCase(unittest.TestCase):

    def setUp(self):
        np.random.seed(seed=0)

    def assertSameOutputs(self, outputs1, outputs2, decimal=7):
        self.assertEqual(len(outputs1), len(outputs2))
        for o1, o2 in zip(outputs1, outputs2):
            self.assertEqual(o1.dtype, o2.dtype)
            np.testing.assert_almost_equal(o1, o2, decimal=decimal)

    def add_test_case(self, name, test_func):
        if not name.startswith('test_'):
            raise ValueError('Test name must start with test_: {}'.format(name))
        if hasattr(self, name):
            raise ValueError('Duplicated test name: {}'.format(name))
        setattr(self, name, test_func)


class DownloadingTestCase(TestCase):

    def _download(self, model):
        model_dir = self._model_dir(model)
        assert not os.path.exists(model_dir)
        os.makedirs(model_dir)
        for f in ['predict_net.pb', 'init_net.pb', 'value_info.json']:
            url = getURLFromName(model, f)
            dest = os.path.join(model_dir, f)
            try:
                try:
                    downloadFromURLToFile(url, dest,
                                          show_progress=False)
                except TypeError:
                    # show_progress not supported prior to
                    # Caffe2 78c014e752a374d905ecfb465d44fa16e02a28f1
                    # (Sep 17, 2017)
                    downloadFromURLToFile(url, dest)
            except Exception as e:
                print("Abort: {reason}".format(reason=e))
                print("Cleaning up...")
                deleteDirectory(model_dir)
                raise AssertionError("Test model downloading failed")
