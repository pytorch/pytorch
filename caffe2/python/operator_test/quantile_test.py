from __future__ import absolute_import, division, print_function

import unittest

import caffe2.python.hypothesis_test_util as hu
import numpy as np
from caffe2.python import core, workspace


class TestQuantile(hu.HypothesisTestCase):
    def _test_quantile(self, inputs, quantile, abs, tol):
        net = core.Net("test_net")
        net.Proto().type = "dag"
        input_tensors = []
        for i, input in enumerate(inputs):
            workspace.FeedBlob("t_{}".format(i), input)
            input_tensors.append("t_{}".format(i))
        net.Quantile(
            input_tensors, ["quantile_value"], quantile=quantile, abs=abs, tol=tol
        )
        workspace.RunNetOnce(net)
        quantile_value_blob = workspace.FetchBlob("quantile_value")
        assert np.size(quantile_value_blob) == 1
        quantile_value = quantile_value_blob[0]

        input_cat = np.concatenate([input.flatten() for input in inputs])
        input_cat = np.abs(input_cat) if abs == 1 else input_cat
        target_cnt = np.ceil(np.size(input_cat) * quantile)
        actual_cnt = np.sum(input_cat <= quantile_value)
        # prune with return value will remove no less than
        # "quantile" portion of  elements
        assert actual_cnt >= target_cnt
        # Expect that (hi-lo) < tol * (|lo| + |hi|)
        # if tol < 1.0 -> hi * lo > 0, then we are expecting
        # 1. if hi >0,
        #           |hi|-|lo| < tol * (|lo| + |hi|)
        #          hi - lo  < (2 tol) /(1 + tol)  |hi| < 2 tol |hi|
        # 2. if hi < 0,
        #           |lo|- |hi| < tol * (|lo| + |hi|)
        #          hi - lo  < (2 tol) /(1 - tol)  |hi| < 2.5 tol |hi| if tol < 0.2
        quantile_value_lo = quantile_value - 2.5 * tol * np.abs(quantile_value)
        lo_cnt = np.sum(input_cat <= quantile_value_lo)
        # prune with a slightly smaller value will remove
        # less than "quantile" portion of elements
        assert lo_cnt <= target_cnt

    def test_quantile_1(self):
        inputs = []
        num_tensors = 5
        for i in range(num_tensors):
            dim = np.random.randint(5, 100)
            inputs.append(np.random.rand(dim))
        self._test_quantile(inputs=inputs, quantile=0.2, abs=1, tol=1e-4)

    def test_quantile_2(self):
        inputs = []
        num_tensors = 5
        for i in range(num_tensors):
            dim = np.random.randint(5, 100)
            inputs.append(np.random.rand(dim))
        self._test_quantile(inputs=inputs, quantile=1e-6, abs=0, tol=1e-3)

    def test_quantile_3(self):
        inputs = []
        num_tensors = 5
        for i in range(num_tensors):
            dim1 = np.random.randint(5, 100)
            dim2 = np.random.randint(5, 100)
            inputs.append(np.random.rand(dim1, dim2))
        self._test_quantile(inputs=inputs, quantile=1 - 1e-6, abs=1, tol=1e-5)

    def test_quantile_4(self):
        inputs = []
        num_tensors = 5
        for i in range(num_tensors):
            dim1 = np.random.randint(5, 100)
            dim2 = np.random.randint(5, 100)
            inputs.append(np.random.rand(dim1, dim2))
            inputs.append(np.random.rand(dim1))
        self._test_quantile(inputs=inputs, quantile=0.168, abs=1, tol=1e-4)


if __name__ == "__main__":
    global_options = ["caffe2"]
    core.GlobalInit(global_options)
    unittest.main()
