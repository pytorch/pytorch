from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import workspace, model_helpers
from caffe2.python.model_helper import ModelHelperBase

import caffe2.python.hypothesis_test_util as hu
from hypothesis import given
import hypothesis.strategies as st
import numpy as np


class ModelHelpersTest(hu.HypothesisTestCase):

    @given(n=st.integers(2, 5), m=st.integers(2, 5), **hu.gcs)
    def test_dropout(self, n, m, gc, dc):
        X = np.random.rand(n, m).astype(np.float32) - 0.5
        workspace.FeedBlob("x", X)
        model = ModelHelperBase(name="test_model")
        out = model_helpers.Dropout(model, "x", "out")
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        out = workspace.FetchBlob("x")
        np.testing.assert_equal(X, out)

    @given(n=st.integers(2, 5), m=st.integers(2, 5),
           k=st.integers(2, 5), **hu.gcs)
    def test_fc(self, n, m, k, gc, dc):
        X = np.random.rand(m, k).astype(np.float32) - 0.5

        workspace.FeedBlob("x", X)
        model = ModelHelperBase(name="test_model")
        out = model_helpers.FC(model, "x", "out_1", k, n)
        out = model_helpers.PackedFC(model, out, "out_2", n, n)
        out = model_helpers.FC_Decomp(model, out, "out_3", n, n)
        out = model_helpers.FC_Prune(model, out, "out_4", n, n)

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
