from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import workspace, model_helpers
from caffe2.python.model_helper import ModelHelper

import unittest
import numpy as np


class ModelHelpersTest(unittest.TestCase):
    def setUp(self):

        def myhelper(model, val=-1):
            return val

        if not model_helpers.has_helper(myhelper):
            model_helpers.Register(myhelper)
        self.myhelper = myhelper

        def myhelper2(model, val=-1):
            return val

        if not model_helpers.has_helper(myhelper2):
            model_helpers.Register(myhelper2)
        self.myhelper2 = myhelper2

    def test_dropout(self):
        p = 0.2
        X = np.ones((100, 100)).astype(np.float32) - p
        workspace.FeedBlob("x", X)
        model = ModelHelper(name="test_model")
        model_helpers.Dropout(model, "x", "out")
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        out = workspace.FetchBlob("out")
        self.assertLess(abs(out.mean() - (1 - p)), 0.05)

    def test_fc(self):
        m, n, k = (15, 15, 15)
        X = np.random.rand(m, k).astype(np.float32) - 0.5

        workspace.FeedBlob("x", X)
        model = ModelHelper(name="test_model")
        out = model_helpers.FC(model, "x", "out_1", k, n)
        out = model_helpers.PackedFC(model, out, "out_2", n, n)
        out = model_helpers.FC_Decomp(model, out, "out_3", n, n)
        out = model_helpers.FC_Prune(model, out, "out_4", n, n)

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

    def test_arg_scope(self):
        myhelper = self.myhelper
        myhelper2 = self.myhelper2
        n = 15
        with model_helpers.arg_scope([myhelper], val=n):
            res = model_helpers.myhelper(None)
        self.assertEqual(n, res)

        with model_helpers.arg_scope([myhelper, myhelper2], val=n):
            res1 = model_helpers.myhelper(None)
            res2 = model_helpers.myhelper2(None)
        self.assertEqual([n, n], [res1, res2])

    def test_arg_scope_single(self):
        X = np.random.rand(64, 3, 32, 32).astype(np.float32) - 0.5

        workspace.FeedBlob("x", X)
        model = ModelHelper(name="test_model")
        with model_helpers.arg_scope(
            model_helpers.Conv,
            stride=2,
            pad=2,
            weight_init=('XavierFill', {}),
            bias_init=('ConstantFill', {})
        ):
            model_helpers.Conv(
                model=model,
                blob_in="x",
                blob_out="out",
                dim_in=3,
                dim_out=64,
                kernel=3,
            )

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        out = workspace.FetchBlob("out")
        self.assertEqual(out.shape, (64, 64, 17, 17))

    def test_arg_scope_nested(self):
        myhelper = self.myhelper
        n = 16
        with model_helpers.arg_scope([myhelper], val=-3), \
             model_helpers.arg_scope([myhelper], val=-2):
            with model_helpers.arg_scope([myhelper], val=n):
                res = model_helpers.myhelper(None)
                self.assertEqual(n, res)
            res = model_helpers.myhelper(None)
            self.assertEqual(res, -2)

        res = model_helpers.myhelper(None, val=15)
        self.assertEqual(res, 15)

    def test_double_register(self):
        myhelper = self.myhelper
        with self.assertRaises(AttributeError):
            model_helpers.Register(myhelper)

    def test_has_helper(self):
        self.assertTrue(model_helpers.has_helper(model_helpers.Conv))
        self.assertTrue(model_helpers.has_helper("Conv"))

        def myhelper3():
            pass

        self.assertFalse(model_helpers.has_helper(myhelper3))
