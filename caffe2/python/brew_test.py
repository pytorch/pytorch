from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import workspace, brew
from caffe2.python.model_helper import ModelHelper
from caffe2.python.cnn import CNNModelHelper

import unittest
import numpy as np


class BrewTest(unittest.TestCase):
    def setUp(self):

        def myhelper(model, val=-1):
            return val

        if not brew.has_helper(myhelper):
            brew.Register(myhelper)
        self.myhelper = myhelper

        def myhelper2(model, val=-1):
            return val

        if not brew.has_helper(myhelper2):
            brew.Register(myhelper2)
        self.myhelper2 = myhelper2
        self.model = ModelHelper(name="test_model")

    def test_dropout(self):
        p = 0.2
        X = np.ones((100, 100)).astype(np.float32) - p
        workspace.FeedBlob("x", X)
        model = ModelHelper(name="test_model")
        brew.dropout(model, "x", "out")
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        out = workspace.FetchBlob("out")
        self.assertLess(abs(out.mean() - (1 - p)), 0.05)

    def test_fc(self):
        m, n, k = (15, 15, 15)
        X = np.random.rand(m, k).astype(np.float32) - 0.5

        workspace.FeedBlob("x", X)
        model = ModelHelper(name="test_model")
        brew.fc(model, "x", "out_1", k, n)
        model.Validate()
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

    def test_validate(self):
        model = ModelHelper(name="test_model")
        model.params.append("aaa")
        model.params.append("bbb")
        self.assertEqual(model._Validate(), [])

        model.params.append("xxx")
        model.params.append("bbb")
        self.assertEqual(model._Validate(), ["bbb"])

    def test_arg_scope(self):
        myhelper = self.myhelper
        myhelper2 = self.myhelper2
        n = 15
        with brew.arg_scope([myhelper], val=n):
            res = brew.myhelper(self.model)
        self.assertEqual(n, res)

        with brew.arg_scope([myhelper, myhelper2], val=n):
            res1 = brew.myhelper(self.model)
            res2 = brew.myhelper2(self.model)
        self.assertEqual([n, n], [res1, res2])

    def test_arg_scope_single(self):
        X = np.random.rand(64, 3, 32, 32).astype(np.float32) - 0.5

        workspace.FeedBlob("x", X)
        model = ModelHelper(name="test_model")
        with brew.arg_scope(
            brew.conv,
            stride=2,
            pad=2,
            weight_init=('XavierFill', {}),
            bias_init=('ConstantFill', {})
        ):
            brew.conv(
                model=model,
                blob_in="x",
                blob_out="out",
                dim_in=3,
                dim_out=64,
                kernel=3,
            )
        model.Validate()
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        out = workspace.FetchBlob("out")
        self.assertEqual(out.shape, (64, 64, 17, 17))

    def test_arg_scope_nested(self):
        myhelper = self.myhelper
        n = 16
        with brew.arg_scope([myhelper], val=-3), \
             brew.arg_scope([myhelper], val=-2):
            with brew.arg_scope([myhelper], val=n):
                res = brew.myhelper(self.model)
                self.assertEqual(n, res)
            res = brew.myhelper(self.model)
            self.assertEqual(res, -2)

        res = brew.myhelper(self.model, val=15)
        self.model.Validate()
        self.assertEqual(res, 15)

    def test_double_register(self):
        myhelper = self.myhelper
        with self.assertRaises(AttributeError):
            brew.Register(myhelper)

    def test_has_helper(self):
        self.assertTrue(brew.has_helper(brew.conv))
        self.assertTrue(brew.has_helper("conv"))

        def myhelper3():
            pass

        self.assertFalse(brew.has_helper(myhelper3))

    def test_model_helper(self):
        X = np.random.rand(64, 32, 32, 3).astype(np.float32) - 0.5

        workspace.FeedBlob("x", X)
        my_arg_scope = {'order': 'NHWC'}
        model = ModelHelper(name="test_model", arg_scope=my_arg_scope)
        with brew.arg_scope(
            brew.conv,
            stride=2,
            pad=2,
            weight_init=('XavierFill', {}),
            bias_init=('ConstantFill', {})
        ):
            brew.conv(
                model=model,
                blob_in="x",
                blob_out="out",
                dim_in=3,
                dim_out=64,
                kernel=3,
            )
        model.Validate()
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        out = workspace.FetchBlob("out")
        self.assertEqual(out.shape, (64, 17, 17, 64))

    def test_cnn_model_helper_deprecated(self):
        X = np.random.rand(64, 32, 32, 3).astype(np.float32) - 0.5

        workspace.FeedBlob("x", X)
        # CNNModelHelper is going to be deprecated soon. This test is only
        # covering some CNNModelHelper logic
        model = CNNModelHelper(name="test_model", order='NHWC')
        self.assertEqual(model.arg_scope['order'], 'NHWC')
