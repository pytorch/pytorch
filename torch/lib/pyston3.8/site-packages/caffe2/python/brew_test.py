




from caffe2.python import brew, core, scope, workspace
from caffe2.python.modeling.parameter_info import ParameterTags
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
        brew.dropout(model, "x", "out", is_test=False)
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

    def test_relu(self):
        Xpos = np.ones((5, 5)).astype(np.float32) - 0.5
        Xneg = np.ones((5, 5)).astype(np.float32) - 1.5

        workspace.FeedBlob("xpos", Xpos)
        workspace.FeedBlob("xneg", Xneg)
        model = ModelHelper(name="test_model")
        brew.relu(model, "xpos", "out_xpos")
        brew.relu(model, "xneg", "out_xneg")
        model.Validate()
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        pos = workspace.FetchBlob("out_xpos")
        self.assertAlmostEqual(pos.mean(), 0.5)
        neg = workspace.FetchBlob("out_xneg")
        self.assertAlmostEqual(neg.mean(), 0)

    def test_tanh(self):
        X = np.ones((5, 5)).astype(np.float32) - 0.5

        workspace.FeedBlob("x", X)
        model = ModelHelper(name="test_model")
        brew.tanh(model, "x", "out_tanh")
        model.Validate()
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        out = workspace.FetchBlob("out_tanh")
        self.assertAlmostEqual(out.mean(), np.tanh(0.5), places=5)

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
                kernel=[8, 3]
            )
        model.Validate()
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        out = workspace.FetchBlob("out")
        self.assertEqual(out.shape, (64, 15, 17, 64))

    def test_cnn_model_helper_deprecated(self):
        X = np.random.rand(64, 32, 32, 3).astype(np.float32) - 0.5

        workspace.FeedBlob("x", X)
        # CNNModelHelper is going to be deprecated soon. This test is only
        # covering some CNNModelHelper logic
        model = CNNModelHelper(name="test_model", order='NHWC')
        self.assertEqual(model.arg_scope['order'], 'NHWC')

    def test_get_params(self):
        def param(x):
            return core.ScopedBlobReference(x)

        def to_str_list(x):
            return sorted([str(p) for p in x])

        model = ModelHelper(name="test_model")
        model.AddParameter(param("a"))
        model.AddParameter(param("b"), tags=ParameterTags.COMPUTED_PARAM)
        with scope.NameScope("c"):
            model.AddParameter(param("a"))
            model.AddParameter(param("d"), tags=ParameterTags.COMPUTED_PARAM)
            self.assertEqual(to_str_list(model.GetParams()), ['c/a'])
            self.assertEqual(to_str_list(model.GetComputedParams()), ['c/d'])
            self.assertEqual(to_str_list(model.GetAllParams()), ['c/a', 'c/d'])
            # Get AllParams from the global Scope
            self.assertEqual(to_str_list(model.GetAllParams('')), [
                             'a', 'b', 'c/a', 'c/d'])
        self.assertEqual(to_str_list(model.GetParams()), ['a', 'c/a'])
        self.assertEqual(to_str_list(model.GetComputedParams()), ['b', 'c/d'])
        self.assertEqual(to_str_list(model.GetAllParams()),
                         ['a', 'b', 'c/a', 'c/d'])
        self.assertEqual(to_str_list(model.GetAllParams('')),
                         ['a', 'b', 'c/a', 'c/d'])
        # Get AllParams from the scope 'c'
        self.assertEqual(to_str_list(model.GetAllParams('c')), ['c/a', 'c/d'])
        self.assertEqual(to_str_list(model.GetAllParams('c/')), ['c/a', 'c/d'])

    def test_param_consistence(self):
        model = ModelHelper(name='test_mode')
        cnv = brew.conv(model, 'data', 'cnv', 32, 32, 4)
        step_model = ModelHelper(name='step_model', param_model=model)
        a = brew.fc(step_model, cnv, 'a', 100, 200)
        brew.fc(model, a, 'b', 200, 5)
        # test the _parameters_info is shared between model and step_model
        self.assertEqual(model._parameters_info, step_model._parameters_info)

    def test_cond(self):
        workspace.FeedBlob("cond", np.array(True))
        workspace.FeedBlob("then_value", np.array(1))
        workspace.FeedBlob("else_value", np.array(2))

        then_model = ModelHelper(name="then_test_model")
        then_model.net.Copy("then_value", "output_blob")

        else_model = ModelHelper(name="else_test_model")
        else_model.net.Copy("else_value", "output_blob")

        model = ModelHelper(name="test_model")
        brew.cond(
            model=model,
            cond_blob="cond",
            external_blobs=["then_value", "else_value", "output_blob"],
            then_model=then_model,
            else_model=else_model)

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        output_value = workspace.FetchBlob("output_blob")
        self.assertEqual(output_value, 1)
        workspace.FeedBlob("cond", np.array(False))
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        output_value = workspace.FetchBlob("output_blob")
        self.assertEqual(output_value, 2)

    def test_loop(self):
        workspace.FeedBlob("cond", np.array(True))
        workspace.FeedBlob("ONE", np.array(1))
        workspace.FeedBlob("TWO", np.array(2))
        workspace.FeedBlob("TEN", np.array(10))
        workspace.FeedBlob("counter", np.array(0))
        workspace.FeedBlob("output_blob", np.array(0))

        loop_model = ModelHelper(name="loop_test_model")
        loop_model.net.Add(["output_blob", "TWO"], "output_blob")

        cond_model = ModelHelper(name="cond_test_model")
        cond_model.net.Add(["counter", "ONE"], "counter")
        comp_res = cond_model.net.LT(["counter", "TEN"])
        cond_model.net.Copy(comp_res, "cond")

        model = ModelHelper(name="test_model")
        brew.loop(
            model=model,
            cond_blob="cond",
            external_blobs=["cond", "ONE", "TWO", "TEN", "counter", "output_blob"],
            loop_model=loop_model,
            cond_model=cond_model)

        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)
        output_value = workspace.FetchBlob("output_blob")
        self.assertEqual(output_value, 18)


@unittest.skipIf(not workspace.has_gpu_support, "No gpu support.")
class BrewGPUTest(unittest.TestCase):
    def test_relu(self):
        Xpos = np.ones((5, 5)).astype(np.float32) - 0.5
        Xneg = np.ones((5, 5)).astype(np.float32) - 1.5

        workspace.FeedBlob("xpos", Xpos)
        workspace.FeedBlob("xneg", Xneg)
        model = ModelHelper(name="test_model")
        brew.relu(model, "xpos", "out_xpos", use_cudnn=True)
        brew.relu(model, "xneg", "out_xneg", use_cudnn=True)
        model.Validate()
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        pos = workspace.FetchBlob("out_xpos")
        self.assertAlmostEqual(pos.mean(), 0.5)
        neg = workspace.FetchBlob("out_xneg")
        self.assertAlmostEqual(neg.mean(), 0)

    def test_tanh(self):
        X = np.ones((5, 5)).astype(np.float32) - 0.5

        workspace.FeedBlob("x", X)
        model = ModelHelper(name="test_model")
        brew.tanh(model, "x", "out_tanh", use_cudnn=True)
        model.Validate()
        workspace.RunNetOnce(model.param_init_net)
        workspace.RunNetOnce(model.net)

        out = workspace.FetchBlob("out_tanh")
        self.assertAlmostEqual(out.mean(), np.tanh(0.5), places=5)
