from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from caffe2.proto import caffe2_pb2
import caffe2.python.optimizer as optimizer
from caffe2.python.optimizer import (
    build_sgd, build_multi_precision_sgd, build_ftrl, build_gftrl, build_wngrad,
    build_adagrad, build_adadelta, build_adam, build_yellowfin, build_rms_prop,
    add_weight_decay, SgdOptimizer)
from caffe2.python.optimizer_context import UseOptimizer
from caffe2.python.optimizer_test_util import (
    OptimizerTestBase, LRModificationTestBase
)
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import math
import unittest


class TestLars(OptimizerTestBase, TestCase):
    def testSparse(self):
        raise unittest.SkipTest("no sparse support")

    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = False
        return build_sgd(model, base_learning_rate=0.1, lars=0.5, **kwargs)

    def check_optimizer(self, optimizer):
        self.assertTrue(optimizer.get_auxiliary_parameters().shared)
        self.assertFalse(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().shared:
            tensor = workspace.FetchBlob(param)
            np.testing.assert_allclose(np.array([1.0]), tensor, atol=1e-5)


class TestMomentumSgd(OptimizerTestBase, TestCase):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = False
        return build_sgd(model, base_learning_rate=0.1, momentum=0.1, **kwargs)

    def check_optimizer(self, optimizer):
        self.assertTrue(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().shared:
            tensor = workspace.FetchBlob(param)
            np.testing.assert_allclose(np.array([1.0]), tensor, atol=1e-5)


class TestSgd(OptimizerTestBase, LRModificationTestBase, TestCase):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = False
        return build_sgd(model, base_learning_rate=0.1, **kwargs)

    def check_optimizer(self, optimizer):
        self.assertTrue(optimizer.get_auxiliary_parameters().shared)
        self.assertFalse(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().shared:
            tensor = workspace.FetchBlob(param)
            np.testing.assert_allclose(np.array([1.0]), tensor, atol=1e-5)


class TestMultiPrecisionSgd(
    OptimizerTestBase, LRModificationTestBase, TestCase
):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = False
        return build_multi_precision_sgd(
            model, base_learning_rate=0.1, **kwargs
        )

    def check_optimizer(self, optimizer):
        self.assertTrue(optimizer.get_auxiliary_parameters().shared)
        self.assertFalse(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().shared:
            tensor = workspace.FetchBlob(param)
            np.testing.assert_allclose(np.array([1.0]), tensor, atol=1e-5)

    @unittest.skipIf(not workspace.has_gpu_support, "No GPU support")
    def testGPUDense(self):
        super(TestMultiPrecisionSgd, self).testGPUDense(core.DataType.FLOAT16)


class TestFtrl(OptimizerTestBase, TestCase):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = True
        return build_ftrl(
            model,
            engine=None,
            alpha=1.0,
            beta=0.1,
            lambda1=0.0,
            lambda2=0.0,
            **kwargs
        )

    def check_optimizer(self, optimizer):
        self.assertFalse(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)


class TestGFtrl(OptimizerTestBase, TestCase):
    def testSparse(self):
        raise unittest.SkipTest("no sparse support")

    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = True
        return build_gftrl(
            model,
            engine=None,
            alpha=1.0,
            beta=0.1,
            lambda1=0.0,
            lambda2=0.0,
            **kwargs
        )

    def check_optimizer(self, optimizer):
        self.assertFalse(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)


class TestAdagrad(OptimizerTestBase, LRModificationTestBase, TestCase):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = False
        return build_adagrad(model, base_learning_rate=1.0, lars=0.5, **kwargs)

    def check_optimizer(self, optimizer):
        self.assertFalse(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)


class TestRowWiseAdagrad(OptimizerTestBase, TestCase):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = True
        return build_adagrad(
            model, base_learning_rate=1.0, lars=0.5, rowWise=True, **kwargs
        )

    def check_optimizer(self, optimizer):
        self.assertFalse(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)

    def testDense(self):
        raise unittest.SkipTest("no dense support")

    def testGPUDense(self):
        raise unittest.SkipTest("no dense support")


class TestWngrad(OptimizerTestBase, LRModificationTestBase, TestCase):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = True
        return build_wngrad(model, base_learning_rate=25.0, **kwargs)

    def check_optimizer(self, optimizer):
        self.assertFalse(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)


class TestAdadelta(OptimizerTestBase, LRModificationTestBase, TestCase):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = False
        return build_adadelta(model, base_learning_rate=1.0, decay=0.995, **kwargs)

    def check_optimizer(self, optimizer):
        self.assertFalse(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)


class TestAdam(OptimizerTestBase, LRModificationTestBase, TestCase):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = False
        return build_adam(model, base_learning_rate=0.1, **kwargs)

    def check_optimizer(self, optimizer):
        self.assertTrue(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        self.assertTrue(workspace.HasBlob("optimizer_iteration"))
        iteration_tensor = workspace.FetchBlob("optimizer_iteration")
        np.testing.assert_allclose(np.array([2000]),
                                   iteration_tensor,
                                   atol=1e-5)
        for param in optimizer.get_auxiliary_parameters().shared:
            workspace.FetchBlob(param)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)


class TestSparseRAdam(OptimizerTestBase, LRModificationTestBase, TestCase):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = True
        return build_adam(model, base_learning_rate=0.1, enableRAdam=True, **kwargs)

    def check_optimizer(self, optimizer):
        self.assertTrue(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        self.assertTrue(workspace.HasBlob("optimizer_iteration"))
        iteration_tensor = workspace.FetchBlob("optimizer_iteration")
        np.testing.assert_allclose(np.array([2000]),
                                   iteration_tensor,
                                   atol=1e-5)
        for param in optimizer.get_auxiliary_parameters().shared:
            workspace.FetchBlob(param)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)


class TestYellowFin(OptimizerTestBase, TestCase):
    # YellowFin: An automatic tuner for momentum SGD
    # (https://arxiv.org/abs/1706.03471)
    def build_optimizer(self, model):
        self._skip_gpu = False
        return build_yellowfin(model, base_learning_rate=0.1)

    def check_optimizer(self, optimizer):
        self.assertTrue(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        self.assertTrue(workspace.HasBlob("optimizer_iteration"))
        iteration_tensor = workspace.FetchBlob("optimizer_iteration")
        np.testing.assert_allclose(np.array([2000]),
                                   iteration_tensor,
                                   atol=1e-5)
        for param in optimizer.get_auxiliary_parameters().shared:
            workspace.FetchBlob(param)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)

    def testSparse(self):
        raise unittest.SkipTest("no sparse support")

    def deb(self, val, beta, i, zero_debias):
        if zero_debias:
            return val / (1.0 - beta ** i)
        else:
            return val

    def get_lr_mu(self, distance, grad_var, h_min, h_max):
        # First tune based on dynamic range
        if grad_var == 0:
            dr = h_max / h_min
            mu = ((np.sqrt(dr) - 1) / (np.sqrt(dr) + 1)) ** 2
            lr_min = (1 + np.sqrt(mu)) ** 2 / h_max
            return lr_min, mu

        p = distance ** 2 * h_min ** 2 / 2 / grad_var
        w3 = (-math.sqrt(p * p + 4.0 / 27.0 * p * p * p) - p) / 2.0
        w = (1.0 if w3 > 0.0 else -1.0) * math.pow(math.fabs(w3), 1.0 / 3.0)
        y = w - p / 3.0 / w
        root = y + 1
        root = min(root, 1.0 - 1e-6)
        dr = h_max / h_min
        mu = max(((np.sqrt(dr) - 1) / (np.sqrt(dr) + 1)) ** 2, root**2)
        lr_min = (1 - np.sqrt(mu)) ** 2 / h_min
        return lr_min, mu

    def caffe2_yellowfin(self, zero_debias, grad_coef, n_dim, n_iter, gpu):
        caffe2_res = {}

        alpha = 1.0
        mu = 0.0
        beta = 0.999
        curv_win_width = 20
        epsilon = 1e-6

        net = core.Net("net")
        param_init_net = core.Net("param_init_net")
        workspace.ResetWorkspace()

        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
            iteration = param_init_net.ConstantFill(
                [],
                "iteration",
                shape=[1],
                value=0,
                dtype=core.DataType.INT64)
            iter_mutex = param_init_net.CreateMutex([], ["iteration_mutex"])
            net.AtomicIter([iter_mutex, iteration], [iteration])
        pre_grad = param_init_net.ConstantFill(
            [],
            "pre_grad",
            shape=[n_dim],
            value=grad_coef
        )
        if gpu:
            iteration = net.CopyCPUToGPU(
                [iteration],
                "iteration_cpu"
            )
        iteration_float = net.Cast([iteration], "iteration_float")
        grad = net.Mul([pre_grad, iteration_float], "grad", broadcast=True)
        w = param_init_net.ConstantFill([], "w", shape=[n_dim], value=0.0)

        # a hack to create an object with __dict__
        param_info = lambda: None
        param_info.blob = w
        param_info.grad = grad

        optimizer.YellowFinOptimizer(
            alpha=alpha,
            mu=mu,
            beta=beta,
            curv_win_width=curv_win_width,
            epsilon=epsilon,
            zero_debias=zero_debias
        )._run(
            net,
            param_init_net,
            param_info
        )

        workspace.RunNetOnce(param_init_net)
        workspace.CreateNet(net, overwrite=True)
        for i in range(n_iter):
            workspace.RunNet(net)
            scalars_memory_blob = workspace.FetchBlob("w_scalars_memory")
            g_norm2_avg = scalars_memory_blob[1]
            g_norm2_min_avg = scalars_memory_blob[2]
            g_norm2_max_avg = scalars_memory_blob[3]
            distance_avg = scalars_memory_blob[4]
            g_avg_blob = workspace.FetchBlob("w_g_avg")
            res_lr = workspace.FetchBlob("w_lr_avg")[0]
            res_mu = workspace.FetchBlob("w_mu_avg")[0]
            g_deb = self.deb(g_avg_blob, beta, i + 1, zero_debias)
            variance = max(
                self.deb(g_norm2_avg, beta, i + 1, zero_debias) -
                g_deb.dot(g_deb),
                epsilon
            )
            if i > 0:
                caffe2_res[i] = {
                    'h_max': np.exp(self.deb(g_norm2_max_avg,
                                             beta,
                                             i + 1,
                                             zero_debias)),
                    'h_min': np.exp(self.deb(g_norm2_min_avg,
                                             beta,
                                             i + 1,
                                             zero_debias)),
                    'var': variance,
                    'dist': self.deb(distance_avg, beta, i + 1, zero_debias),
                    'lr': res_lr,
                    'mu': res_mu
                }
        return caffe2_res

    def numpy_yellowfin(self, zero_debias, grad_coef, n_dim, n_iter, gpu):
        numpy_res = {}

        target_h_max = 0.0
        target_h_min = 0.0
        target_g_norm_squared_avg = 0.0
        target_g_norm_avg = 0.0
        target_g_avg = 0.0
        target_dist_avg = 0.0
        target_lr = 1.0
        target_mu = 0.0

        for i in range(n_iter):
            grad_val = (i + 1) * grad_coef
            target_g_norm_squared_avg = 0.999 * target_g_norm_squared_avg + \
                0.001 * np.sum((grad_val * np.ones([n_dim, ])) ** 2)
            target_g_norm_avg = 0.999 * target_g_norm_avg + \
                0.001 * np.linalg.norm(grad_val * np.ones([n_dim, ]))
            target_g_avg = 0.999 * target_g_avg + 0.001 * grad_val

            target_h_max = 0.999 * target_h_max + \
                0.001 * np.log(grad_val ** 2 * n_dim)
            target_h_min = 0.999 * target_h_min + \
                0.001 * np.log((max(1, i + 2 - 20) * grad_coef) ** 2 * n_dim)
            if zero_debias:
                target_var = target_g_norm_squared_avg / \
                    (1 - 0.999 ** (i + 1)) - \
                    target_g_avg ** 2 * n_dim / (1 - 0.999 ** (i + 1)) ** 2
            else:
                target_var = target_g_norm_squared_avg - \
                    target_g_avg ** 2 * n_dim
            target_dist_avg = 0.999 * target_dist_avg + \
                0.001 * target_g_norm_avg / target_g_norm_squared_avg

            if i > 0:
                if zero_debias:
                    lr, mu = self.get_lr_mu(
                        target_dist_avg / (1.0 - 0.999 ** (i + 1)),
                        target_var,
                        np.exp(target_h_min / (1.0 - 0.999 ** (i + 1))),
                        np.exp(target_h_max / (1.0 - 0.999 ** (i + 1))))
                    target_lr = 0.999 * target_lr + 0.001 * lr
                    target_mu = 0.999 * target_mu + 0.001 * mu
                    numpy_res[i] = {
                        'h_max': np.exp(target_h_max / (1 - 0.999 ** (i + 1))),
                        'h_min': np.exp(target_h_min / (1 - 0.999 ** (i + 1))),
                        'var': target_var,
                        'dist': target_dist_avg / (1 - 0.999 ** (i + 1)),
                        'lr': target_lr,
                        'mu': target_mu
                    }
                else:
                    lr, mu = self.get_lr_mu(
                        target_dist_avg,
                        target_var,
                        np.exp(target_h_min),
                        np.exp(target_h_max))
                    target_lr = 0.999 * target_lr + 0.001 * lr
                    target_mu = 0.999 * target_mu + 0.001 * mu
                    numpy_res[i] = {
                        'h_max': np.exp(target_h_max),
                        'h_min': np.exp(target_h_min),
                        'var': target_var,
                        'dist': target_dist_avg,
                        'lr': target_lr,
                        'mu': target_mu
                    }
        return numpy_res

    def compare_yellowfin_models(self,
                                 model0,
                                 model1,
                                 zero_debias,
                                 grad_coef,
                                 n_dim,
                                 n_iter,
                                 gpu):
        model0_res = model0(zero_debias, grad_coef, n_dim, n_iter, gpu)
        model1_res = model1(zero_debias, grad_coef, n_dim, n_iter, gpu)
        assert_equal(len(model0_res), len(model1_res))
        for i in range(1, len(model0_res)):
            assert_equal(model0_res[i].keys(), model1_res[i].keys())
            for feat in model0_res[i].keys():
                err_msg = \
                    'i=' + str(i) + ',\n' + \
                    'feat=' + feat + ',\n' + \
                    'grad_coef=' + str(grad_coef) + ',\n' + \
                    'zero_debias=' + str(zero_debias)
                assert_allclose(model0_res[i][feat],
                                model1_res[i][feat],
                                rtol=1e-2,
                                err_msg=err_msg)

    @unittest.skip("Results might vary too much. Only for individual use.")
    def test_caffe2_cpu_vs_numpy(self):
        n_dim = 1000000
        n_iter = 50
        cpu_device_opt = core.DeviceOption(caffe2_pb2.CPU)
        with core.DeviceScope(cpu_device_opt):
            for zero_debias, grad_coef in [
                (False, 1.0),
                (False, 0.1),
                (False, 0.01),
                (True, 1.0)
            ]:
                self.compare_yellowfin_models(
                    self.caffe2_yellowfin,
                    self.numpy_yellowfin,
                    zero_debias,
                    grad_coef,
                    n_dim,
                    n_iter,
                    gpu=False
                )

    @unittest.skip("Results might vary too much. Only for individual use.")
    @unittest.skipIf(not workspace.has_gpu_support, "No gpu support")
    def test_caffe2_gpu_vs_numpy(self):
        n_dim = 1000000
        n_iter = 50
        gpu_device_opt = core.DeviceOption(workspace.GpuDeviceType, 0)
        with core.DeviceScope(gpu_device_opt):
            for zero_debias in [False, True]:
                for grad_coef in [1.0, 0.1, 0.01]:
                    self.compare_yellowfin_models(
                        self.caffe2_yellowfin,
                        self.numpy_yellowfin,
                        zero_debias,
                        grad_coef,
                        n_dim,
                        n_iter,
                        gpu=True
                    )


class TestRmsProp(OptimizerTestBase, LRModificationTestBase, TestCase):
    def build_optimizer(self, model, **kwargs):
        self._skip_gpu = False
        return build_rms_prop(
            model, base_learning_rate=0.1, epsilon=0.1, **kwargs
        )

    def check_optimizer(self, optimizer):
        self.assertFalse(optimizer.get_auxiliary_parameters().shared)
        self.assertTrue(optimizer.get_auxiliary_parameters().local)
        for param in optimizer.get_auxiliary_parameters().local:
            workspace.FetchBlob(param)

    def testSparse(self):
        raise unittest.SkipTest("no sparse support")


class TestMultiOptimizers(TestCase):
    def test_multiple_optimizers(self):
        from caffe2.python import brew, core, optimizer
        from caffe2.python.model_helper import ModelHelper

        model = ModelHelper(name="test")
        fc1 = brew.fc(model, 'data', 'fc1', 100, 50)
        fc2 = brew.fc(model, fc1, 'fc2', 50, 25)
        pred = brew.fc(model, fc2, 'fc3', 25, 10)
        (softmax, loss) = model.SoftmaxWithLoss(
            [pred, 'label'],
            ['softmax', 'loss'],
        )
        model.AddGradientOperators([loss])

        param_to_device = optimizer._get_param_to_device(model)

        def infer_blob_device(blob_name):
            return optimizer.get_param_device(
                blob_name, "{}_grad".format(blob_name), param_to_device
            )

        sgd_1 = optimizer.SgdOptimizer(base_learning_rate=0.1)
        sgd_2 = optimizer.SgdOptimizer(base_learning_rate=0.2)
        adagrad = optimizer.AdagradOptimizer()

        # Check same optimizer share the same learning rate.
        with core.DeviceScope(infer_blob_device("fc1_w")):
            sgd_1(model.net, model.param_init_net, "fc1_w", "fc1_w_grad")
        with core.DeviceScope(infer_blob_device("fc1_b")):
            sgd_1(model.net, model.param_init_net, "fc1_b", "fc1_b_grad")
        fc1_lr_blobs = []
        for op in model.net.Proto().op:
            if op.type == 'WeightedSum' and op.input[0] == 'fc1_w' or \
                    op.input[0] == 'fc1_b':
                fc1_lr_blobs.append(op.input[3])
        self.assertEqual(fc1_lr_blobs[0], fc1_lr_blobs[1])

        # Check different instance of the same optimizer has a different lr.
        with core.DeviceScope(infer_blob_device("fc2_w")):
            sgd_2(model.net, model.param_init_net, "fc2_w", "fc2_w_grad")
        with core.DeviceScope(infer_blob_device("fc2_b")):
            sgd_2(model.net, model.param_init_net, "fc2_b", "fc2_b_grad")
        fc2_lr_blobs = []
        for op in model.net.Proto().op:
            if op.type == 'WeightedSum' and op.input[0] == 'fc2_w' or \
                    op.input[0] == 'fc2_b':
                self.assertTrue(op.input[3] not in fc1_lr_blobs)
                fc2_lr_blobs.append(op.input[3])
        self.assertEqual(fc2_lr_blobs[0], fc2_lr_blobs[1])

        # Check different optimizer type case
        with core.DeviceScope(infer_blob_device("fc3_w")):
            adagrad(model.net, model.param_init_net, "fc3_w", "fc3_w_grad")
        with core.DeviceScope(infer_blob_device("fc3_b")):
            adagrad(model.net, model.param_init_net, "fc3_b", "fc3_b_grad")
        fc3_lr_blobs = []
        for op in model.net.Proto().op:
            if op.type == 'Adagrad' and op.input[0] == 'fc3_w' or \
                    op.input[0] == 'fc3_b':
                self.assertTrue(op.input[3] not in fc2_lr_blobs)
                self.assertTrue(op.input[3] not in fc1_lr_blobs)
                fc3_lr_blobs.append(op.input[3])
        self.assertEqual(fc3_lr_blobs[0], fc3_lr_blobs[1])


class TestWeightDecay(TestCase):

    def test_weight_decay(self):
        from caffe2.python import brew
        from caffe2.python.model_helper import ModelHelper

        model = ModelHelper(name="test", arg_scope={'order': 'NCHW'})
        cnv = brew.conv(model, 'data', 'cnv', 32, 32, 4)
        a = brew.fc(model, cnv, 'a', 100, 200)
        pred = brew.fc(model, a, 'b', 200, 5)
        (softmax, loss) = model.SoftmaxWithLoss(
            [pred, 'label'],
            ['softmax', 'loss'],
        )
        model.AddGradientOperators([loss])

        add_weight_decay(model, weight_decay=1e-4)
        build_sgd(model, 0.11)

        expected_weight_grad = {'b_w_grad', 'a_w_grad', 'cnv_w_grad'}

        # Check the proto that all weights are decayed and not non-weights
        # are decayed.
        for op in model.net.Proto().op:
            if op.type == 'WeightedSum' and 'wd_0_0' in op.input:
                if op.output[0] not in expected_weight_grad:
                    print(
                        "Unexpected param for weight_decay: {}".
                        format(op.output[0])
                    )
                self.assertTrue(op.output[0] in expected_weight_grad)
                expected_weight_grad.remove(op.output[0])

        self.assertEqual(
            expected_weight_grad,
            set(),
            "Not all weights were decayed: {}".format(expected_weight_grad)
        )


class TestOptimizerContext(TestCase):

    def test_optimizer_context(self):
        from caffe2.python import brew, optimizer
        from caffe2.python.model_helper import ModelHelper

        model = ModelHelper(name="test", arg_scope={'order': 'NCHW'})
        count = optimizer._optimizer_instance_count['SgdOptimizer']
        cnv_optim = SgdOptimizer(0.15)
        weight_optim = SgdOptimizer(0.2)
        bias_optim = SgdOptimizer(0.1)

        with UseOptimizer(cnv_optim):
            cnv = brew.conv(model, 'data', 'cnv', 32, 32, 4)
        with UseOptimizer({'WEIGHT': weight_optim, 'BIAS': bias_optim}):
            a = brew.fc(model, cnv, 'a', 100, 200)
        pred = brew.fc(model, a, 'b', 200, 5)
        (softmax, loss) = model.SoftmaxWithLoss(
            [pred, 'label'],
            ['softmax', 'loss'],
        )
        model.AddGradientOperators([loss])

        add_weight_decay(model, weight_decay=1e-4)
        # use the following optimizer if none specified in param_info
        build_sgd(model, 0.11)
        expected_weight_grad = {'b_w_grad', 'a_w_grad', 'cnv_w_grad'}
        expected_learning_rate = {
            "SgdOptimizer_{}_lr_cpu".format(count): -0.15,
            "SgdOptimizer_{}_lr_cpu".format(count + 1): -0.2,
            "SgdOptimizer_{}_lr_cpu".format(count + 2): -0.1,
            "SgdOptimizer_{}_lr_cpu".format(count + 3): -0.11
        }

        for op in model.net.Proto().op:
            # Check the proto that all weights are decayed and not non-weights
            # are decayed.
            if op.type == 'WeightedSum' and 'wd_0_0' in op.input:
                if op.output[0] not in expected_weight_grad:
                    print(
                        "Unexpected param for weight_decay: {}".
                        format(op.output[0])
                    )
                self.assertTrue(op.output[0] in expected_weight_grad)
                expected_weight_grad.remove(op.output[0])
            # Check the learning rate for each parameter
            if op.type == 'LearningRate':
                val = 0
                for arg in op.arg:
                    if arg.name == 'base_lr':
                        val = arg.f
                self.assertAlmostEqual(
                    val,
                    expected_learning_rate[op.output[0]]
                )

        self.assertEqual(
            expected_weight_grad,
            set(),
            "Not all weights were decayed: {}".format(expected_weight_grad)
        )
