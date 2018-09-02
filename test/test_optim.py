import math
import unittest
import functools
from copy import deepcopy
import torch
from torch._six import inf
import torch.optim as optim
import torch.legacy.optim as old_optim
import torch.nn.functional as F
from torch.optim import SGD
from torch.autograd import Variable
from torch import sparse
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import CyclicLR
from common import TestCase, run_tests, TEST_WITH_UBSAN, skipIfRocm


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(tensor):
    x, y = tensor
    return torch.DoubleTensor((-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2)))


def wrap_old_fn(old_fn, **config):
    def wrapper(closure, params, state):
        return old_fn(closure, params, config, state)
    return wrapper


class TestOptim(TestCase):
    def _test_rosenbrock(self, constructor, old_fn):
        params_t = torch.Tensor([1.5, 1.5])
        state = {}

        params = Variable(torch.Tensor([1.5, 1.5]), requires_grad=True)
        optimizer = constructor([params])

        solution = torch.Tensor([1, 1])
        initial_dist = params.data.dist(solution)

        def eval():
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            # loss.backward() will give **slightly** different
            # gradients, than drosenbtock, because of a different ordering
            # of floating point operations. In most cases it doesn't matter,
            # but some optimizers are so sensitive that they can temporarily
            # diverge up to 1e-4, just to converge again. This makes the
            # comparison more stable.
            params.grad.data.copy_(drosenbrock(params.data))
            return loss

        for i in range(2000):
            optimizer.step(eval)
            old_fn(lambda _: (rosenbrock(params_t), drosenbrock(params_t)),
                   params_t, state)
            self.assertEqual(params.data, params_t)

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_rosenbrock_sparse(self, constructor, sparse_only=False):
        params_t = torch.Tensor([1.5, 1.5])

        params = Variable(params_t, requires_grad=True)
        optimizer = constructor([params])

        if not sparse_only:
            params_c = Variable(params_t.clone(), requires_grad=True)
            optimizer_c = constructor([params_c])

        solution = torch.Tensor([1, 1])
        initial_dist = params.data.dist(solution)

        def eval(params, sparse_grad, w):
            # Depending on w, provide only the x or y gradient
            optimizer.zero_grad()
            loss = rosenbrock(params)
            loss.backward()
            grad = drosenbrock(params.data)
            # NB: We torture test the optimizer by returning an
            # uncoalesced sparse tensor
            if w:
                i = torch.LongTensor([[0, 0]])
                x = grad[0]
                v = torch.DoubleTensor([x / 4., x - x / 4.])
            else:
                i = torch.LongTensor([[1, 1]])
                y = grad[1]
                v = torch.DoubleTensor([y - y / 4., y / 4.])
            x = sparse.DoubleTensor(i, v, torch.Size([2]))
            if sparse_grad:
                params.grad.data = x
            else:
                params.grad.data = x.to_dense()
            return loss

        for i in range(2000):
            # Do cyclic coordinate descent
            w = i % 2
            optimizer.step(functools.partial(eval, params, True, w))
            if not sparse_only:
                optimizer_c.step(functools.partial(eval, params_c, False, w))
                self.assertEqual(params.data, params_c.data)

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_basic_cases_template(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)
        optimizer = constructor(weight, bias)

        # to check if the optimizer can be printed as a string
        optimizer.__repr__()

        def fn():
            optimizer.zero_grad()
            y = weight.mv(input)
            if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
                y = y.cuda(bias.get_device())
            loss = (y + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = fn().item()
        for i in range(200):
            optimizer.step(fn)
        self.assertLess(fn().item(), initial_value)

    def _test_state_dict(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)

        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            i = input_cuda if weight.is_cuda else input
            loss = (weight.mv(i) + bias).pow(2).sum()
            loss.backward()
            return loss

        optimizer = constructor(weight, bias)
        fn = functools.partial(fn_base, optimizer, weight, bias)

        # Prime the optimizer
        for i in range(20):
            optimizer.step(fn)
        # Clone the weights and construct new optimizer for them
        weight_c = Variable(weight.data.clone(), requires_grad=True)
        bias_c = Variable(bias.data.clone(), requires_grad=True)
        optimizer_c = constructor(weight_c, bias_c)
        fn_c = functools.partial(fn_base, optimizer_c, weight_c, bias_c)
        # Load state dict
        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_c.load_state_dict(state_dict_c)
        # Run both optimizations in parallel
        for i in range(20):
            optimizer.step(fn)
            optimizer_c.step(fn_c)
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)
        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        # Check that state dict can be loaded even when we cast parameters
        # to a different type and move to a different device.
        if not torch.cuda.is_available():
            return

        input_cuda = Variable(input.data.float().cuda())
        weight_cuda = Variable(weight.data.float().cuda(), requires_grad=True)
        bias_cuda = Variable(bias.data.float().cuda(), requires_grad=True)
        optimizer_cuda = constructor(weight_cuda, bias_cuda)
        fn_cuda = functools.partial(fn_base, optimizer_cuda, weight_cuda, bias_cuda)

        state_dict = deepcopy(optimizer.state_dict())
        state_dict_c = deepcopy(optimizer.state_dict())
        optimizer_cuda.load_state_dict(state_dict_c)

        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)

        for i in range(20):
            optimizer.step(fn)
            optimizer_cuda.step(fn_cuda)
            self.assertEqual(weight, weight_cuda)
            self.assertEqual(bias, bias_cuda)

    def _test_basic_cases(self, constructor, ignore_multidevice=False):
        self._test_state_dict(
            torch.randn(10, 5),
            torch.randn(10),
            torch.randn(5),
            constructor
        )
        self._test_basic_cases_template(
            torch.randn(10, 5),
            torch.randn(10),
            torch.randn(5),
            constructor
        )
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2)[..., 0],
            torch.randn(10, 2)[..., 0],
            torch.randn(5),
            constructor
        )
        # CUDA
        if not torch.cuda.is_available():
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(),
            torch.randn(10).cuda(),
            torch.randn(5).cuda(),
            constructor
        )
        # Multi-GPU
        if not torch.cuda.device_count() > 1 or ignore_multidevice:
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(0),
            torch.randn(10).cuda(1),
            torch.randn(5).cuda(0),
            constructor
        )

    def _build_params_dict(self, weight, bias, **kwargs):
        return [dict(params=[weight]), dict(params=[bias], **kwargs)]

    def _build_params_dict_single(self, weight, bias, **kwargs):
        return [dict(params=bias, **kwargs)]

    def test_sgd(self):
        self._test_rosenbrock(
            lambda params: optim.SGD(params, lr=1e-3),
            wrap_old_fn(old_optim.sgd, learningRate=1e-3)
        )
        self._test_rosenbrock(
            lambda params: optim.SGD(params, lr=1e-3, momentum=0.9,
                                     dampening=0, weight_decay=1e-4),
            wrap_old_fn(old_optim.sgd, learningRate=1e-3, momentum=0.9,
                        dampening=0, weightDecay=1e-4)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.SGD([weight, bias], lr=1e-3)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.SGD(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.SGD(
                self._build_params_dict_single(weight, bias, lr=1e-2),
                lr=1e-3)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.SGD(
                self._build_params_dict_single(weight, bias, lr=1e-2))
        )
        with self.assertRaisesRegex(ValueError, "Invalid momentum value: -0.5"):
            optim.SGD(None, lr=1e-2, momentum=-0.5)

    def test_sgd_sparse(self):
        self._test_rosenbrock_sparse(
            lambda params: optim.SGD(params, lr=5e-3)
        )

    def test_adam(self):
        self._test_rosenbrock(
            lambda params: optim.Adam(params, lr=1e-2),
            wrap_old_fn(old_optim.adam, learningRate=1e-2)
        )
        self._test_rosenbrock(
            lambda params: optim.Adam(params, lr=1e-2, weight_decay=1e-2),
            wrap_old_fn(old_optim.adam, learningRate=1e-2, weightDecay=1e-2)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adam([weight, bias], lr=1e-3)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adam([weight, bias], lr=1e-3,
                                            amsgrad=True)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adam(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3, amsgrad=True)
        )
        with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 0: 1.0"):
            optim.Adam(None, lr=1e-2, betas=(1.0, 0.0))

    def test_sparse_adam(self):
        self._test_rosenbrock_sparse(
            lambda params: optim.SparseAdam(params, lr=4e-2),
            True
        )
        with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 0: 1.0"):
            optim.SparseAdam(None, lr=1e-2, betas=(1.0, 0.0))

    def test_adadelta(self):
        self._test_rosenbrock(
            lambda params: optim.Adadelta(params),
            wrap_old_fn(old_optim.adadelta)
        )
        self._test_rosenbrock(
            lambda params: optim.Adadelta(params, rho=0.95),
            wrap_old_fn(old_optim.adadelta, rho=0.95)
        )
        self._test_rosenbrock(
            lambda params: optim.Adadelta(params, weight_decay=1e-2),
            wrap_old_fn(old_optim.adadelta, weightDecay=1e-2)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adadelta([weight, bias])
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adadelta(
                self._build_params_dict(weight, bias, rho=0.95))
        )
        with self.assertRaisesRegex(ValueError, "Invalid rho value: 1.1"):
            optim.Adadelta(None, lr=1e-2, rho=1.1)

    def test_adagrad(self):
        self._test_rosenbrock(
            lambda params: optim.Adagrad(params, lr=1e-1),
            wrap_old_fn(old_optim.adagrad, learningRate=1e-1)
        )
        self._test_rosenbrock(
            lambda params: optim.Adagrad(params, lr=1e-1, lr_decay=1e-3),
            wrap_old_fn(old_optim.adagrad, learningRate=1e-1, learningRateDecay=1e-3)
        )
        self._test_rosenbrock(
            lambda params: optim.Adagrad(params, lr=1e-1, weight_decay=1e-2),
            wrap_old_fn(old_optim.adagrad, learningRate=1e-1, weightDecay=1e-2)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adagrad([weight, bias], lr=1e-1)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adagrad([weight, bias], lr=1e-1,
                                               initial_accumulator_value=0.1)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adagrad(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-1)
        )
        with self.assertRaisesRegex(ValueError, "Invalid lr_decay value: -0.5"):
            optim.Adagrad(None, lr=1e-2, lr_decay=-0.5)

    def test_adagrad_sparse(self):
        self._test_rosenbrock_sparse(
            lambda params: optim.Adagrad(params, lr=1e-1)
        )

    @skipIfRocm
    def test_adamax(self):
        self._test_rosenbrock(
            lambda params: optim.Adamax(params, lr=1e-1),
            wrap_old_fn(old_optim.adamax, learningRate=1e-1)
        )
        self._test_rosenbrock(
            lambda params: optim.Adamax(params, lr=1e-1, weight_decay=1e-2),
            wrap_old_fn(old_optim.adamax, learningRate=1e-1, weightDecay=1e-2)
        )
        self._test_rosenbrock(
            lambda params: optim.Adamax(params, lr=1e-1, betas=(0.95, 0.998)),
            wrap_old_fn(old_optim.adamax, learningRate=1e-1, beta1=0.95, beta2=0.998)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adamax([weight, bias], lr=1e-1)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adamax(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-1)
        )
        with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 1: 1.0"):
            optim.Adamax(None, lr=1e-2, betas=(0.0, 1.0))

    def test_rmsprop(self):
        self._test_rosenbrock(
            lambda params: optim.RMSprop(params, lr=1e-2),
            wrap_old_fn(old_optim.rmsprop, learningRate=1e-2)
        )
        self._test_rosenbrock(
            lambda params: optim.RMSprop(params, lr=1e-2, weight_decay=1e-2),
            wrap_old_fn(old_optim.rmsprop, learningRate=1e-2, weightDecay=1e-2)
        )
        self._test_rosenbrock(
            lambda params: optim.RMSprop(params, lr=1e-2, alpha=0.95),
            wrap_old_fn(old_optim.rmsprop, learningRate=1e-2, alpha=0.95)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.RMSprop([weight, bias], lr=1e-2)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.RMSprop(
                self._build_params_dict(weight, bias, lr=1e-3),
                lr=1e-2)
        )
        with self.assertRaisesRegex(ValueError, "Invalid momentum value: -1.0"):
            optim.RMSprop(None, lr=1e-2, momentum=-1.0)

    def test_asgd(self):
        self._test_rosenbrock(
            lambda params: optim.ASGD(params, lr=1e-3),
            wrap_old_fn(old_optim.asgd, eta0=1e-3)
        )
        self._test_rosenbrock(
            lambda params: optim.ASGD(params, lr=1e-3, alpha=0.8),
            wrap_old_fn(old_optim.asgd, eta0=1e-3, alpha=0.8)
        )
        self._test_rosenbrock(
            lambda params: optim.ASGD(params, lr=1e-3, t0=1e3),
            wrap_old_fn(old_optim.asgd, eta0=1e-3, t0=1e3)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.ASGD([weight, bias], lr=1e-3, t0=100)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.ASGD(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3, t0=100)
        )
        with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -0.5"):
            optim.ASGD(None, lr=1e-2, weight_decay=-0.5)

    @skipIfRocm
    def test_rprop(self):
        self._test_rosenbrock(
            lambda params: optim.Rprop(params, lr=1e-3),
            wrap_old_fn(old_optim.rprop, stepsize=1e-3)
        )
        self._test_rosenbrock(
            lambda params: optim.Rprop(params, lr=1e-3, etas=(0.6, 1.1)),
            wrap_old_fn(old_optim.rprop, stepsize=1e-3, etaminus=0.6, etaplus=1.1)
        )
        self._test_rosenbrock(
            lambda params: optim.Rprop(params, lr=1e-3, step_sizes=(1e-4, 3)),
            wrap_old_fn(old_optim.rprop, stepsize=1e-3, stepsizemin=1e-4, stepsizemax=3)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Rprop([weight, bias], lr=1e-3)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Rprop(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-3)
        )
        with self.assertRaisesRegex(ValueError, "Invalid eta values: 1.0, 0.5"):
            optim.Rprop(None, lr=1e-2, etas=(1.0, 0.5))

    def test_lbfgs(self):
        self._test_rosenbrock(
            lambda params: optim.LBFGS(params),
            wrap_old_fn(old_optim.lbfgs)
        )
        self._test_rosenbrock(
            lambda params: optim.LBFGS(params, lr=5e-2, max_iter=5),
            wrap_old_fn(old_optim.lbfgs, learningRate=5e-2, maxIter=5)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.LBFGS([weight, bias]),
            ignore_multidevice=True
        )

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_lbfgs_return_type(self):
        params = [torch.randn(10, 5), torch.randn(10)]
        opt1 = optim.LBFGS(params, 0.01, tolerance_grad=inf)
        opt2 = optim.LBFGS(params, 0.01, tolerance_grad=-inf)

        def closure():
            return torch.Tensor([10])

        res1 = opt1.step(closure)
        res2 = opt2.step(closure)
        self.assertEqual(type(res1), type(res2))

    def test_invalid_param_type(self):
        with self.assertRaises(TypeError):
            optim.SGD(Variable(torch.randn(5, 5)), lr=3)


class SchedulerTestNet(torch.nn.Module):
    def __init__(self):
        super(SchedulerTestNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class LambdaLRTestObject:
    def __init__(self, value):
        self.value = value

    def __call__(self, epoch):
        return self.value * epoch

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False


class TestLRScheduler(TestCase):
    def setUp(self):
        self.net = SchedulerTestNet()
        self.opt = SGD(
            [{'params': self.net.conv1.parameters()},
             {'params': self.net.conv2.parameters(), 'lr': 0.5}],
            lr=0.05)

    def test_step_lr(self):
        # lr = 0.05     if epoch < 3
        # lr = 0.005    if 30 <= epoch < 6
        # lr = 0.0005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3
        targets = [single_targets, list(map(lambda x: x * epochs, single_targets))]
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test(scheduler, targets, epochs)

    def test_multi_step_lr(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        targets = [single_targets, list(map(lambda x: x * epochs, single_targets))]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test(scheduler, targets, epochs)

    def test_exp_lr(self):
        epochs = 10
        single_targets = [0.05 * (0.9 ** x) for x in range(epochs)]
        targets = [single_targets, list(map(lambda x: x * epochs, single_targets))]
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test(scheduler, targets, epochs)

    def test_cos_anneal_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [eta_min + (0.05 - eta_min) *
                          (1 + math.cos(math.pi * x / epochs)) / 2
                          for x in range(epochs)]
        targets = [single_targets, list(map(lambda x: x * epochs, single_targets))]
        scheduler = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        self._test(scheduler, targets, epochs)

    def test_reduce_lr_on_plateau1(self):
        epochs = 10
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [[0.5] * 20]
        metrics = [10 - i * 0.0167 for i in range(20)]
        scheduler = ReduceLROnPlateau(self.opt, threshold_mode='abs', mode='min',
                                      threshold=0.01, patience=5, cooldown=5)
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau2(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [[0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2]
        metrics = [10 - i * 0.0165 for i in range(22)]
        scheduler = ReduceLROnPlateau(self.opt, patience=5, cooldown=0, threshold_mode='abs',
                                      mode='min', threshold=0.1)
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau3(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [[0.5] * (2 + 6) + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [-0.8] * 2 + [-0.234] * 20
        scheduler = ReduceLROnPlateau(self.opt, mode='max', patience=5, cooldown=5,
                                      threshold_mode='abs')
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau4(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [[0.5] * 20]
        metrics = [1.5 * (1.025 ** i) for i in range(20)]  # 1.025 > 1.1**0.25
        scheduler = ReduceLROnPlateau(self.opt, mode='max', patience=3,
                                      threshold_mode='rel', threshold=0.1)
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau5(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [1.5 * (1.005 ** i) for i in range(20)]
        scheduler = ReduceLROnPlateau(self.opt, mode='max', threshold_mode='rel',
                                      threshold=0.1, patience=5, cooldown=5)
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau6(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [[0.5] * 20]
        metrics = [1.5 * (0.85 ** i) for i in range(20)]
        scheduler = ReduceLROnPlateau(self.opt, mode='min', threshold_mode='rel',
                                      threshold=0.1)
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau7(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [1] * 7 + [0.6] + [0.5] * 12
        scheduler = ReduceLROnPlateau(self.opt, mode='min', threshold_mode='rel',
                                      threshold=0.1, patience=5, cooldown=5)
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau8(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        targets = [[0.5] * 6 + [0.4] * 14, [0.5] * 6 + [0.3] * 14]
        metrics = [1.5 * (1.005 ** i) for i in range(20)]
        scheduler = ReduceLROnPlateau(self.opt, mode='max', threshold_mode='rel',
                                      min_lr=[0.4, 0.3],
                                      threshold=0.1, patience=5, cooldown=5)
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_cycle_lr_invalid_mode(self):
        with self.assertRaises(ValueError):
            scheduler = CyclicLR(self.opt, mode="CATS")

    def test_cycle_lr_triangular_mode_one_lr(self):
        target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        targets = [target, target]
        scheduler = CyclicLR(self.opt, base_lr=1, max_lr=5, step_size_up=4,
                             mode='triangular')
        self._test_cycle_lr(scheduler, targets, len(target))

    def test_cycle_lr_triangular2_mode_one_lr(self):
        target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1] + \
                 [1.25, 1.50, 1.75, 2.00, 1.75]
        targets = [target, target]
        scheduler = CyclicLR(self.opt, base_lr=1, max_lr=5, step_size_up=4,
                             mode='triangular2')
        self._test_cycle_lr(scheduler, targets, len(target))

    def test_cycle_lr_exp_range_mode_one_lr(self):
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        target = list(map(lambda x: base_lr + x[1] * diff_lr * gamma**x[0], enumerate(xs)))
        targets = [target, target]
        scheduler = CyclicLR(self.opt, base_lr=base_lr,
                             max_lr=max_lr, step_size_up=4,
                             mode='exp_range', gamma=gamma)
        self._test_cycle_lr(scheduler, targets, len(target))

    def test_cycle_lr_triangular_mode(self):
        target_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        target_2 = list(map(lambda x: x + 1, target_1))
        targets = [target_1, target_2]
        scheduler = CyclicLR(self.opt, base_lr=[1, 2], max_lr=[5, 6], step_size_up=4,
                             mode='triangular')
        self._test_cycle_lr(scheduler, targets, len(target_1))

    def test_cycle_lr_triangular2_mode(self):
        target_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1] + \
                   [1.25, 1.50, 1.75, 2.00, 1.75]
        target_2 = list(map(lambda x: x + 2, target_1))
        targets = [target_1, target_2]
        scheduler = CyclicLR(self.opt, base_lr=[1, 3], max_lr=[5, 7], step_size_up=4,
                             mode='triangular2')
        self._test_cycle_lr(scheduler, targets, len(target_1))

    def test_cycle_lr_exp_range_mode(self):
        base_lr_1, max_lr_1 = 1, 5
        base_lr_2, max_lr_2 = 5, 12

        diff_lr_1 = max_lr_1 - base_lr_1
        diff_lr_2 = max_lr_2 - base_lr_2

        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        target_1 = list(map(lambda x: base_lr_1 + x[1] * diff_lr_1 * gamma**x[0], enumerate(xs)))
        target_2 = list(map(lambda x: base_lr_2 + x[1] * diff_lr_2 * gamma**x[0], enumerate(xs)))
        targets = [target_1, target_2]
        scheduler = CyclicLR(self.opt, base_lr=[base_lr_1, base_lr_2],
                             max_lr=[max_lr_1, max_lr_2], step_size_up=4,
                             mode='exp_range', gamma=gamma)
        self._test_cycle_lr(scheduler, targets, len(target_1))

    def test_triangular_mode_step_size_up_down(self):
        target = [1.0, 2.0, 3.0, 4.0, 5.0, 13.0 / 3, 11.0 / 3, 9.0 / 3, 7.0 / 3, 5.0 / 3, 1.0]
        targets = [target, target]
        scheduler = CyclicLR(self.opt, base_lr=1, max_lr=5,
                             step_size_up=4,
                             step_size_down=6,
                             mode='triangular')
        self._test_cycle_lr(scheduler, targets, len(target))

    def test_triangular2_mode_step_size_up_down(self):
        base_target = ([
            1.0, 3.0, 5.0, 13.0 / 3, 11.0 / 3, 9.0 / 3, 7.0 / 3, 5.0 / 3, 1.0, 2.0, 3.0, 8.0 / 3,
            7.0 / 3, 6.0 / 3, 5.0 / 3, 4.0 / 3, 1.0, 3.0 / 2, 2.0, 11.0 / 6, 10.0 / 6, 9.0 / 6,
            8.0 / 6, 7.0 / 6
        ])
        deltas = [2 * i for i in range(0, 2)]
        base_lrs = [1 + delta for delta in deltas]
        max_lrs = [5 + delta for delta in deltas]
        targets = [[x + delta for x in base_target] for delta in deltas]
        scheduler = CyclicLR(
            self.opt,
            base_lr=base_lrs,
            max_lr=max_lrs,
            step_size_up=2,
            step_size_down=6,
            mode='triangular2')
        self._test_cycle_lr(scheduler, targets, len(base_target))

    def test_exp_range_mode_step_size_up_down(self):
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = ([
            0.0, 0.5, 1.0, 5.0 / 6, 4.0 / 6, 3.0 / 6, 2.0 / 6, 1.0 / 6, 0.0, 0.5, 1.0, 5.0 / 6,
            4.0 / 6
        ])
        target = [base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)]
        targets = [target, target]
        scheduler = CyclicLR(self.opt, base_lr=base_lr, max_lr=max_lr,
                             step_size_up=2, step_size_down=6,
                             mode='exp_range', gamma=gamma)
        self._test_cycle_lr(scheduler, targets, len(target))

    def test_lambda_lr(self):
        epochs = 10
        self.opt.param_groups[0]['lr'] = 0.05
        self.opt.param_groups[1]['lr'] = 0.4
        targets = [[0.05 * (0.9 ** x) for x in range(epochs)], [0.4 * (0.8 ** x)
                                                                for x in range(epochs)]]
        scheduler = LambdaLR(self.opt,
                             lr_lambda=[lambda x1: 0.9 ** x1, lambda x2: 0.8 ** x2])
        self._test(scheduler, targets, epochs)

    def test_step_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: StepLR(self.opt, gamma=0.1, step_size=3),
            lambda: StepLR(self.opt, gamma=0.01 / 2, step_size=1))

    def test_multi_step_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9]),
            lambda: MultiStepLR(self.opt, gamma=0.01, milestones=[1, 4, 6]))

    def test_exp_step_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: ExponentialLR(self.opt, gamma=0.1),
            lambda: ExponentialLR(self.opt, gamma=0.01))

    def test_cosine_lr_state_dict(self):
        epochs = 10
        eta_min = 1e-10
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min),
            lambda: CosineAnnealingLR(self.opt, T_max=epochs // 2, eta_min=eta_min / 2),
            epochs=epochs)

    def test_reduce_lr_on_plateau_state_dict(self):
        scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.1, patience=2)
        for score in [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 3.0, 2.0, 1.0]:
            scheduler.step(score)
        scheduler_copy = ReduceLROnPlateau(self.opt, mode='max', factor=0.5, patience=10)
        scheduler_copy.load_state_dict(scheduler.state_dict())
        for key in scheduler.__dict__.keys():
            if key not in {'optimizer', 'is_better'}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key], allow_inf=True)

    def test_lambda_lr_state_dict_fn(self):
        scheduler = LambdaLR(self.opt, lr_lambda=lambda x: x)
        state = scheduler.state_dict()
        self.assertIsNone(state['lr_lambdas'][0])

        scheduler_copy = LambdaLR(self.opt, lr_lambda=lambda x: x)
        scheduler_copy.load_state_dict(state)
        for key in scheduler.__dict__.keys():
            if key not in {'optimizer', 'lr_lambdas'}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key], allow_inf=True)

    def test_lambda_lr_state_dict_obj(self):
        scheduler = LambdaLR(self.opt, lr_lambda=LambdaLRTestObject(10))
        state = scheduler.state_dict()
        self.assertIsNotNone(state['lr_lambdas'][0])

        scheduler_copy = LambdaLR(self.opt, lr_lambda=LambdaLRTestObject(-1))
        scheduler_copy.load_state_dict(state)
        for key in scheduler.__dict__.keys():
            if key not in {'optimizer'}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key], allow_inf=True)

    def _check_scheduler_state_dict(self, constr, constr2, epochs=10):
        scheduler = constr()
        for _ in range(epochs):
            scheduler.step()
        scheduler_copy = constr2()
        scheduler_copy.load_state_dict(scheduler.state_dict())
        for key in scheduler.__dict__.keys():
            if key != 'optimizer':
                self.assertAlmostEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])
        self.assertAlmostEqual(scheduler.get_lr(), scheduler_copy.get_lr())

    def _test(self, scheduler, targets, epochs=10):
        for epoch in range(epochs):
            scheduler.step(epoch)
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertAlmostEqual(target[epoch], param_group['lr'],
                                       msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                           epoch, target[epoch], param_group['lr']), delta=1e-5)

    def _test_reduce_lr_on_plateau(self, scheduler, targets, metrics, epochs=10, verbose=False):
        for epoch in range(epochs):
            scheduler.step(metrics[epoch])
            if verbose:
                print('epoch{}:\tlr={}'.format(epoch, self.opt.param_groups[0]['lr']))
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertAlmostEqual(target[epoch], param_group['lr'],
                                       msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                           epoch, target[epoch], param_group['lr']), delta=1e-5)

    def _test_cycle_lr(self, scheduler, targets, batch_iterations, verbose=False):
        for batch_num in range(batch_iterations):
            scheduler.step(batch_num)
            if verbose:
                print('batch{}:\tlr={}'.format(batch_num, self.opt.param_groups[0]['lr']))
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertAlmostEqual(
                    target[batch_num], param_group['lr'],
                    msg='LR is wrong in batch_num {}: expected {}, got {}'.format(
                        batch_num, target[batch_num], param_group['lr']), delta=1e-5)


if __name__ == '__main__':
    run_tests()
