import unittest
import functools
from copy import deepcopy
import torch
import torch.optim as optim
import torch.legacy.optim as old_optim
from torch.autograd import Variable

from common import TestCase, run_tests


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

    def _test_basic_cases_template(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)
        optimizer = constructor(weight, bias)

        def fn():
            optimizer.zero_grad()
            y = weight.mv(input)
            if y.is_cuda and bias.is_cuda and y.get_device() != bias.get_device():
                y = y.cuda(bias.get_device())
            loss = (y + bias).pow(2).sum()
            loss.backward()
            return loss

        initial_value = fn().data[0]
        for i in range(200):
            optimizer.step(fn)
        self.assertLess(fn().data[0], initial_value)

    def _test_state_dict(self, weight, bias, input, constructor):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)

        def fn_base(optimizer, weight, bias):
            optimizer.zero_grad()
            loss = (weight.mv(input) + bias).pow(2).sum()
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
            lambda weight, bias: optim.Adagrad(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-1)
        )

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
            lambda weight, bias: optim.Adagrad([weight, bias], lr=1e-1)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adagrad(
                self._build_params_dict(weight, bias, lr=1e-2),
                lr=1e-1)
        )

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
            lambda weight, bias: optim.Adagrad([weight, bias], lr=1e-2)
        )
        self._test_basic_cases(
            lambda weight, bias: optim.Adagrad(
                self._build_params_dict(weight, bias, lr=1e-3),
                lr=1e-2)
        )

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

    def test_invalid_param_type(self):
        with self.assertRaises(TypeError):
            optim.SGD(Variable(torch.randn(5, 5)), lr=3)


if __name__ == '__main__':
    run_tests()
