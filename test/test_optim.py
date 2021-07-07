import warnings
import math
import unittest
import functools
from copy import deepcopy
import torch
from torch._six import inf
import torch.optim as optim
import torch.optim._multi_tensor as optim_mt
import torch.nn.functional as F
from torch.optim import SGD
from torch.autograd import Variable
from torch import sparse
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR, StepLR, \
    MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, \
    _LRScheduler, CyclicLR, CosineAnnealingWarmRestarts, OneCycleLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_UBSAN, load_tests, \
    skipIfRocm

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


def rosenbrock(tensor):
    x, y = tensor
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


def drosenbrock(tensor):
    x, y = tensor
    return torch.tensor((-400 * x * (y - x ** 2) - 2 * (1 - x), 200 * (y - x ** 2)))


class TestOptim(TestCase):
    exact_dtype = True

    def _test_rosenbrock_sparse(self, constructor, scheduler_constructors=None,
                                sparse_only=False):
        if scheduler_constructors is None:
            scheduler_constructors = []
        params_t = torch.tensor([1.5, 1.5])

        params = Variable(params_t, requires_grad=True)
        optimizer = constructor([params])
        schedulers = []
        for scheduler_constructor in scheduler_constructors:
            schedulers.append(scheduler_constructor(optimizer))

        if not sparse_only:
            params_c = Variable(params_t.clone(), requires_grad=True)
            optimizer_c = constructor([params_c])

        solution = torch.tensor([1, 1])
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
                v = torch.tensor([x / 4., x - x / 4.])
            else:
                i = torch.LongTensor([[1, 1]])
                y = grad[1]
                v = torch.tensor([y - y / 4., y / 4.])
            x = sparse.DoubleTensor(i, v, torch.Size([2])).to(dtype=v.dtype)
            with torch.no_grad():
                if sparse_grad:
                    params.grad = x
                else:
                    params.grad = x.to_dense()
            return loss

        for i in range(2000):
            # Do cyclic coordinate descent
            w = i % 2
            optimizer.step(functools.partial(eval, params, True, w))
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(rosenbrock(params))
                else:
                    scheduler.step()
            if not sparse_only:
                optimizer_c.step(functools.partial(eval, params_c, False, w))
                self.assertEqual(params.data, params_c.data)

        self.assertLessEqual(params.data.dist(solution), initial_dist)

    def _test_basic_cases_template(self, weight, bias, input, constructor, scheduler_constructors):
        weight = Variable(weight, requires_grad=True)
        bias = Variable(bias, requires_grad=True)
        input = Variable(input)
        optimizer = constructor(weight, bias)
        schedulers = []
        for scheduler_constructor in scheduler_constructors:
            schedulers.append(scheduler_constructor(optimizer))

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
        for _i in range(200):
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    val_loss = fn()
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
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
        for _i in range(20):
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
        for _i in range(20):
            optimizer.step(fn)
            optimizer_c.step(fn_c)
            self.assertEqual(weight, weight_c)
            self.assertEqual(bias, bias_c)
        # Make sure state dict wasn't modified
        self.assertEqual(state_dict, state_dict_c)
        # Make sure state dict is deterministic with equal but not identical parameters
        self.assertEqual(optimizer.state_dict(), optimizer_c.state_dict())
        # Make sure repeated parameters have identical representation in state dict
        optimizer_c.param_groups.extend(optimizer_c.param_groups)
        self.assertEqual(optimizer.state_dict()['param_groups'][-1],
                         optimizer_c.state_dict()['param_groups'][-1])

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

        for _i in range(20):
            optimizer.step(fn)
            optimizer_cuda.step(fn_cuda)
            self.assertEqual(weight, weight_cuda)
            self.assertEqual(bias, bias_cuda)

        # validate deepcopy() copies all public attributes
        def getPublicAttr(obj):
            return set(k for k in obj.__dict__ if not k.startswith('_'))
        self.assertEqual(getPublicAttr(optimizer), getPublicAttr(deepcopy(optimizer)))

    def _test_basic_cases(self, constructor, scheduler_constructors=None,
                          ignore_multidevice=False):
        if scheduler_constructors is None:
            scheduler_constructors = []
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
            constructor,
            scheduler_constructors
        )
        # non-contiguous parameters
        self._test_basic_cases_template(
            torch.randn(10, 5, 2)[..., 0],
            torch.randn(10, 2)[..., 0],
            torch.randn(5),
            constructor,
            scheduler_constructors
        )
        # CUDA
        if not torch.cuda.is_available():
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(),
            torch.randn(10).cuda(),
            torch.randn(5).cuda(),
            constructor,
            scheduler_constructors
        )
        # Multi-GPU
        if not torch.cuda.device_count() > 1 or ignore_multidevice:
            return
        self._test_basic_cases_template(
            torch.randn(10, 5).cuda(0),
            torch.randn(10).cuda(1),
            torch.randn(5).cuda(0),
            constructor,
            scheduler_constructors
        )

    def _build_params_dict(self, weight, bias, **kwargs):
        return [{'params': [weight]}, dict(params=[bias], **kwargs)]

    def _build_params_dict_single(self, weight, bias, **kwargs):
        return [dict(params=bias, **kwargs)]

    def test_sgd(self):
        for optimizer in [optim.SGD, optim_mt.SGD]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict_single(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict_single(weight, bias, lr=1e-2))
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3),
                [lambda opt: StepLR(opt, gamma=0.9, step_size=10)]
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3),
                [lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                 lambda opt: ReduceLROnPlateau(opt)]
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3),
                [lambda opt: StepLR(opt, gamma=0.99, step_size=10),
                 lambda opt: ExponentialLR(opt, gamma=0.99),
                 lambda opt: ReduceLROnPlateau(opt)]
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, momentum=1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, momentum=1, weight_decay=1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], nesterov=True, lr=1e-3, momentum=1, weight_decay=1)
            )
            with self.assertRaisesRegex(ValueError, "Invalid momentum value: -0.5"):
                optimizer(None, lr=1e-2, momentum=-0.5)

    def test_sgd_sparse(self):
        for optimizer in [optim.SGD, optim_mt.SGD]:
            self._test_rosenbrock_sparse(
                lambda params: optimizer(params, lr=5e-3)
            )
            self._test_rosenbrock_sparse(
                lambda params: optimizer(params, lr=0.005),
                [lambda opt: StepLR(opt, gamma=0.99999, step_size=300)]
            )

    def test_multi_tensor_optimizers(self):
        if not torch.cuda.is_available():
            return

        optimizer_pairs_with_flags = [
            ((optim.Adam, optim._multi_tensor.Adam), dict(weight_decay=1., amsgrad=True)),
            ((optim.Adam, optim._multi_tensor.Adam), dict(weight_decay=1., amsgrad=False)),
            ((optim.Adam, optim._multi_tensor.Adam), dict(weight_decay=0., amsgrad=True)),
            ((optim.Adam, optim._multi_tensor.Adam), dict(weight_decay=0., amsgrad=False)),
            ((optim.AdamW, optim._multi_tensor.AdamW), dict(weight_decay=1., amsgrad=True)),
            ((optim.AdamW, optim._multi_tensor.AdamW), dict(weight_decay=1., amsgrad=False)),
            ((optim.AdamW, optim._multi_tensor.AdamW), dict(weight_decay=0., amsgrad=True)),
            ((optim.AdamW, optim._multi_tensor.AdamW), dict(weight_decay=0., amsgrad=False)),
            ((optim.NAdam, optim._multi_tensor.NAdam), dict(weight_decay=0., momentum_decay=6e-3)),
            ((optim.NAdam, optim._multi_tensor.NAdam), dict(weight_decay=1., momentum_decay=6e-3)),
            ((optim.NAdam, optim._multi_tensor.NAdam), dict(weight_decay=0., momentum_decay=4e-3)),
            ((optim.NAdam, optim._multi_tensor.NAdam), dict(weight_decay=0.01, momentum_decay=4e-3)),
            ((optim.SGD, optim._multi_tensor.SGD), dict(lr=0.2, momentum=1, dampening=0, weight_decay=1, nesterov=True)),
            ((optim.SGD, optim._multi_tensor.SGD), dict(lr=0.2, momentum=1, dampening=0.5, weight_decay=1, nesterov=False)),
            ((optim.RAdam, optim._multi_tensor.RAdam), dict(weight_decay=0)),
            ((optim.RAdam, optim._multi_tensor.RAdam), dict(weight_decay=1)),
            ((optim.RMSprop, optim._multi_tensor.RMSprop), dict(weight_decay=1, momentum=1, centered=True)),
            ((optim.RMSprop, optim._multi_tensor.RMSprop), dict(weight_decay=1, momentum=0, centered=True)),
            ((optim.RMSprop, optim._multi_tensor.RMSprop), dict(weight_decay=1, momentum=1, centered=False)),
            ((optim.RMSprop, optim._multi_tensor.RMSprop), dict(weight_decay=0, momentum=1, centered=False)),
            ((optim.Rprop, optim._multi_tensor.Rprop), dict(lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50))),
            ((optim.ASGD, optim._multi_tensor.ASGD), dict(weight_decay=0)),
            ((optim.ASGD, optim._multi_tensor.ASGD), dict(weight_decay=1)),
            ((optim.Adamax, optim._multi_tensor.Adamax), dict(weight_decay=0)),
            ((optim.Adamax, optim._multi_tensor.Adamax), dict(weight_decay=1)),
            ((optim.Adadelta, optim._multi_tensor.Adadelta), dict(weight_decay=0)),
            ((optim.Adadelta, optim._multi_tensor.Adadelta), dict(weight_decay=1)),
            ((optim.Adagrad, optim._multi_tensor.Adagrad), dict(weight_decay=0)),
            ((optim.Adagrad, optim._multi_tensor.Adagrad), dict(weight_decay=1)),
        ]

        kIterations = 11
        device = 'cuda'

        for optimizers, params in optimizer_pairs_with_flags:
            res = []
            for opt in optimizers:
                weight = torch.tensor([[-0.2109, -0.4976], [-0.1413, -0.3420], [-0.2524, 0.6976]],
                                      dtype=torch.float64, device=device, requires_grad=True)
                bias = torch.tensor([-0.1085, -0.2979, 0.6892], dtype=torch.float64, device=device, requires_grad=True)
                weight2 = torch.tensor([[-0.0508, -0.3941, -0.2843]],
                                       dtype=torch.float64, device=device, requires_grad=True)
                bias2 = torch.tensor([-0.0711], dtype=torch.float64, device=device, requires_grad=True)
                input = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float64, device=device).reshape(3, 2)

                model = torch.nn.Sequential(torch.nn.Linear(2, 3),
                                            torch.nn.Sigmoid(),
                                            torch.nn.Linear(3, 1),
                                            torch.nn.Sigmoid())
                model.to(torch.float64).to(device)

                pretrained_dict = model.state_dict()
                pretrained_dict['0.weight'] = weight
                pretrained_dict['0.bias'] = bias
                pretrained_dict['2.weight'] = weight2
                pretrained_dict['2.bias'] = bias2
                model.load_state_dict(pretrained_dict)

                optimizer = opt(model.parameters(), **params)

                for _ in range(kIterations):
                    optimizer.zero_grad()
                    output = model(input)
                    loss = output.sum()
                    loss.backward()

                    if iter == 0:
                        model.parameters().__next__().grad = None

                    optimizer.step()

                res.append(model.parameters())

            for p1, p2 in zip(res[0], res[1]):
                self.assertEqual(p1, p2, atol=5e-5, rtol=0)

    def test_adam(self):
        for optimizer in [optim.Adam, optim_mt.Adam]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, amsgrad=True)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=0.1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3, amsgrad=True)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3),
                [lambda opt: ExponentialLR(opt, gamma=0.9)]
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, amsgrad=True),
                [lambda opt: ExponentialLR(opt, gamma=0.9),
                 lambda opt: ReduceLROnPlateau(opt)]
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3, amsgrad=True),
                [lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                 lambda opt: ReduceLROnPlateau(opt)]
            )
            with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 0: 1.0"):
                optimizer(None, lr=1e-2, betas=(1.0, 0.0))

            with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
                optimizer(None, lr=1e-2, weight_decay=-1)

    def test_adamw(self):
        for optimizer in [optim.AdamW, optim_mt.AdamW]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=1, amsgrad=True)
            )
            with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
                optimizer(None, lr=1e-2, weight_decay=-1)

    def test_sparse_adam(self):
        self._test_rosenbrock_sparse(
            lambda params: optim.SparseAdam(params, lr=4e-2),
            [],
            True
        )
        with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 0: 1.0"):
            optim.SparseAdam(None, lr=1e-2, betas=(1.0, 0.0))
        with self.assertRaisesRegex(ValueError, "SparseAdam requires dense parameter tensors"):
            optim.SparseAdam([torch.zeros(3, layout=torch.sparse_coo)])
        with self.assertRaisesRegex(ValueError, "SparseAdam requires dense parameter tensors"):
            optim.SparseAdam([{"params": [torch.zeros(3, layout=torch.sparse_coo)]}])

    # ROCm precision is too low to pass this test
    @skipIfRocm
    def test_adadelta(self):
        for optimizer in [optim.Adadelta, optim_mt.Adadelta]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias])
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, rho=0.95))
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, rho=0.95)),
                [lambda opt: StepLR(opt, gamma=0.9, step_size=10),
                 lambda opt: ReduceLROnPlateau(opt)]
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], weight_decay=1)
            )
            with self.assertRaisesRegex(ValueError, "Invalid rho value: 1.1"):
                optimizer(None, lr=1e-2, rho=1.1)

    def test_nadam(self):
        for optimizer in [optim.NAdam, optim_mt.NAdam]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=0.1, momentum_decay=6e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=0.1, momentum_decay=6e-3),
                [lambda opt: ExponentialLR(opt, gamma=0.9)]
            )
            with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 0: 1.0"):
                optimizer(None, lr=1e-2, betas=(1.0, 0.0))
            with self.assertRaisesRegex(ValueError, "Invalid momentum_decay value: -0.2"):
                optimizer(None, lr=1e-2, momentum_decay=-0.2)

    def test_adagrad(self):
        for optimizer in [optim.Adagrad, optim_mt.Adagrad]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    [weight, bias], lr=1e-1, initial_accumulator_value=0.1
                )
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-1),
                [lambda opt: ReduceLROnPlateau(opt)]
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-1),
                [lambda opt: ReduceLROnPlateau(opt),
                 lambda opt: ExponentialLR(opt, gamma=0.99)]
            )
            with self.assertRaisesRegex(ValueError, "Invalid lr_decay value: -0.5"):
                optimizer(None, lr=1e-2, lr_decay=-0.5)

    def test_adagrad_sparse(self):
        for optimizer in [optim.Adagrad, optim_mt.Adagrad]:
            self._test_rosenbrock_sparse(
                lambda params: optimizer(params, lr=1e-1)
            )
            self._test_rosenbrock_sparse(
                lambda params: optimizer(params, lr=0.1),
                [lambda opt: StepLR(opt, gamma=1 - 1e-5, step_size=500),
                 lambda opt: ReduceLROnPlateau(opt, threshold=1e-4)]
            )

    def test_adamax(self):
        for optimizer in [optim.Adamax, optim_mt.Adamax]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-1, weight_decay=1)
            )
            with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 1: 1.0"):
                optimizer(None, lr=1e-2, betas=(0.0, 1.0))

    def test_radam(self):
        for optimizer in [optim.RAdam, optim_mt.RAdam]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, weight_decay=0.1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3),
                [lambda opt: ExponentialLR(opt, gamma=0.9),
                    lambda opt: ReduceLROnPlateau(opt)]
            )
            with self.assertRaisesRegex(ValueError, "Invalid beta parameter at index 0: 1.0"):
                optimizer(None, lr=1e-2, betas=(1.0, 0.0))

            with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -1"):
                optimizer(None, lr=1e-2, weight_decay=-1)

    def test_rmsprop(self):
        for optimizer in [optim.RMSprop, optim_mt.RMSprop]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-2)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2, centered=True)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2, centered=True, momentum=0.1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2, momentum=0.1)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2, momentum=0.1, weight_decay=1)
            )
            with self.assertRaisesRegex(ValueError, "Invalid momentum value: -1.0"):
                optimizer(None, lr=1e-2, momentum=-1.0)

    def test_asgd(self):
        for optimizer in [optim.ASGD, optim_mt.ASGD]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3, t0=100)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3, t0=100)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-3),
                    lr=1e-2, weight_decay=1)
            )
            with self.assertRaisesRegex(ValueError, "Invalid weight_decay value: -0.5"):
                optimizer(None, lr=1e-2, weight_decay=-0.5)

    def test_rprop(self):
        for optimizer in [optim.Rprop, optim_mt.Rprop]:
            self._test_basic_cases(
                lambda weight, bias: optimizer([weight, bias], lr=1e-3)
            )
            self._test_basic_cases(
                lambda weight, bias: optimizer(
                    self._build_params_dict(weight, bias, lr=1e-2),
                    lr=1e-3)
            )
            with self.assertRaisesRegex(ValueError, "Invalid eta values: 1.0, 0.5"):
                optimizer(None, lr=1e-2, etas=(1.0, 0.5))

    def test_lbfgs(self):
        self._test_basic_cases(
            lambda weight, bias: optim.LBFGS([weight, bias]),
            ignore_multidevice=True
        )
        self._test_basic_cases(
            lambda weight, bias: optim.LBFGS([weight, bias], line_search_fn="strong_wolfe"),
            ignore_multidevice=True
        )

    @unittest.skipIf(TEST_WITH_UBSAN, "division-by-zero error with UBSAN")
    def test_lbfgs_return_type(self):
        params = [torch.randn(10, 5), torch.randn(10)]
        opt1 = optim.LBFGS(params, 0.01, tolerance_grad=inf)
        opt2 = optim.LBFGS(params, 0.01, tolerance_grad=-inf)

        def closure():
            return torch.tensor([10])

        res1 = opt1.step(closure)
        res2 = opt2.step(closure)
        self.assertEqual(type(res1), type(res2))

    def test_invalid_param_type(self):
        with self.assertRaises(TypeError):
            optim.SGD(Variable(torch.randn(5, 5)), lr=3)

    def test_duplicate_params_in_param_group(self):
        param = Variable(torch.randn(5, 5))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            optim.SGD([param, param], lr=0.1)
            self.assertEqual(len(w), 1)
            self.assertIn('a parameter group with duplicate parameters', str(w[0].message))

    def test_no_grad_for_all_params(self):
        param = torch.randn(5, 5, requires_grad=False)

        optimizer_list = [
            optim.Adadelta,
            optim.AdamW,
            optim.Adam,
            optim.Adagrad,
            optim.Adamax,
            optim.RMSprop,
            optim.SGD,
            optim.SparseAdam,
            optim.ASGD,
        ]
        for optim_ctr in optimizer_list:
            opt = optim_ctr([param, param], lr=0.1)
            # make sure step can still run even if
            # all params have no grad
            opt.step()


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
    exact_dtype = True

    def setUp(self):
        super(TestLRScheduler, self).setUp()
        self.net = SchedulerTestNet()
        self.opt = SGD(
            [{'params': self.net.conv1.parameters()}, {'params': self.net.conv2.parameters(), 'lr': 0.5}],
            lr=0.05)

    def test_error_when_getlr_has_epoch(self):
        class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, gamma, milestones, last_epoch=-1):
                self.init_lr = [group['lr'] for group in optimizer.param_groups]
                self.gamma = gamma
                self.milestones = milestones
                super().__init__(optimizer, last_epoch)

            def get_lr(self, step):
                global_step = self.last_epoch
                gamma_power = ([0] + [i + 1 for i, m in enumerate(self.milestones) if global_step >= m])[-1]
                return [init_lr * (self.gamma ** gamma_power) for init_lr in self.init_lr]

        optimizer = torch.optim.SGD([torch.rand(1)], lr=1)

        with self.assertRaises(TypeError):
            scheduler = MultiStepLR(optimizer, gamma=1, milestones=[10, 20])

    def test_no_cyclic_references(self):
        import gc
        param = Variable(torch.empty(10), requires_grad=True)
        optim = SGD([param], lr=0.5)
        scheduler = LambdaLR(optim, lambda epoch: 1.0)
        del scheduler

        # Prior to Python 3.7, local variables in a function will be referred by the current frame.
        import sys
        if sys.version_info < (3, 7):
            import inspect
            referrers = gc.get_referrers(optim)
            self.assertTrue(
                len(referrers) == 1 and referrers[0] is inspect.currentframe(),
                "Optimizer should contain no cyclic references (except current frame)")
            del referrers
        else:
            self.assertTrue(
                len(gc.get_referrers(optim)) == 0,
                "Optimizer should contain no cyclic references")

        gc.collect()
        del optim
        self.assertEqual(
            gc.collect(), 0, msg="Optimizer should be garbage-collected on __del__")

    def test_old_pattern_warning(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate', old_pattern)

    def test_old_pattern_warning_with_arg(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern2():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate', old_pattern2)

    def test_old_pattern_warning_resuming(self):
        epochs = 35
        for i, group in enumerate(self.opt.param_groups):
            group['initial_lr'] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate', old_pattern)

    def test_old_pattern_warning_resuming_with_arg(self):
        epochs = 35
        for i, group in enumerate(self.opt.param_groups):
            group['initial_lr'] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern2():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate', old_pattern2)

    def test_old_pattern_warning_with_overridden_optim_step(self):
        epochs = 35
        for i, group in enumerate(self.opt.param_groups):
            group['initial_lr'] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # emulate use-case with optimizer.step overridden
        import types

        old_step = self.opt.step

        def new_step(o, *args, **kwargs):
            retval = old_step(*args, **kwargs)
            return retval

        self.opt.step = types.MethodType(new_step, self.opt)

        def old_pattern2():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r'how-to-adjust-learning-rate', old_pattern2)

    def test_new_pattern_no_warning(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()
            self.assertTrue(len(ws) == 0, "No warning should be raised")

    def test_new_pattern_no_warning_with_arg(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()
            self.assertTrue(len(ws) == 0, "No warning should be raised")

    def test_new_pattern_no_warning_with_overridden_optim_step(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # emulate use-case with optimizer.step overridden
        import types

        old_step = self.opt.step

        def new_step(o, *args, **kwargs):
            retval = old_step(*args, **kwargs)
            return retval

        self.opt.step = types.MethodType(new_step, self.opt)

        def new_pattern():
            for e in range(epochs):
                self.opt.step()
                scheduler.step()

        self.assertWarnsRegex(UserWarning, r'`optimizer.step\(\)` has been overridden', new_pattern)

    def _test_lr_is_constant_for_constant_epoch(self, scheduler):
        l = []

        for _ in range(10):
            scheduler.step(2)
            l.append(self.opt.param_groups[0]['lr'])
        self.assertEqual(min(l), max(l))

    def test_step_lr_is_constant_for_constant_epoch(self):
        scheduler = StepLR(self.opt, 2)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_exponential_lr_is_constant_for_constant_epoch(self):
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_step_lr(self):
        # lr = 0.05     if epoch < 3
        # lr = 0.005    if 30 <= epoch < 6
        # lr = 0.0005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test(scheduler, targets, epochs)

    def test_get_last_lr_step_lr(self):
        from torch.nn import Parameter
        epochs = 10
        optimizer = torch.optim.SGD([Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
        targets = [[0.1] * 3 + [0.01] * 3 + [0.001] * 3 + [0.0001]]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_get_last_lr_multi_step_lr(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if 5 <= epoch < 9
        # lr = 0.00005   if 9 <= epoch
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 1
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_multi_step_lr(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test(scheduler, targets, epochs)

    def test_multi_step_lr_with_epoch(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_with_epoch(scheduler, targets, epochs)

    def test_exp_lr(self):
        epochs = 10
        single_targets = [0.05 * (0.9 ** x) for x in range(epochs)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test(scheduler, targets, epochs)

    def test_cos_anneal_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [eta_min + (0.05 - eta_min) *
                          (1 + math.cos(math.pi * x / epochs)) / 2
                          for x in range(epochs)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        self._test(scheduler, targets, epochs)

    def test_closed_form_step_lr(self):
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        closed_form_scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_multi_step_lr(self):
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        closed_form_scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_exp_lr(self):
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        closed_form_scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_cos_anneal_lr(self):
        eta_min = 1e-10
        epochs = 20
        T_max = 5
        scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        closed_form_scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        self._test_against_closed_form(scheduler, closed_form_scheduler, epochs)

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
        scheduler = ReduceLROnPlateau(self.opt, mode='max', threshold_mode='rel', min_lr=[0.4, 0.3],
                                      threshold=0.1, patience=5, cooldown=5)
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_compound_step_and_multistep_lr(self):
        epochs = 10
        schedulers = [None] * 2
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        targets = [[0.05] * 2 + [0.005] * 1 + [5e-4] * 2 + [5e-5] + [5e-6] * 3 + [5e-8]]
        self._test(schedulers, targets, epochs)

    def test_compound_step_and_exp_lr(self):
        epochs = 10
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9 ** x) for x in range(3)]
        single_targets += [0.005 * (0.9 ** x) for x in range(3, 6)]
        single_targets += [0.0005 * (0.9 ** x) for x in range(6, 9)]
        single_targets += [0.00005 * (0.9 ** x) for x in range(9, 12)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_exp_and_multistep_lr(self):
        epochs = 10
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9 ** x) for x in range(2)]
        single_targets += [0.005 * (0.9 ** x) for x in range(2, 5)]
        single_targets += [0.0005 * (0.9 ** x) for x in range(5, 9)]
        single_targets += [0.00005 * (0.9 ** x) for x in range(9, 11)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_step_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [eta_min + (0.05 - eta_min) *
                          (1 + math.cos(math.pi * x / epochs)) / 2
                          for x in range(epochs)]
        single_targets = [x * 0.1 ** (i // 3) for i, x in enumerate(single_targets)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_multistep_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [eta_min + (0.05 - eta_min) *
                          (1 + math.cos(math.pi * x / epochs)) / 2
                          for x in range(epochs)]
        multipliers = [1] * 2 + [0.1] * 3 + [0.01] * 4 + [0.001]
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_exp_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [eta_min + (0.05 - eta_min) *
                          (1 + math.cos(math.pi * x / epochs)) / 2
                          for x in range(epochs)]
        multipliers = [0.1 ** i for i in range(epochs)]
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.1)
        self._test(schedulers, targets, epochs)

    def test_compound_reduce_lr_on_plateau1(self):
        epochs = 10
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        single_targets = [0.5] * 20
        multipliers = [0.1 ** (i // 3) for i in range(20)]
        single_targets = [x * y for x, y in zip(multipliers, single_targets)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0167 for i in range(20)]
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(self.opt, threshold_mode='abs', mode='min',
                                          threshold=0.01, patience=5, cooldown=5)
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau2(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        single_targets = [0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2
        multipliers = [1] * 3 + [0.1] * 5 + [0.01] * 4 + [0.001] * 10
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0165 for i in range(22)]
        schedulers = [None] * 2
        schedulers[0] = ReduceLROnPlateau(self.opt, patience=5, cooldown=0, threshold_mode='abs',
                                          mode='min', threshold=0.1)
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[3, 8, 12])
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau3(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.5
        single_targets = [0.5] * (2 + 6) + [0.05] * (5 + 6) + [0.005] * 4
        multipliers = [0.1 ** i for i in range(epochs)]
        single_targets = [x * y for x, y in zip(multipliers, single_targets)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [-0.8] * 2 + [-0.234] * 20
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(self.opt, mode='max', patience=5, cooldown=5,
                                          threshold_mode='abs')
        schedulers[1] = ExponentialLR(self.opt, gamma=0.1)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau4(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group['lr'] = 0.05
        epochs = 10
        eta_min = 1e-10
        single_targets = [eta_min + (0.05 - eta_min) *
                          (1 + math.cos(math.pi * x / epochs)) / 2
                          for x in range(epochs)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [1.5 * (1.025 ** i) for i in range(20)]  # 1.025 > 1.1**0.25
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(self.opt, mode='max', patience=3,
                                          threshold_mode='rel', threshold=0.1)
        schedulers[1] = CosineAnnealingLR(self.opt, epochs, eta_min)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_cycle_lr_invalid_mode(self):
        with self.assertRaises(ValueError):
            scheduler = CyclicLR(self.opt, base_lr=0, max_lr=0, mode="CATS")

    def test_cycle_lr_triangular_mode_one_lr(self):
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        momentum_target = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(self.opt, base_lr=1, max_lr=5, step_size_up=4,
                             cycle_momentum=True, base_momentum=1, max_momentum=5,
                             mode='triangular')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular_mode_one_lr_no_momentum(self):
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_targets = [lr_target, lr_target]
        momentum_target = [self.opt.defaults['momentum']] * len(lr_target)
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(self.opt, base_lr=1, max_lr=5, step_size_up=4,
                             cycle_momentum=False, mode='triangular')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular2_mode_one_lr(self):
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5,
                     1, 1.25, 1.50, 1.75, 2.00, 1.75]
        momentum_target = [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.5, 4.0,
                           3.5, 3.0, 3.5, 4.0, 4.5, 5.0, 4.75, 4.5, 4.25, 4.0, 4.25]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(self.opt, base_lr=1, max_lr=5, step_size_up=4,
                             cycle_momentum=True, base_momentum=1, max_momentum=5,
                             mode='triangular2')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_exp_range_mode_one_lr(self):
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        lr_target = [base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)]
        momentum_target = [max_lr - x * diff_lr * gamma**i for i, x in enumerate(xs)]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(self.opt, base_lr=base_lr,
                             max_lr=max_lr, step_size_up=4,
                             cycle_momentum=True, base_momentum=base_lr, max_momentum=max_lr,
                             mode='exp_range', gamma=gamma)
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular_mode(self):
        lr_target_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_target_2 = [x + 1 for x in lr_target_1]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        momentum_target_2 = [x + 1 for x in momentum_target_1]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicLR(self.opt, base_lr=[1, 2], max_lr=[5, 6], step_size_up=4,
                             cycle_momentum=True, base_momentum=[1, 2], max_momentum=[5, 6],
                             mode='triangular')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))

    def test_cycle_lr_triangular2_mode(self):
        lr_target_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 1.5, 2.0, 2.5, 3.0, 2.5, 2.0, 1.5, 1,
                       1.25, 1.50, 1.75, 2.00, 1.75]
        lr_target_2 = [x + 2 for x in lr_target_1]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.5, 4.0, 3.5,
                             3.0, 3.5, 4.0, 4.5, 5.0, 4.75, 4.5, 4.25, 4.0, 4.25]
        momentum_target_2 = [x + 2 for x in momentum_target_1]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicLR(self.opt, base_lr=[1, 3], max_lr=[5, 7], step_size_up=4,
                             cycle_momentum=True, base_momentum=[1, 3], max_momentum=[5, 7],
                             mode='triangular2')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))

    def test_cycle_lr_exp_range_mode(self):
        base_lr_1, max_lr_1 = 1, 5
        base_lr_2, max_lr_2 = 5, 12

        diff_lr_1 = max_lr_1 - base_lr_1
        diff_lr_2 = max_lr_2 - base_lr_2

        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        lr_target_1 = [base_lr_1 + x * diff_lr_1 * gamma**i for i, x in enumerate(xs)]
        lr_target_2 = [base_lr_2 + x * diff_lr_2 * gamma**i for i, x in enumerate(xs)]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [max_lr_1 - x * diff_lr_1 * gamma**i for i, x in enumerate(xs)]
        momentum_target_2 = [max_lr_2 - x * diff_lr_2 * gamma**i for i, x in enumerate(xs)]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicLR(self.opt, base_lr=[base_lr_1, base_lr_2],
                             max_lr=[max_lr_1, max_lr_2], step_size_up=4,
                             cycle_momentum=True, base_momentum=[base_lr_1, base_lr_2],
                             max_momentum=[max_lr_1, max_lr_2],
                             mode='exp_range', gamma=gamma)
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))

    def test_cycle_lr_triangular_mode_step_size_up_down(self):
        lr_target = [1.0, 2.0, 3.0, 4.0, 5.0, 13.0 / 3, 11.0 / 3, 9.0 / 3, 7.0 / 3, 5.0 / 3, 1.0]
        lr_targets = [lr_target, lr_target]
        momentum_target = [5.0, 4.0, 3.0, 2.0, 1.0, 5.0 / 3, 7.0 / 3, 3.0, 11.0 / 3, 13.0 / 3, 5.0]
        momentum_targets = [momentum_target, momentum_target]

        scheduler = CyclicLR(self.opt, base_lr=1, max_lr=5,
                             step_size_up=4,
                             step_size_down=6,
                             cycle_momentum=True,
                             base_momentum=1, max_momentum=5,
                             mode='triangular')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular2_mode_step_size_up_down(self):
        lr_base_target = ([
            1.0, 3.0, 5.0, 13.0 / 3, 11.0 / 3, 9.0 / 3, 7.0 / 3, 5.0 / 3, 1.0, 2.0, 3.0, 8.0 / 3,
            7.0 / 3, 6.0 / 3, 5.0 / 3, 4.0 / 3, 1.0, 3.0 / 2, 2.0, 11.0 / 6, 10.0 / 6, 9.0 / 6,
            8.0 / 6, 7.0 / 6
        ])
        momentum_base_target = ([
            5.0, 3.0, 1.0, 5.0 / 3, 7.0 / 3, 3.0, 11.0 / 3, 13.0 / 3, 5.0, 4.0, 3.0, 10.0 / 3,
            11.0 / 3, 4.0, 13.0 / 3, 14.0 / 3, 5.0, 4.5, 4.0, 25.0 / 6, 13.0 / 3, 4.5, 14.0 / 3,
            29.0 / 6
        ])
        deltas = [2 * i for i in range(0, 2)]
        base_lrs = [1 + delta for delta in deltas]
        max_lrs = [5 + delta for delta in deltas]
        lr_targets = [[x + delta for x in lr_base_target] for delta in deltas]
        momentum_targets = [[x + delta for x in momentum_base_target] for delta in deltas]
        scheduler = CyclicLR(
            self.opt,
            base_lr=base_lrs,
            max_lr=max_lrs,
            step_size_up=2,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=base_lrs,
            max_momentum=max_lrs,
            mode='triangular2')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_base_target))

    def test_cycle_lr_exp_range_mode_step_size_up_down(self):
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = ([
            0.0, 0.5, 1.0, 5.0 / 6, 4.0 / 6, 3.0 / 6, 2.0 / 6, 1.0 / 6, 0.0, 0.5, 1.0, 5.0 / 6,
            4.0 / 6
        ])
        lr_target = [base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)]
        lr_targets = [lr_target, lr_target]
        momentum_target = [max_lr - x * diff_lr * gamma**i for i, x in enumerate(xs)]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(self.opt, base_lr=base_lr, max_lr=max_lr,
                             step_size_up=2, step_size_down=6,
                             cycle_momentum=True, base_momentum=base_lr,
                             max_momentum=max_lr,
                             mode='exp_range', gamma=gamma)
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_with_momentumless_optimizer(self):
        # Note [Temporarily set optimizer to Adam]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # The TestLRScheduler object carries around an SGD optimizer to avoid having to
        # instantiate one for every test. This gets in the way for our very specific case
        # in which we need to use Adam (or really any optimizer that doesn't use momentum)
        # in order to test that the momentum bug in CyclicLR is fixed (the bug is described
        # in more detail in https://github.com/pytorch/pytorch/issues/19003 ).
        old_opt = self.opt
        self.opt = optim.Adam(
            [{'params': self.net.conv1.parameters()}, {'params': self.net.conv2.parameters(), 'lr': 0.5}],
            lr=0.05)

        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_targets = [lr_target, lr_target]
        momentum_target = [None] * len(lr_target)
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(self.opt, base_lr=1, max_lr=5, step_size_up=4,
                             cycle_momentum=False, mode='triangular')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

        self.opt = old_opt  # set optimizer back to SGD

    def test_cycle_lr_cycle_momentum_fail_with_momentumless_optimizer(self):
        with self.assertRaises(ValueError):
            adam_opt = optim.Adam(self.net.parameters())
            scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=True)

    def test_onecycle_lr_invalid_anneal_strategy(self):
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(self.opt, max_lr=1e-3, total_steps=10, anneal_strategy="CATS")

    def test_onecycle_lr_invalid_pct_start(self):
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(self.opt, max_lr=1e-3, total_steps=10, pct_start=1.1)

    def test_onecycle_lr_cannot_calculate_total_steps(self):
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(self.opt, max_lr=1e-3)

    def test_onecycle_lr_linear_annealing(self):
        lr_target = [1, 13, 25, 21.5, 18, 14.5, 11, 7.5, 4, 0.5]
        momentum_target = [22, 11.5, 1, 4, 7, 10, 13, 16, 19, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(self.opt, max_lr=25, final_div_factor=2, base_momentum=1, max_momentum=22,
                               total_steps=10, anneal_strategy='linear')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_onecycle_lr_linear_annealing_three_phases(self):
        lr_target = [1, 9, 17, 25, 17, 9, 1, 0.75, 0.5, 0.25]
        momentum_target = [22, 15, 8, 1, 8, 15, 22, 22, 22, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(self.opt, max_lr=25, div_factor=25,
                               base_momentum=1, max_momentum=22,
                               total_steps=10, anneal_strategy='linear',
                               pct_start=0.4, final_div_factor=4,
                               three_phase=True)
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_onecycle_lr_cosine_annealing(self):
        def annealing_cos(start, end, pct):
            cos_out = math.cos(math.pi * pct) + 1
            return end + (start - end) / 2.0 * cos_out
        lr_target = [1, 13, 25, annealing_cos(25, 0.5, 1 / 7.0), annealing_cos(25, 0.5, 2 / 7.0),
                     annealing_cos(25, 0.5, 3 / 7.0), annealing_cos(25, 0.5, 4 / 7.0), annealing_cos(25, 0.5, 5 / 7.0),
                     annealing_cos(25, 0.5, 6 / 7.0), 0.5]
        momentum_target = [22, 11.5, 1, annealing_cos(1, 22, 1 / 7.0), annealing_cos(1, 22, 2 / 7.0),
                           annealing_cos(1, 22, 3 / 7.0), annealing_cos(1, 22, 4 / 7.0), annealing_cos(1, 22, 5 / 7.0),
                           annealing_cos(1, 22, 6 / 7.0), 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(self.opt, max_lr=25, final_div_factor=2, base_momentum=1, max_momentum=22,
                               total_steps=10)
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_cycle_lr_with_adam(self):
        old_opt = self.opt
        self.opt = optim.Adam(
            [{'params': self.net.conv1.parameters()}, {'params': self.net.conv2.parameters(), 'lr': 0.5}],
            lr=0.05)

        lr_target = [1, 13, 25, 21.5, 18, 14.5, 11, 7.5, 4, 0.5]
        momentum_target = [22, 11.5, 1, 4, 7, 10, 13, 16, 19, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(self.opt, max_lr=25, final_div_factor=2, base_momentum=1, max_momentum=22,
                               total_steps=10, anneal_strategy='linear')
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10, use_beta1=True)
        self.opt = old_opt  # set optimizer back to SGD

    def test_lambda_lr(self):
        epochs = 10
        self.opt.param_groups[0]['lr'] = 0.05
        self.opt.param_groups[1]['lr'] = 0.4
        targets = [[0.05 * (0.9 ** x) for x in range(epochs)], [0.4 * (0.8 ** x) for x in range(epochs)]]
        scheduler = LambdaLR(self.opt,
                             lr_lambda=[lambda x1: 0.9 ** x1, lambda x2: 0.8 ** x2])
        self._test(scheduler, targets, epochs)

    def test_multiplicative_lr(self):
        epochs = 10
        self.opt.param_groups[0]['lr'] = 0.05
        self.opt.param_groups[1]['lr'] = 0.4
        targets = [[0.05 * (0.9 ** x) for x in range(epochs)], [0.4 * (0.8 ** x) for x in range(epochs)]]
        scheduler = MultiplicativeLR(self.opt, lr_lambda=[lambda x1: 0.9, lambda x2: 0.8])
        self._test(scheduler, targets, epochs)

    def test_CosineAnnealingWarmRestarts_lr1(self):
        iters = 100
        eta_min = 1e-10
        T_mults = [1, 2, 4]
        for T_mult in T_mults:
            T_i = 10
            T_cur = 0
            targets = [[0.05], [0.5]]
            scheduler = CosineAnnealingWarmRestarts(self.opt, T_0=T_i, T_mult=T_mult, eta_min=eta_min)
            for _ in range(1, iters, 1):
                T_cur += 1
                if T_cur >= T_i:
                    T_cur = T_cur - T_i
                    T_i = int(T_mult) * T_i
                targets[0] += [eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2]
                targets[1] += [eta_min + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2]
            self._test(scheduler, targets, iters)

    def test_CosineAnnealingWarmRestarts_lr2(self):
        iters = 30
        eta_min = 1e-10
        T_mults = [1, 2, 4]
        for T_mult in T_mults:
            T_i = 10
            T_cur = 0
            targets = [[0.05], [0.5]]
            scheduler = CosineAnnealingWarmRestarts(self.opt, T_0=T_i, T_mult=T_mult, eta_min=eta_min)
            for _ in torch.arange(0.1, iters, 0.1):
                T_cur = round(T_cur + 0.1, 1)
                if T_cur >= T_i:
                    T_cur = T_cur - T_i
                    T_i = int(T_mult) * T_i
                targets[0] += [eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2]
                targets[1] += [eta_min + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2]
            self._test_CosineAnnealingWarmRestarts(scheduler, targets, iters)

    def test_CosineAnnealingWarmRestarts_lr3(self):
        epochs_for_T_mults = [[0, 1, 2, 3, 4, 5, 12, 27, 3, 4, 5, 6, 13],
                              [0, 1, 2, 3, 4, 5, 25, 32, 33, 34, 80, 81, 3],
                              [0, 0.1, 0.2, 0.3, 1.3, 2.3, 17.5, 18.5, 19.5, 29.5, 30.5, 31.5, 50]]
        T_curs_for_T_mults = [[1, 2, 3, 4, 5, 2, 7, 3, 4, 5, 6, 3],
                              [1, 2, 3, 4, 5, 15, 2, 3, 4, 10, 11, 3],
                              [0.1, 0.2, 0.3, 1.3, 2.3, 7.5, 8.5, 9.5, 19.5, 20.5, 21.5, 10]]
        T_is_for_T_mults = [[10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                            [10, 10, 10, 10, 10, 20, 40, 40, 40, 80, 80, 10],
                            [10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 90]]
        eta_min = 1e-10
        T_mults = [1, 2, 3]
        for epochs, T_mult, T_curs, T_is in zip(epochs_for_T_mults, T_mults, T_curs_for_T_mults, T_is_for_T_mults):
            targets = [[0.05], [0.5]]
            scheduler = CosineAnnealingWarmRestarts(self.opt, T_0=10, T_mult=T_mult, eta_min=eta_min)
            for T_cur, T_i in zip(T_curs, T_is):
                targets[0] += [eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2]
                targets[1] += [eta_min + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2]
            self._test_interleaved_CosineAnnealingWarmRestarts(scheduler, targets, epochs)

    def test_swalr_no_anneal(self):
        epochs, swa_start, swa_lr = 10, 5, 0.01
        initial_lrs = [group['lr'] for group in self.opt.param_groups]
        targets = [[lr] * (swa_start + 1) + [swa_lr] * (epochs - swa_start - 1)
                   for lr in initial_lrs]
        swa_scheduler = SWALR(self.opt, anneal_epochs=1, swa_lr=swa_lr)
        self._test_swalr(swa_scheduler, None, targets, swa_start, epochs)

    def test_swalr_cosine_anneal_after_multiplicative(self):
        # same swa_lr for different param_groups
        epochs, swa_start, swa_lr, anneal_epochs = 15, 5, 0.01, 5
        mult_factor = 0.9
        scheduler = MultiplicativeLR(self.opt, lr_lambda=lambda epoch: mult_factor)
        swa_scheduler = SWALR(self.opt, anneal_epochs=anneal_epochs, swa_lr=swa_lr)

        def anneal_coef(t):
            if t + 1 >= anneal_epochs:
                return 0.
            return (1 + math.cos(math.pi * (t + 1) / anneal_epochs)) / 2

        initial_lrs = [group['lr'] for group in self.opt.param_groups]
        targets_before_swa = [[lr * mult_factor**i for i in range(swa_start + 1)]
                              for lr in initial_lrs]
        swa_epochs = epochs - swa_start - 1
        targets = [lrs + [lrs[-1] * anneal_coef(t) + swa_lr * (1 - anneal_coef(t)) for t in range(swa_epochs)]
                   for lrs in targets_before_swa]

        self._test_swalr(swa_scheduler, scheduler, targets, swa_start, epochs)

    def test_swalr_linear_anneal_after_multiplicative(self):
        # separate swa_lr for different param_groups
        epochs, swa_start, swa_lrs, anneal_epochs = 15, 5, [0.01, 0.02], 4
        mult_factor = 0.9
        scheduler = MultiplicativeLR(self.opt, lr_lambda=lambda epoch: mult_factor)
        swa_scheduler = SWALR(self.opt, anneal_epochs=anneal_epochs,
                              anneal_strategy="linear", swa_lr=swa_lrs)

        def anneal_coef(t):
            if t + 1 >= anneal_epochs:
                return 0.
            return 1 - (t + 1) / anneal_epochs

        initial_lrs = [group['lr'] for group in self.opt.param_groups]
        targets_before_swa = [[lr * mult_factor**i for i in range(swa_start + 1)]
                              for lr in initial_lrs]
        swa_epochs = epochs - swa_start - 1
        targets = [lrs + [lrs[-1] * anneal_coef(t) + swa_lr * (1 - anneal_coef(t)) for t in range(swa_epochs)]
                   for lrs, swa_lr in zip(targets_before_swa, swa_lrs)]

        self._test_swalr(swa_scheduler, scheduler, targets, swa_start, epochs)

    def _test_swalr(self, swa_scheduler, scheduler, targets, swa_start, epochs):
        for epoch in range(epochs):
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(target[epoch], param_group['lr'],
                                 msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                     epoch, target[epoch], param_group['lr']), atol=1e-5, rtol=0)
            if epoch >= swa_start:
                swa_scheduler.step()
            elif scheduler is not None:
                scheduler.step()

    def test_swalr_hypers(self):
        # Test that SWALR raises errors for incorrect hyper-parameters
        with self.assertRaisesRegex(ValueError, "anneal_strategy must"):
            swa_scheduler = SWALR(self.opt, anneal_strategy="exponential", swa_lr=1.)

        with self.assertRaisesRegex(ValueError, "anneal_epochs must"):
            swa_scheduler = SWALR(self.opt, anneal_epochs=-1, swa_lr=1.)
        with self.assertRaisesRegex(ValueError, "anneal_epochs must"):
            swa_scheduler = SWALR(self.opt, anneal_epochs=1.7, swa_lr=1.)
        with self.assertRaisesRegex(ValueError, "swa_lr must"):
            swa_scheduler = SWALR(self.opt, swa_lr=[1., 0.1, 0.01])

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
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    def test_lambda_lr_state_dict_fn(self):
        scheduler = LambdaLR(self.opt, lr_lambda=lambda x: x)
        state = scheduler.state_dict()
        self.assertIsNone(state['lr_lambdas'][0])

        scheduler_copy = LambdaLR(self.opt, lr_lambda=lambda x: x)
        scheduler_copy.load_state_dict(state)
        for key in scheduler.__dict__.keys():
            if key not in {'optimizer', 'lr_lambdas'}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    def test_lambda_lr_state_dict_obj(self):
        scheduler = LambdaLR(self.opt, lr_lambda=LambdaLRTestObject(10))
        state = scheduler.state_dict()
        self.assertIsNotNone(state['lr_lambdas'][0])

        scheduler_copy = LambdaLR(self.opt, lr_lambda=LambdaLRTestObject(-1))
        scheduler_copy.load_state_dict(state)
        for key in scheduler.__dict__.keys():
            if key not in {'optimizer'}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    def test_CosineAnnealingWarmRestarts_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingWarmRestarts(self.opt, T_0=10, T_mult=2),
            lambda: CosineAnnealingWarmRestarts(self.opt, T_0=100))

    def test_swa_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: SWALR(self.opt, anneal_epochs=3, swa_lr=0.5),
            lambda: SWALR(self.opt, anneal_epochs=10, anneal_strategy="linear", swa_lr=5.))

    def _check_scheduler_state_dict(self, constr, constr2, epochs=10):
        scheduler = constr()
        for _ in range(epochs):
            scheduler.step()
        scheduler_copy = constr2()
        scheduler_copy.load_state_dict(scheduler.state_dict())
        for key in scheduler.__dict__.keys():
            if key != 'optimizer':
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])
        self.assertEqual(scheduler.get_last_lr(), scheduler_copy.get_last_lr())

    def _test_get_last_lr(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, _LRScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            result = [scheduler.get_last_lr() for scheduler in schedulers]
            [scheduler.step() for scheduler in schedulers]
            target = [[t[epoch] for t in targets]] * len(schedulers)
            for t, r in zip(target, result):
                self.assertEqual(target, result,
                                 msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                     epoch, t, r), atol=1e-5, rtol=0)

    def _test_with_epoch(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, _LRScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            [scheduler.step(epoch) for scheduler in schedulers]  # step before assert: skip initial lr
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(target[epoch], param_group['lr'],
                                 msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                     epoch, target[epoch], param_group['lr']), atol=1e-5, rtol=0)

    def _test(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, _LRScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(target[epoch], param_group['lr'],
                                 msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                     epoch, target[epoch], param_group['lr']), atol=1e-5, rtol=0)
            [scheduler.step() for scheduler in schedulers]

    def _test_CosineAnnealingWarmRestarts(self, scheduler, targets, epochs=10):
        for index, epoch in enumerate(torch.arange(0, epochs, 0.1)):
            epoch = round(epoch.item(), 1)
            scheduler.step(epoch)
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(target[index], param_group['lr'],
                                 msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                     epoch, target[index], param_group['lr']), atol=1e-5, rtol=0)

    def _test_interleaved_CosineAnnealingWarmRestarts(self, scheduler, targets, epochs):
        for index, epoch in enumerate(epochs):
            scheduler.step(epoch)
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(target[index], param_group['lr'],
                                 msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                     epoch, target[index], param_group['lr']), atol=1e-5, rtol=0)

    def _test_against_closed_form(self, scheduler, closed_form_scheduler, epochs=10):
        self.setUp()
        targets = []
        for epoch in range(epochs):
            closed_form_scheduler.step(epoch)
            targets.append([group['lr'] for group in self.opt.param_groups])
        self.setUp()
        for epoch in range(epochs):
            scheduler.step()
            for i, param_group in enumerate(self.opt.param_groups):
                self.assertEqual(targets[epoch][i], param_group['lr'],
                                 msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                     epoch, targets[epoch][i], param_group['lr']), atol=1e-5, rtol=0)

    def _test_reduce_lr_on_plateau(self, schedulers, targets, metrics, epochs=10, verbose=False):
        if isinstance(schedulers, _LRScheduler) or isinstance(schedulers, ReduceLROnPlateau):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(metrics[epoch])
                else:
                    scheduler.step()
            if verbose:
                print('epoch{}:\tlr={}'.format(epoch, self.opt.param_groups[0]['lr']))
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(target[epoch], param_group['lr'],
                                 msg='LR is wrong in epoch {}: expected {}, got {}'.format(
                                     epoch, target[epoch], param_group['lr']), atol=1e-5, rtol=0)

    def _test_cycle_lr(self, scheduler, lr_targets, momentum_targets, batch_iterations, verbose=False, use_beta1=False):
        for batch_num in range(batch_iterations):
            if verbose:
                if 'momentum' in self.opt.param_groups[0].keys():
                    print('batch{}:\tlr={},momentum={}'.format(batch_num, self.opt.param_groups[0]['lr'],
                                                               self.opt.param_groups[0]['momentum']))
                elif use_beta1 and 'betas' in self.opt.param_groups[0].keys():
                    print('batch{}:\tlr={},beta1={}'.format(batch_num, self.opt.param_groups[0]['lr'],
                                                            self.opt.param_groups[0]['betas'][0]))
                else:
                    print('batch{}:\tlr={}'.format(batch_num, self.opt.param_groups[0]['lr']))

            for param_group, lr_target, momentum_target in zip(self.opt.param_groups, lr_targets, momentum_targets):
                self.assertEqual(
                    lr_target[batch_num], param_group['lr'],
                    msg='LR is wrong in batch_num {}: expected {}, got {}'.format(
                        batch_num, lr_target[batch_num], param_group['lr']), atol=1e-5, rtol=0)

                if use_beta1 and 'betas' in param_group.keys():
                    self.assertEqual(
                        momentum_target[batch_num], param_group['betas'][0],
                        msg='Beta1 is wrong in batch_num {}: expected {}, got {}'.format(
                            batch_num, momentum_target[batch_num], param_group['betas'][0]), atol=1e-5, rtol=0)
                elif 'momentum' in param_group.keys():
                    self.assertEqual(
                        momentum_target[batch_num], param_group['momentum'],
                        msg='Momentum is wrong in batch_num {}: expected {}, got {}'.format(
                            batch_num, momentum_target[batch_num], param_group['momentum']), atol=1e-5, rtol=0)
            scheduler.step()

    def test_cosine_then_cyclic(self):
        # https://github.com/pytorch/pytorch/issues/21965

        max_lr = 0.3
        base_lr = 0.1
        optim_lr = 0.5

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=optim_lr)
        lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.1)
        lr_scheduler_2 = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=1, step_size_down=3
        )

        for i in range(40):
            if i <= lr_scheduler_1.T_max:
                lr_scheduler_1.step()
            else:
                lr_scheduler_2.step()
            last_lr = optimizer.param_groups[0]["lr"]

        self.assertLessEqual(last_lr, max_lr)


class SWATestDNN(torch.nn.Module):
    def __init__(self, input_features):
        super(SWATestDNN, self).__init__()
        self.n_features = 100
        self.fc1 = torch.nn.Linear(input_features, self.n_features)
        self.bn = torch.nn.BatchNorm1d(self.n_features)

    def compute_preactivation(self, x):
        return self.fc1(x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        return x


class SWATestCNN(torch.nn.Module):
    def __init__(self, input_channels):
        super(SWATestCNN, self).__init__()
        self.n_features = 10
        self.conv1 = torch.nn.Conv2d(input_channels, self.n_features, kernel_size=3, padding=1)
        self.bn = torch.nn.BatchNorm2d(self.n_features, momentum=0.3)

    def compute_preactivation(self, x):
        return self.conv1(x)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        return x


class TestSWAUtils(TestCase):

    def _test_averaged_model(self, net_device, swa_device):
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.BatchNorm2d(5, momentum=0.3),
            torch.nn.Conv2d(5, 2, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 10)
        ).to(net_device)

        averaged_dnn = AveragedModel(dnn, device=swa_device)
        averaged_params = [torch.zeros_like(param) for param in dnn.parameters()]
        n_updates = 10
        for i in range(n_updates):
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                p_avg += p.detach() / n_updates
            averaged_dnn.update_parameters(dnn)

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertEqual(p_avg, p_swa)
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_swa.device == swa_device)
            self.assertTrue(p.device == net_device)
        self.assertTrue(averaged_dnn.n_averaged.device == swa_device)

    def test_averaged_model_all_devices(self):
        cpu = torch.device("cpu")
        self._test_averaged_model(cpu, cpu)
        if torch.cuda.is_available():
            cuda = torch.device(0)
            self._test_averaged_model(cuda, cpu)
            self._test_averaged_model(cpu, cuda)
            self._test_averaged_model(cuda, cuda)

    def test_averaged_model_mixed_device(self):
        if not torch.cuda.is_available():
            return
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.Linear(5, 10)
        )
        dnn[0].cuda()
        dnn[1].cpu()
        averaged_dnn = AveragedModel(dnn)
        averaged_params = [torch.zeros_like(param) for param in dnn.parameters()]
        n_updates = 10
        for i in range(n_updates):
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                p_avg += p.detach() / n_updates
            averaged_dnn.update_parameters(dnn)

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertEqual(p_avg, p_swa)
            # Check that AveragedModel is on the correct device
            self.assertTrue(p_avg.device == p_swa.device)

    def test_averaged_model_state_dict(self):
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.Linear(5, 10)
        )
        averaged_dnn = AveragedModel(dnn)
        averaged_dnn2 = AveragedModel(dnn)
        n_updates = 10
        for i in range(n_updates):
            for p in dnn.parameters():
                p.detach().add_(torch.randn_like(p))
            averaged_dnn.update_parameters(dnn)
        averaged_dnn2.load_state_dict(averaged_dnn.state_dict())
        for p_swa, p_swa2 in zip(averaged_dnn.parameters(), averaged_dnn2.parameters()):
            self.assertEqual(p_swa, p_swa2)
        self.assertTrue(averaged_dnn.n_averaged == averaged_dnn2.n_averaged)

    def test_averaged_model_exponential(self):
        # Test AveragedModel with EMA as avg_fn
        dnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 5, kernel_size=3),
            torch.nn.Linear(5, 10)
        )
        alpha = 0.9

        def avg_fn(p_avg, p, n_avg):
            return alpha * p_avg + (1 - alpha) * p
        averaged_dnn = AveragedModel(dnn, avg_fn=avg_fn)
        averaged_params = [torch.zeros_like(param) for param in dnn.parameters()]
        n_updates = 10
        for i in range(n_updates):
            updated_averaged_params = []
            for p, p_avg in zip(dnn.parameters(), averaged_params):
                p.detach().add_(torch.randn_like(p))
                if i == 0:
                    updated_averaged_params.append(p.clone())
                else:
                    updated_averaged_params.append((p_avg * alpha +
                                                   p * (1 - alpha)).clone())
            averaged_dnn.update_parameters(dnn)
            averaged_params = updated_averaged_params

        for p_avg, p_swa in zip(averaged_params, averaged_dnn.parameters()):
            self.assertEqual(p_avg, p_swa)

    def _test_update_bn(self, dnn, dl_x, dl_xy, cuda):

        preactivation_sum = torch.zeros(dnn.n_features)
        preactivation_squared_sum = torch.zeros(dnn.n_features)
        if cuda:
            preactivation_sum = preactivation_sum.cuda()
            preactivation_squared_sum = preactivation_squared_sum.cuda()
        total_num = 0
        for x in dl_x:
            x = x[0]
            if cuda:
                x = x.cuda()

            dnn.forward(x)
            preactivations = dnn.compute_preactivation(x)
            if len(preactivations.shape) == 4:
                preactivations = preactivations.transpose(1, 3)
            preactivations = preactivations.contiguous().view(-1, dnn.n_features)
            total_num += preactivations.shape[0]

            preactivation_sum += torch.sum(preactivations, dim=0)
            preactivation_squared_sum += torch.sum(preactivations**2, dim=0)

        preactivation_mean = preactivation_sum / total_num
        preactivation_var = preactivation_squared_sum / total_num
        preactivation_var = preactivation_var - preactivation_mean**2

        update_bn(dl_xy, dnn, device=x.device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

        def _reset_bn(module):
            if issubclass(module.__class__,
                          torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
        # reset batch norm and run update_bn again
        dnn.apply(_reset_bn)
        update_bn(dl_xy, dnn, device=x.device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)
        # using the dl_x loader instead of dl_xy
        dnn.apply(_reset_bn)
        update_bn(dl_x, dnn, device=x.device)
        self.assertEqual(preactivation_mean, dnn.bn.running_mean)
        self.assertEqual(preactivation_var, dnn.bn.running_var, atol=1e-1, rtol=0)

    def test_update_bn_dnn(self):
        # Test update_bn for a fully-connected network with BatchNorm1d
        objects, input_features = 100, 5
        x = torch.rand(objects, input_features)
        y = torch.rand(objects)
        ds_x = torch.utils.data.TensorDataset(x)
        ds_xy = torch.utils.data.TensorDataset(x, y)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dl_xy = torch.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        dnn = SWATestDNN(input_features=input_features)
        dnn.train()
        self._test_update_bn(dnn, dl_x, dl_xy, False)
        if torch.cuda.is_available():
            dnn = SWATestDNN(input_features=input_features)
            dnn.train()
            self._test_update_bn(dnn.cuda(), dl_x, dl_xy, True)
        self.assertTrue(dnn.training)

    def test_update_bn_cnn(self):
        # Test update_bn for convolutional network and BatchNorm2d
        objects = 100
        input_channels = 3
        height, width = 5, 5
        x = torch.rand(objects, input_channels, height, width)
        y = torch.rand(objects)
        ds_x = torch.utils.data.TensorDataset(x)
        ds_xy = torch.utils.data.TensorDataset(x, y)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dl_xy = torch.utils.data.DataLoader(ds_xy, batch_size=5, shuffle=True)
        dnn = SWATestCNN(input_channels=input_channels)
        dnn.train()
        self._test_update_bn(dnn, dl_x, dl_xy, False)
        if torch.cuda.is_available():
            dnn = SWATestCNN(input_channels=input_channels)
            dnn.train()
            self._test_update_bn(dnn.cuda(), dl_x, dl_xy, True)
        self.assertTrue(dnn.training)

    def test_bn_update_eval_momentum(self):
        # check that update_bn preserves eval mode
        objects = 100
        input_channels = 3
        height, width = 5, 5
        x = torch.rand(objects, input_channels, height, width)
        ds_x = torch.utils.data.TensorDataset(x)
        dl_x = torch.utils.data.DataLoader(ds_x, batch_size=5, shuffle=True)
        dnn = SWATestCNN(input_channels=input_channels)
        dnn.eval()
        update_bn(dl_x, dnn)
        self.assertFalse(dnn.training)

        # check that momentum is preserved
        self.assertEqual(dnn.bn.momentum, 0.3)


if __name__ == '__main__':
    run_tests()
