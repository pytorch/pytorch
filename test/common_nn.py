import sys
import tempfile
import unittest
from copy import deepcopy
from itertools import product

import torch
import torch.cuda
from torch.autograd import Variable
from common import TestCase, to_gpu, freeze_rng_state
from torch.autograd.gradcheck import get_numerical_jacobian, iter_tensors, contiguous
import torch.backends.cudnn

# tarfile module tries to obtain a file object name in python 3.3
if sys.version_info[:2] == (3, 3):
    TemporaryFile = tempfile.NamedTemporaryFile
else:
    TemporaryFile = tempfile.TemporaryFile

TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = TEST_CUDA and torch.cuda.device_count() >= 2
TEST_CUDNN = TEST_CUDA and torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1))
TEST_CUDNN_VERSION = TEST_CUDNN and torch.backends.cudnn.version()
PRECISION = 1e-5

module_tests = [
    dict(
        module_name='Linear',
        constructor_args=(10, 8),
        input_size=(4, 10),
        reference_fn=lambda i, p: torch.mm(i, p[0].t()) + p[1].view(1, -1).expand(4, 8)
    ),
    dict(
        module_name='Linear',
        constructor_args=(10, 8, False),
        input_size=(4, 10),
        desc='no_bias',
        reference_fn=lambda i, p: torch.mm(i, p[0].t())
    ),
    dict(
        module_name='Threshold',
        constructor_args=(2, 1),
        input_size=(2, 3, 4, 5),
        check_inplace=True,
        desc='threshold_value'
    ),
    dict(
        module_name='Threshold',
        constructor_args=(2, 10),
        input_size=(2, 3, 4, 5),
        desc='large_value'
    ),
    dict(
        module_name='ReLU',
        input_size=(2, 3, 4, 5),
        check_inplace=True,
    ),
    dict(
        module_name='ReLU6',
        input_size=(2, 3, 4, 5),
        check_inplace=True,
    ),
    dict(
        module_name='RReLU',
        input_size=(1, 2, 2),
        test_cuda=False,
        check_gradgrad=False,
    ),
    dict(
        module_name='RReLU',
        constructor_args=(0.1, 0.9),
        input_size=(4, 4, 5),
        desc='with_up_down',
        test_cuda=False,
        check_gradgrad=False,
    ),
    dict(
        module_name='Hardtanh',
        input_size=(3, 2, 5),
        reference_fn=lambda i, _: i.clamp(-1, 1),
    ),
    dict(
        module_name='Sigmoid',
        input_size=(2, 3, 4, 5)
    ),
    dict(
        module_name='Tanh',
        input_size=(2, 3, 4, 5)
    ),
    dict(
        module_name='Softmax',
        input_size=(10, 20),
        reference_fn=lambda i, _: torch.exp(i).div(torch.exp(i).sum(1, True).expand(10, 20)),
    ),
    dict(
        module_name='Softmax2d',
        input_size=(1, 3, 10, 20),
        reference_fn=lambda i, _: torch.exp(i).div(torch.exp(i).sum(1, False)),
    ),
    dict(
        module_name='LogSoftmax',
        input_size=(10, 20),
        reference_fn=lambda i, _: torch.exp(i).div_(torch.exp(i).sum(1, True).expand(10, 20)).log_(),
    ),
    dict(
        module_name='LogSoftmax',
        input_size=(1, 3, 10, 20),
        reference_fn=lambda i, _: torch.exp(i).div_(torch.exp(i).sum(1, False)).log_(),
        desc='multiparam',
    ),
    dict(
        module_name='ELU',
        constructor_args=(2.,),
        input_size=(3, 2, 5),
    ),
    # TODO: reference function
    dict(
        module_name='Hardshrink',
        constructor_args=(2.,),
        input_size=(4, 3, 2, 4),
        check_gradgrad=False,
    ),
    dict(
        module_name='LeakyReLU',
        input_size=(3, 2, 5),
        check_inplace=True
    ),
    dict(
        module_name='LeakyReLU',
        constructor_args=(0.5,),
        input_size=(3, 2, 5),
        check_inplace=True,
        desc='with_negval'
    ),
    dict(
        module_name='LogSigmoid',
        input_size=(2, 3, 4),
        reference_fn=lambda i, _: i.sigmoid().log(),
        check_gradgrad=False,
    ),
    dict(
        module_name='Softplus',
        input_size=(10, 20),
        reference_fn=lambda i, _: torch.log(1 + torch.exp(i)),
        check_gradgrad=False,
    ),
    dict(
        module_name='Softplus',
        constructor_args=(2,),
        input_size=(10, 20),
        reference_fn=lambda i, _: 1. / 2. * torch.log(1 + torch.exp(2 * i)),
        desc='beta',
        check_gradgrad=False,
    ),
    dict(
        module_name='Softshrink',
        input_size=(3, 2, 5),
        check_gradgrad=False,
    ),
    dict(
        module_name='Softshrink',
        constructor_args=(1,),
        input_size=(3, 2, 5),
        desc='lambda',
        check_gradgrad=False,
    ),
    dict(
        module_name='CrossMapLRN2d',
        constructor_args=(5, 5e-3, 1e-3, 2),
        input_size=(2, 3, 6, 6),
        check_gradgrad=False,
    ),
    dict(
        module_name='PReLU',
        input_size=(2, 3, 4),
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
        desc='1d',
    ),
    dict(
        module_name='PReLU',
        constructor_args=(3,),
        input_size=(2, 3, 4),
        desc='1d_multiparam',
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='PReLU',
        input_size=(2, 3, 4, 5),
        desc='2d',
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='PReLU',
        constructor_args=(3,),
        input_size=(2, 3, 4, 5),
        desc='2d_multiparam',
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='PReLU',
        input_size=(2, 3, 4, 5, 6),
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
        desc='3d',
    ),
    dict(
        module_name='PReLU',
        constructor_args=(3,),
        input_size=(2, 3, 4, 5, 6),
        desc='3d_multiparam',
        reference_fn=lambda i, p: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0],
    ),
    dict(
        module_name='Softsign',
        input_size=(3, 2, 5),
        reference_fn=lambda i, _: i.div(1 + torch.abs(i)),
    ),
    dict(
        module_name='Softmin',
        input_size=(10, 20),
        check_gradgrad=False,
    ),
    dict(
        module_name='Tanhshrink',
        input_size=(2, 3, 4, 5)
    ),
]

criterion_tests = [
    dict(module_name='L1Loss',
         input_size=(2, 3, 4),
         target=torch.randn(2, 3, 4),
         reference_fn=lambda i, t, _: 1. / i.numel() *
         sum((a - b).abs().sum() for a, b in zip(i, t)),
         ),
    dict(
        module_name='NLLLoss',
        input=torch.rand(15, 10).log(),
        target=torch.Tensor(15).uniform_().mul(10).floor().long(),
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(None, False),
        input=torch.rand(15, 10).log(),
        target=torch.Tensor(15).uniform_().mul(10).floor().long(),
        desc='no_size_average'
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(None, True, 2),
        input=torch.rand(15, 10).log(),
        target=torch.Tensor(15).uniform_().mul(10).floor().long(),
        desc='ignore_index'
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(torch.rand(10),),
        input=torch.rand(15, 10).add(1e-2).log(),
        target=torch.Tensor(15).uniform_().mul(10).floor().long(),
        desc='weights',
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(torch.rand(10), True, 2),
        input=torch.rand(15, 10).add(1e-2).log(),
        target=torch.Tensor(15).uniform_().mul(10).floor().long(),
        desc='weights_ignore_index'
    ),
    dict(
        module_name='NLLLoss',
        constructor_args=(torch.rand(10), True, -1),
        input=torch.rand(15, 10).add(1e-2).log(),
        target=torch.Tensor(15).uniform_().mul(10 + 1).floor().long() - 1,
        desc='weights_ignore_index_neg'
    ),
    dict(
        module_name='KLDivLoss',
        input=torch.rand(10, 10).log(),
        target=torch.rand(10, 10),
        check_gradgrad=False,
    ),
    dict(
        module_name='MSELoss',
        input=torch.randn(2, 3, 4, 5),
        target=torch.randn(2, 3, 4, 5),
        reference_fn=lambda i, t, _: (i - t).abs().pow(2).sum() / i.numel(),
        check_gradgrad=False,
    ),
    dict(
        module_name='BCELoss',
        input=torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target=torch.randn(15, 10).gt(0).double(),
        check_gradgrad=False,
    ),
    dict(
        module_name='BCELoss',
        constructor_args=(torch.rand(10),),
        input=torch.rand(15, 10).clamp_(1e-2, 1 - 1e-2),
        target=torch.randn(15, 10).gt(0).double(),
        desc='weights',
        check_gradgrad=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        input=torch.randn(15, 10),
        target=torch.Tensor(15).uniform_().mul(10).floor().long(),
        check_gradgrad=False,
    ),
    dict(
        module_name='CrossEntropyLoss',
        constructor_args=(torch.rand(10),),
        input=torch.randn(15, 10),
        target=torch.Tensor(15).uniform_().mul(10).floor().long(),
        desc='weights',
        check_gradgrad=False,
    ),
    dict(
        module_name='NLLLoss2d',
        input_size=(2, 3, 5, 5),
        target=torch.rand(2, 5, 5).mul(3).floor().long(),
    ),
    dict(
        module_name='NLLLoss2d',
        constructor_args=(torch.rand(3),),
        input_size=(2, 3, 5, 5),
        target=torch.rand(2, 5, 5).mul(3).floor().long(),
        desc='weights',
    ),
    dict(
        module_name='NLLLoss2d',
        constructor_args=(None, True, 3),
        input_size=(2, 3, 5, 5),
        target=torch.rand(2, 5, 5).mul(4).floor().long(),
        desc='ignore_index',
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        input=torch.rand(10),
        target=torch.randn(10).gt(0).double().mul_(2).sub(1),
        check_gradgrad=False,
    ),
    dict(
        module_name='HingeEmbeddingLoss',
        constructor_args=(0.5,),
        input=torch.rand(10),
        target=torch.randn(10).gt(0).double().mul_(2).sub(1),
        desc='margin',
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiLabelMarginLoss',
        input_size=(5, 10),
        target=torch.rand(5, 10).mul(10).floor().long(),
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiLabelSoftMarginLoss',
        input_size=(5, 10),
        target=torch.rand(5, 10).mul(2).floor(),
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiLabelSoftMarginLoss',
        constructor_args=(torch.rand(10),),
        input_size=(5, 10),
        target=torch.rand(5, 10).mul(2).floor(),
        desc='weights',
        check_gradgrad=False,
    ),
    dict(
        module_name='MultiMarginLoss',
        input_size=(5, 10),
        target=torch.rand(5).mul(8).floor().long(),
        check_gradgrad=False,
    ),
    dict(
        module_name='SmoothL1Loss',
        input_size=(5, 10),
        target=torch.randn(5, 10),
        check_gradgrad=False,
    ),
    dict(
        module_name='SoftMarginLoss',
        input_size=(5, 5),
        target=torch.randn(5, 5).sign(),
        check_gradgrad=False,
    ),
    dict(
        module_name='CosineEmbeddingLoss',
        input=(torch.rand(15, 10), torch.rand(15, 10)),
        target=torch.randn(15).sign(),
        check_gradgrad=False,
    ),
    dict(
        module_name='CosineEmbeddingLoss',
        constructor_args=(0.7,),
        input=(torch.rand(15, 10), torch.rand(15, 10)),
        target=torch.randn(15).sign(),
        desc='margin',
        check_gradgrad=False,
    ),
    dict(
        module_name='MarginRankingLoss',
        input=(torch.randn(50).mul(10), torch.randn(50).mul(10)),
        target=torch.randn(50).sign(),
        check_gradgrad=False,
    ),
    dict(
        module_name='MarginRankingLoss',
        constructor_args=(2,),
        input=(torch.randn(50).mul(10), torch.randn(50).mul(10)),
        target=torch.randn(50).sign(),
        desc='margin',
        check_gradgrad=False,
    ),
]


class NNTestCase(TestCase):

    def _jacobian(self, input, num_out):
        if isinstance(input, tuple):
            return tuple(self._jacobian(elem, num_out) for elem in input)
        elif isinstance(input, list):
            return [self._jacobian(elem, num_out) for elem in input]
        else:
            return torch.zeros(input.nelement(), num_out)

    def _flatten_tensors(self, x):
        if torch.is_tensor(x):
            if x.is_sparse:
                return x.to_dense().view(-1)
            else:
                return x.view(-1)
        elif isinstance(x, Variable):
            return self._flatten_tensors(x.data)
        else:
            return tuple(self._flatten_tensors(a) for a in x)

    def _zero_grad_input(self, input):
        if isinstance(input, Variable):
            if input.requires_grad and input.grad is not None:
                input.grad.data.zero_()
                input.grad.detach_()
        elif torch.is_tensor(input):
            return
        else:
            for i in input:
                self._zero_grad_input(i)

    def _analytical_jacobian(self, module, input, jacobian_input=True, jacobian_parameters=True):
        output = self._forward(module, input)
        output_t = output.data if isinstance(output, Variable) else output
        d_out = output_t.new().resize_(output_t.size())
        flat_d_out = d_out.view(-1)

        if jacobian_input:
            jacobian_inp = self._jacobian(input, d_out.nelement())
            flat_jacobian_input = list(iter_tensors(jacobian_inp))

        if jacobian_parameters:
            param, d_param = self._get_parameters(module)
            num_param = sum(p.numel() for p in param)
            jacobian_param = torch.zeros(num_param, d_out.nelement())

        for i in range(flat_d_out.nelement()):
            d_out.zero_()
            flat_d_out[i] = 1

            if jacobian_parameters:
                self._zero_grad_parameters(module)
            # Variables will accumulate gradient from multiple steps
            if jacobian_input:
                self._zero_grad_input(input)
            d_input = self._backward(module, input, output, d_out)

            if jacobian_input:
                for jacobian_x, d_x in zip(flat_jacobian_input, iter_tensors(d_input)):
                    jacobian_x[:, i] = d_x
            if jacobian_parameters:
                jacobian_param[:, i] = torch.cat(self._flatten_tensors(d_param), 0)

        res = tuple()
        if jacobian_input:
            res += jacobian_inp,
        if jacobian_parameters:
            res += jacobian_param,

        return res

    def _numerical_jacobian(self, module, input, jacobian_input=True, jacobian_parameters=True):
        output = self._forward(module, input)
        output_size = output.nelement()

        if jacobian_parameters:
            param, d_param = self._get_parameters(module)

        def fw(input):
            out = self._forward(module, input)
            if isinstance(out, Variable):
                return out.data
            return out

        res = tuple()
        input = contiguous(input)
        if jacobian_input:
            res += get_numerical_jacobian(fw, input, input, eps=1e-6),
        if jacobian_parameters:
            res += torch.cat(list(get_numerical_jacobian(fw, input, p, eps=1e-6) for p in param), 0),
        return res

    def check_jacobian(self, module, input, jacobian_input=True):
        jacobian_parameters = bool(self._get_parameters(module)[0])
        analytical = self._analytical_jacobian(module, input, jacobian_input, jacobian_parameters)
        numerical = self._numerical_jacobian(module, input, jacobian_input, jacobian_parameters)
        analytical_t = iter_tensors(analytical)
        numerical_t = iter_tensors(numerical)
        # TODO: compare structure
        self.assertLessEqual(
            max(a.add(-1, n).abs().max() for a, n in zip(analytical_t, numerical_t)),
            PRECISION
        )

    def check_criterion_jacobian(self, criterion, input, target):
        eps = 1e-6
        self._forward_criterion(criterion, input, target)
        analytical_d_x = self._backward_criterion(criterion, input, target)
        numerical_d_x = deepcopy(analytical_d_x)

        input_t = iter_tensors(input)
        numerical_t = iter_tensors(numerical_d_x)
        for x, d_x in zip(input_t, numerical_t):
            x = x.view(-1)
            d_x = d_x.view(-1)
            for i in range(x.nelement()):
                original = x[i]
                x[i] = original + eps
                fx1 = self._forward_criterion(criterion, input, target)
                x[i] = original - eps
                fx2 = self._forward_criterion(criterion, input, target)
                deriv = (fx1 - fx2) / (2. * eps)
                d_x[i] = deriv
                x[i] = original

        # TODO: check structure
        analytical_t = iter_tensors(analytical_d_x)
        numerical_t = iter_tensors(numerical_d_x)
        self.assertLessEqual(
            max(a.add(-1, n).abs().max() for a, n in zip(analytical_t, numerical_t)),
            PRECISION
        )


class TestBase(object):

    def __init__(self, constructor, constructor_args=tuple(), input_size=None,
                 input=None, desc='', reference_fn=None, fullname=None, **kwargs):
        if input_size is None and input is None:
            raise RuntimeError("Specify either an input tensor, or it's size!")
        self.constructor = constructor
        self.constructor_args = constructor_args
        self.input = input
        self.input_size = input_size
        self.desc = desc
        self.fullname = fullname
        self.reference_fn = reference_fn

    def get_name(self):
        if self.fullname is not None:
            return 'test_' + self.fullname

        test_name = 'test_' + self.constructor.__name__
        if self.desc:
            test_name += '_' + self.desc
        return test_name

    def _unpack_input(self, input):
        if isinstance(input, Variable):
            return input.data
        elif torch.is_tensor(input):
            return input
        else:
            return type(input)(self._unpack_input(i) for i in input)

    def _get_input(self):
        if self.input is not None:
            return self.input

        def map_input_sizes(sizes):
            if isinstance(sizes, list):
                return [map_input_sizes(s) for s in sizes]
            elif torch.is_tensor(sizes):
                return sizes.double()
            else:
                return torch.randn(*sizes)

        assert self.input_size is not None
        return map_input_sizes(self.input_size)

    def __call__(self, test_case):
        raise NotImplementedError


class ModuleTest(TestBase):

    def __init__(self, *args, **kwargs):
        super(ModuleTest, self).__init__(*args, **kwargs)
        self.jacobian_input = kwargs.get('jacobian_input', True)
        self.should_test_cuda = kwargs.get('test_cuda', True)

    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()

        if self.reference_fn is not None:
            out = test_case._forward(module, input)
            if isinstance(out, Variable):
                out = out.data
            ref_input = self._unpack_input(deepcopy(input))
            expected_out = self.reference_fn(ref_input, test_case._get_parameters(module)[0])
            test_case.assertEqual(out, expected_out)

        self.test_noncontig(test_case, module, input)

        # TODO: do this with in-memory files as soon as torch.save will support it
        with TemporaryFile() as f:
            test_case._forward(module, input)
            torch.save(module, f)
            f.seek(0)
            module_copy = torch.load(f)
            test_case.assertEqual(test_case._forward(module, input), test_case._forward(module_copy, input))

        self._do_test(test_case, module, input)

    def noncontiguize(self, obj):
        if isinstance(obj, list):
            return [self.noncontiguize(o) for o in obj]
        tensor = obj.data if isinstance(obj, Variable) else obj
        ndim = tensor.dim()
        noncontig = torch.stack([tensor.clone().zero_(), tensor], ndim).select(ndim, 1)
        assert noncontig.numel() == 1 or not noncontig.is_contiguous()
        if isinstance(obj, Variable):
            return Variable(noncontig, requires_grad=obj.requires_grad)
        return noncontig

    def test_noncontig(self, test_case, module, input):
        test_case._zero_grad_parameters(module)
        test_case._zero_grad_input(input)
        with freeze_rng_state():
            output = test_case._forward(module, input)
            grad_output = output
            if isinstance(grad_output, Variable):
                grad_output = grad_output.data.clone()
            else:
                grad_output = grad_output.clone()
                output = output.clone()
            grad_output.normal_()
            d_input = deepcopy(test_case._backward(module, input, output, grad_output))
            d_param = deepcopy(test_case._get_parameters(module)[1])

        nc_input = self.noncontiguize(input)
        nc_grad_output = self.noncontiguize(grad_output)
        for contig_i, contig_g in product((True, False), repeat=2):
            i = input if contig_i else nc_input
            go = grad_output if contig_g else nc_grad_output
            test_case._zero_grad_parameters(module)
            test_case._zero_grad_input(i)
            with freeze_rng_state():
                try:
                    out = test_case._forward(module, i)
                except Exception:
                    # Some modules will fail because of non contiguous inputs and we're ok with that
                    continue
                grad = test_case._backward(module, i, out, go)

                test_case.assertEqual(out, output)
                test_case.assertEqual(grad, d_input, 1e-4)
                test_case.assertEqual(test_case._get_parameters(module)[1], d_param)

    def test_cuda(self, test_case):
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            type_map = {torch.DoubleTensor: torch.cuda.FloatTensor}
            gpu_input = to_gpu(cpu_input, type_map=type_map)

            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).float().cuda()
            cpu_param = test_case._get_parameters(cpu_module)
            gpu_param = test_case._get_parameters(gpu_module)
            for cpu_p, gpu_p in zip(cpu_param[0], gpu_param[0]):
                if isinstance(cpu_p, Variable):
                    cpu_p = cpu_p.data
                if isinstance(gpu_p, Variable):
                    gpu_p = gpu_p.data
                gpu_p.copy_(cpu_p)

            test_case._zero_grad_input(cpu_input)
            test_case._zero_grad_input(gpu_input)
            test_case._zero_grad_parameters(cpu_module)
            test_case._zero_grad_parameters(gpu_module)
            cpu_output = test_case._forward(cpu_module, cpu_input)
            gpu_output = test_case._forward(gpu_module, gpu_input)
            test_case.assertEqual(cpu_output, gpu_output, 2e-4)

            for i in range(5):
                cpu_output_t = cpu_output.data if isinstance(cpu_output, Variable) else cpu_output
                cpu_gradOutput = cpu_output_t.clone().bernoulli_()
                gpu_gradOutput = cpu_gradOutput.type('torch.cuda.FloatTensor')
                cpu_gradInput = test_case._backward(cpu_module, cpu_input, cpu_output, cpu_gradOutput)
                gpu_gradInput = test_case._backward(gpu_module, gpu_input, gpu_output, gpu_gradOutput)
                test_case.assertEqual(cpu_gradInput, gpu_gradInput, 2e-4)
                for cpu_d_p, gpu_d_p in zip(cpu_param[1], gpu_param[1]):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, 2e-4)

            self.test_noncontig(test_case, gpu_module, gpu_input)
        except NotImplementedError:
            pass
        # TODO: remove this after CUDA scatter_ is implemented
        except AttributeError as e:
            if len(e.args) == 1 and "'FloatTensor' object has no attribute 'scatter_'" in e.args[0]:
                pass
            else:
                raise


class CriterionTest(TestBase):

    def __init__(self, *args, **kwargs):
        super(CriterionTest, self).__init__(*args, **kwargs)
        self.target = self._get_target(kwargs['target'])
        self.should_test_cuda = kwargs.get('test_cuda', True)

    def _get_target(self, target):
        return target

    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()

        # Check that these methods don't raise errors
        module.__repr__()
        str(module)

        if self.reference_fn is not None:
            out = test_case._forward_criterion(module, input, self.target)
            target = self.target
            if isinstance(target, Variable):
                target = target.data
            expected_out = self.reference_fn(deepcopy(self._unpack_input(input)),
                                             deepcopy(target), module)
            test_case.assertEqual(out, expected_out)

        test_case.check_criterion_jacobian(module, input, self.target)
        self._do_extra_tests(test_case, module, input, self.target)

    def test_cuda(self, test_case):
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            type_map = {
                torch.DoubleTensor: torch.cuda.FloatTensor,
            }
            gpu_input = to_gpu(cpu_input, type_map=type_map)

            cpu_target = self.target
            gpu_target = to_gpu(self.target, type_map=type_map)

            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).float().cuda()

            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target)
            test_case.assertEqual(cpu_output, gpu_output, 4e-4)

            cpu_gradInput = test_case._backward_criterion(cpu_module, cpu_input, cpu_target)
            gpu_gradInput = test_case._backward_criterion(gpu_module, gpu_input, gpu_target)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, 4e-4)
        except NotImplementedError:
            pass

    def _do_extra_tests(self, test_case, module, input, target):
        pass
