import unittest
from copy import deepcopy

import torch
from torch.autograd import Variable
from common import TestCase, to_gpu, get_numerical_jacobian, iter_tensors, contiguous

try:
    import torch.cuda
    import torch.legacy.cunn
    TEST_CUDA = True
except Exception:
    # TODO: catch ImportError once it works with "setup.py develop"
    TEST_CUDA = False

PRECISION = 1e-5

module_tests = [
    dict(
        module_name='Linear',
        constructor_args=(10, 8),
        input_size=(4, 10),
        reference_fn=lambda i,p: torch.mm(i, p[0].t()) + p[1].view(1, -1).expand(4, 8)
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
        check_inplace=True
    ),
    dict(
        module_name='ReLU6',
        input_size=(2, 3, 4, 5),
        check_inplace=True
    ),
    dict(
        module_name='HardTanh',
        input_size=(3, 2, 5),
        reference_fn=lambda i,_: i.clamp(-1, 1)
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
        reference_fn=lambda i,_: torch.exp(i).div(torch.exp(i).sum(1).expand(10, 20))
    ),
    dict(
        module_name='Softmax2d',
        input_size=(1, 3, 10, 20),
        reference_fn=lambda i,_: torch.exp(i).div(torch.exp(i).sum(1).expandAs(i))
    ),
    dict(
        module_name='BatchNorm',
        constructor_args=(10,),
        input_size=(4, 10),
        desc='affine'
    ),
    dict(
        module_name='BatchNorm',
        constructor_args=(10, 1e-3, 0.3, False),
        input_size=(4, 10),
        desc='not_affine'
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3,),
        input_size=(2, 3, 6, 6),
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, 0.8),
        input_size=(2, 3, 6, 6),
        desc='momentum',
    ),
    dict(
        module_name='BatchNorm2d',
        constructor_args=(3, 1e-3, 0.8, False),
        input_size=(2, 3, 6, 6),
        desc='no_affine',
    ),
    dict(
        module_name='LogSoftmax',
        input_size=(10, 20),
        reference_fn=lambda i,_: torch.exp(i).div_(torch.exp(i).sum(1).expand(10, 20)).log_()
    ),
]


criterion_tests = [
    dict(module_name='AbsCriterion',
        input_size=(2, 3, 4),
        target=torch.randn(2, 3, 4),
        reference_fn=lambda i,t,_: 1./i.numel() * \
            sum((a-b).abs().sum() for a,b in zip(i, t))
    ),
    dict(
        module_name='ClassNLLCriterion',
        input=torch.rand(15, 10).log(),
        target=torch.Tensor(15).uniform_().mul(10).floor().long(),
    ),
    dict(
        module_name='ClassNLLCriterion',
        constructor_args=(torch.rand(10),),
        input=torch.rand(15, 10).add(1e-2).log(),
        target=torch.Tensor(15).uniform_().mul(10).floor().long(),
        desc='weights',
    ),
]


class NNTestCase(TestCase):

    def _jacobian(self, input, num_out):
        if isinstance(input, tuple):
            return tuple(self._jacobian(elem, num_out) for elem in input)
        elif isinstance(input, list):
            return [self._jacobian(elem, num_out) for elem in input]
        else:
            return torch.zeros(input.nElement(), num_out)

    def _flatten_tensors(self, x):
        if torch.isTensor(x):
            return x.view(-1)
        elif isinstance(x, Variable):
            return x.data.view(-1)
        else:
            return tuple(self._flatten_tensors(a) for a in x)

    def _zero_grad_input(self, input):
        if isinstance(input, Variable):
            input.grad.zero_()
        elif torch.isTensor(input):
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
            jacobian_input = self._jacobian(input, d_out.nElement())
            flat_jacobian_input = list(iter_tensors(jacobian_input))

        if jacobian_parameters:
            param, d_param = self._get_parameters(module)
            num_param = sum(p.numel() for p in param)
            jacobian_param = torch.zeros(num_param, d_out.nElement())

        for i in range(flat_d_out.nElement()):
            d_out.zero_()
            flat_d_out[i] = 1

            if jacobian_parameters:
                self._zero_grad_parameters(module)
            # Variables will accumulate gradient from multiple steps
            self._zero_grad_input(input)
            d_input = self._backward(module, input, output, d_out)

            if jacobian_input:
                for jacobian_x, d_x in zip(flat_jacobian_input, iter_tensors(d_input)):
                    jacobian_x[:,i] = d_x
            if jacobian_parameters:
                jacobian_param[:,i] = torch.cat(self._flatten_tensors(d_param), 0)

        res = tuple()
        if jacobian_input:
            res += jacobian_input,
        if jacobian_parameters:
            res += jacobian_param,

        return res

    def _numerical_jacobian(self, module, input, jacobian_input=True, jacobian_parameters=True):
        output = self._forward(module, input)
        output_size = output.nElement()

        if jacobian_parameters:
            param, d_param = self._get_parameters(module)

        def fw(input):
            out = self._forward(module, input)
            if isinstance(out, Variable):
                return out.data
            return out

        res = tuple()
        # TODO: enable non-contig tests
        input = contiguous(input)
        if jacobian_input:
            res += get_numerical_jacobian(fw, input, input),
        if jacobian_parameters:
            res += torch.cat(list(get_numerical_jacobian(fw, input, p) for p in param), 0),
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
            for i in range(x.nElement()):
                original = x[i]
                x[i] = original + eps
                fx1 = self._forward_criterion(criterion, input, target)
                x[i] = original - eps
                fx2 = self._forward_criterion(criterion, input, target)
                deriv = (fx1 - fx2) / (2.*eps)
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
        elif torch.isTensor(input):
            return input
        else:
            return type(input)(self._unpack_input(i) for i in input)

    def _get_input(self):
        if self.input is not None:
            return self.input

        def map_input_sizes(sizes):
            if isinstance(sizes, list):
                return [map_input_sizes(s) for s in sizes]
            elif torch.isTensor(sizes):
                return sizes
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

        self._do_test(test_case, module, input)

    def test_cuda(self, test_case):
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            gpu_input = to_gpu(cpu_input, tensor_type=torch.cuda.FloatTensor)

            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).cuda()
            test_case._zero_grad_parameters(cpu_module)
            test_case._zero_grad_parameters(gpu_module)
            cpu_param = test_case._get_parameters(cpu_module)
            gpu_param = test_case._get_parameters(gpu_module)
            for cpu_p, gpu_p in zip(cpu_param[0], gpu_param[0]):
                if isinstance(cpu_p, Variable):
                    cpu_p = cpu_p.data
                if isinstance(gpu_p, Variable):
                    gpu_p = gpu_p.data
                gpu_p.copy_(cpu_p)

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
        self.target = kwargs.get('target', None)
        self.should_test_cuda = kwargs.get('test_cuda', True)

    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()

        if self.reference_fn is not None:
            out = test_case._forward_criterion(module, input, self.target)
            expected_out = self.reference_fn(deepcopy(self._unpack_input(input)),
                    deepcopy(self.target), module)
            test_case.assertEqual(out, expected_out)

        test_case.check_criterion_jacobian(module, input, self.target)

    def test_cuda(self, test_case):
        if not TEST_CUDA or not self.should_test_cuda:
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            gpu_input = to_gpu(cpu_input, tensor_type=torch.cuda.FloatTensor)

            cpu_target = self.target
            gpu_target = to_gpu(self.target, tensor_type=torch.cuda.FloatTensor)

            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).cuda()

            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target)
            test_case.assertEqual(cpu_output, gpu_output, 2e-4)

            cpu_gradInput = test_case._backward_criterion(cpu_module, cpu_input, cpu_target)
            gpu_gradInput = test_case._backward_criterion(gpu_module, gpu_input, gpu_target)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, 2e-4)
        except NotImplementedError:
            pass
