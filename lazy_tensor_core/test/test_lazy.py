from functools import partial
from typing import Sequence
import torch

from torch.testing._internal.common_utils import clone_input_helper, run_tests, TestCase
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import OpDTypes, ops, instantiate_device_type_tests

import lazy_tensor_core
lazy_tensor_core._LAZYC._ltc_init_ts_backend()
import lazy_tensor_core.core.lazy_model as ltm
import lazy_tensor_core.debug.metrics as metrics

torch.manual_seed(42)

# class TestLazyTensor(JitTestCase):
#     def testConvolutionBackward(self):
#         def clone_move(t):
#             dev = 'lazy'
#             copy_t = t.detach().clone().requires_grad_(True).to(device=dev)
#             return copy_t

#         inp = torch.rand(1, 3, 128, 128, device='cuda', requires_grad=True)
#         inp_copy = clone_move(inp)
#         grad = torch.rand(1, 32, 121, 121, device='cuda')  # no requires_grad
#         grad_copy = clone_move(grad)
#         weight = torch.rand(32, 3, 8, 8, device='cuda', requires_grad=True)
#         weight_copy = clone_move(weight)
#         bias = torch.rand(32, device='cuda', requires_grad=True)
#         bias_copy = clone_move(bias)

#         # run eager
#         conv_out = torch.nn.functional.conv2d(inp, weight, bias)
#         (inp_grad, weight_grad, bias_grad) = torch.autograd.grad([conv_out], [inp, weight, bias], [grad])

#         # run lazy
#         conv_copy_out = torch.nn.functional.conv2d(inp_copy, weight_copy, bias_copy)
#         (inp_copy_grad, weight_copy_grad, bias_copy_grad) = torch.autograd.grad(
#             [conv_copy_out], [inp_copy, weight_copy, bias_copy], [grad_copy])

#         jit_graph = lazy_tensor_core._LAZYC._get_ltc_tensors_backend([bias_copy_grad])

#         # check numerics
#         torch.testing.assert_allclose(bias_copy_grad.cpu(), bias_grad.cpu())
#         torch.testing.assert_allclose(weight_copy_grad.cpu(), weight_grad.cpu())
#         torch.testing.assert_allclose(inp_copy_grad.cpu(), inp_grad.cpu())



bad_ops_list = set([
    'index_select', # SEGFAULT
    'linalg.pinv', # lazy_tensor Input tensor is not a lazy tensor: CPUFloatType
    'nn.functional.group_norm' # lazy_tensor Input tensor is not a lazy tensor: CPUFloatType
])

lazy_ops_list = [
    '_log_softmax',
    '_log_softmax_backward_data',
    '_softmax',
    '_softmax_backward_data',
    'add.Tensor',
    'addcdiv',
    'addcmul',
    'addmm',
    'avg_pool2d',
    'avg_pool2d_backward',
    'baddbmm',
    'binary_cross_entropy',
    'binary_cross_entropy_backward',
    'bitwise_and.Tensor',
    'bmm',
    'constant_pad_nd',
    'convolution',
    'convolution_backward',
    'cos',
    'clamp',
    'div.Tensor',
    'div.Tensor_mode',
    'elu',
    'elu_backward',
    'embedding',
    'embedding_dense_backward',
    'eq.Scalar',
    'eq.Tensor',
    'exp',
    'floor',
    'frac',
    'ge.Scalar',
    'ge.Tensor',
    'gelu',
    'gelu_backward',
    'gt.Scalar',
    'gt.Tensor',
    'index_select',
    'kl_div_backward',
    'le.Scalar',
    'le.Tensor',
    'leaky_relu',
    'leaky_relu_backward',
    'log2',
    'logdet',
    'log_sigmoid_backward',
    'log_sigmoid_forward',
    'lt.Scalar',
    'lt.Tensor',
    'masked_fill_.Scalar',
    'masked_fill_.Tensor',
    'max_pool2d_with_indices',
    'max_pool2d_with_indices_backward',
    'mean',
    'mean.dim',
    'mm',
    'mul.Tensor',
    'mv',
    'native_dropout',
    'native_dropout_backward',
    'native_layer_norm',
    'native_layer_norm_backward',
    'ne.Scalar',
    'ne.Tensor',
    'nll_loss_backward',
    'nll_loss_forward',
    'nll_loss2d_backward',
    'nll_loss2d_forward',
    'pow.Tensor_Scalar',
    'pow.Tensor_Tensor',
    'relu',
    'relu_',
    'rsqrt',
    'sigmoid',
    'sigmoid_backward',
    'silu',
    'smooth_l1_loss',
    'smooth_l1_loss_backward',
    'softplus',
    'softplus_backward',
    'sort',
    'sqrt',
    'std',
    'std.dim',
    'std.correction',
    'sum',
    'sum.dim_IntList',
    'tanh',
    'tanh_backward',
    'threshold',
    'threshold_backward',
    'topk',
    'trace',
    'triu',
    'trunc',
    'zero_',
    'any',
    'bitwise_or.Tensor',
    'log',
    'max.dim',
    'maximum',
    'minimum',
    'neg',
    'remainder.Tensor',
    'upsample_bilinear2d',
    'upsample_nearest2d',
    'upsample_nearest2d_backward',
    'as_strided',
    'as_strided_',
    'bernoulli',
    'bernoulli_.float',
    'cat',
    'clone',
    '_copy_from',
    '_copy_from_and_resize',
    'empty.memory_format',
    'empty_strided',
    'expand',
    'fill_.Scalar',
    'native_batch_norm',
    'native_batch_norm_backward',
    'normal_',
    'max_pool3d_with_indices',
    'max_pool3d_with_indices_backward',
    'permute',
    'random_',
    'random_.from',
    'random_.to',
    'repeat',
    'select.int',
    'slice.Tensor',
    'squeeze',
    'squeeze.dim',
    'squeeze_',
    'squeeze_.dim',
    'stack',
    't',
    't_',
    'transpose.int',
    'transpose_',
    'unsqueeze',
    'unsqueeze_',
    'sub.Tensor',
    'sub.Scalar',
    'view',
    'alias',
    'max_pool3d'
]


def get_name(op):
    l = [op.name]
    if op.variant_test_name != '':
        l.append(op.variant_test_name)
    return '.'.join(l)

def clone_input_helper(input, fn):
    if isinstance(input, torch.Tensor):
        return fn(input)

    if isinstance(input, Sequence):
        return tuple(map(partial(clone_input_helper, fn=fn), input))

    return input

class TestLazyOpInfo(TestCase):


    @ops([op for op in op_db if get_name(op) not in bad_ops_list], allowed_dtypes=(torch.float,))
    def test_correctness(self, device, dtype, op):
        print(f"op = {get_name(op)} {get_name(op)=='linalg.pinv'}")
        samples = op.sample_inputs("lazy", dtype, requires_grad=False)
        sample = list(samples)[0]
        args = [sample.input] + list(sample.args)
        kwargs = sample.kwargs
        lazy_r = op(*args, **kwargs)
        ltm.mark_step()
        ltm.wait_device_ops()
        only_t_lazy_r = []
        clone_input_helper(lazy_r, lambda x: only_t_lazy_r.append(x.to(device="cpu")))

        cpu_args = clone_input_helper(args, lambda x : x.detach().to(device="cpu"))
        cpu_kwargs = clone_input_helper(kwargs, lambda x : x.detach().to(device="cpu"))
        cpu_r = op(*cpu_args, **cpu_kwargs)
        only_t_cpu_r = []
        clone_input_helper(cpu_r, lambda x: only_t_cpu_r.append(x))

        for (l, c) in zip(only_t_lazy_r, only_t_cpu_r):
            self.assertTrue(torch.allclose(l, c))


    @ops([op for op in op_db if get_name(op) in lazy_ops_list and get_name(op) not in bad_ops_list], allowed_dtypes=(torch.float,))
    def test_dispatched_to_lazy(self, device, dtype, op):
        print(get_name(op))

        samples = op.sample_inputs("lazy", dtype, requires_grad=False)
        sample = list(samples)[0]
        args = [sample.input] + list(sample.args)
        kwargs = sample.kwargs
        ltm.mark_step()
        ltm.wait_device_ops()
        metrics.reset_metrics()
        print(type(args[0]).__name__)
        r = op(*args, **kwargs)
        ltm.mark_step()
        ltm.wait_device_ops()
        print(metrics.metric_names())

# only_for = "lazy" doesn't work for some reason
instantiate_device_type_tests(TestLazyOpInfo, globals(), only_for="cpu")


if __name__ == '__main__':
    run_tests()
