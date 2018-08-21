import math
import random
import unittest
import collections
from copy import deepcopy

import torch
import torch.legacy.nn as nn
from common import to_gpu, freeze_rng_state, run_tests, skipIfRocm, TEST_WITH_ROCM
from common_nn import NNTestCase, ModuleTest, CriterionTest, iter_tensors, \
    module_tests, criterion_tests, PRECISION
from torch.autograd.gradcheck import get_numerical_jacobian
from torch.autograd import Variable


class OldModuleTest(ModuleTest):

    def __init__(self, *args, **kwargs):
        super(OldModuleTest, self).__init__(*args, **kwargs)
        self.check_inplace = kwargs.get('check_inplace', False)
        # Never check gradgrad for legacy NN
        self.check_gradgrad = False

    def _do_test(self, test_case, module, input):
        # TODO: check update parameters
        # TODO: test IO
        module.training()
        with torch.no_grad():
            test_case.check_jacobian(module, input, self.jacobian_input)
        module.evaluate()
        with torch.no_grad():
            test_case.check_jacobian(module, input, self.jacobian_input)

        # Test .type()
        module.float().double().forward(input)

        # Test .clearState()
        module.clearState()

        # test if module can be printed
        module.__repr__()

        if self.check_inplace:
            input2 = deepcopy(input)
            module_ip = self.constructor(*self.constructor_args, inplace=True)
            with freeze_rng_state():
                output = module.forward(input)
            test_case.assertEqual(input, input2)
            with freeze_rng_state():
                output2 = module_ip.forward(input2)
            if not torch.equal(output, input):
                test_case.assertNotEqual(input, input2)
            test_case.assertEqual(output, input2)

# TODO: hessian tests
tests = [
    OldModuleTest(nn.Add,
                  constructor_args=(torch.Size([5, 4]),),
                  input_size=(3, 5, 4),
                  desc='3D'),
    OldModuleTest(nn.Add,
                  constructor_args=(1, True),
                  input_size=(3, 1, 4),
                  desc='scalar'),
    OldModuleTest(nn.AddConstant,
                  constructor_args=(3.5,),
                  input_size=(3, 5, 4),
                  reference_fn=lambda i, _: i + 3.5,
                  check_inplace=True,
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.BatchNormalization,
                  constructor_args=(10,),
                  input_size=(4, 10),
                  desc='affine',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.BatchNormalization,
                  constructor_args=(10, 1e-3, 0.3, False),
                  input_size=(4, 10),
                  desc='not_affine',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.SpatialBatchNormalization,
                  constructor_args=(3,),
                  input_size=(2, 3, 6, 6),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.SpatialBatchNormalization,
                  constructor_args=(3, 1e-3, 0.8),
                  input_size=(2, 3, 6, 6),
                  desc='momentum',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.SpatialBatchNormalization,
                  constructor_args=(3, 1e-3, 0.8, False),
                  input_size=(2, 3, 6, 6),
                  desc='no_affine'),
    OldModuleTest(nn.VolumetricBatchNormalization,
                  constructor_args=(3,),
                  input_size=(2, 3, 4, 4, 4),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.VolumetricBatchNormalization,
                  constructor_args=(3, 1e-3, 0.7),
                  input_size=(2, 3, 4, 4, 4),
                  desc='momentum',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.VolumetricBatchNormalization,
                  constructor_args=(3, 1e-3, 0.7, False),
                  input_size=(2, 3, 4, 4, 4),
                  desc='no_affine'),
    OldModuleTest(nn.CMul,
                  constructor_args=(5, 6),
                  input_size=(10, 5, 6),
                  desc='3D',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.CMul,
                  constructor_args=(50, 4),
                  input_size=(1, 50, 4),
                  desc='3D_single_example',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.CMul,
                  constructor_args=(1, 5),
                  input_fn=lambda: torch.randn(10, 3, 5)[:, 1],
                  desc='3D_noncontiguous',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Exp,
                  input_size=(2, 3, 4),
                  reference_fn=lambda i, _: i.exp(),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Log,
                  input_fn=lambda: torch.rand(2, 3, 2) + 0.1,
                  reference_fn=lambda i, _: i.log(),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Clamp,
                  constructor_args=(-2., 5.),
                  input_fn=lambda: torch.randn(3, 2, 50) * 6,
                  reference_fn=lambda i, _: i.clamp(-2, 5)),
    OldModuleTest(nn.Abs,
                  input_size=(3, 20, 5),
                  reference_fn=lambda i, _: i.abs(),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Bilinear,
                  constructor_args=(2, 3, 10),
                  input_size=[(4, 2), (4, 3)],
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Bilinear,
                  constructor_args=(5, 4, 2),
                  input_size=[(2, 5), (2, 4)],
                  desc='small_output',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Euclidean,
                  constructor_args=(5, 7),
                  input_size=(10, 5),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.WeightedEuclidean,
                  constructor_args=(5, 7),
                  input_size=(10, 5),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Cosine,
                  constructor_args=(5, 7),
                  input_size=(10, 5),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.CAddTable,
                  input_size=[(5, 7), (5, 7)],
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.CSubTable,
                  input_size=[(5, 7), (5, 7)],
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.CDivTable,
                  input_fn=lambda: [torch.randn(1, 7), torch.rand(1, 7) + 0.1],
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.CMulTable,
                  input_size=[(5, 7), (5, 7)],
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Square,
                  input_size=(10, 2, 4),
                  reference_fn=lambda i, _: i.mul(i)),
    OldModuleTest(nn.Sqrt,
                  input_fn=lambda: torch.rand(10, 2, 4) + 0.01,
                  reference_fn=lambda i, _: i.sqrt()),
    OldModuleTest(nn.Squeeze,
                  input_size=(2, 1, 1, 4, 5),
                  reference_fn=lambda i, _: i.squeeze()),
    OldModuleTest(nn.Squeeze,
                  constructor_args=(1,),
                  input_size=(2, 1, 1, 4, 5),
                  reference_fn=lambda i, _: i.squeeze(1),
                  desc='dim'),
    OldModuleTest(nn.Unsqueeze,
                  constructor_args=(1,),
                  input_size=(2, 4, 5),
                  reference_fn=lambda i, _: i.view(2, 1, 4, 5)),
    OldModuleTest(nn.Unsqueeze,
                  constructor_args=(0,),
                  input_size=(2, 4, 5),
                  reference_fn=lambda i, _: i.view(1, 2, 4, 5),
                  desc='fist_dim'),
    OldModuleTest(nn.Unsqueeze,
                  constructor_args=(3,),
                  input_size=(2, 4, 5),
                  reference_fn=lambda i, _: i.view(2, 4, 5, 1),
                  desc='last_dim'),
    OldModuleTest(nn.View,
                  constructor_args=(-1, 2, 20),
                  input_size=(2, 2, 4, 5),
                  reference_fn=lambda i, _: i.view(-1, 2, 20),
                  desc='infer_batch'),
    OldModuleTest(nn.View,
                  constructor_args=(2, 2, 2, 5),
                  input_size=(2, 4, 5),
                  reference_fn=lambda i, _: i.view(2, 2, 2, 5),
                  desc='split_dim'),
    OldModuleTest(nn.View,
                  constructor_args=(2, -1, 2, 5),
                  input_size=(2, 4, 5),
                  reference_fn=lambda i, _: i.view(2, -1, 2, 5),
                  desc='infer_middle'),
    OldModuleTest(nn.Sum,
                  constructor_args=(1,),
                  input_size=(2, 4, 5),
                  reference_fn=lambda i, _: i.sum(1, keepdim=False),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Sum,
                  constructor_args=(1, True),
                  input_size=(2, 4, 5),
                  reference_fn=lambda i, _: i.sum(1, keepdim=False).div(i.size(1)),
                  desc='sizeAverage',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Mean,
                  constructor_args=(1,),
                  input_size=(2, 4, 5),
                  reference_fn=lambda i, _: torch.mean(i, 1, keepdim=False),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(lambda: nn.Sequential().add(nn.GradientReversal()).add(nn.GradientReversal()),
                  input_size=(4, 3, 2, 2),
                  fullname='GradientReversal',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Identity,
                  input_size=(4, 3, 2, 4),
                  reference_fn=lambda i, _: i),
    OldModuleTest(nn.DotProduct,
                  input_size=[(10, 4), (10, 4)],
                  reference_fn=lambda i, _: torch.Tensor(list(
                      a.dot(b) for a, b in zip(i[0], i[1]))),
                  test_cuda = (not TEST_WITH_ROCM)
                  ),
    OldModuleTest(nn.CosineDistance,
                  input_size=[(10, 4), (10, 4)],
                  reference_fn=lambda i, _: torch.Tensor(list(
                      a.dot(b) / (a.norm(2) * b.norm(2)) for a, b in zip(i[0], i[1]))),
                  test_cuda = (not TEST_WITH_ROCM)
                  ),
    OldModuleTest(nn.JoinTable,
                  constructor_args=(0,),
                  input_size=[(10, 4), (10, 4)],
                  reference_fn=lambda i, _: torch.cat(i, 0),
                  desc='first_dim'),
    OldModuleTest(nn.JoinTable,
                  constructor_args=(2,),
                  input_size=[(2, 4, 2), (2, 4, 2)],
                  reference_fn=lambda i, _: torch.cat(i, 2),
                  desc='positive_dim_index'),
    OldModuleTest(nn.JoinTable,
                  constructor_args=(-1,),
                  input_size=[(2, 4, 2, 4), (2, 4, 2, 4)],
                  reference_fn=lambda i, _: torch.cat(i, 3),
                  desc='negative_dim_index'),
    OldModuleTest(nn.MM,
                  input_size=[(4, 5, 3), (4, 3, 2)],
                  reference_fn=lambda i, _: torch.bmm(*i)),
    OldModuleTest(nn.MV,
                  input_size=[(4, 5, 3), (4, 3)],
                  reference_fn=lambda i, _: torch.bmm(i[0], i[1].view(i[1].size(0), i[1].size(1), 1)).squeeze()),
    OldModuleTest(nn.Max,
                  input_size=(4, 5, 3),
                  reference_fn=lambda i, _: torch.max(i, 0, False)[0]),
    OldModuleTest(nn.Max,
                  constructor_args=(1,),
                  input_size=(4, 5, 3),
                  reference_fn=lambda i, _: torch.max(i, 1, False)[0],
                  desc='with_dimension'),
    OldModuleTest(nn.Min,
                  input_size=(4, 5, 3),
                  reference_fn=lambda i, _: torch.min(i, 0, False)[0]),
    OldModuleTest(nn.Min,
                  constructor_args=(1,),
                  input_size=(4, 5, 3),
                  reference_fn=lambda i, _: torch.min(i, 1, False)[0],
                  desc='with_dimension'),
    OldModuleTest(nn.MixtureTable,
                  input_size=[(5, 3), (5, 3, 6)],
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.LookupTable,
                  constructor_args=(4, 3),
                  input_fn=lambda: torch.randperm(2).repeat(1, 2),
                  jacobian_input=False,
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Mul,
                  input_size=(2, 3, 4, 2),
                  reference_fn=lambda i, p: i * p[0][0],
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.MulConstant,
                  constructor_args=(4,),
                  input_size=(2, 3, 4, 2),
                  reference_fn=lambda i, _: i * 4,
                  check_inplace=True,
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Narrow,
                  constructor_args=(0, 0),
                  input_size=(2, 3, 4, 2),
                  reference_fn=lambda i, _: i.narrow(0, 0, 1)),
    OldModuleTest(nn.Narrow,
                  constructor_args=(1, 1, 2),
                  input_size=(2, 3, 4, 2),
                  reference_fn=lambda i, _: i.narrow(1, 1, 2),
                  desc='length'),
    OldModuleTest(nn.Transpose,
                  constructor_args=((1, 2), (1, 3)),
                  input_size=(2, 3, 4, 5),
                  reference_fn=lambda i, _: i.transpose(1, 2).transpose(1, 3)),
    OldModuleTest(nn.Transpose,
                  constructor_args=((1, 2),),
                  input_size=(2, 3, 4, 5),
                  reference_fn=lambda i, _: i.transpose(1, 2),
                  desc='single_arg'),
    # TODO: this seems to be very slow
    OldModuleTest(nn.Replicate,
                  constructor_args=(2, 1),
                  input_size=(10, 3, 4, 5),
                  reference_fn=lambda i, _: i.view(10, 1, 3, 4, 5).expand(10, 2, 3, 4, 5),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Padding,
                  constructor_args=(0, 2, -10),
                  input_size=(2, 3, 4, 5)),
    OldModuleTest(nn.Padding,
                  constructor_args=(0, 2, -10, 1),
                  input_size=(2, 3, 4, 5),
                  desc='index'),
    OldModuleTest(nn.Padding,
                  constructor_args=(0, -2, -10, 1),
                  input_size=(2, 3, 4, 5),
                  desc='negative_pad'),
    OldModuleTest(nn.PartialLinear,
                  constructor_args=(5, 6),
                  input_size=(4, 5),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(lambda: nn.PartialLinear(5, 6).setPartition(torch.Tensor((2, 4))),
                  input_size=(4, 5),
                  fullname='PartialLinear_setPartition',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Power,
                  constructor_args=(2,),
                  input_size=(2, 3, 4, 5),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Power,
                  constructor_args=(1.5,),
                  input_fn=lambda: torch.rand(3, 4, 5),
                  desc='fractional',
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.Reshape,
                  constructor_args=(4, 5),
                  input_size=(3, 4 * 5),
                  desc='add_dim'),
    OldModuleTest(nn.Reshape,
                  constructor_args=(4 * 5,),
                  input_size=(3, 4, 5),
                  desc='squash_dim'),
    OldModuleTest(nn.Select,
                  constructor_args=(1, 2),
                  input_size=(3, 4, 5),
                  reference_fn=lambda i, _: i.select(1, 2)),
    OldModuleTest(nn.SelectTable,
                  constructor_args=(1,),
                  input_size=[(1,), (2,), (3,), (4,)],
                  reference_fn=lambda i, _: i[1]),
    OldModuleTest(nn.SpatialAveragePooling,
                  constructor_args=(2, 2),
                  input_size=(2, 3, 6, 6)),
    OldModuleTest(nn.SpatialAveragePooling,
                  constructor_args=(2, 2, 2, 2),
                  input_size=(2, 3, 6, 6),
                  desc='stride'),
    OldModuleTest(nn.SpatialAveragePooling,
                  constructor_args=(2, 2, 2, 2, 1, 1),
                  input_size=(2, 3, 6, 6),
                  desc='stride_pad'),
    OldModuleTest(nn.SpatialAdaptiveMaxPooling,
                  constructor_args=(4, 4),
                  input_size=(2, 3, 8, 8),
                  reference_fn=lambda i, _: nn.SpatialMaxPooling(2, 2).forward(i)),
    OldModuleTest(nn.SpatialAdaptiveMaxPooling,
                  constructor_args=(4, 4),
                  input_size=(2, 3, 7, 11),
                  desc='irregular'),
    OldModuleTest(nn.SpatialConvolution,
                  constructor_args=(3, 4, 3, 3),
                  input_size=(2, 3, 6, 6)),
    OldModuleTest(nn.SpatialConvolution,
                  constructor_args=(3, 4, 3, 3, 2, 2),
                  input_size=(2, 3, 6, 6),
                  desc='strided'),
    OldModuleTest(nn.SpatialConvolution,
                  constructor_args=(3, 4, 3, 3, 2, 2, 1, 1),
                  input_size=(2, 3, 6, 6),
                  desc='padding'),
    OldModuleTest(nn.SpatialConvolutionLocal,
                  constructor_args=(3, 2, 4, 4, 2, 2),
                  input_size=(1, 3, 4, 4)),
    OldModuleTest(nn.SpatialConvolutionLocal,
                  constructor_args=(3, 2, 6, 6, 2, 2, 2, 2),
                  input_size=(2, 3, 6, 6),
                  desc='stride'),
    OldModuleTest(nn.SpatialConvolutionLocal,
                  constructor_args=(3, 2, 6, 6, 2, 2, 2, 2, 1, 1),
                  input_size=(2, 3, 6, 6),
                  desc='stride_pad'),
    OldModuleTest(nn.SpatialDivisiveNormalization,
                  constructor_args=(3,),
                  input_size=(2, 3, 8, 8),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.SpatialContrastiveNormalization,
                  constructor_args=(3,),
                  input_size=(2, 3, 8, 8),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.SpatialDilatedConvolution,
                  constructor_args=(3, 2, 3, 3, 2, 2, 1, 1, 2, 2),
                  input_size=(2, 3, 8, 8)),
    OldModuleTest(nn.SpatialDilatedConvolution,
                  constructor_args=(3, 2, 3, 3, 2, 2, 1, 1, 2, 2),
                  input_size=(2, 3, 8, 8),
                  desc='stride_pad'),
    OldModuleTest(nn.SpatialMaxPooling,
                  constructor_args=(3, 3, 2, 2, 1, 1),
                  input_size=(1, 3, 7, 7)),
    OldModuleTest(nn.SpatialReflectionPadding,
                  constructor_args=(1, 2, 3, 4),
                  input_size=(2, 3, 8, 8)),
    OldModuleTest(nn.SpatialReplicationPadding,
                  constructor_args=(1, 2, 3, 4),
                  input_size=(2, 3, 4, 4)),
    OldModuleTest(nn.SpatialZeroPadding,
                  constructor_args=(1, 2, 3, 4),
                  input_size=(2, 3, 4, 4)),
    OldModuleTest(nn.SpatialConvolutionMap,
                  constructor_args=(nn.SpatialConvolutionMap.maps.oneToOne(3), 3, 3),
                  input_size=(3, 5, 5),
                  desc='oneToOne'),
    OldModuleTest(nn.SpatialConvolutionMap,
                  constructor_args=(nn.SpatialConvolutionMap.maps.oneToOne(3), 3, 3, 2, 2),
                  input_size=(3, 5, 5),
                  desc='oneToOne_stride'),
    OldModuleTest(nn.SpatialConvolutionMap,
                  constructor_args=(nn.SpatialConvolutionMap.maps.full(3, 4), 3, 3),
                  input_size=(3, 5, 5),
                  desc='full'),
    OldModuleTest(nn.SpatialFullConvolutionMap,
                  constructor_args=(nn.SpatialConvolutionMap.maps.oneToOne(3), 3, 3),
                  input_size=(3, 5, 5),
                  desc='oneToOne'),
    OldModuleTest(nn.SpatialFullConvolutionMap,
                  constructor_args=(nn.SpatialConvolutionMap.maps.oneToOne(3), 3, 3, 2, 2),
                  input_size=(3, 5, 5),
                  desc='oneToOne_stride'),
    OldModuleTest(nn.SpatialFullConvolutionMap,
                  constructor_args=(nn.SpatialConvolutionMap.maps.full(3, 4), 3, 3),
                  input_size=(3, 5, 5),
                  desc='full'),
    # TODO: test CUDA
    OldModuleTest(lambda: nn.SpatialFractionalMaxPooling(2, 2, 0.5, 0.5).fixPoolingRegions(),
                  input_size=(1, 3, 5, 5),
                  fullname='SpatialFractionalMaxPooling_ratio',
                  test_cuda=False),
    OldModuleTest(lambda: nn.SpatialFractionalMaxPooling(2, 2, 4, 4).fixPoolingRegions(),
                  input_size=(1, 3, 7, 7),
                  fullname='SpatialFractionalMaxPooling_size',
                  test_cuda=False),
    OldModuleTest(nn.SpatialFullConvolution,
                  constructor_args=(3, 4, 3, 3, 2, 2, 1, 1, 1, 1),
                  input_size=(1, 3, 7, 7)),
    OldModuleTest(nn.SpatialLPPooling,
                  constructor_args=(3, 2, 2, 2, 2, 2),
                  input_size=(1, 3, 7, 7),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.SpatialSubSampling,
                  constructor_args=(3, 3, 3, 2, 2),
                  input_size=(1, 3, 7, 7)),
    OldModuleTest(nn.SpatialSubtractiveNormalization,
                  constructor_args=(3,),
                  input_size=(1, 3, 7, 7),
                  test_cuda = (not TEST_WITH_ROCM)),
    OldModuleTest(nn.SpatialSubtractiveNormalization,
                  constructor_args=(3, torch.rand(3)),
                  input_size=(1, 3, 7, 7),
                  desc='kernel'),
    OldModuleTest(nn.SpatialUpSamplingNearest,
                  constructor_args=(2,),
                  input_size=(1, 3, 4, 4)),

    OldModuleTest(nn.TemporalConvolution,
                  constructor_args=(4, 5, 3),
                  input_size=(2, 10, 4)),
    OldModuleTest(nn.TemporalConvolution,
                  constructor_args=(4, 5, 3, 2),
                  input_size=(2, 10, 4),
                  desc='stride'),
    # TODO: this runs in non-batch mode only
    OldModuleTest(nn.TemporalSubSampling,
                  constructor_args=(4, 3),
                  input_size=(10, 4)),
    OldModuleTest(nn.TemporalSubSampling,
                  constructor_args=(4, 3, 2),
                  input_size=(10, 4),
                  desc='stride'),

    OldModuleTest(nn.VolumetricAveragePooling,
                  constructor_args=(2, 2, 2),
                  input_size=(2, 3, 4, 4, 4)),
    OldModuleTest(nn.VolumetricAveragePooling,
                  constructor_args=(2, 2, 2, 2, 2, 2),
                  input_size=(2, 3, 5, 5, 5),
                  desc='stride'),
    OldModuleTest(nn.VolumetricAveragePooling,
                  constructor_args=(2, 2, 2, 2, 2, 2, 1, 1, 1),
                  input_size=(2, 3, 5, 5, 5),
                  desc='stride_pad'),
    OldModuleTest(nn.VolumetricConvolution,
                  constructor_args=(3, 4, 2, 2, 2),
                  input_size=(2, 3, 3, 3, 3)),
    OldModuleTest(nn.VolumetricConvolution,
                  constructor_args=(3, 4, 2, 2, 2, 2, 2, 2),
                  input_size=(2, 3, 5, 5, 5),
                  desc='stride'),
    OldModuleTest(nn.VolumetricConvolution,
                  constructor_args=(3, 4, 2, 2, 2, 2, 2, 2, 1, 1, 1),
                  input_size=(2, 3, 5, 5, 5),
                  desc='stride_padding'),
    OldModuleTest(nn.VolumetricFullConvolution,
                  constructor_args=(2, 3, 2, 2, 2),
                  input_size=(1, 2, 4, 4, 4)),
    OldModuleTest(nn.VolumetricMaxPooling,
                  constructor_args=(2, 2, 2),
                  input_fn=lambda: (torch.randn(2, 3, 5, 5, 5) * 1000)),
    OldModuleTest(nn.VolumetricMaxPooling,
                  constructor_args=(2, 2, 2, 2, 2, 2),
                  input_fn=lambda: (torch.randn(2, 3, 5, 5, 5) * 1000),
                  desc='stride'),
    OldModuleTest(nn.VolumetricMaxPooling,
                  constructor_args=(2, 2, 2, 2, 2, 2, 1, 1, 1),
                  input_fn=lambda: (torch.randn(2, 3, 5, 5, 5) * 1000),
                  desc='stride_padding'),
    OldModuleTest(nn.VolumetricReplicationPadding,
                  constructor_args=(1, 2, 3, 4, 5, 6),
                  input_size=(2, 3, 5, 5, 5)),

    CriterionTest(nn.L1Cost,
                  input_size=(2, 3, 4, 5),
                  target=None),
    CriterionTest(nn.L1HingeEmbeddingCriterion,
                  input_size=[(2, 3, 4, 5), (2, 3, 4, 5)],
                  target=1),
    CriterionTest(nn.L1HingeEmbeddingCriterion,
                  constructor_args=(2,),
                  input_size=[(2, 3, 4, 5), (2, 3, 4, 5)],
                  target=1,
                  desc='margin'),
    CriterionTest(nn.WeightedMSECriterion,
                  constructor_args_fn=lambda: (torch.rand(3, 4, 5),),
                  input_size=(2, 3, 4, 5),
                  target_size=(2, 3, 4, 5),
                  test_cuda = (not TEST_WITH_ROCM)),
    CriterionTest(nn.MarginCriterion,
                  input_size=(5, 10),
                  target_fn=lambda: torch.randn(5, 10).sign()),
    CriterionTest(nn.ClassSimplexCriterion,
                  constructor_args=(30,),
                  input_fn=lambda: torch.randn(5, 30).mul(10).renorm(2, 0, 1),
                  target_fn=lambda: torch.rand(5).mul(30).floor().long(),
                  desc='margin'),
]
# TODO: FlattenTable gradient
# TODO: NarrowTable gradient
# TODO: CriterionTable
# TODO: MultiCriterion
# TODO: SplitTable

for p in (1, 2, 1.5):
    tests.append(
        OldModuleTest(nn.Normalize,
                      constructor_args=(p,),
                      input_size=(4, 5),
                      # Eh, we need to use p as a default, so it's passed by value
                      reference_fn=lambda i, _, p=p: i.div(i.norm(p, 1, True).expand_as(i)),
                      desc=str(p),
                      test_cuda = (not TEST_WITH_ROCM)),
    )
for p in range(1, 4 + 1):
    tests.append(
        OldModuleTest(nn.PairwiseDistance,
                      constructor_args=(p,),
                      input_size=[(4, 10), (4, 10)],
                      desc=str(p),
                      test_cuda = (not TEST_WITH_ROCM))
    )


def build_spatial_unpooling_net():
    pool = nn.SpatialMaxPooling(2, 2, 2, 2)
    unpool = nn.SpatialMaxUnpooling(pool)
    return nn.Sequential().add(pool).add(unpool)

tests.append(
    OldModuleTest(build_spatial_unpooling_net,
                  input_size=(1, 3, 10, 10),
                  desc='SpatialMaxUnpooling')
)


def build_volumetric_unpooling_net():
    pool = nn.VolumetricMaxPooling(2, 2, 2, 2)
    unpool = nn.VolumetricMaxUnpooling(pool)
    return nn.Sequential().add(pool).add(unpool)

tests.append(
    OldModuleTest(build_volumetric_unpooling_net,
                  input_size=(1, 3, 10, 10),
                  desc='VolumetricMaxUnpooling')
)


def prepare_tests():
    def add_test(test):
        test_name = test.get_name()
        cuda_test_name = test_name + '_cuda'
        if hasattr(TestNN, test_name):
            raise RuntimeError('Found two tests with the same name: ' + test_name)
        if hasattr(TestNN, cuda_test_name):
            raise RuntimeError('Found two tests with the same name: ' + cuda_test_name)
        setattr(TestNN, test_name, lambda self, test=test: test(self))
        setattr(TestNN, cuda_test_name, lambda self, test=test: test.test_cuda(self))
    name_remap = {
        'Conv2d': 'SpatialConvolution',
        'MaxPool2d': 'SpatialMaxPooling',
        'AvgPool2d': 'SpatialAveragePooling',
        'Softmax': 'SoftMax',
        'Softmax2d': 'SpatialSoftMax',
        'LogSoftmax': 'LogSoftMax',
        'BatchNorm1d': 'BatchNormalization',
        'BatchNorm2d': 'SpatialBatchNormalization',
        'BatchNorm3d': 'VolumetricBatchNormalization',
        'Hardtanh': 'HardTanh',
        'Hardshrink': 'HardShrink',
        'Softplus': 'SoftPlus',
        'Softshrink': 'SoftShrink',
        'Softsign': 'SoftSign',
        'Softmin': 'SoftMin',
        'Tanhshrink': 'TanhShrink',
        'CrossMapLRN2d': 'SpatialCrossMapLRN',
        'L1Loss': 'AbsCriterion',
        'NLLLoss': 'ClassNLLCriterion',
        'NLLLoss2d': 'SpatialClassNLLCriterion',
        'KLDivLoss': 'DistKLDivCriterion',
    }
    for test in tests:
        name = test.get_name()
        if ((name == "test_Max" or name == "test_Min" or name == "test_Max_with_dimension" or name == "test_Min_with_dimension") and TEST_WITH_ROCM):
            continue
        add_test(test)
    for test_params in module_tests:
        test_params = deepcopy(test_params)
        name = test_params.pop('module_name')
        name = name_remap.get(name, name)
        # hardshrink is deprecated in nn
        if name == "HardShrink":
            continue

        test_params['constructor'] = getattr(nn, name)
        test = OldModuleTest(**test_params)
        add_test(test)
    for test_params in criterion_tests:
        test_params = deepcopy(test_params)
        name = test_params.pop('module_name')
        name = name_remap.get(name, name.replace('Loss', 'Criterion'))
        # hardshrink is deprecated in nn
        if name == "HardShrink":
            continue

        # nn.NLLLoss2d is deprecated, but there is a NLLLoss test for 2d
        if name == 'ClassNLLCriterion' and 'desc' in test_params.keys() and '2d' in test_params['desc']:
            name = 'SpatialClassNLLCriterion'

        test_params['constructor'] = getattr(nn, name)

        # If legacy constructor args are specified, use them instead
        legacy_args = test_params.pop('legacy_constructor_args', None)
        if legacy_args is not None:
            test_params['constructor_args'] = legacy_args

        test = CriterionTest(**test_params)
        add_test(test)


def require_grad(input):
    if isinstance(input, torch.Tensor):
        input = input.detach()
        input.requires_grad = True
        return input
    elif isinstance(input, collections.Iterable):
        return type(input)(require_grad(e) for e in input)
    return input


class TestNN(NNTestCase):
    _do_cuda_memory_leak_check = True

    def _numerical_jacobian(self, module, input, jacobian_input=True, jacobian_parameters=True):
        def fw(input):
            out = self._forward(module, input)
            if isinstance(out, Variable):
                return out.data
            return out

        res = tuple()
        if jacobian_input:
            input = require_grad(input)
            res += get_numerical_jacobian(fw, input, eps=1e-6),
        if jacobian_parameters:
            params, _ = self._get_parameters(module)
            jacobians = []
            for p in params:
                p = p.detach()
                p.requires_grad = True
                jacobians.append(get_numerical_jacobian(fw, input, p, eps=1e-6))
            res += torch.cat(jacobians, 0),
        return res

    def _forward(self, module, input):
        with freeze_rng_state():
            with torch.no_grad():
                return module.forward(input)

    def _backward(self, module, input, output, grad_output, create_graph=False):
        if isinstance(input, Variable):
            input = input.data

        return module.backward(input, grad_output)

    def _forward_criterion(self, criterion, input, target, extra_args=None):
        if extra_args is None:
            extra_args = tuple()
        with torch.no_grad():
            return criterion.forward(input, target, *extra_args)

    def _backward_criterion(self, criterion, input, target, gradOutput=None, extra_args=None):
        if extra_args is None:
            extra_args = tuple()
        # Ignore gradOutput. It's used for non-legacy tests.
        with torch.no_grad():
            return criterion.backward(input, target, *extra_args)

    def _zero_grad_parameters(self, module):
        return module.zeroGradParameters()

    def _get_parameters(self, module):
        return module.parameters() or ([], [])

    def test_Dropout(self):
        p = 0.2
        input = torch.Tensor(1000).fill_(1 - p)

        module = nn.Dropout(p)
        output = module.forward(input)
        self.assertLess(abs(output.mean() - (1 - p)), 0.05)
        gradInput = module.backward(input, input)
        self.assertLess(abs(gradInput.mean() - (1 - p)), 0.05)

        module = nn.Dropout(p, True)
        output = module.forward(input.clone())
        self.assertLess(abs(output.mean() - (1 - p)), 0.05)
        gradInput = module.backward(input.clone(), input.clone())
        self.assertLess(abs(gradInput.mean() - (1 - p)), 0.05)

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def test_SpatialDropout(self):
        p = 0.2
        b = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        nfeats = 1000
        input = torch.Tensor(b, nfeats, w, h).fill_(1)
        module = nn.SpatialDropout(p)
        module.training()
        output = module.forward(input)
        self.assertLess(abs(output.mean() - (1 - p)), 0.05)
        gradInput = module.backward(input, input)
        self.assertLess(abs(gradInput.mean() - (1 - p)), 0.05)

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def test_VolumetricDropout(self):
        p = 0.2
        bsz = random.randint(1, 5)
        t = random.randint(1, 5)
        w = random.randint(1, 5)
        h = random.randint(1, 5)
        nfeats = 1000
        input = torch.Tensor(bsz, nfeats, t, w, h).fill_(1)
        module = nn.VolumetricDropout(p)
        module.training()
        output = module.forward(input)
        self.assertLess(abs(output.mean() - (1 - p)), 0.05)
        gradInput = module.backward(input, input)
        self.assertLess(abs(gradInput.mean() - (1 - p)), 0.05)

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def test_ReLU_reference(self):
        input = torch.randn(10, 20)
        module = nn.ReLU()
        output = module.forward(input)
        self.assertTrue(output[input.ge(0)].eq(input[input.gt(0)]).all())
        self.assertTrue(output[input.lt(0)].eq(0).all())

    def test_ReLU6_reference(self):
        input = torch.randn(10, 20).mul(10)
        module = nn.ReLU6()
        output = module.forward(input)
        self.assertTrue(output[input.ge(6)].eq(6).all())
        self.assertTrue(output[input.lt(0)].eq(0).all())

    def test_Copy(self):
        input = torch.randn(3, 4).double()
        c = nn.Copy(torch.DoubleTensor, torch.FloatTensor)
        output = c.forward(input)
        self.assertIsInstance(output, torch.FloatTensor)
        self.assertEqual(output, input.float(), 1e-6)
        gradInput = c.backward(input, output.fill_(1))
        self.assertIsInstance(gradInput, torch.DoubleTensor)
        self.assertEqual(gradInput, output.double(), 1e-6)
        c.dontCast = True
        c.double()
        self.assertIsInstance(output, torch.FloatTensor)

        # Check that these don't raise errors
        c.__repr__()
        str(c)

    def test_FlattenTable(self):
        input = [
            torch.rand(1),
            [
                torch.rand(2),
                [
                    torch.rand(3)
                ],
            ],
            torch.rand(4)
        ]
        gradOutput = [
            torch.rand(1),
            torch.rand(2),
            torch.rand(3),
            torch.rand(4)
        ]

        m = nn.FlattenTable()
        output = m.forward(input)
        self.assertEqual(len(output), 4)
        self.assertEqual(output[0], input[0])
        self.assertEqual(output[1], input[1][0])
        self.assertEqual(output[2], input[1][1][0])
        self.assertEqual(output[3], input[2])

        gradInput = m.backward(input, gradOutput)
        self.assertEqual(gradOutput[0], gradInput[0])
        self.assertEqual(gradOutput[1], gradInput[1][0])
        self.assertEqual(gradOutput[2], gradInput[1][1][0])
        self.assertEqual(gradOutput[3], gradInput[2])

        # Check that these don't raise errors
        m.__repr__()
        str(m)

        # More uglyness: FlattenTable doesn't rebuild the table every updateOutput
        # call, so we need to make sure that modifications to the input are
        # detected correctly (and that the table is correctly rebuilt.
        # CASE 1: Nothing changes so the output table shouldn't be redefined
        old_input_map = m.input_map
        old_output = m.output
        m.forward(input)
        self.assertEqual(old_input_map, m.input_map)
        self.assertEqual(old_output, m.output)

        # CASE 2: An element is added to the input table
        old_input_map = m.input_map
        old_output = m.output
        input[1].append(torch.rand(5))
        m.forward(input)
        self.assertNotEqual(old_input_map, m.input_map)
        self.assertNotEqual(old_output, m.output)

        # CASE 3: An element is removed from the input table
        old_input_map = m.input_map
        old_output = m.output
        input.pop()
        m.forward(input)
        self.assertNotEqual(old_input_map, m.input_map)
        self.assertNotEqual(old_output, m.output)

    def test_Concat(self):
        input = torch.randn(4, 2)
        num_modules = random.randint(2, 5)
        linears = [nn.Linear(2, 5) for i in range(num_modules)]

        m = nn.Concat(0)
        for l in linears:
            m.add(l)
            l.zeroGradParameters()
            l.weight.fill_(1)
            l.bias.fill_(0)

        # Check that these don't raise errors
        m.__repr__()
        str(m)

        output = m.forward(input)
        output2 = input.sum(1, True).expand(4, 5).repeat(num_modules, 1)
        self.assertEqual(output2, output)

        gradInput = m.backward(input, torch.ones(output2.size()))
        gradInput2 = torch.ones(4, 2).fill_(num_modules * 5)
        self.assertEqual(gradInput, gradInput2)

        gradWeight = input.sum(0, keepdim=True).expand(5, 2)
        for l in linears:
            self.assertEqual(gradWeight, l.gradWeight)

    def test_Parallel(self):
        input = torch.randn(3, 4, 5)
        m = nn.Parallel(0, 2)
        m.add(nn.View(4, 5, 1))
        m.add(nn.View(4, 5, 1))
        m.add(nn.View(4, 5, 1))

        # Check that these don't raise errors
        m.__repr__()
        str(m)

        output = m.forward(input)
        output2 = input.transpose(0, 2).transpose(0, 1)
        self.assertEqual(output2, output)

        gradInput = m.backward(input, output2)
        self.assertEqual(gradInput, input)

    def test_ParallelTable(self):
        input = torch.randn(3, 4, 5)
        p = nn.ParallelTable()
        p.add(nn.View(4, 5, 1))
        p.add(nn.View(4, 5, 1))
        p.add(nn.View(4, 5, 1))
        m = nn.Sequential()
        m.add(nn.SplitTable(0))
        m.add(p)
        m.add(nn.JoinTable(2))

        # Check that these don't raise errors
        p.__repr__()
        str(p)

        output = m.forward(input)
        output2 = input.transpose(0, 2).transpose(0, 1)
        self.assertEqual(output2, output)

        gradInput = m.backward(input, output2)
        self.assertEqual(gradInput, input)

    def test_ConcatTable(self):
        input = [
            torch.randn(3, 4).float(), torch.randn(3, 4).float(), [torch.randn(3, 4).float()]
        ]
        _gradOutput = [
            torch.randn(3, 3, 4).float(), torch.randn(3, 3, 4).float(), torch.randn(3, 3, 4).float()
        ]
        gradOutput = [
            [_gradOutput[0][0], _gradOutput[1][0], [_gradOutput[2][0]]],
            [_gradOutput[0][1], _gradOutput[1][1], [_gradOutput[2][1]]],
            [_gradOutput[0][2], _gradOutput[1][2], [_gradOutput[2][2]]]
        ]
        module = nn.ConcatTable()
        module.add(nn.Identity())
        module.add(nn.Identity())
        module.add(nn.Identity())
        module.float()

        # Check that these don't raise errors
        module.__repr__()
        str(module)

        output = module.forward(input)
        output2 = [input, input, input]
        self.assertEqual(output2, output)
        gradInput = module.backward(input, gradOutput)
        gradInput2 = [_gradOutput[0].sum(0, keepdim=False), _gradOutput[1].sum(
            0, keepdim=False), [_gradOutput[2].sum(0, keepdim=False)]]
        self.assertTrue(isinstance(gradInput, list))
        self.assertFalse(isinstance(gradInput[0], list))
        self.assertFalse(isinstance(gradInput[1], list))
        self.assertTrue(isinstance(gradInput[2], list))
        self.assertEqual(len(gradInput), 3)
        self.assertEqual(len(gradInput[2]), 1)
        for t1, t2 in zip(iter_tensors(gradInput), iter_tensors(gradInput2)):
            self.assertEqual(t1, t2)

        # test outputs for variable length inputs
        test = nn.ConcatTable()
        test.add(nn.Identity())
        test.add(nn.Identity())

        x = [torch.randn(5), torch.randn(5)]
        y = [torch.randn(5)]

        o1 = len(test.forward(x))
        go1 = len(test.backward(x, [x, x]))
        o2 = len(test.forward(y))
        go2 = len(test.backward(y, [y, y]))
        self.assertEqual(o1, 2)
        self.assertEqual(go1, 2)
        self.assertEqual(o2, 2)
        self.assertEqual(go2, 1)

    def test_DepthConcat(self):
        outputSize = [5, 6, 7, 8]
        input = torch.randn(2, 3, 12, 12)
        gradOutput = torch.randn(2, sum(outputSize), 12, 12)
        concat = nn.DepthConcat(1)
        concat.add(nn.SpatialConvolution(3, outputSize[0], 1, 1, 1, 1))  # > 2, 5, 12, 12
        concat.add(nn.SpatialConvolution(3, outputSize[1], 3, 3, 1, 1))  # > 2, 6, 10, 10
        concat.add(nn.SpatialConvolution(3, outputSize[2], 4, 4, 1, 1))  # > 2, 7, 9, 9
        concat.add(nn.SpatialConvolution(3, outputSize[3], 5, 5, 1, 1))  # > 2, 8, 8, 8
        concat.zeroGradParameters()
        # forward/backward
        outputConcat = concat.forward(input)
        gradInputConcat = concat.backward(input, gradOutput)
        # the spatial dims are the largest, the nFilters is the sum
        output = torch.Tensor(2, sum(outputSize), 12, 12).zero_()  # zero for padding
        narrows = ((slice(None), slice(0, 5), slice(None), slice(None)),
                   (slice(None), slice(5, 11), slice(1, 11), slice(1, 11)),
                   (slice(None), slice(11, 18), slice(1, 10), slice(1, 10)),
                   (slice(None), slice(18, 26), slice(2, 10), slice(2, 10)))
        gradInput = input.clone().zero_()
        for i in range(4):
            conv = concat.get(i)
            gradWeight = conv.gradWeight.clone()
            conv.zeroGradParameters()
            output[narrows[i]].copy_(conv.forward(input))
            gradInput.add_(conv.backward(input, gradOutput[narrows[i]]))
            self.assertEqual(gradWeight, conv.gradWeight)

        self.assertEqual(output, outputConcat)
        self.assertEqual(gradInput, gradInputConcat)

        # Check that these don't raise errors
        concat.__repr__()
        str(concat)

    def test_Contiguous(self):
        input = torch.randn(10, 10, 10)
        noncontig = input[:, 4]
        module = nn.Contiguous()
        assert not noncontig.is_contiguous()
        output = module.forward(noncontig)
        self.assertEqual(output, noncontig)
        self.assertTrue(output.is_contiguous())

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def test_Index(self):
        net = nn.Index(0)

        # test 1D
        input = [torch.Tensor((10, 20, 30)), torch.LongTensor((0, 1, 1, 2))]
        output = net.forward(input)
        self.assertEqual(output, torch.Tensor((10, 20, 20, 30)))

        gradOutput = torch.Tensor((1, 1, 1, 3))
        gradInput = net.backward(input, gradOutput)
        self.assertEqual(gradInput[0], torch.Tensor((1, 2, 3)))

        # test 2D
        input = [torch.Tensor(((10, 20), (30, 40))), torch.LongTensor((0, 0))]
        output = net.forward(input)
        self.assertEqual(output, torch.Tensor(((10, 20), (10, 20))))

        gradOutput = torch.Tensor(((1, 2), (1, 2)))
        gradInput = net.backward(input, gradOutput)
        self.assertEqual(gradInput[0], torch.Tensor(((2, 4), (0, 0))))

        # Check that these don't raise errors
        net.__repr__()
        str(net)

    def test_L1Penalty(self):
        weight = 1
        m = nn.L1Penalty(weight, False, False)

        input = torch.rand(2, 10).add_(-0.5)
        input[0][0] = 0

        m.forward(input)
        grad = m.backward(input, torch.ones(input.size()))

        self.assertEqual(input.abs().sum() * weight, m.loss)

        true_grad = (input.gt(0).type_as(grad) +
                     input.lt(0).type_as(grad).mul_(-1)).mul_(weight)
        self.assertEqual(true_grad, grad)

        # Check that these don't raise errors
        m.__repr__()
        str(m)

    def test_MaskedSelect(self):
        input = torch.randn(4, 5)
        mask = torch.ByteTensor(4, 5).bernoulli_()
        module = nn.MaskedSelect()
        out = module.forward([input, mask])
        self.assertEqual(input.masked_select(mask), out)

        gradOut = torch.Tensor((20, 80))
        input = torch.Tensor(((10, 20), (30, 40)))
        inTarget = torch.Tensor(((20, 0), (0, 80)))
        mask = torch.ByteTensor(((1, 0), (0, 1)))
        module = nn.MaskedSelect()
        module.forward([input, mask])
        gradIn = module.backward([input, mask], gradOut)
        self.assertEqual(inTarget, gradIn[0])

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def test_MultiCriterion(self):
        input = torch.rand(2, 10)
        target = torch.LongTensor((1, 8))
        nll = nn.ClassNLLCriterion()
        nll2 = nn.CrossEntropyCriterion()
        mc = nn.MultiCriterion().add(nll, 0.5).add(nll2)

        output = mc.forward(input, target)
        output2 = nll.forward(input, target) / 2 + nll2.forward(input, target)

        self.assertEqual(output, output2)
        gradInput = mc.backward(input, target)
        gradInput2 = nll.backward(input, target).clone().div(2).add(nll2.backward(input, target))
        self.assertEqual(gradInput, gradInput2)

        # test type
        mc.float()
        gradInput = gradInput.clone()
        input3 = input.float()
        target3 = target
        output3 = mc.forward(input3, target3)
        gradInput3 = mc.backward(input3, target3)
        self.assertEqual(output, output3)
        self.assertEqual(gradInput.float(), gradInput3)

        # Check that these don't raise errors
        mc.__repr__()
        str(mc)

        # test table input
        # TODO: enable when Criterion.clone is ready
        # mc.double()
        # input = [torch.randn(2, 10), [torch.randn(2, 10), torch.randn(2, 10)]]
        # target = [torch.IntTensor((1, 8)), [torch.IntTensor((5, 6)), torch.IntTensor((4, 3))]]
        # pnllc = nn.ParallelCriterion().add(nll).add(nn.ParallelCriterion().add(nll.clone()).add(nll.clone()))
        # pnllc2 = nn.ParallelCriterion().add(nll2).add(nn.ParallelCriterion().add(nll2.clone()).add(nll2.clone()))
        # mc = nn.MultiCriterion().add(pnllc, 0.5).add(pnllc2)
        # output = mc.forward(input, target)
        # output2 = pnllc.forward(input, target)/2 + pnllc2.forward(input, target)
        # self.assertEqual(output, output2)
        # gradInput = mc.backward(input, target)
        # gradInput2 = pnllc.clone().backward(input, target)
        # gradInput2b = pnllc2.backward(input, target)
        # gradInput2[0].div(2).add(gradInput2b[0])
        # gradInput2[1][0].div(2).add(gradInput2b[1][0])
        # gradInput2[1][1].div(2).add(gradInput2b[1][1])
        # self.assertEqual(gradInput[1], gradInput2[0])
        # self.assertEqual(gradInput[1][9], gradInput2[1][0])
        # self.assertEqual(gradInput[1][1], gradInput2[1][1])

    def test_ParallelCriterion(self):
        input = [torch.rand(2, 10), torch.randn(2, 10)]
        target = [torch.LongTensor((1, 8)), torch.randn(2, 10)]
        nll = nn.ClassNLLCriterion()
        mse = nn.MSECriterion()
        pc = nn.ParallelCriterion().add(nll, 0.5).add(mse)
        output = pc.forward(input, target)
        output2 = nll.forward(input[0], target[0]) / 2 + mse.forward(input[1], target[1])
        self.assertEqual(output, output2)
        gradInput2 = [nll.backward(input[0], target[0]).clone().div(2), mse.backward(input[1], target[1])]
        gradInput = pc.backward(input, target)
        self.assertEqual(gradInput[0], gradInput2[0])
        self.assertEqual(gradInput[1], gradInput2[1])

        # test type
        pc.float()
        gradInput[0], gradInput[1] = gradInput[0].clone(), gradInput[1].clone()
        input3 = [input[0].float(), input[1].float()]
        target3 = [target[0], target[1].float()]
        output3 = pc.forward(input3, target3)
        gradInput3 = pc.backward(input3, target3)
        self.assertEqual(output, output3)
        self.assertEqual(gradInput[0].float(), gradInput3[0])
        self.assertEqual(gradInput[1].float(), gradInput3[1])

        # test repeatTarget
        input = [torch.rand(2, 10), torch.randn(2, 10)]
        target = torch.randn(2, 10)
        mse = nn.MSECriterion()
        pc = nn.ParallelCriterion(True).add(mse, 0.5).add(nn.MSECriterion())
        output = pc.forward(input, target)
        output2 = mse.forward(input[0], target) / 2 + mse.forward(input[1], target)
        self.assertEqual(output, output2)
        gradInput = pc.backward(input, target)
        gradInput2 = [mse.backward(input[0], target).clone().div(2), mse.backward(input[1], target)]
        self.assertEqual(gradInput[0], gradInput2[0])
        self.assertEqual(gradInput[1], gradInput2[1])

        # table input
        input = [torch.randn(2, 10), [torch.rand(2, 10), torch.randn(2, 10)]]
        target = [torch.LongTensor((2, 5)), [torch.LongTensor((1, 8)), torch.randn(2, 10)]]
        nll2 = nn.ClassNLLCriterion()
        nll = nn.ClassNLLCriterion()
        mse = nn.MSECriterion()
        pc = nn.ParallelCriterion().add(nll, 0.5).add(mse)
        pc2 = nn.ParallelCriterion().add(nll2, 0.4).add(pc)
        output = pc2.forward(input, target)
        output2 = (nll2.forward(input[0], target[0]) * 0.4 +
                   nll.forward(input[1][0], target[1][0]) / 2 +
                   mse.forward(input[1][1], target[1][1]))
        self.assertEqual(output, output2)
        gradInput2 = [
            nll2.backward(input[0], target[0]).clone().mul(0.4),
            [nll.backward(input[1][1], target[1][0]).clone().div(2), mse.backward(input[1][1], target[1][1])]
        ]
        gradInput = pc2.backward(input, target)
        self.assertEqual(gradInput[0], gradInput2[0])
        self.assertEqual(gradInput[1][0], gradInput2[1][0])
        self.assertEqual(gradInput[1][1], gradInput2[1][1])

        # Check that these don't raise errors
        pc.__repr__()
        str(pc)

    def test_NarrowTable(self):
        input = [torch.Tensor(i) for i in range(1, 6)]

        module = nn.NarrowTable(1)
        output = module.forward(input)
        self.assertEqual(output, input[1:2])

        module = nn.NarrowTable(2, 3)
        output = module.forward(input)
        self.assertEqual(output, input[2:5])

        # Check that these don't raise errors
        module.__repr__()
        str(module)

    def test_accUpdateGradParameters(self):
        module = nn.LookupTable(5, 3)
        module.weight.fill_(2)
        input = torch.LongTensor([1, 3])
        output = module.updateOutput(input)
        module.backwardUpdate(input, output, 0.1)
        self.assertEqual(module.weight[0, 0], 2)
        self.assertEqual(module.weight[3, 0], 1.8)

    def _build_net(self):
        return (nn.Sequential()
                .add(nn.Concat(0)
                     .add(nn.Linear(2, 5))
                     .add(nn.Linear(2, 5)))
                .add(nn.ReLU())
                .add(nn.Linear(10, 20)))

    def test_parameters(self):
        net = self._build_net()
        concat = net.modules[0]
        param, grad = net.parameters()

        self.assertEqual(len(param), 6)
        self.assertEqual(len(grad), 6)

        self.assertObjectIn(concat.modules[0].weight, param)
        self.assertObjectIn(concat.modules[0].bias, param)
        self.assertObjectIn(concat.modules[1].weight, param)
        self.assertObjectIn(concat.modules[1].bias, param)
        self.assertObjectIn(net.modules[2].weight, param)
        self.assertObjectIn(net.modules[2].bias, param)

        self.assertObjectIn(concat.modules[0].gradWeight, grad)
        self.assertObjectIn(concat.modules[0].gradBias, grad)
        self.assertObjectIn(concat.modules[1].gradWeight, grad)
        self.assertObjectIn(concat.modules[1].gradBias, grad)
        self.assertObjectIn(net.modules[2].gradWeight, grad)
        self.assertObjectIn(net.modules[2].gradBias, grad)

    def test_flattenParameters(self):
        net = self._build_net()
        param, grad_param = net.flattenParameters()
        self.assertEqual(param.dim(), 1)
        self.assertEqual(param.size(0), 250)
        self.assertEqual(grad_param.dim(), 1)
        self.assertEqual(grad_param.size(0), 250)

    def test_findModules(self):
        net = self._build_net()
        modules, containers = net.findModules(nn.Linear)
        self.assertEqual(len(modules), 3)
        self.assertEqual(len(modules), len(containers))
        self.assertObjectIn(net.modules[0].modules[0], modules)
        self.assertObjectIn(net.modules[0].modules[1], modules)
        self.assertObjectIn(net.modules[2], modules)
        self.assertObjectIn(net.modules[0], containers)
        self.assertEqual(containers.count(net.modules[0]), 2)
        self.assertObjectIn(net, containers)
        for m, c in zip(modules, containers):
            self.assertObjectIn(m, c.modules)

    def test_apply(self):
        net = self._build_net()
        seen_modules = set()

        def callback(module):
            self.assertNotIn(module, seen_modules)
            seen_modules.add(module)
        net.apply(callback)
        self.assertEqual(len(seen_modules), 6)

    def test_listModules(self):
        net = self._build_net()
        module_list = list()

        def callback(module):
            module_list.append(module)
        net.apply(callback)
        self.assertEqual(module_list, net.listModules())

    def test_replace(self):
        ref_net = self._build_net()
        net = self._build_net()

        def callback(module):
            if isinstance(module, nn.ReLU):
                return nn.Tanh()
            return module
        net.replace(callback)

        for module, reference in zip(net.listModules(), ref_net.listModules()):
            if isinstance(reference, nn.ReLU):
                self.assertIsInstance(module, nn.Tanh)
            else:
                self.assertIsInstance(module, type(reference))


prepare_tests()


if __name__ == '__main__':
    run_tests()
