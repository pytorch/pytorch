# -*- coding: utf-8 -*-
from torch.testing._internal.common_utils import run_tests

import copy
import numpy as np
import io
import logging
from itertools import product

import torch
import torch.quantization as tq

from torch import nn
from torch.ao.nn.sparse import quantized as ao_nn_sq
from torch.ao.nn.sparse.quantized.utils import LinearBlockSparsePattern

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_quantized import (
    override_cpu_allocator_for_qnnpack,
    override_qengines,
    qengine_is_qnnpack,
    qengine_is_fbgemm,
)

# TODO: Once more test files are created, move the contents to a ao folder.

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

class TestQuantizedSparseKernels(TestCase):
    @override_qengines
    def test_sparse_qlinear(self):
        batch_size = 12
        input_channels = 16
        output_channels = 4
        decimal_val = 4
        row_block_size = 1
        col_block_size = 4

        # X86 implementation of sparse ops in qnnpack only support
        # block pattern 1x4.
        # arm kernels have support for both 1x4 and 8x1.
        # This distinction is only because x86 implementations exist
        # only to enable testing of integration path.
        # We do plan to add 8x1 as well so that testing does not have to
        # special case like this. At the moment it is deprioritized due
        # to other higher priority works.
        if qengine_is_qnnpack() and not (row_block_size == 1 and col_block_size == 4):
            return

        dense_prepack = torch.ops.quantized.linear_prepack
        dense_qlinear = torch.ops.quantized.linear
        dense_qlinear_dynamic = torch.ops.quantized.linear_dynamic

        sparse_prepack = torch.ops.sparse.qlinear_prepack
        sparse_qlinear = torch.ops.sparse.qlinear
        sparse_qlinear_dynamic = torch.ops.sparse.qlinear_dynamic

        X_scale = 0.2
        X_zp = 2
        X_fp32 = torch.randn(batch_size, input_channels, dtype=torch.float32)
        float_bias = torch.randn(output_channels, dtype=torch.float32)

        W_scales = torch.rand(output_channels, dtype=torch.float32)
        W_zps = torch.zeros(output_channels, dtype=torch.int32)
        W_fp32 = torch.randn(output_channels, input_channels, dtype=torch.float32)

        with override_cpu_allocator_for_qnnpack(qengine_is_qnnpack()):
            X_q = torch.quantize_per_tensor(
                X_fp32, scale=X_scale, zero_point=X_zp, dtype=torch.quint8
            )

            for use_channelwise, dynamic_mode in product([True, False], [True, False]):
                if qengine_is_fbgemm() and dynamic_mode:
                    logging.info("dynamic sparse qlinear is only available in qnnpack")
                    continue
                if qengine_is_qnnpack() and not dynamic_mode:
                    logging.info("static sparse qlinear is only available in fbgemm")
                    continue
                if use_channelwise:
                    W_q = torch.quantize_per_channel(
                        W_fp32, scales=W_scales, zero_points=W_zps, axis=0, dtype=torch.qint8
                    )
                else:
                    W_q = torch.quantize_per_tensor(
                        W_fp32, scale=W_scales[0], zero_point=W_zps[0], dtype=torch.qint8
                    )

                Y_scale = 1.1234
                Y_zp = 5
                W_prepack_dense = dense_prepack(W_q, float_bias)
                W_prepack_sparse = sparse_prepack(W_q, float_bias, row_block_size, col_block_size)

                if dynamic_mode:
                    Y = sparse_qlinear_dynamic(X_fp32, W_prepack_sparse)
                    Y_ref = dense_qlinear_dynamic(X_fp32, W_prepack_dense)

                    np.testing.assert_array_almost_equal(Y_ref.numpy(), Y.numpy(), decimal=decimal_val)
                else:
                    Y_q = sparse_qlinear(X_q, W_prepack_sparse, Y_scale, Y_zp)
                    Y_q_ref = dense_qlinear(X_q, W_prepack_dense, Y_scale, Y_zp)

                    np.testing.assert_array_almost_equal(
                        Y_q_ref.int_repr().numpy(), Y_q.int_repr().numpy(), decimal=decimal_val
                    )


class TestQuantizedSparseLayers(TestCase):
    class SparseQuantizedModel(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.linear = nn.Linear(in_channels, out_channels)

        def forward(self, x):
            return self.linear(x)

    @override_qengines
    def test_sparse_qlinear(self):
        batch_size = 12
        input_channels = 4
        output_channels = 7
        model = self.SparseQuantizedModel(input_channels, output_channels)

        # For sparse kernels both the activation and weight ZP = 0
        X_scale = 0.2
        X_zp = 2
        W_scale = 1e-2
        W_zp = 0

        X_fp32 = torch.randn(batch_size, input_channels, dtype=torch.float32)
        float_bias = torch.randn(output_channels, dtype=torch.float32)

        W_fp32 = torch.randn(output_channels, input_channels, dtype=torch.float32)
        mask = torch.randint(0, 2, W_fp32.shape)
        W_fp32 *= mask

        with override_cpu_allocator_for_qnnpack(qengine_is_qnnpack()):
            X_q = torch.quantize_per_tensor(
                X_fp32, scale=X_scale, zero_point=X_zp, dtype=torch.quint8
            )
            X_fp32 = X_q.dequantize()

            W_q = torch.quantize_per_tensor(W_fp32, W_scale, W_zp, torch.qint8)

            model.weight = nn.Parameter(W_q.dequantize())
            model.eval()

            # Note: At the moment, for sparse kernels
            # fbgemm supports only static quantized sparse linear
            # qnnpack supports only dynamically quantized sparse linear
            # Hence we have two different tests.
            # fbgemm tests static flow, qnnpack tests dynamic.
            # Should be unified later on and tests should be fixed
            # appropriately.
            if qengine_is_fbgemm():
                model.qconfig = tq.get_default_qconfig('fbgemm')
                qmodel = copy.deepcopy(model)
                sqmodel = copy.deepcopy(model)

                tq.prepare(qmodel, inplace=True)
                tq.prepare(sqmodel, inplace=True)

                with torch.no_grad():
                    qmodel(X_fp32)
                    sqmodel(X_fp32)

                # Make sure the quantization parameters are computed the same way
                qparams = qmodel.linear.qconfig.weight().calculate_qparams()
                sqparams = sqmodel.linear.qconfig.weight().calculate_qparams()
                self.assertEqual(qparams, sqparams)

                # Make sure mapping of sparse kernels does not affect the non-sparse
                sparse_mapping = tq.get_default_static_quant_module_mappings()
                sparse_mapping[nn.Linear] = ao_nn_sq.Linear
                tq.convert(sqmodel, inplace=True, mapping=sparse_mapping)
                tq.convert(qmodel, inplace=True)

                assert isinstance(sqmodel.linear, ao_nn_sq.Linear), "Convert failed"
                assert isinstance(qmodel.linear, nn.quantized.Linear), "Mapping failed"

                # Make sure numerics are right
                Y_ref = qmodel(X_q)
                Y_hat = sqmodel(X_q)
                self.assertEqual(Y_ref.dequantize(), Y_hat.dequantize())

            if qengine_is_qnnpack():
                qconfig = {nn.Linear : tq.qconfig.default_dynamic_qconfig}
                dqmodel = copy.deepcopy(model)
                sdqmodel = copy.deepcopy(model)

                tq.propagate_qconfig_(dqmodel, qconfig)
                tq.propagate_qconfig_(sdqmodel, qconfig)

                # Make sure the quantization parameters are computed the same way
                qparams = dqmodel.linear.qconfig.weight().calculate_qparams()
                sqparams = sdqmodel.linear.qconfig.weight().calculate_qparams()
                self.assertEqual(qparams, sqparams)

                # Make sure mapping of sparse kernels does not affect the non-sparse
                sparse_mapping = copy.deepcopy(tq.get_default_dynamic_quant_module_mappings())
                sparse_mapping[nn.Linear] = ao_nn_sq.dynamic.Linear
                with LinearBlockSparsePattern(1, 4):
                    tq.convert(sdqmodel, inplace=True, mapping=sparse_mapping)
                tq.convert(dqmodel, mapping=tq.get_default_dynamic_quant_module_mappings(), inplace=True)

                assert isinstance(sdqmodel.linear, ao_nn_sq.dynamic.Linear), "Convert failed"
                assert isinstance(dqmodel.linear, nn.quantized.dynamic.Linear), "Mapping failed"

                # Make sure numerics are right
                Y_ref = dqmodel(X_fp32)
                Y_hat = sdqmodel(X_fp32)
                self.assertEqual(Y_ref, Y_hat)

    @override_qengines
    def test_sparse_qlinear_serdes(self):
        batch_size = 12
        input_channels = 4
        output_channels = 7
        model = self.SparseQuantizedModel(input_channels, output_channels)

        # For sparse kernels both the activation and weight ZP = 0
        X_scale = 0.2
        X_zp = 0
        W_scale = 1e-2
        W_zp = 0

        with override_cpu_allocator_for_qnnpack(qengine_is_qnnpack()):
            X_fp32 = torch.randn(batch_size, input_channels, dtype=torch.float32)
            float_bias = torch.randn(output_channels, dtype=torch.float32)

            X_q = torch.quantize_per_tensor(
                X_fp32, scale=X_scale, zero_point=X_zp, dtype=torch.quint8
            )
            X_fp32 = X_q.dequantize()

            W_fp32 = torch.randn(output_channels, input_channels, dtype=torch.float32)
            mask = torch.randint(0, 2, W_fp32.shape)
            W_fp32 *= mask
            W_q = torch.quantize_per_tensor(W_fp32, W_scale, W_zp, torch.qint8)

            model.weight = nn.Parameter(W_q.dequantize())
            model.eval()

            # Note: At the moment, for sparse kernels
            # fbgemm supports only static quantized sparse linear
            # qnnpack supports only dynamically quantized sparse linear
            # Hence we have two different tests.
            # fbgemm tests static flow, qnnpack tests dynamic.
            # Should be unified later on and tests should be fixed
            # appropriately.
            if qengine_is_fbgemm():
                model.qconfig = tq.get_default_qconfig('fbgemm')
                qmodel = copy.deepcopy(model)
                sqmodel = copy.deepcopy(model)

                tq.prepare(qmodel, inplace=True)
                tq.prepare(sqmodel, inplace=True)

                with torch.no_grad():
                    qmodel(X_fp32)
                    sqmodel(X_fp32)

                # Make sure the quantization parameters are computed the same way
                qparams = qmodel.linear.qconfig.weight().calculate_qparams()
                sqparams = sqmodel.linear.qconfig.weight().calculate_qparams()
                self.assertEqual(qparams, sqparams)

                # Make sure mapping of sparse kernels does not affect the non-sparse
                sparse_mapping = tq.get_default_static_quant_module_mappings()
                sparse_mapping[nn.Linear] = ao_nn_sq.Linear
                tq.convert(sqmodel, inplace=True, mapping=sparse_mapping)
                tq.convert(qmodel, inplace=True)

                assert isinstance(sqmodel.linear, ao_nn_sq.Linear), "Convert failed"
                assert isinstance(qmodel.linear, nn.quantized.Linear), "Mapping failed"

                scripted_sqmodel = torch.jit.script(sqmodel)
                scripted_sqmodel.eval()
                buffer = io.BytesIO()
                torch.jit.save(scripted_sqmodel, buffer)
                buffer.seek(0)
                sqmodel = torch.jit.load(buffer)

                # Make sure numerics are right
                Y_ref = qmodel(X_q)
                Y_hat = sqmodel(X_q)
                self.assertEqual(Y_ref.dequantize(), Y_hat.dequantize())

            if qengine_is_qnnpack():
                qconfig = {nn.Linear : tq.qconfig.default_dynamic_qconfig}
                dqmodel = copy.deepcopy(model)
                sdqmodel = copy.deepcopy(model)

                tq.propagate_qconfig_(dqmodel, qconfig)
                tq.propagate_qconfig_(sdqmodel, qconfig)

                # Make sure the quantization parameters are computed the same way
                qparams = dqmodel.linear.qconfig.weight().calculate_qparams()
                sqparams = sdqmodel.linear.qconfig.weight().calculate_qparams()
                self.assertEqual(qparams, sqparams)

                # Make sure mapping of sparse kernels does not affect the non-sparse
                sparse_mapping = copy.deepcopy(tq.get_default_dynamic_quant_module_mappings())
                sparse_mapping[nn.Linear] = ao_nn_sq.dynamic.Linear
                with LinearBlockSparsePattern(1, 4):
                    tq.convert(sdqmodel, inplace=True, mapping=sparse_mapping)
                tq.convert(dqmodel, mapping=tq.get_default_dynamic_quant_module_mappings(), inplace=True)

                assert isinstance(sdqmodel.linear, ao_nn_sq.dynamic.Linear), "Convert failed"
                assert isinstance(dqmodel.linear, nn.quantized.dynamic.Linear), "Mapping failed"

                scripted_sdqmodel = torch.jit.script(sdqmodel)
                scripted_sdqmodel.eval()
                buffer = io.BytesIO()
                torch.jit.save(scripted_sdqmodel, buffer)
                buffer.seek(0)
                sdqmodel = torch.jit.load(buffer)

                # Make sure numerics are right
                Y_ref = dqmodel(X_fp32)
                Y_hat = sdqmodel(X_fp32)
                self.assertEqual(Y_ref, Y_hat)

if __name__ == '__main__':
    run_tests()
