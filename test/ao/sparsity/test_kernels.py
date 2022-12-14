# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]

from torch.testing._internal.common_utils import run_tests

import copy
import numpy as np
import io
import logging
from itertools import product

import torch
import torch.ao.quantization as tq

from torch import nn
from torch.ao.pruning.sparsifier.utils import fqn_to_module

from torch.testing._internal.common_utils import TestCase, skipIfTorchDynamo
from torch.testing._internal.common_quantized import (
    override_cpu_allocator_for_qnnpack,
    override_qengines,
    qengine_is_qnnpack,
    qengine_is_fbgemm,
    qengine_is_onednn,
    qengine_is_x86,
)

# TODO: Once more test files are created, move the contents to a ao folder.

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

class TestQuantizedSparseKernels(TestCase):
    @skipIfTorchDynamo("TorchDynamo fails here for unknown reasons")
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
        # ONEDNN and X86 do not support this yet
        if qengine_is_onednn() or qengine_is_x86():
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


def _sparse_layer_test_helper(
    model_class,
    sparse_mapping,
    ref_mapping,
    qconfig_dict,
    fqn_to_check,
    test_class,
    test_scripting,
):
    # SET UP TEST PARAMETERS, INPUTS AND WEIGHTS
    # ------------------------------------------
    batch_size = 12
    input_channels = 4
    output_channels = 7
    model = model_class(input_channels, output_channels)

    # For sparse kernels both the activation and weight ZP = 0
    X_scale = 0.2
    X_zp = 2
    W_scale = 1e-2
    W_zp = 0

    X_fp32 = torch.randn(batch_size, input_channels, dtype=torch.float32)
    float_bias = torch.randn(output_channels, dtype=torch.float32)

    # generate a weight which we'll insert into the model
    W_fp32 = torch.randn(output_channels, input_channels, dtype=torch.float32)
    mask = torch.randint(0, 2, W_fp32.shape)
    W_fp32 *= mask
    with override_cpu_allocator_for_qnnpack(qengine_is_qnnpack()):
        X_q = torch.quantize_per_tensor(
            X_fp32, scale=X_scale, zero_point=X_zp, dtype=torch.quint8
        )
        X_fp32 = X_q.dequantize()

        W_q = torch.quantize_per_tensor(W_fp32, W_scale, W_zp, torch.qint8)

        # PREPARE MODELS FOR QUANTIZATION
        # -------------------------------
        model.linear.weight = nn.Parameter(W_q.dequantize())
        model.eval()

        # Add `sparse_params` to the model. The test for correct
        # sparse_param addition is in the sparsifier tests
        model.linear.sparse_params = {"sparse_block_shape": (1, 4)}

        # generate model versions
        qmodel = copy.deepcopy(model)
        sqmodel = copy.deepcopy(model)

        # generate model versions and apply qconfigs
        tq.propagate_qconfig_(qmodel, qconfig_dict)
        tq.propagate_qconfig_(sqmodel, qconfig_dict)

        tq.prepare(qmodel, inplace=True)
        tq.prepare(sqmodel, inplace=True)

        # calibrate
        with torch.no_grad():
            qmodel(X_fp32)
            sqmodel(X_fp32)

        # ACTUAL TESTING BEGINS HERE
        # --------------------------

        # Make sure the quantization parameters are computed the same way
        qparams = qmodel.linear.qconfig.weight().calculate_qparams()
        sqparams = sqmodel.linear.qconfig.weight().calculate_qparams()
        test_class.assertEqual(qparams, sqparams)

        sqmodule_to_check = fqn_to_module(sqmodel, fqn_to_check)
        sqmodule_start_class = sqmodule_to_check.__class__
        sqmodule_expected_converted_class = sparse_mapping[sqmodule_start_class]

        qmodule_to_check = fqn_to_module(qmodel, fqn_to_check)
        qmodule_start_class = qmodule_to_check.__class__
        qmodule_expected_converted_class = ref_mapping[qmodule_start_class]

        # need to determine whether dynamic quantization is being performed since
        # input dtype will be different at the end
        is_dynamic = isinstance(
            qmodule_to_check.activation_post_process, tq.PlaceholderObserver
        )

        tq.convert(sqmodel, inplace=True, mapping=sparse_mapping)
        tq.convert(qmodel, inplace=True, mapping=ref_mapping)

        # this code is a duplicate of above since the references do not
        # update to the post-convert modules
        sqmodule_to_check = fqn_to_module(sqmodel, fqn_to_check)
        qmodule_to_check = fqn_to_module(qmodel, fqn_to_check)

        # check that the modules were converted as expected
        assert isinstance(
            sqmodule_to_check, sqmodule_expected_converted_class
        ), "Convert failed"
        assert isinstance(
            qmodule_to_check, qmodule_expected_converted_class
        ), "Mapping failed"

        row_block_size, col_block_size = sqmodel.linear._packed_params._weight_bias()[
            2:
        ]
        assert row_block_size == 1 and col_block_size == 4

        # only run during serialization/deserialization tests
        # makes sure script/save/load doesn't malform the sqmodel
        if test_scripting:
            scripted_sqmodel = torch.jit.script(sqmodel)
            scripted_sqmodel.eval()
            buffer = io.BytesIO()
            torch.jit.save(scripted_sqmodel, buffer)
            buffer.seek(0)
            sqmodel = torch.jit.load(buffer)

        # use correct input dtype
        if is_dynamic:
            Y_ref = qmodel(X_fp32)
            Y_hat = sqmodel(X_fp32)
            test_class.assertEqual(Y_ref, Y_hat)
        else:
            Y_ref = qmodel(X_q)
            Y_hat = sqmodel(X_q)
            test_class.assertEqual(Y_ref.dequantize(), Y_hat.dequantize())

class SparseQuantizedModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.linear(x)

class TestQuantizedSparseLayers(TestCase):
    @override_qengines
    def test_sparse_qlinear(self):
        # Note: At the moment, for sparse kernels
        # fbgemm supports only static quantized sparse linear
        # qnnpack supports only dynamically quantized sparse linear
        # Hence we have two different tests.
        # fbgemm tests static flow, qnnpack tests dynamic.
        # Should be unified later on and tests should be fixed
        # appropriately.
        model_class = SparseQuantizedModel
        fqn_to_check = "linear"
        if qengine_is_fbgemm():
            sparse_mapping = tq.get_default_static_sparse_quant_module_mappings()
            ref_mapping = tq.get_default_static_quant_module_mappings()
            qconfig_dict = {nn.Linear: tq.get_default_qconfig("fbgemm")}
        elif qengine_is_qnnpack():
            sparse_mapping = tq.get_default_dynamic_sparse_quant_module_mappings()
            ref_mapping = tq.get_default_dynamic_quant_module_mappings()
            qconfig_dict = {nn.Linear: tq.qconfig.default_dynamic_qconfig}
        else:
            return

        _sparse_layer_test_helper(
            model_class=model_class,
            sparse_mapping=sparse_mapping,
            ref_mapping=ref_mapping,
            qconfig_dict=qconfig_dict,
            fqn_to_check=fqn_to_check,
            test_class=self,
            test_scripting=False,
        )

    @override_qengines
    def test_sparse_qlinear_serdes(self):
        # Note: At the moment, for sparse kernels
        # fbgemm supports only static quantized sparse linear
        # qnnpack supports only dynamically quantized sparse linear
        # Hence we have two different tests.
        # fbgemm tests static flow, qnnpack tests dynamic.
        # Should be unified later on and tests should be fixed
        # appropriately.
        model_class = SparseQuantizedModel
        fqn_to_check = "linear"
        if qengine_is_fbgemm():
            sparse_mapping = tq.get_default_static_sparse_quant_module_mappings()
            ref_mapping = tq.get_default_static_quant_module_mappings()
            qconfig_dict = {nn.Linear: tq.get_default_qconfig("fbgemm")}
        elif qengine_is_qnnpack():
            sparse_mapping = tq.get_default_dynamic_sparse_quant_module_mappings()
            ref_mapping = tq.get_default_dynamic_quant_module_mappings()
            qconfig_dict = {nn.Linear: tq.qconfig.default_dynamic_qconfig}
        else:
            return

        _sparse_layer_test_helper(
            model_class=model_class,
            sparse_mapping=sparse_mapping,
            ref_mapping=ref_mapping,
            qconfig_dict=qconfig_dict,
            fqn_to_check=fqn_to_check,
            test_class=self,
            test_scripting=True,
        )


if __name__ == "__main__":
    run_tests()
