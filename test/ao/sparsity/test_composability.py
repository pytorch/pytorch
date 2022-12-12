# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]


import logging

import torch
import torch.ao.quantization as tq
from torch import nn
from torch.ao import pruning
from torch.testing._internal.common_utils import TestCase
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, convert_to_reference_fx, prepare_qat_fx
from torch.ao.pruning import fqn_to_module

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

sparse_defaults = {
    "sparsity_level": 0.8,
    "sparse_block_shape": (1, 4),
    "zeros_per_block": 4,
}

def _get_model_and_sparsifier_and_sparse_config(qconfig=None):
    model = nn.Sequential(
        nn.Linear(4, 4),  # 0
        nn.ReLU(),
        nn.Linear(4, 4),  # 2
        nn.ReLU(),
        tq.QuantStub(),
        nn.Linear(4, 4),  # 5
        nn.ReLU(),
        tq.DeQuantStub(),
    )
    if qconfig:
        model[4].qconfig = qconfig
        model[5].qconfig = qconfig

    sparsifier = pruning.WeightNormSparsifier(**sparse_defaults)

    sparse_config = [
        {
            "tensor_fqn": '5.weight',
            "sparsity_level": 0.7,
            "sparse_block_shape": (1, 4),
            "zeros_per_block": 4,
        },
        {"tensor_fqn": "0.weight"},
    ]
    return model, sparsifier, sparse_config

def _squash_mask_calibrate_and_convert(model, sparsifier, input):
    sparsifier.step()
    sparsifier.squash_mask()
    model(input)
    tq.convert(model, inplace=True)

def _calculate_sparsity(tensor):
    return ((tensor == 0).sum() / tensor.numel()).item()

# This series of tests are to check the composability goals for sparsity and quantization. Namely
# that performing quantization and sparsity model manipulations in various orderings
# does not cause problems
class TestComposability(TestCase):
    # This test checks whether performing quantization prepare before sparse prepare
    # causes any issues and verifies that the correct observers are inserted and that
    # the quantized model works as expected
    def test_q_prep_before_s_prep(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qconfig("fbgemm"))

        tq.prepare(mod, inplace=True)
        sparsifier.prepare(mod, config=sparse_config)

        # check that correct modules had parametrizations added
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))
        # check that correct observers were inserted
        self.assertTrue(hasattr(mod[5], "activation_post_process"))

        _squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    # This test checks whether performing sparsity prepare before quantization prepare
    # causes any issues. In particular, previous quantization flow was unable to match
    # the post sparse prepare module names (adding parametrizations changes the module class names)
    # which would result in those parametrized modules not being quantized. This test verifies that
    # the fix for this was successful.
    def test_s_prep_before_q_prep(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qconfig("fbgemm"))

        sparsifier.prepare(mod, config=sparse_config)
        tq.prepare(mod, inplace=True)

        # check that correct modules had parametrizations added and
        # that none were lost during prepare
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(hasattr(mod[5], "activation_post_process"))

        _squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    # if the sparsified modules have not undergone the final squash mask operation, its possible
    # that the problem outlined in test_s_prep_before_q_prep would occur. This test verifies
    # both that the fix to the convert flow avoids this issue and that the resulting quantized
    # module uses the sparse version of the weight value.
    def test_convert_without_squash_mask(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qconfig("fbgemm"))

        sparsifier.prepare(mod, config=sparse_config)
        tq.prepare(mod, inplace=True)

        # check that correct modules had parametrizations added and
        # that none were lost during prepare
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(mod[5].weight)
        mod(torch.randn(1, 4, 4, 4))
        tq.convert(mod, inplace=True)

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

    # This tests whether performing sparse prepare before fusion causes any issues. The
    # worry was that the link created between the sparsifier and the modules that need to
    # be sparsified would be broken.
    def test_s_prep_before_fusion(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qconfig("fbgemm"))
        sparsifier.prepare(mod, config=sparse_config)
        tq.fuse_modules(mod, [["5", "6"]], inplace=True)
        mod[5].qconfig = tq.get_default_qconfig("fbgemm")
        tq.prepare(mod, inplace=True)

        # check that correct modules had parametrizations added and
        # that none were lost during prepare or fusion
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5][0], "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        _squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    # This tests whether performing fusion before sparse prepare causes and issues. The
    # main worry was that the links to the modules in the sparse config would be broken by fusion.
    def test_fusion_before_s_prep(self):
        (
            mod,
            sparsifier,
            _,
        ) = _get_model_and_sparsifier_and_sparse_config(tq.get_default_qconfig("fbgemm"))
        tq.fuse_modules(mod, [["5", "6"]], inplace=True)

        # its absolutely broken by fusion but will still work if you put the correct fqn in
        sparse_config = [
            {
                "tensor_fqn": "5.0.weight",
                "sparsity_level": 0.7,
                "sparse_block_shape": (1, 4),
                "zeros_per_block": 4,
            },
            {"tensor_fqn": "0.weight"},
        ]

        sparsifier.prepare(mod, config=sparse_config)
        mod[5].qconfig = tq.get_default_qconfig("fbgemm")
        tq.prepare(mod, inplace=True)

        # check that correct modules had parametrizations added and
        # that none were lost during prepare
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5][0], "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(mod[5][0].weight)
        mod(torch.randn(1, 4, 4, 4))
        tq.convert(mod, inplace=True)

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

    # This tests whether performing sparse prepare before qat prepare causes issues.
    # The primary worries were that qat_prep wouldn't recognize the parametrized
    # modules and that the convert step for qat would remove the paramerizations
    # from the modules.
    def test_s_prep_before_qat_prep(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config(
            tq.get_default_qat_qconfig("fbgemm")
        )
        sparsifier.prepare(mod, config=sparse_config)
        tq.prepare_qat(mod, inplace=True)
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        self.assertTrue(isinstance(mod[5], torch.ao.nn.qat.Linear))
        _squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )
        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

    # This tests whether performing qat prepare before sparse prepare causes issues.
    def test_qat_prep_before_s_prep(self):
        mod, sparsifier, _ = _get_model_and_sparsifier_and_sparse_config(
            tq.get_default_qat_qconfig("fbgemm")
        )
        tq.prepare_qat(mod, inplace=True)

        # need to setup sparse_config on new modules
        sparse_config = [
            {
                "tensor_fqn": "5.weight",
                "sparsity_level": 0.7,
                "sparse_block_shape": (1, 4),
                "zeros_per_block": 4,
            },
            {"tensor_fqn": "0.weight"},
        ]
        sparsifier.prepare(mod, config=sparse_config)

        # check that correct modules had parametrizations added and
        # that none were lost during qat prepare
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        self.assertTrue(isinstance(mod[5], torch.ao.nn.qat.Linear))

        _squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.ao.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = _calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

def _module_has_activation_post_process(model, fqn_of_module):
    for node in model.graph.nodes:
        # look for an observer whose arg is the target module
        if "activation_post_process" in node.name:
            if node.args[0].target == fqn_of_module:
                return True
    return False

class TestFxComposability(TestCase):
    r"""This series of tests checks that various steps of the quantization and sparsity flow
    compose cleanly despite variation in sequencing.
    """
    def test_q_prep_fx_before_s_prep(self):
        r"""
        This test checks that the ordering of prepare_fx -> sparse prepare -> convert_fx
        compose cleanly without issue and that the final result is sparsified without
        having to call squash mask between sparse prepare and convert_fx. This also tests the
        automatic fusion that occurs during prepare_fx.
        """
        (
            mod,
            sparsifier,
            _,
        ) = _get_model_and_sparsifier_and_sparse_config()

        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qconfig("fbgemm")
        qconfig_mapping = tq.QConfigMapping() \
            .set_module_name("4", qconfig) \
            .set_module_name("5", qconfig)


        mod = prepare_fx(mod, qconfig_mapping, (example,))

        # its absolutely broken by auto fusion in fx
        # but will still work if you put the correct fqn in
        sparse_config = [
            {
                "tensor_fqn": "5.0.weight",
                "sparsity_level": 0.7,
                "sparse_block_shape": (1, 4),
                "zeros_per_block": 4,
            },
            {"tensor_fqn": "0.0.weight"},
        ]
        sparsifier.prepare(mod, config=sparse_config)

        # check that correct modules had parametrizations added and
        # that none were lost during prepare
        self.assertTrue(hasattr(fqn_to_module(mod, "0.0"), "parametrizations"))
        self.assertTrue(hasattr(fqn_to_module(mod, "5.0"), "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(_module_has_activation_post_process(mod, "5"))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        mod(example)
        mod = convert_fx(mod)

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(fqn_to_module(mod, "5"), torch.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, "5")._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

    def test_q_prep_fx_s_prep_ref_conv(self):
        r"""
        This checks that the ordering: prepare_fx -> sparse prepare -> convert_to_reference_fx
        compose cleanly without issue and that the final result is sparsified without
        having to call squash mask before convert_to_reference_fx.
        """
        (
            mod,
            sparsifier,
            _,
        ) = _get_model_and_sparsifier_and_sparse_config()

        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qconfig("fbgemm")
        qconfig_mapping = tq.QConfigMapping() \
            .set_module_name("4", qconfig) \
            .set_module_name("5", qconfig)

        mod = prepare_fx(mod, qconfig_mapping, (example,))

        # its absolutely broken by auto fusion in fx
        # but will still work if you put the correct fqn in
        sparse_config = [
            {
                "tensor_fqn": "5.0.weight",
                "sparsity_level": 0.7,
                "sparse_block_shape": (1, 4),
                "zeros_per_block": 4,
            },
            {"tensor_fqn": "0.0.weight"},
        ]
        sparsifier.prepare(mod, config=sparse_config)

        # check that correct modules had parametrizations added and
        # that none were lost during prepare
        self.assertTrue(hasattr(fqn_to_module(mod, "0.0"), "parametrizations"))
        self.assertTrue(hasattr(fqn_to_module(mod, "5.0"), "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(_module_has_activation_post_process(mod, "5"))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        mod(example)
        mod = convert_to_reference_fx(mod)

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(fqn_to_module(mod, "5"), torch.nn.intrinsic.LinearReLU))
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))
        self.assertTrue(isinstance(fqn_to_module(mod, "5.0"), torch.nn.quantized._reference.Linear))

        # check that module was actually sparsified
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

    def test_s_prep_before_q_prep_fx(self):
        r"""
        This test checks that the ordering of sparse prepare -> prepare_fx -> convert_fx
        compose cleanly without issue and that the final result is sparsified without
        having to call squash mask before convert_fx.
        """
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config()
        sparsifier.prepare(mod, config=sparse_config)

        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qconfig("fbgemm")
        qconfig_mapping = tq.QConfigMapping() \
            .set_module_name("4", qconfig) \
            .set_module_name("5", qconfig)
        mod = prepare_fx(mod, qconfig_mapping, (example,))

        # check that correct modules had parametrizations added and
        # that none were lost during prepare
        self.assertTrue(hasattr(fqn_to_module(mod, "0.0"), "parametrizations"))
        self.assertTrue(hasattr(fqn_to_module(mod, "5.0"), "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(_module_has_activation_post_process(mod, "5"))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        mod(example)
        mod = convert_fx(mod)

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(fqn_to_module(mod, "5"), torch.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, "5")._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

    def test_s_prep_before_qat_prep_fx(self):
        r"""
        This test checks that the ordering of sparse prepare -> prepare_qat_fx -> convert_fx
        compose cleanly without issue and that the final result is sparsified without
        having to call squash mask before convert_fx.
        """
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config()
        sparsifier.prepare(mod, config=sparse_config)

        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qat_qconfig("fbgemm")
        qconfig_mapping = tq.QConfigMapping() \
            .set_module_name("4", qconfig) \
            .set_module_name("5", qconfig)
        mod = prepare_qat_fx(mod, qconfig_mapping, (example,))

        # check that correct modules had parametrizations added and
        # that none were lost during prepare
        self.assertTrue(hasattr(fqn_to_module(mod, "0.0"), "parametrizations"))
        self.assertTrue(hasattr(fqn_to_module(mod, "5"), "parametrizations"))
        self.assertTrue(isinstance(fqn_to_module(mod, "5"), torch.ao.nn.intrinsic.qat.LinearReLU))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(_module_has_activation_post_process(mod, "5"))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, "5.weight"))
        mod(example)
        mod = convert_fx(mod)

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(fqn_to_module(mod, "5"), torch.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, "5")._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

    def test_s_prep_q_prep_fx_ref(self):
        r"""
        This checks that the ordering: sparse prepare -> prepare_fx -> convert_to_reference_fx
        compose cleanly without issue and that the final result is sparsified without
        having to call squash mask before convert_to_reference_fx.
        """
        (
            mod,
            sparsifier,
            sparse_config,
        ) = _get_model_and_sparsifier_and_sparse_config()
        sparsifier.prepare(mod, config=sparse_config)

        example = torch.randn(1, 4, 4, 4)
        qconfig = tq.get_default_qconfig("fbgemm")
        qconfig_mapping = tq.QConfigMapping() \
            .set_module_name("4", qconfig) \
            .set_module_name("5", qconfig)
        mod = prepare_fx(mod, qconfig_mapping, (example,))

        # check that correct modules had parametrizations added and
        # that none were lost during prepare
        self.assertTrue(hasattr(fqn_to_module(mod, "0.0"), "parametrizations"))
        self.assertTrue(hasattr(fqn_to_module(mod, "5.0"), "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(_module_has_activation_post_process(mod, "5"))
        sparsifier.step()
        sparsity_level = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        mod(example)
        mod = convert_to_reference_fx(mod)

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(fqn_to_module(mod, "5"), torch.nn.intrinsic.LinearReLU))
        self.assertEqual(mod(example).shape, torch.Size([1, 4, 4, 4]))
        self.assertTrue(isinstance(fqn_to_module(mod, "5.0"), torch.nn.quantized._reference.Linear))

        # check that module was actually sparsified
        cur_sparsity = _calculate_sparsity(fqn_to_module(mod, "5.0.weight"))
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])
