# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]


import logging

import torch
import torch.ao.quantization as tq
from torch import nn
from torch.ao import sparsity
from torch.testing._internal.common_utils import TestCase

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

sparse_defaults = {
    "sparsity_level": 0.8,
    "sparse_block_shape": (1, 4),
    "zeros_per_block": 4,
}

# This series of tests are to check the composability goals for sparsity and quantization. Namely
# that performing quantization and sparsity model manipulations in various orderings
# does not cause problems
class TestComposability(TestCase):
    def _get_model_and_sparsifier_and_sparse_config(self, qconfig=None):
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
        if qconfig is None:
            model[4].qconfig = tq.get_default_qconfig("fbgemm")
            model[5].qconfig = tq.get_default_qconfig("fbgemm")
        else:
            model[4].qconfig = qconfig
            model[5].qconfig = qconfig

        sparsifier = sparsity.WeightNormSparsifier(**sparse_defaults)

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

    def _squash_mask_calibrate_and_convert(self, model, sparsifier, input):
        sparsifier.step()
        sparsifier.squash_mask()
        model(input)
        tq.convert(model, inplace=True)

    def _calculate_sparsity(self, tensor):
        return ((tensor == 0).sum() / tensor.numel()).item()

    # This test checks whether performing quantization prepare before sparse prepare
    # causes any issues and verifies that the correct observers are inserted and that
    # the quantized model works as expected
    def test_q_prep_before_s_prep(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = self._get_model_and_sparsifier_and_sparse_config()

        tq.prepare(mod, inplace=True)
        sparsifier.prepare(mod, config=sparse_config)

        # check that correct modules had parametrizations added
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))
        # check that correct observers were inserted
        self.assertTrue(hasattr(mod[5], "activation_post_process"))

        self._squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.nn.quantized.Linear))
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
        ) = self._get_model_and_sparsifier_and_sparse_config()

        sparsifier.prepare(mod, config=sparse_config)
        tq.prepare(mod, inplace=True)

        # check that correct modules had parametrizations added and
        # that none were lost during prepare
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(hasattr(mod[5], "activation_post_process"))

        self._squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.nn.quantized.Linear))
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
        ) = self._get_model_and_sparsifier_and_sparse_config()

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
        sparsity_level = self._calculate_sparsity(mod[5].weight)
        mod(torch.randn(1, 4, 4, 4))
        tq.convert(mod, inplace=True)

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = self._calculate_sparsity(mod[5]._weight_bias()[0])
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
        ) = self._get_model_and_sparsifier_and_sparse_config()
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
        self._squash_mask_calibrate_and_convert(
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
        ) = self._get_model_and_sparsifier_and_sparse_config()
        tq.fuse_modules(mod, [["5", "6"]], inplace=True)

        # its absolutely broken by fusion but will still work if you put the correct fqn in
        sparse_config = [
            {
                "tensor_fqn": "5.0.weight",
                "sparsity_level": 0.7,
                "sparse_block_shape": (1, 4),
                "zeros_per_block": 4,
            },
            {"tensor_fqn": ".0.weight"},
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
        sparsity_level = self._calculate_sparsity(mod[5][0].weight)
        mod(torch.randn(1, 4, 4, 4))
        tq.convert(mod, inplace=True)

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = self._calculate_sparsity(mod[5]._weight_bias()[0])
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
        ) = self._get_model_and_sparsifier_and_sparse_config(
            tq.get_default_qat_qconfig("fbgemm")
        )
        sparsifier.prepare(mod, config=sparse_config)
        tq.prepare_qat(mod, inplace=True)
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        self.assertTrue(isinstance(mod[5], torch.nn.qat.Linear))
        self._squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )
        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = self._calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

    # This tests whether performing qat prepare before sparse prepare causes issues.
    def test_qat_prep_before_s_prep(self):
        mod, sparsifier, _ = self._get_model_and_sparsifier_and_sparse_config(
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
            {"tensor_fqn": ".0.weight"},
        ]
        sparsifier.prepare(mod, config=sparse_config)

        # check that correct modules had parametrizations added and
        # that none were lost during qat prepare
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))

        # check that correct observers were inserted and that matching
        # occured successfully
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        self.assertTrue(isinstance(mod[5], torch.nn.qat.Linear))

        self._squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )

        # check that final module is the expected quantized module and that the model runs
        self.assertTrue(isinstance(mod[5], torch.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        # check that module was actually sparsified
        cur_sparsity = self._calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])
