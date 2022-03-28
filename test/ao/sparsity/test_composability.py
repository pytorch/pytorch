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


class TestComposability(TestCase):
    def _get_model_and_sparsifier_and_sparse_config(self):
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
        model[4].qconfig = tq.get_default_qconfig("fbgemm")
        model[5].qconfig = tq.get_default_qconfig("fbgemm")

        sparsifier = sparsity.WeightNormSparsifier(**sparse_defaults)

        sparse_config = [
            {
                "module": model[5],
                "sparsity_level": 0.7,
                "sparse_block_shape": (1, 4),
                "zeros_per_block": 4,
            },
            model[0],
        ]
        return model, sparsifier, sparse_config

    def _squash_mask_calibrate_and_convert(self, model, sparsifier, input):
        sparsifier.step()
        sparsifier.squash_mask()
        model(input)
        tq.convert(model, inplace=True)

    def _calculate_sparsity(self, tensor):
        return ((tensor == 0).sum() / tensor.numel()).item()

    def test_q_prep_before_s_prep(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = self._get_model_and_sparsifier_and_sparse_config()

        tq.prepare(mod, inplace=True)
        sparsifier.prepare(mod, config=sparse_config)
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        self._squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )
        self.assertTrue(isinstance(mod[5], torch.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    def test_s_prep_before_q_prep(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = self._get_model_and_sparsifier_and_sparse_config()

        sparsifier.prepare(mod, config=sparse_config)
        tq.prepare(mod, inplace=True)
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        self._squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )
        self.assertTrue(isinstance(mod[5], torch.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    def test_convert_without_squash_mask(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = self._get_model_and_sparsifier_and_sparse_config()

        sparsifier.prepare(mod, config=sparse_config)
        tq.prepare(mod, inplace=True)
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        sparsifier.step()
        sparsity_level = self._calculate_sparsity(mod[5].weight)
        mod(torch.randn(1, 4, 4, 4))
        tq.convert(mod, inplace=True)
        self.assertTrue(isinstance(mod[5], torch.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        cur_sparsity = self._calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])

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
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5][0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        self._squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )
        self.assertTrue(isinstance(mod[5], torch.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

    def test_fusion_before_s_prep(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = self._get_model_and_sparsifier_and_sparse_config()
        tq.fuse_modules(mod, [["5", "6"]], inplace=True)
        sparsifier.prepare(mod, config=sparse_config)
        mod[5].qconfig = tq.get_default_qconfig("fbgemm")
        tq.prepare(mod, inplace=True)
        self.assertTrue(hasattr(mod[0], "parametrizations"))
        self.assertTrue(hasattr(mod[5][0], "parametrizations"))
        self.assertTrue(hasattr(mod[5], "activation_post_process"))
        sparsifier.step()
        sparsity_level = self._calculate_sparsity(mod[5][0].weight)
        mod(torch.randn(1, 4, 4, 4))
        tq.convert(mod, inplace=True)
        self.assertTrue(isinstance(mod[5], torch.nn.intrinsic.quantized.LinearReLU))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))

        cur_sparsity = self._calculate_sparsity(mod[5]._weight_bias()[0])
        self.assertGreaterAlmostEqual(cur_sparsity, sparsity_level)
        self.assertGreaterAlmostEqual(
            sparsity_level, sparse_config[0]["sparsity_level"]
        )
        self.assertGreaterAlmostEqual(cur_sparsity, sparse_config[0]["sparsity_level"])
