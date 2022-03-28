# -*- coding: utf-8 -*-
# Owner(s): ["module: unknown"]


import logging

import torch
import torch.quantization as tq
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
            nn.Identity(),
            # nn.ReLU(), not testing fusion yet
            tq.DeQuantStub(),
        )
        model[5].qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")
        model[4].qconfig = torch.ao.quantization.get_default_qconfig("fbgemm")

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

    def _check_parametrizations_and_observers(self, model):
        self.assertTrue(hasattr(model[0], "parametrizations"))
        self.assertTrue(hasattr(model[5], "parametrizations"))
        self.assertTrue(hasattr(model[5], "activation_post_process"))

    def _squash_mask_calibrate_and_convert(self, model, sparsifier, input):
        sparsifier.step()
        sparsifier.squash_mask()
        model(input)
        tq.convert(model, inplace=True)

    def test_q_prep_before_s_prep(self):
        (
            mod,
            sparsifier,
            sparse_config,
        ) = self._get_model_and_sparsifier_and_sparse_config()

        tq.prepare(mod, inplace=True)
        sparsifier.prepare(mod, config=sparse_config)
        self._check_parametrizations_and_observers(mod)
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
        torch.quantization.prepare(mod, inplace=True)
        self._check_parametrizations_and_observers(mod)
        self._squash_mask_calibrate_and_convert(
            mod, sparsifier, torch.randn(1, 4, 4, 4)
        )
        self.assertTrue(isinstance(mod[5], torch.nn.quantized.Linear))
        self.assertEqual(mod(torch.randn(1, 4, 4, 4)).shape, torch.Size([1, 4, 4, 4]))
