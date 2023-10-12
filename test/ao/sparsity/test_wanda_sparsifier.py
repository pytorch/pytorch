import itertools
import logging
import re

import torch
from torch import nn
from torch.ao.pruning import BaseSparsifier, WeightNormSparsifier, FakeSparsity, NearlyDiagonalSparsifier, WandaSparsifier
from torch.nn.utils.parametrize import is_parametrized

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_pruning import SimpleLinear, MockSparseLinear, ImplementedSparsifier

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


class TestWandaSparsifier(TestCase):
    """
    Test WandA Sparsifier.
    """

    def test_prepare(self):
        model = SimpleLinear()
        sparsifier = WandaSparsifier()
        sparsifier.prepare(model, config=None)
        for g in sparsifier.groups:
            module = g['module']
            # Check mask exists
            assert hasattr(module.parametrizations['weight'][0], 'mask')
            # Check parametrization exists and is correct
            assert is_parametrized(module, 'weight')
            assert type(module.parametrizations.weight[0]) == FakeSparsity
            # check activation observer is present
            assert hasattr(module, 'activation_post_process')

    def test_squash_mask(self):
        model = SimpleLinear()
        sparsifier = WandaSparsifier()
        sparsifier.prepare(model, config=None)
        sparsifier.squash_mask()
        for g in sparsifier.groups:
            module = g['module']
            assert not is_parametrized(module, 'weight')
            assert not hasattr(module, 'mask')
            assert not hasattr(module, 'activation_post_process')

    def test_semi_structured(self):
        sparsity_levels = [-1.0, 0.0, 0.5, 1.0, 2.0]
        sparse_block_shapes = [(1, 1), (1, 4), (2, 2), (4, 1)]
        zeros_per_blocks = [0, 1, 2, 3, 4]

        testcases = itertools.tee(itertools.product(sparsity_levels,
                                                    sparse_block_shapes,
                                                    zeros_per_blocks))
        # Create a config and model with all the testcases
        model = nn.Sequential()
        sparsifier = WeightNormSparsifier()

        sparsity_per_layer_config = []
        p = re.compile(r'[-\.\s]')
        for sl, sbs, zpb in testcases[0]:
            # Make sure the number of zeros is not > values in a block
            if zpb > sbs[0] * sbs[1]:
                continue
            layer_name = f'{sl}_{sbs}_{zpb}'
            layer_name = p.sub('_', layer_name)

            layer = nn.Linear(12, 12, bias=False)
            layer.weight = nn.Parameter(torch.ones(12, 12))
            model.add_module(layer_name, layer)
            config = {
                'tensor_fqn': layer_name + ".weight",
                'sparsity_level': sl,
                'sparse_block_shape': sbs,
                'zeros_per_block': zpb
            }
            sparsity_per_layer_config.append(config)

        sparsifier.prepare(model, sparsity_per_layer_config)
        sparsifier.step()
        sparsifier.squash_mask()
        model.eval()

        for sl, sbs, zpb in testcases[1]:
            if zpb > sbs[0] * sbs[1]:
                continue
            layer_name = f'{sl}_{sbs}_{zpb}'
            layer_name = p.sub('_', layer_name)
            layer = getattr(model, layer_name)

            # Level of sparsity is achieved
            sparse_mask = (layer.weight == 0).float()
            if zpb == 0:
                assert sparse_mask.mean() == 0
            else:
                # Ratio of individual zeros in the tensor
                true_sl = min(max(sl, 0.0), 1.0)
                true_sl = true_sl * zpb / sbs[0] / sbs[1]
                assert sparse_mask.mean() == true_sl

    def test_one_layer_mlp_2x4(self):
        model = nn.Sequential(nn.Linear(8,1))
        weights = torch.tensor([[1,2,3,4, 5, 6, 7, 8]])
        model[0].weight.data.copy_(weights.data)
        X = torch.ones(1,8)

        sparsifier = WandaSparsifier(sparsity_level=0.5, semi_structured_block_size=4)
        sparsifier.prepare(model, config=None)

        model(X)

        sparsifier.step()
        sparsifier.squash_mask()

        sparsity = (model[0].weight == 0).float().mean()
        assert sparsity == 0.5, "sparsity should be 0.5"
        expected_fc = torch.tensor([[0,0,3,4, 0, 0, 7, 8]], dtype=torch.float32)
        assert torch.isclose(model[0].weight.data, expected_fc, rtol=1e-05, atol=1e-07).all()

    def test_one_layer_mlp_unstructured(self):
        model = nn.Sequential(nn.Linear(4,1))
        weights = torch.tensor([[1,2,3,4]], dtype=torch.float32)
        model[0].weight.data.copy_(weights.data)
        X = torch.tensor([[100, 10, 1, 0.1]], dtype=torch.float32)

        sparsifier = WandaSparsifier(sparsity_level=0.5)
        sparsifier.prepare(model, config=None)

        model(X)

        sparsifier.step()
        sparsifier.squash_mask()

        sparsity = (model[0].weight == 0).float().mean()
        assert sparsity == 0.5, "sparsity should be 0.5"
        expected_fc = torch.tensor([[1,2,0,0]], dtype=torch.float32)
        assert torch.allclose(model[0].weight.data, expected_fc, rtol=1e-05, atol=1e-07)


    def test_two_layer_mlp_2x4(self):
        model = nn.Sequential(nn.Linear(128, 200), nn.ReLU(), nn.Linear(200, 10))      ## C_in by C_out
        X1 = torch.randn(100,128)           ## B1 by C_in
        X2 = torch.randn(50, 128)           ## B2 by C_in

        sparsifier = WandaSparsifier(sparsity_level=0.5)
        sparsifier.prepare(model, config=None)

        model(X1)
        model(X2)
        sparsifier.step()

        cnt = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                cnt += 1
                sparsity_level = (m.weight == 0).float().mean()
                # print(f"Level of sparsity for Linear layer {cnt}: {sparsity_level.item():.2%}")
                assert sparsity_level == 0.5, f"sparsity for linear layer {cnt} should be 0.5"

        sparsifier.squash_mask()
