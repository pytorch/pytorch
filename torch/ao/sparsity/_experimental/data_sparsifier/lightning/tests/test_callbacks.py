from torch.ao.sparsity._experimental.data_sparsifier.data_norm_sparsifier import DataNormSparsifier
import torch
import torch.nn as nn
from typing import List
from torch.ao.sparsity._experimental.data_sparsifier.lightning.callbacks.data_sparsity import PostTrainingDataSparsity
from torch.ao.sparsity._experimental.data_sparsifier.lightning.callbacks._data_sparstity_utils import _get_valid_name
from torch.ao.sparsity._experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests
import importlib
import unittest


class DummyModel(nn.Module):
    def __init__(self, iC: int, oC: List[int]):
        super().__init__()
        self.linears = nn.Sequential()
        i = iC
        for idx, c in enumerate(oC):
            self.linears.append(nn.Linear(i, c, bias=False))
            if idx < len(oC) - 1:
                self.linears.append(nn.ReLU())
            i = c


def _make_lightning_module(iC: int, oC: List[int]):
    import pytorch_lightning as pl  # type: ignore[import]

    class DummyLightningModule(pl.LightningModule):
        def __init__(self, ic: int, oC: List[int]):
            super().__init__()
            self.model = DummyModel(iC, oC)

        def forward(self):
            pass

    return DummyLightningModule(iC, oC)


class TestPostTrainingCallback(TestCase):
    def _check_on_fit_end(self, pl_module, callback, sparsifier_args):
        """Makes sure that each component of is working as expected while calling the
        post-training callback.
        Specifically, check the following -
            1. sparsifier config is the same as input config
            2. data sparsifier is correctly attached to the model
            3. sparsity is achieved after .step()
            4. non-sparsified values are the same as original values
        """
        callback.on_fit_end(42, pl_module)  # 42 is a dummy value

        # check sparsifier config
        for key, value in sparsifier_args.items():
            assert callback.data_sparsifier.defaults[key] == value

        # assert that the model is correctly attached to the sparsifier
        for name, param in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            if type(param) not in SUPPORTED_TYPES:
                assert valid_name not in callback.data_sparsifier.state
                assert valid_name not in callback.data_sparsifier.data_groups
                continue
            assert valid_name in callback.data_sparsifier.data_groups
            assert valid_name in callback.data_sparsifier.state

            mask = callback.data_sparsifier.get_mask(name=valid_name)

            # assert that some level of sparsity is achieved
            assert (1.0 - mask.float().mean()) > 0.0

            # make sure that non-zero values in data after squash mask are equal to original values
            sparsified_data = callback.data_sparsifier.get_data(name=valid_name, return_original=False)
            assert torch.all(sparsified_data[sparsified_data != 0] == param[sparsified_data != 0])

    @unittest.skipIf(not importlib.util.find_spec("pytorch_lightning"), "No pytorch_lightning")
    def test_post_training_callback(self):
        sparsifier_args = {
            'sparsity_level': 0.5,
            'sparse_block_shape': (1, 4),
            'zeros_per_block': 4
        }
        callback = PostTrainingDataSparsity(DataNormSparsifier, sparsifier_args)
        pl_module = _make_lightning_module(100, [128, 256, 16])

        self._check_on_fit_end(pl_module, callback, sparsifier_args)


if __name__ == "__main__":
    run_tests()
