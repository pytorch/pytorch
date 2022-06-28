from torch.ao import sparsity
import torch
import pytorch_lightning as pl
import torch.nn as nn
from typing import List
from torch.ao.sparsity.experimental.data_sparsifier.lightning.callbacks.data_sparsity import PostTrainingDataSparsity
from torch.ao.sparsity.experimental.data_sparsifier.lightning.callbacks._data_sparstity_utils import _get_valid_name
from torch.ao.sparsity.experimental.data_sparsifier.base_data_sparsifier import SUPPORTED_TYPES
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests


class DummyModel(nn.Module):
    def __init__(self, iC: int, oC: List[int]):
        super().__init__()
        self.linears = nn.Sequential()
        self.emb = nn.Embedding(1024, 32)
        self.emb_bag = nn.EmbeddingBag(1024, 32)
        i = iC
        for idx, c in enumerate(oC):
            self.linears.append(nn.Linear(i, c, bias=False))
            if idx < len(oC) - 1:
                self.linears.append(nn.ReLU())
            i = c


class DummyLightningModule(pl.LightningModule):
    def __init__(self, iC: int, oC: List[int]):
        super().__init__()
        self.model = DummyModel(iC, oC)

    def forward(self):
        pass


class _PostTrainingCallbackTestCase(TestCase):
    def _check_on_fit_end(self, pl_module, callback, sparsifier_args):
        callback.on_fit_end(42, pl_module)  # 42 is a dummy value as trainer not used

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

    def run_all_checks(self, pl_module, callback, sparsifier_args):
        self._check_on_fit_end(pl_module, callback, sparsifier_args)


class TestPostTrainingCallback(_PostTrainingCallbackTestCase):
    def test_post_training_callback(self):
        sparsifier_args = {
            'sparsity_level': 0.5,
            'sparse_block_shape': (1, 4),
            'zeros_per_block': 4
        }
        callback = PostTrainingDataSparsity(data_sparsifier_type=sparsity.DataNormSparsifier,
                                            data_sparsifier_args=sparsifier_args)
        pl_module = DummyLightningModule(100, [128, 256, 16])

        self.run_all_checks(pl_module, callback, sparsifier_args)


if __name__ == "__main__":
    run_tests()
