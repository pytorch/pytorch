# Owner(s): ["module: unknown"]
import unittest

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.sac_estimator import SACEstimator
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
)


class TestSACEstimator(TestCase):
    def _sac_estimation(
        self,
        estimate_mode: str,
        model: torch.nn.Module,
        inp: torch.Tensor,
    ):
        sace = SACEstimator()
        with sace(estimate_mode_type=estimate_mode):
            loss = model(inp).sum()
        loss.backward()
        sace.pwlf_sac_tradeoff_curve(n_segments=2, save_tradeoff_graphs=False)

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/115653")
    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_transformer_sac_estimation(
        self,
    ):
        """Runs a basic GPT-2 model"""
        dev = torch.cuda.current_device()
        vocab_size = 8192
        bsz, seq_len = 8, 1024
        model_args = ModelArgs(
            n_layers=4,
            n_heads=12,
            vocab_size=vocab_size,
            max_seq_len=seq_len,
            dim=768,
            dropout_p=0.1,
        )
        with FakeTensorMode():
            with torch.device(dev):
                model = Transformer(model_args)
            inp = torch.randint(
                0, model_args.vocab_size, (bsz, model_args.max_seq_len), device=dev
            )

            self._sac_estimation("operator-level-benchmark", model, inp)
            self._sac_estimation("operator-level-cost-model", model, inp)


if __name__ == "__main__":
    run_tests()
