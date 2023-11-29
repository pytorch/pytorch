# Owner(s): ["oncall: distributed"]

import copy
import functools
from typing import Iterable, List, Union

import torch
import torch.nn as nn
from _test_fully_shard_common import MLP
from torch.distributed._composable import replicate
from torch.distributed._composable.fsdp import clip_grad_norm_, fully_shard
from torch.distributed._tensor import DTensor
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests


class TestFullyShardClipGradNorm(FSDPTest):
    @property
    def world_size(self) -> int:
        return min(4, torch.cuda.device_count())

    def _build_model_and_optim(self, lin_dim: int = 4, build_extra_ddp: bool = False):
        torch.manual_seed(42)
        model = nn.Sequential(
            MLP(lin_dim, torch.device("cpu"), dim_multiplier=3),
            MLP(lin_dim, torch.device("cpu")),
            MLP(lin_dim, torch.device("cpu"), dim_multiplier=3),
        )

        if build_extra_ddp:
            ref_model = copy.deepcopy(model).cuda()
            replicate(ref_model, device_ids=[self.rank])
            ref_optim = torch.optim.SGD(ref_model.parameters(), lr=1e-2)

        for mlp in model:
            fully_shard(mlp)
        fully_shard(model)
        optim = torch.optim.SGD(model.parameters(), lr=1e-2)

        if build_extra_ddp:
            return model, optim, ref_model, ref_optim
        else:
            return model, optim

    @skip_if_lt_x_gpu(2)
    def test_clip_grad_norm_1d(self):
        self.run_subtests(
            {
                "max_norm": [1, 2.5],
                "norm_type": [1, 2, float("inf")],
            },
            self._test_clip_grad_norm_1d,
        )

    def _test_clip_grad_norm_1d(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int],
    ):
        lin_dim = 4
        model, optim, ref_model, ref_optim = self._build_model_and_optim(
            lin_dim, build_extra_ddp=True
        )

        torch.manual_seed(42 + self.rank + 1)
        vector_norm_fn = functools.partial(torch.linalg.vector_norm, ord=norm_type)

        for iter in range(10):
            x = torch.rand((32, lin_dim), device="cuda")
            total_norms: List[torch.Tensor] = []
            for _model, _optim in ((ref_model, ref_optim), (model, optim)):
                _optim.zero_grad()
                loss = _model(x).sum()
                loss.backward()

                orig_grads_noncloned = [
                    param.grad if _model is ref_model else param.grad._local_tensor
                    for param in _model.parameters()
                ]
                orig_grads = [grad.detach().clone() for grad in orig_grads_noncloned]
                greater_mask = [
                    vector_norm_fn(grad).item() > max_norm for grad in orig_grads
                ]
                self.assertTrue(
                    any(greater_mask),
                    "at least one grad should be greater than max_norm before grad clipping",
                )

                if _model is ref_model:
                    total_norm = torch.nn.utils.clip_grad_norm_(
                        _model.parameters(), max_norm=max_norm, norm_type=norm_type
                    )
                    total_norms.append(total_norm)
                    for param in _model.parameters():
                        self.assertTrue(vector_norm_fn(param.grad).item() <= max_norm)
                else:
                    total_norm = clip_grad_norm_(
                        _model.parameters(), max_norm=max_norm, norm_type=norm_type
                    )
                    total_norms.append(total_norm)
                    for param in _model.parameters():
                        self.assertTrue(
                            vector_norm_fn(param.grad._local_tensor).item() <= max_norm
                        )

                # all-zero grads remains the same
                # since zero / total_norm is still zero
                # otherwise grad should be clipped
                # because orignal norm > max_norm
                for param, orig_grad in zip(_model.parameters(), orig_grads):
                    if torch.count_nonzero(orig_grad):
                        if _model == ref_model:
                            self.assertFalse(torch.equal(param.grad, orig_grad))
                        else:
                            self.assertFalse(
                                torch.equal(param.grad._local_tensor, orig_grad)
                            )
                    else:
                        if _model == ref_model:
                            self.assertEqual(param.grad, orig_grad)
                        else:
                            self.assertEqual(param.grad._local_tensor, orig_grad)

                _optim.step()
            self.assertEqual(total_norms[0], total_norms[1])

    @skip_if_lt_x_gpu(2)
    def test_clip_grad_norm_no_grads(self):
        self.run_subtests(
            {
                "max_norm": [1, 2.5],
                "norm_type": [1, 2, float("inf")],
            },
            self._test_clip_grad_norm_no_grads,
        )

    def _test_clip_grad_norm_no_grads(
        self,
        max_norm: Union[float, int],
        norm_type: Union[float, int],
    ):
        lin_dim = 4
        model, _ = self._build_model_and_optim(lin_dim)
        total_norm = clip_grad_norm_(
            model.parameters(), max_norm=max_norm, norm_type=norm_type
        )
        self.assertEqual(total_norm.dtype, torch.float32)
        self.assertEqual(total_norm, torch.tensor(0.0, device="cuda"))

    @skip_if_lt_x_gpu(2)
    def test_clip_grad_norm_empty_parameters(self):
        parameters: Iterable[DTensor] = []
        max_norm = 1
        norm_type = 2
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="on rank "
            rf"{self.rank} with empty parameters -- returning zero norm on CPU in fp32",
        ):
            total_norm = clip_grad_norm_(
                parameters, max_norm=max_norm, norm_type=norm_type
            )
            self.assertEqual(total_norm, torch.tensor(0.0))


if __name__ == "__main__":
    run_tests()
