# Owner(s): ["oncall: distributed"]

import sys
from unittest.mock import patch

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._runtime_utils import (
    _get_handle_to_prefetch,
    _get_training_state,
)
from torch.distributed.fsdp.flat_param import HandleTrainingState
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

NUM_ITERS = 1
BACKWARD_PREFETCH_OPTIONS = [
    None,
    BackwardPrefetch.BACKWARD_PRE,
    BackwardPrefetch.BACKWARD_POST,
]


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestBackwardPrefetch(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _dist_train(self, backward_prefetch=BackwardPrefetch.BACKWARD_PRE):
        rank = self.rank
        orig_get_handle_to_prefetch = _get_handle_to_prefetch

        torch.manual_seed(0)
        policy = ModuleWrapPolicy(
            {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}
        )
        model = FSDP(
            nn.Transformer(d_model=1024, nhead=8, device="cuda"),
            device_id=torch.cuda.current_device(),
            auto_wrap_policy=policy,
            use_orig_params=True,
            backward_prefetch=backward_prefetch,
        )
        optim = torch.optim.SGD(model.parameters(), lr=1e-2)

        # prepare input
        torch.manual_seed(rank + 1)
        src = torch.randn((10, 1, 1024), device="cuda")
        tgt = torch.randn((20, 1, 1024), device="cuda")

        # monkey patch
        none_handle_count = 0
        func_call_count = 0

        def patched_get_handle_to_prefetch(*args, **kwargs):
            handle = orig_get_handle_to_prefetch(*args, **kwargs)

            assert (
                len(args) == 2
            ), "expect _get_handle_to_prefetch(state, current_handle)"
            state = args[0]
            current_handle = args[1]
            training_state = _get_training_state(current_handle)
            if (
                training_state == HandleTrainingState.BACKWARD_PRE
                and state.backward_prefetch == BackwardPrefetch.BACKWARD_PRE
            ) or (
                training_state == HandleTrainingState.BACKWARD_POST
                and state.backward_prefetch == BackwardPrefetch.BACKWARD_POST
            ):
                nonlocal none_handle_count
                nonlocal func_call_count
                if handle is not None:
                    none_handle_count += 1
                func_call_count += 1
            return handle

        # track num of calls to _get_handle_to_prefetch
        # track num of non-None prefetch handles
        with patch(
            "torch.distributed.fsdp._runtime_utils._get_handle_to_prefetch",
            patched_get_handle_to_prefetch,
        ):
            for _ in range(NUM_ITERS):
                optim.zero_grad()
                loss = model(src, tgt).sum()
                loss.backward()
                optim.step()
        if backward_prefetch is None:
            self.assertTrue(
                func_call_count == 0, f"_get_handle_to_prefetch: {func_call_count}"
            )
        elif backward_prefetch in [
            BackwardPrefetch.BACKWARD_PRE,
            BackwardPrefetch.BACKWARD_POST,
        ]:
            self.assertTrue(
                func_call_count > 0 and none_handle_count > 0,
                f"_get_handle_to_prefetch: {func_call_count} non-None handles: {none_handle_count}",
            )

    @skip_if_lt_x_gpu(2)
    def test_backward_prefetch_pre(self):
        self._dist_train(BackwardPrefetch.BACKWARD_PRE)

    @skip_if_lt_x_gpu(2)
    def test_backward_prefetch_post(self):
        self._dist_train(BackwardPrefetch.BACKWARD_POST)

    @skip_if_lt_x_gpu(2)
    def test_backward_prefetch_disabled(self):
        self._dist_train(None)


if __name__ == "__main__":
    run_tests()
