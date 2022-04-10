# Owner(s): ["oncall: distributed"]

import sys
from math import inf

import torch
from torch import distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    _calc_grad_norm,
)
from torch.nn import utils as nn_utils
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DeterministicModel,
    FSDPTest,
    _collect_total_grad_norm_fsdp,
    _collect_total_grad_norm_local,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
    parametrize,
    instantiate_parametrized_tests,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestClipGradNorm(FSDPTest):
    def _run_fsdp_one_iteration(self, norm_type, nested_fsdp, cpu_offload):
        """Test FSDP with clip grad norm."""
        fsdp_model = DeterministicModel(nested_fsdp, cpu_offload=cpu_offload)
        local_model = DeterministicModel(False)
        input = torch.rand(14, 2, device=self.rank)
        fsdp_model = FSDP(fsdp_model, cpu_offload=cpu_offload)
        self.assertTrue(len(input) >= self.world_size)
        out = local_model(input[: self.world_size])
        out.sum().backward()
        in_data = torch.tensor(input[self.rank], device=self.rank)
        out_fsdp = fsdp_model(in_data)
        out_fsdp.sum().backward()
        total_norms_fsdp = _collect_total_grad_norm_fsdp(
            fsdp_model, norm_type, self.rank
        )
        total_norms_local = _collect_total_grad_norm_local(local_model, norm_type)
        total_norms_local /= self.world_size
        norm_cap = total_norms_fsdp / 2.0
        self.assertEqual(total_norms_local, total_norms_fsdp)
        fsdp_model.clip_grad_norm_(norm_cap, norm_type=norm_type)
        nn_utils.clip_grad_norm_(
            local_model.parameters(), norm_cap, norm_type=norm_type
        )
        total_norms_after_clip_fsdp = _collect_total_grad_norm_fsdp(
            fsdp_model, norm_type, self.rank
        )
        total_norms_after_clip_local = _collect_total_grad_norm_local(
            local_model, norm_type
        )
        self.assertTrue(total_norms_after_clip_fsdp <= norm_cap)
        self.assertEqual(total_norms_after_clip_local, total_norms_after_clip_fsdp)

    @skip_if_lt_x_gpu(2)
    @parametrize("norm_type", [2.0, inf])
    @parametrize("nested_fsdp", [True, False])
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=True), CPUOffload(offload_params=False)],
    )
    def test_fsdp_clip_grad_norm(self, norm_type, nested_fsdp, cpu_offload):
        """Test FSDP with clip grad norm."""
        self._run_fsdp_one_iteration(norm_type, nested_fsdp, cpu_offload)


class TestCalcuGradNorm(FSDPTest):
    @skip_if_lt_x_gpu(2)
    @parametrize("norm_type", [2.0, inf])
    @parametrize("nested_fsdp", [True, False])
    def test_fsdp_calc_grad_norm(self, norm_type, nested_fsdp):
        """Test grad norm cal API."""
        model = FSDP(DeterministicModel(nested_fsdp))
        input = torch.rand(15, 2, device=self.rank)
        out = model(input)
        out.sum().backward()
        total_norm = _calc_grad_norm(model.params_with_grad, norm_type)
        total_norm_expected = _collect_total_grad_norm_local(model, norm_type)
        self.assertEqual(total_norm, total_norm_expected)

    @skip_if_lt_x_gpu(2)
    @parametrize("norm_type", [1.3, 2.5])
    def test_fsdp_calc_grad_norm_error(self, norm_type):
        """Test the abnormal cases of grad norm cal API."""
        model = DeterministicModel(False)
        input = torch.rand(12, 2, device=self.rank)
        out = model(input)
        out.sum().backward()
        error_msg = f"Order {norm_type} not supported for matrix norm"
        with self.assertRaisesRegex(RuntimeError, error_msg):
            total_norm = _calc_grad_norm(model.parameters(), norm_type)


instantiate_parametrized_tests(TestClipGradNorm)
instantiate_parametrized_tests(TestCalcuGradNorm)

if __name__ == "__main__":
    run_tests()
