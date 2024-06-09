# Owner(s): ["oncall: distributed"]

import copy
import functools
import itertools
import sys
import unittest
from typing import List, Optional

import torch
from torch import distributed as dist
from torch.cuda.amp.common import amp_definitely_not_available
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    CUDAInitMode,
    DummyProcessGroup,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
    NonUniformReqGradNWM,
    subtest_name,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TestCase,
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


params = "cpu_offload,sharding_strategy,mixed_precision,use_orig_params"
cpu_offload_config = [CPUOffload(offload_params=True), CPUOffload(offload_params=False)]
sharding_strategy_config = [ShardingStrategy.SHARD_GRAD_OP, None]
mixed_precision = ["enable_mixed_precision", None]
use_orig_params = ["enable_use_orig_params", None]

configs = list(
    itertools.product(
        cpu_offload_config, sharding_strategy_config, mixed_precision, use_orig_params
    )
)
test_name_mapping = {
    str(CPUOffload(offload_params=True)): "offload_true",
    str(CPUOffload(offload_params=False)): "offload_false",
    str(ShardingStrategy.SHARD_GRAD_OP): "shard_grad_op",
    "enable_mixed_precision": "mixed_precision",
    "enable_use_orig_params": "use_orig_params",
}

subtest_name = functools.partial(subtest_name, test_name_mapping)


class TestShardGradScaler(TestCase):
    @unittest.skipIf(
        amp_definitely_not_available(), "no supported device (cuda, xla) found"
    )
    def test_grad_scaling(self):
        pg = DummyProcessGroup(0, 1)
        scaler = ShardedGradScaler(init_scale=2.0, process_group=pg, enabled=True)
        t0 = torch.full((1,), 4.0, dtype=torch.float32, device="cpu")
        t1 = torch.full((1,), 8.0, dtype=torch.float32, device="cpu")
        outputs = [t1.clone(), (t0.clone(), t1.clone()), [t0.clone(), t1.clone()]]
        outputs = scaler.scale(outputs)
        self.assertTrue(
            outputs[0] == 16.0 and outputs[1][0] == 8.0 and outputs[1][1] == 16.0
        )
        self.assertTrue(outputs[2][0] == 8.0 and outputs[2][1] == 16.0)
        self.assertTrue(scaler._scale.device == t1.device)

    @unittest.skipIf(
        amp_definitely_not_available(), "no supported device (cuda, xla) found"
    )
    def test_scaling_unscaling_sparse(self):
        pg = DummyProcessGroup(0, 1)
        scaler = ShardedGradScaler(init_scale=2.0, process_group=pg, enabled=True)
        inv_scale = torch.full((1,), 0.5, dtype=torch.float, device="cpu")
        found_inf = torch.full((1,), 0, dtype=torch.float, device="cpu")

        i = torch.tensor([[0, 1, 1], [2, 0, 2]], device="cpu", dtype=torch.int64)
        v = torch.tensor([16.0, 32.0, 64.0], dtype=torch.float, device="cpu")
        s = torch.sparse_coo_tensor(
            i, v, torch.Size([2, 3]), device="cpu", dtype=torch.float
        )

        # unscale sparse tensors
        s1 = s.clone()
        s1.grad = s.clone()
        opt = torch.optim.SGD([s1], lr=1.0)
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf)[s1.device]
        self.assertEqual(found_inf, 0.0)
        self.assertEqual(s1.grad.to_dense(), (s / 2).to_dense())

        # unscale sparse tensor: inf
        v = torch.tensor([16.0, 32.0, float("inf")], dtype=torch.float, device="cpu")
        s1.grad = torch.sparse_coo_tensor(
            i, v, torch.Size([2, 3]), device="cpu", dtype=torch.float
        )
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf)[s1.device]
        self.assertEqual(found_inf, 1.0)

        # unscale sparse tensor: overflow (marked as inf)
        i = torch.tensor([[1, 1, 1], [0, 0, 2]], device="cpu", dtype=torch.int64)
        # coalescing sparse tensor here will cause the value to be Inf
        v = torch.tensor([2**15, 2**15, 1.0], dtype=torch.float16, device="cpu")
        s1 = torch.sparse_coo_tensor(
            i, v, torch.Size([2, 3]), device="cpu", dtype=torch.float16
        )
        s1.grad = s1.clone()
        found_inf.zero_()
        found_inf = scaler._unscale_grads_(opt, inv_scale, found_inf)[s1.device]
        self.assertEqual(found_inf, 1.0)

    @unittest.skipIf(
        amp_definitely_not_available(), "no supported device (cuda, xla) found"
    )
    def test_inf_gradients_skip_optim_step(self):
        pg = DummyProcessGroup(0, 1)
        scaler = ShardedGradScaler(init_scale=2.0, process_group=pg, enabled=True)
        loss = torch.full((1,), 4.0, dtype=torch.float32, device="cpu")
        t0 = torch.tensor([float("inf")], dtype=torch.float32, device="cpu")
        t0.grad = t0.clone()
        opt = torch.optim.SGD([t0], lr=1.0)
        scaler.scale(loss)
        ret_val = scaler.step(opt)
        self.assertTrue(ret_val is None)


class TestShardedGradScalerParityWithDDP(FSDPTest):
    def _get_init_modes_for_test(self, cpu_offload):
        modes = [CUDAInitMode.CUDA_AFTER, CUDAInitMode.CUDA_BEFORE]
        # Note that CUDAInitMode.CUDA_NEVER works currently only with CPU
        # offload as we explicitly bring the param back to CUDA device. In
        # general, it will not work since we try to all_gather p.data which is
        # on CPU but NCCL only supports GPU.
        if cpu_offload.offload_params:
            modes.append(CUDAInitMode.CUDA_NEVER)

        return modes

    @skip_if_lt_x_gpu(2)
    @parametrize(params, configs, subtest_name)
    def test_fsdp_ddp_parity_with_grad_scaler(
        self,
        cpu_offload: CPUOffload,
        sharding_strategy: Optional[ShardingStrategy],
        mixed_precision: Optional[str],
        use_orig_params: Optional[str],
    ):
        init_modes = self._get_init_modes_for_test(cpu_offload)
        mp = (
            MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )
            if mixed_precision is not None
            else None
        )
        # the ``NonUniformReqGradNWM`` model requires we set `init_scale`
        # more conservatively than default to avoid infs with the initial steps
        if use_orig_params == "enable_use_orig_params":
            use_orig = True
            model_cls = NonUniformReqGradNWM
            sharded_grad_scaler_kwargs = {"init_scale": 2.0**11}
        else:
            use_orig = False
            model_cls = NestedWrappedModule  # type: ignore[assignment]
            sharded_grad_scaler_kwargs = None
        for cuda_init_mode in init_modes:
            self._test_fsdp_parity(
                model_cls,
                FSDPInitMode.RECURSIVE,
                cuda_init_mode=cuda_init_mode,
                cpu_offload=cpu_offload,
                sharding_strategy=sharding_strategy,
                mixed_precision=mp,
                enable_sharded_grad_scaler=True,
                use_orig_params=use_orig,
                sharded_grad_scaler_kwargs=sharded_grad_scaler_kwargs,
            )

    def _build_model_and_optim(
        self,
        cpu_offload: CPUOffload = CPUOffload(offload_params=False),
        use_orig_params: bool = False,
    ):
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            deterministic=True,
        )
        ref_model = DDP(
            copy.deepcopy(model),
            device_ids=[self.rank],
        )
        ref_optim = torch.optim.Adam(ref_model.parameters(), lr=1e-2)
        fsdp_kwargs = {
            "use_orig_params": use_orig_params,
            "cpu_offload": cpu_offload,
            "auto_wrap_policy": ModuleWrapPolicy(
                {
                    TransformerEncoderLayer,
                    TransformerDecoderLayer,
                }
            ),
        }
        model = FSDP(model, **fsdp_kwargs)
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        return model, optim, ref_model, ref_optim

    @skip_if_lt_x_gpu(2)
    def test_sharded_grad_scaler_found_inf(self):
        self.run_subtests(
            {
                "use_orig_params": [False, True],
                "cpu_offload": [
                    CPUOffload(offload_params=True),
                    CPUOffload(offload_params=False),
                ],
            },
            self._test_sharded_grad_scaler_found_inf,
        )

    def _test_sharded_grad_scaler_found_inf(
        self,
        use_orig_params: bool,
        cpu_offload: CPUOffload,
    ):
        model, optim, ref_model, ref_optim = self._build_model_and_optim(
            cpu_offload=cpu_offload,
            use_orig_params=use_orig_params,
        )
        grad_scaler = ShardedGradScaler(init_scale=2.0)
        ref_grad_scaler = torch.cuda.amp.GradScaler(init_scale=2.0)
        scaled_losses: List[torch.Tensor] = []
        device = torch.device("cuda")
        torch.manual_seed(42 + self.rank + 1)

        for iter in range(10):
            for _model, _optim, _grad_scaler in (
                (ref_model, ref_optim, ref_grad_scaler),
                (model, optim, grad_scaler),
            ):
                module = _model.module
                inp = module.get_input(device)
                _optim.zero_grad()
                output = _model(*inp)
                loss = module.get_loss(inp, output)
                scaled_loss = _grad_scaler.scale(loss)
                scaled_losses.append(scaled_loss)
                scaled_loss.backward()
                orig_params = [
                    param.detach().clone()
                    for param in _model.parameters()
                    if param.grad is not None
                ]
                should_find_inf = iter % 2 == 0
                if should_find_inf and (
                    _model is ref_model or (_model is model and self.rank == 0)
                ):
                    # other ranks should find infs from rank 0
                    # after collectives
                    for param in _model.parameters():
                        if param.grad is None:
                            continue
                        param.grad.fill_(float("inf"))
                        break
                _grad_scaler.step(_optim)
                orig_scale = _grad_scaler.get_scale()
                _grad_scaler.update()
                if should_find_inf:
                    self.assertEqual(
                        _grad_scaler.get_scale(),
                        orig_scale * _grad_scaler.get_backoff_factor(),
                        (
                            f"rank: {self.rank} iter: {iter} expect origin scale {orig_scale} "
                            f"to be backed off by {_grad_scaler.get_backoff_factor()} "
                            f"but got {_grad_scaler.get_scale()}"
                        ),
                    )
                else:
                    self.assertEqual(
                        _grad_scaler.get_scale(),
                        orig_scale,
                        (
                            f"rank: {self.rank} iter: {iter} expect same scale {orig_scale} "
                            f"but got {_grad_scaler.get_scale()}"
                        ),
                    )
                for param, orig_param in zip(
                    [param for param in _model.parameters() if param.grad is not None],
                    orig_params,
                ):
                    if should_find_inf:
                        self.assertEqual(
                            param,
                            orig_param,
                            (
                                f"rank: {self.rank} iter: {iter} expect the same params before "
                                f"and after optim.step but got {param} vs {orig_param}"
                            ),
                        )
                    else:
                        self.assertNotEqual(
                            param,
                            orig_param,
                            (
                                f"rank: {self.rank} iter: {iter} expect the updated params after "
                                f"optim.step but got {param} vs {orig_param}"
                            ),
                        )
            self.assertEqual(
                scaled_losses[0],
                scaled_losses[1],
                f"iter: {iter} {scaled_losses[0]} vs {scaled_losses[1]}",
            )


instantiate_parametrized_tests(TestShardGradScaler)
instantiate_parametrized_tests(TestShardedGradScalerParityWithDDP)

if __name__ == "__main__":
    run_tests()
