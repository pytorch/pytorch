# Owner(s): ["oncall: distributed"]
import itertools
import sys
from typing import Union

import torch
import torch.nn as nn
from torch import distributed as dist
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    DEVICEInitMode,
    FSDPInitMode,
    FSDPTestContinuous,
    get_devtype,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


device_type = torch.device(get_devtype())

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestClipGradNorm(FSDPTestContinuous):
    """Tests :meth:`FullyShardedDataParallel.clip_grad_norm_`."""

    @skip_if_lt_x_gpu(2)
    def test_non_root(self, device):
        """
        Tests that calling ``clip_grad_norm_()`` on a non-root FSDP instance
        raises an error.
        """

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = nn.Linear(5, 5)
                self.lin2 = nn.Linear(5, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin2(self.lin1(x))

        model = Model().to(device_type.type)
        model.lin2 = FSDP(model.lin2)
        fsdp_model = FSDP(model)
        # fsdp_model(torch.randn((2, 5), device=torch.device(self.device_type))).sum().backward()
        fsdp_model(torch.randn((2, 5), device=device_type)).sum().backward()
        error_regex = "should only be called on the root FSDP instance"
        with self.assertRaisesRegex(RuntimeError, error_regex):
            fsdp_model.lin2.clip_grad_norm_(max_norm=2)

    @skip_if_lt_x_gpu(2)
    def test_ddp_parity(self, device):
        """
        Tests FSDP with ``FullyShardedDataParallel.clip_grad_norm_()` against
        DDP with ``torch.nn.utils.clip_grad_norm_()` when using full precision.
        """
        self.run_subtests(
            {
                "device": [device],
                "max_norm": [1, 2.5],
                "norm_type": [1, 2, float("inf")],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.NO_SHARD,
                    "mixed_strategy",
                ],
                "use_orig_params": [False, True],
                "offload_params": [False, True],
            },
            self._test_ddp_parity,
        )

    def _test_ddp_parity(
        self,
        device,
        max_norm: Union[float, int],
        norm_type: Union[float, int],
        sharding_strategy: Union[ShardingStrategy, str],
        use_orig_params: bool,
        offload_params: bool,
    ):
        local_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            deterministic=True,
        )
        ddp_model = DDP(local_model, device_ids=[device_type])
        fsdp_kwargs = {
            "cpu_offload": CPUOffload(offload_params=offload_params),
            "use_orig_params": use_orig_params,
            "device_id": device_type.type,
        }
        if sharding_strategy == "mixed_strategy":
            fsdp_model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                DEVICEInitMode.DEVICE_BEFORE,
                deterministic=True,
            )
            # Apply `NO_SHARD` to the encoder
            fsdp_model.transformer.encoder = FSDP(
                fsdp_model.transformer.encoder,
                sharding_strategy=ShardingStrategy.NO_SHARD,
                **fsdp_kwargs,
            )
            # Apply `FULL_SHARD` to the decoder
            fsdp_model.transformer.decoder = FSDP(
                fsdp_model.transformer.decoder,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                **fsdp_kwargs,
            )
            # TODO: FSDP's `clip_grad_norm_()` is not a static method, so we
            # must make the root module an FSDP instance
            fsdp_model = FSDP(
                fsdp_model, sharding_strategy=ShardingStrategy.FULL_SHARD, **fsdp_kwargs
            )
        else:
            fsdp_kwargs.update(
                {
                    "sharding_strategy": sharding_strategy,
                    "auto_wrap_policy": ModuleWrapPolicy(
                        {
                            TransformerEncoderLayer,
                            TransformerDecoderLayer,
                        }
                    ),
                }
            )
            fsdp_model = TransformerWithSharedParams.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_BEFORE,
                deterministic=True,
                fsdp_kwargs=fsdp_kwargs,
            )
        LR = 1e-2
        ddp_optim = torch.optim.Adam(ddp_model.parameters(), lr=LR)
        fsdp_optim = torch.optim.Adam(fsdp_model.parameters(), lr=LR)
        device = torch.device(self.device_type)
        LARGE_FACTOR = 100
        inp = ddp_model.module.get_input(device)
        for model in (ddp_model, fsdp_model):
            out = model(*inp)
            if isinstance(model, (DDP, FSDP)):
                loss = model.module.get_loss(inp, out)
            else:
                loss = model.get_loss(inp, out)
            loss.backward()
        # Multiply gradients by a large factor to ensure that gradients will
        # actually be clipped
        for param in itertools.chain(ddp_model.parameters(), fsdp_model.parameters()):
            if (
                param.grad is not None
            ):  # gradients may be `None` for `use_orig_params=True`
                param.grad *= LARGE_FACTOR
        orig_ddp_grads = [
            param.grad.detach().clone() for param in ddp_model.parameters()
        ]
        orig_fsdp_grads = [
            param.grad.detach().clone() if param.grad is not None else None
            for param in fsdp_model.parameters()
        ]
        ddp_total_norm = torch.nn.utils.clip_grad_norm_(
            ddp_model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
        )
        fsdp_total_norm = fsdp_model.clip_grad_norm_(
            max_norm=max_norm, norm_type=norm_type
        )
        self.assertEqual(ddp_total_norm, fsdp_total_norm)
        # Check that the gradients were modified by `clip_grad_norm_()`
        for param, orig_grad in zip(ddp_model.parameters(), orig_ddp_grads):
            if torch.equal(param.grad, orig_grad):
                raise AssertionError(
                    "Expected gradient to be modified by clip_grad_norm_()"
                )
        for param, orig_grad in zip(fsdp_model.parameters(), orig_fsdp_grads):
            if param.grad is None:
                self.assertEqual(param.grad, orig_grad)  # `None`
            else:
                if torch.equal(param.grad, orig_grad):
                    raise AssertionError(
                        "Expected gradient to be modified by clip_grad_norm_()"
                    )
        # Run an optimizer step to ensure gradients matched after clipping
        ddp_optim.step()
        fsdp_optim.step()
        with FSDP.summon_full_params(fsdp_model):
            for (n1, p1), (n2, p2) in zip(
                ddp_model.module.named_parameters(),
                fsdp_model.named_parameters(),
            ):
                self.assertEqual(n1, n2)
                self.assertEqual(p1, p2)
        if offload_params:
            # TODO: Gradient computation on CPU and GPU differ slightly causing
            # drift unrelated to `clip_grad_norm_()`.
            # https://github.com/pytorch/pytorch/issues/89133
            return
        # Run a few more iterations
        # TODO: We cannot run too many iterations, or else there is drift:
        # https://github.com/pytorch/pytorch/issues/89136
        for i in range(3):
            set_to_none = i % 2 == 0  # exercise both
            ddp_optim.zero_grad(set_to_none=set_to_none)
            fsdp_optim.zero_grad(set_to_none=set_to_none)
            inp = ddp_model.module.get_input(device)
            for model in (ddp_model, fsdp_model):
                out = model(*inp)
                out.sum().backward()
            ddp_total_norm = torch.nn.utils.clip_grad_norm_(
                ddp_model.parameters(),
                max_norm=max_norm,
                norm_type=norm_type,
            )
            fsdp_total_norm = fsdp_model.clip_grad_norm_(
                max_norm=max_norm, norm_type=norm_type
            )
            self.assertEqual(ddp_total_norm, fsdp_total_norm)
            ddp_optim.step()
            fsdp_optim.step()

    @skip_if_lt_x_gpu(2)
    def test_low_precision_grads(self, device):
        """Tests ``clip_grad_norm_()`` when using low precision gradients."""
        self.run_subtests(
            {
                "device": [device],
                "max_norm": [1, 2.5],
                "norm_type": [1, 2, float("inf")],
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.NO_SHARD,
                ],
                "use_orig_params": [False, True],
            },
            self._test_low_precision_grads,
        )

    def _test_low_precision_grads(
        self,
        device,
        max_norm: Union[float, int],
        norm_type: Union[float, int],
        sharding_strategy: ShardingStrategy,
        use_orig_params: bool,
    ):
        fsdp_kwargs = {
            "sharding_strategy": sharding_strategy,
            "use_orig_params": use_orig_params,
            "mixed_precision": MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                keep_low_precision_grads=True,
            ),
            "device_id": device_type.type,
        }
        fsdp_model = FSDP(
            NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                DEVICEInitMode.DEVICE_BEFORE,
                deterministic=True,
                fsdp_kwargs=fsdp_kwargs,
            ),
            **fsdp_kwargs,
        )
        inp = fsdp_model.module.get_input(torch.device(self.device_type))
        out = fsdp_model(*inp)
        out.sum().backward()
        for param in fsdp_model.parameters():
            if param.grad is not None:
                self.assertEqual(param.grad.dtype, torch.float16)
        total_norm = fsdp_model.clip_grad_norm_(max_norm=max_norm, norm_type=norm_type)
        # Check that the total norm is in FP16 to match the gradient dtype
        self.assertEqual(total_norm.dtype, torch.float16)
        # As a best effort, check that each gradient has norm at most the max
        # norm (since DDP does not support mixed precision natively, we cannot
        # directly compare for parity)
        for param in fsdp_model.parameters():
            if param.grad is not None:
                self.assertTrue(
                    torch.linalg.vector_norm(param.grad, norm_type).item() <= max_norm,
                )

    @skip_if_lt_x_gpu(2)
    def test_no_gradients(self, device):
        """
        Tests that calling ``clip_grad_norm_()`` when the FDSP module has no
        gradients simply returns a scalar zero tensor in FP32 without erroring.
        """
        self.run_subtests(
            {"device": [device], "use_orig_params": [False, True]},
            self._test_no_gradients,
        )

    def _test_no_gradients(self, device, use_orig_params: bool):
        lin_module = nn.Linear(24, 24)
        mixed_precision_config = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )
        fsdp_module = FSDP(
            lin_module,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=mixed_precision_config,
            device_id=device_type.type,
            use_orig_params=use_orig_params,
        )
        inp = torch.randn(32, 24, device=self.device_type)
        fsdp_module(inp)
        with self.assertWarnsRegex(
            expected_warning=UserWarning,
            expected_regex="on rank "
            rf"{self.rank} with no gradients -- returning the total "
            "norm in the default dtype torch.float32",
        ):
            total_norm = fsdp_module.clip_grad_norm_(1)
        self.assertEqual(total_norm.dtype, torch.float32)
        self.assertEqual(total_norm, torch.tensor(0.0, device=self.device_type))

    @skip_if_lt_x_gpu(2)
    def test_non_uniform_grad_dtype(self, device):
        """Tests clip_grad_norm_ with non-uniform gradient dtypes via MixedPrecision."""
        device_type = torch.device(device).type
        model = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5))
        # bf16 and fp32 params with low-precision grads kept
        model[0] = FSDP(
            model[0],
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                keep_low_precision_grads=True,
                cast_forward_inputs=True,
            ),
            device_id=device_type,
        )
        model[1] = FSDP(
            model[1],
            mixed_precision=MixedPrecision(
                param_dtype=torch.float32,
                cast_forward_inputs=True,
            ),
            device_id=device_type,
        )
        fsdp_model = FSDP(model, device_id=device_type)
        fsdp_model(torch.randn((2, 5), device=device_type)).sum().backward()

        # check that dtypes are actually mixed
        grad_dtypes = {
            p.grad.dtype for p in fsdp_model.parameters() if p.grad is not None
        }
        self.assertGreater(len(grad_dtypes), 1, "Expected mixed grad dtypes")

        total_norm = fsdp_model.clip_grad_norm_(max_norm=1)

        # check that total_norm is computed in fp32
        self.assertEqual(total_norm.dtype, torch.float32)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestClipGradNorm, globals(), only_for=devices, allow_xpu=True
)
if __name__ == "__main__":
    run_tests()
