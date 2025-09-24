# Owner(s): ["oncall: distributed"]

import functools
import itertools
import os
import tempfile
import unittest
from enum import auto, Enum
from typing import Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp._wrap_utils import _validate_frozen_params
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    _or_policy,
    _Policy,
    _wrap_module_cls_individually,
    always_wrap_policy,
    CustomPolicy,
    enable_wrap,
    ModuleWrapPolicy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    wrap,
)
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.modules.batchnorm import _BatchNorm
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    _move_to_device,
    DEVICEInitMode,
    DummyProcessGroup,
    FSDPInitMode,
    FSDPTest,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    find_free_port,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_CUDA,
    TEST_XPU,
    TestCase,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = torch.distributed.get_default_backend_for_device(device_type)


class BatchNormNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False)
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm2d(10)
        self.bn3 = nn.BatchNorm3d(10)
        self.sync_bn = nn.SyncBatchNorm(10)


class LoraModel(nn.Module):
    """This is a toy LoRA decoder model."""

    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([LoraDecoder() for _ in range(4)])
        self.norm = nn.LayerNorm(32)
        self.embed_tokens.weight.requires_grad_(False)
        self.norm.weight.requires_grad_(False)
        self.norm.bias.requires_grad_(False)


class LoraDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = LoraAttention()
        self.mlp = LoraMLP()
        self.inp_layernorm = nn.LayerNorm(32)
        self.post_attn_layernorm = nn.LayerNorm(32)
        self.inp_layernorm.weight.requires_grad_(False)
        self.inp_layernorm.bias.requires_grad_(False)
        self.post_attn_layernorm.weight.requires_grad_(False)
        self.post_attn_layernorm.bias.requires_grad_(False)


class LoraAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(32, 32, bias=False)
        self.lora_A = nn.Linear(32, 8, bias=False)
        self.lora_B = nn.Linear(8, 32, bias=False)
        self.k_proj = nn.Linear(32, 32, bias=False)
        self.v_proj = nn.Linear(32, 32, bias=False)
        self.o_proj = nn.Linear(32, 32, bias=False)
        self.q_proj.weight.requires_grad_(False)
        self.k_proj.weight.requires_grad_(False)
        self.v_proj.weight.requires_grad_(False)
        self.o_proj.weight.requires_grad_(False)


class LoraMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj1 = nn.Linear(32, 128, bias=False)
        self.proj2 = nn.Linear(128, 32, bias=False)
        self.proj1.weight.requires_grad_(False)
        self.proj2.weight.requires_grad_(False)


class WrapMethod(Enum):
    FSDP_CTOR = auto()
    # FSDP_CTOR is the supported way forward, but keep WRAP_API in case we miss
    # any use cases and fix them to work with FSDP_CTOR over time.
    WRAP_API = auto()


class TestFSDPWrap(FSDPTest):
    """
    Tests main API for wrapping FSDP, which is to pass auto_wrap_policy into
    FSDP constructor.
    """

    def setUp(self) -> None:
        super().setUp()

    class NestedSequentialModel:
        @staticmethod
        def get_model(device=True):
            sequential = nn.Sequential(
                nn.Linear(5, 5),
                nn.Linear(5, 5),
                nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5)),
            )
            if device:
                sequential = sequential.to(device=device_type)
            return sequential

        @staticmethod
        def verify_model_all_wrapped(cls, model):
            cls.assertTrue(isinstance(model, FSDP))
            cls.assertTrue(isinstance(model.module[0], FSDP))
            cls.assertTrue(isinstance(model.module[1], FSDP))
            cls.assertTrue(isinstance(model.module[2], FSDP))
            cls.assertTrue(isinstance(model.module[2].module[0], FSDP))
            cls.assertTrue(isinstance(model.module[2].module[1], FSDP))

        @staticmethod
        def verify_model(cls, model):
            cls.assertTrue(isinstance(model, FSDP))
            cls.assertTrue(isinstance(model.module[0], nn.Linear))
            cls.assertTrue(isinstance(model.module[1], nn.Linear))
            cls.assertTrue(isinstance(model.module[2], FSDP))
            # following modules were not wrapped by the policy.
            cls.assertTrue(isinstance(model.module[2].module[0], nn.Linear))
            cls.assertTrue(isinstance(model.module[2].module[1], nn.Linear))

    def _get_linear(self, fin, fout):
        return nn.Linear(fin, fout, bias=False)

    def _get_already_wrapped_fsdp(
        self, device_init_mode=DEVICEInitMode.DEVICE_BEFORE, nested=False
    ) -> FSDP:
        fn_self = self

        class MyModel(nn.Module):
            def __init__(self, nested):
                super().__init__()
                # TODO: test the various init modes.
                move_to_device = device_init_mode == DEVICEInitMode.DEVICE_BEFORE
                # if nested=True, the FSDP module will be nested one layer deep
                # and we should pick that up.
                if nested:
                    self.lin1 = nn.Sequential(
                        _move_to_device(fn_self._get_linear(1, 1), move_to_device),
                        FSDP(
                            _move_to_device(fn_self._get_linear(1, 1), move_to_device)
                        ),
                    )
                else:
                    self.lin1 = FSDP(
                        _move_to_device(fn_self._get_linear(1, 1), move_to_device)
                    )
                self.lin2 = FSDP(
                    _move_to_device(fn_self._get_linear(1, 1), move_to_device)
                )
                self.lin3 = FSDP(
                    _move_to_device(fn_self._get_linear(1, 1), move_to_device)
                )

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return self.lin3(self.lin2(self.lin1(input)))

        model = MyModel(nested=nested)
        return model

    @skip_if_lt_x_gpu(2)
    @parametrize("nested", [True, False])
    @parametrize(
        "device_init_mode", [DEVICEInitMode.DEVICE_AFTER, DEVICEInitMode.DEVICE_BEFORE]
    )
    def test_error_already_wrapped(self, nested, device_init_mode):
        """
        Test that an error is raised if we attempt to wrap when submodules are
        already FSDP.
        """
        wrapped_fsdp = self._get_already_wrapped_fsdp(
            nested=nested, device_init_mode=device_init_mode
        )
        if device_init_mode == DEVICEInitMode.DEVICE_AFTER:
            wrapped_fsdp = wrapped_fsdp.to(device=device_type)

        wrapped_module_name = "lin1.1" if nested else "lin1"
        with self.assertRaisesRegex(
            ValueError,
            "FSDP auto wrapping requires modules to not already have FSDP "
            f"applied but found {wrapped_module_name} in",
        ):
            FSDP(wrapped_fsdp, auto_wrap_policy=size_based_auto_wrap_policy)

    @skip_if_lt_x_gpu(2)
    @parametrize("use_or_policy", [True, False])
    def test_wrap_batchnorm_individually(self, use_or_policy):
        def never_wrap_policy(*args, **kwargs):
            return False

        wrap_batchnorm_individually = functools.partial(
            _wrap_module_cls_individually,
            module_classes=[
                _BatchNorm,
            ],
        )
        policy = (
            functools.partial(
                _or_policy, policies=[never_wrap_policy, wrap_batchnorm_individually]
            )
            if use_or_policy
            else wrap_batchnorm_individually
        )
        model = BatchNormNet()
        fsdp = FSDP(model, auto_wrap_policy=policy)
        # Batchnorms should be wrapped
        for layer in [fsdp.bn1, fsdp.bn2, fsdp.bn3, fsdp.sync_bn]:
            self.assertTrue(isinstance(layer, FSDP))

        self.assertFalse(isinstance(fsdp.lin, FSDP))

    @skip_if_lt_x_gpu(2)
    def test_bn_always_wrapped_individually(self):
        """
        Ensures that by using _or_policy with _wrap_module_cls_individually, even
        if the other policy results in a module containing a BN unit being
        wrapped, the contained BN unit will still be individually wrapped.
        """

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bn_container = BatchNormNet()

        def wrap_bn_container(module, recurse, *args, **kwargs):
            if recurse:
                return True
            return isinstance(module, BatchNormNet)

        wrap_batchnorm_individually = functools.partial(
            _wrap_module_cls_individually,
            module_classes=[
                _BatchNorm,
            ],
        )

        my_policy = functools.partial(
            _or_policy, policies=[wrap_bn_container, wrap_batchnorm_individually]
        )
        mod = MyModule()
        fsdp = FSDP(mod, auto_wrap_policy=my_policy)

        # Wrapping should be FSDP(FSDP(BatchNormNet(FSDP(BN))))
        # and not FSDP(FSDP(BatchNormNet(BN))) (in the latter the inner
        # BN is not individually wrapped.)

        for bn in [
            fsdp.bn_container.bn1,
            fsdp.bn_container.bn2,
            fsdp.bn_container.bn3,
            fsdp.bn_container.sync_bn,
        ]:
            self.assertTrue(isinstance(bn, FSDP))

        # if we just wrapped BN container, individual batchnorms are not
        # wrapped.
        mod = MyModule()
        fsdp = FSDP(mod, auto_wrap_policy=wrap_bn_container)
        self.assertTrue(isinstance(mod.bn_container, FSDP))
        for bn in [
            fsdp.bn_container.bn1,
            fsdp.bn_container.bn2,
            fsdp.bn_container.bn3,
            fsdp.bn_container.sync_bn,
        ]:
            self.assertFalse(isinstance(bn, FSDP))

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)],
    )
    @parametrize(
        "backward_prefetch",
        [BackwardPrefetch.BACKWARD_POST, BackwardPrefetch.BACKWARD_PRE],
    )
    @parametrize("forward_prefetch", [False, True])
    @parametrize(
        "device_init_mode", [DEVICEInitMode.DEVICE_AFTER, DEVICEInitMode.DEVICE_BEFORE]
    )
    def test_main_wrap_api(
        self,
        cpu_offload: CPUOffload,
        backward_prefetch: BackwardPrefetch,
        forward_prefetch: bool,
        device_init_mode: DEVICEInitMode,
    ):
        if (
            device_init_mode == DEVICEInitMode.DEVICE_AFTER
            and cpu_offload.offload_params
        ):
            # they don't work together, expected
            return

        move_to_device = device_init_mode == DEVICEInitMode.DEVICE_BEFORE

        class Nested(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.nested_lin = _move_to_device(
                    nn.Linear(1, 1, bias=False), move_to_device
                )

            def forward(self, input):
                return self.nested_lin(input)

        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin1 = _move_to_device(nn.Linear(1, 1, bias=False), move_to_device)
                self.lin2 = _move_to_device(nn.Linear(1, 1, bias=False), move_to_device)
                self.lin3 = _move_to_device(nn.Linear(1, 1, bias=False), move_to_device)
                self.lin4 = Nested()

            def forward(self, input):
                return self.lin4(self.lin3(self.lin2(self.lin1(input))))

        model = MyModel()
        wrapped_model = FSDP(
            model,
            auto_wrap_policy=functools.partial(
                size_based_auto_wrap_policy,
                min_num_params=0,  # wrap all modules
            ),
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            forward_prefetch=forward_prefetch,
        )
        if device_init_mode == DEVICEInitMode.DEVICE_AFTER:
            wrapped_model = wrapped_model.to(device=device_type)

        modules_in_fsdp_graph_order = [
            wrapped_model.module.lin1,
            wrapped_model.module.lin2,
            wrapped_model.module.lin3,
            wrapped_model.module.lin4.module.nested_lin,
            wrapped_model.module.lin4,
            wrapped_model,
        ]

        for module in modules_in_fsdp_graph_order:
            self.assertTrue(isinstance(module, FSDP))
            self._check_cpu_offload(module, cpu_offload)
            self._check_backward_prefetch(module, backward_prefetch)
            self._check_forward_prefetch(module, forward_prefetch)

        # Run model a few times for sanity check.
        optim = torch.optim.SGD(wrapped_model.parameters(), lr=1e-2, momentum=0.9)
        inp = torch.ones(1).to(device=device_type)
        for _ in range(6):
            optim.zero_grad()
            loss = wrapped_model(inp).sum()
            loss.backward()
            optim.step()

    @skip_if_lt_x_gpu(1)
    def test_zero_argument(self):
        class ZeroArguModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.a = torch.tensor([1.0])

            def forward(self):
                return self.a

        model = FSDP(ZeroArguModel())
        self.assertEqual(model(), torch.tensor([1.0]))


class TestAutoWrap(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # For all the tests here, we use a fake group
        self.process_group = DummyProcessGroup(rank=0, size=1)

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    @parametrize("wrap_method", [WrapMethod.FSDP_CTOR, WrapMethod.WRAP_API])
    def test_wrap(self, wrap_method):
        if wrap_method == WrapMethod.WRAP_API:
            with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
                layer = wrap(nn.Linear(5, 5))
        else:
            assert wrap_method == WrapMethod.FSDP_CTOR
            layer = FSDP(
                nn.Linear(5, 5),
                process_group=self.process_group,
                auto_wrap_policy=functools.partial(
                    size_based_auto_wrap_policy, min_num_params=1
                ),
            )
        self.assertTrue(isinstance(layer, FSDP))
        self.assertEqual(layer.rank, self.process_group.rank())
        self.assertEqual(layer.world_size, self.process_group.size())

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_wrap_disabled_outside_context(self):
        pg = self.process_group

        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = wrap(nn.Linear(5, 5), process_group=pg)

        model = MyModel()
        with enable_wrap(wrapper_cls=FSDP, process_group=pg):
            model = wrap(model)

        self.assertTrue(isinstance(model, FSDP))
        self.assertFalse(isinstance(model.lin, FSDP))
        self.assertTrue(isinstance(model.lin, nn.Linear))

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_wrap_override_defaults(self):
        new_process_group = DummyProcessGroup(rank=0, size=2)
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
            layer = wrap(nn.Linear(5, 5), process_group=new_process_group)
        self.assertTrue(isinstance(layer, FSDP))
        self.assertTrue(layer.process_group is new_process_group)
        self.assertEqual(layer.rank, 0)
        self.assertEqual(layer.world_size, 2)

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "Test Requires CUDA or XPU")
    def test_always_wrap(self):
        """
        Test to ensure that if `always_wrap_policy` is
        passed into FSDP, all submodules are wrapped.
        """
        seq = TestFSDPWrap.NestedSequentialModel.get_model(device=True)
        model = FSDP(
            seq, process_group=self.process_group, auto_wrap_policy=always_wrap_policy
        )
        TestFSDPWrap.NestedSequentialModel.verify_model_all_wrapped(self, model)

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_transformer_auto_wrap_policy(self):
        """Tests the ``transformer_auto_wrap_policy``."""
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerEncoderLayer, TransformerDecoderLayer},
        )
        self._test_transformer_wrapping(auto_wrap_policy)

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_module_wrap_policy(self):
        """Tests the ``ModuleWrapPolicy``."""
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        self._test_transformer_wrapping(auto_wrap_policy)

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_module_wrap_policy_callable(self):
        """Tests the ``ModuleWrapPolicy`` as a ``Callable``."""
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        callable_policy = functools.partial(_or_policy, policies=[auto_wrap_policy])
        self._test_transformer_wrapping(callable_policy)

    def _test_transformer_wrapping(self, auto_wrap_policy: Union[Callable, _Policy]):
        fsdp_kwargs = {"auto_wrap_policy": auto_wrap_policy}
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            DEVICEInitMode.DEVICE_BEFORE,
            fsdp_kwargs,
        )
        modules = list(fsdp_model.modules())
        encoder_layers = set(fsdp_model.module.transformer.encoder.layers)
        decoder_layers = set(fsdp_model.module.transformer.decoder.layers)
        for module in modules:
            if (
                module is fsdp_model
                or module in encoder_layers
                or module in decoder_layers
            ):
                self.assertTrue(isinstance(module, FSDP))
            else:
                self.assertFalse(isinstance(module, FSDP))

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_custom_policy(self):
        """
        Tests ``CustomPolicy`` with both a lambda function that uses uniform
        kwargs (so only returns ``False`` or ``True``) and a lambda function
        that uses non-uniform kwargs (so returns a dict to override the root
        kwargs).
        """
        for use_uniform_kwargs in [False, True]:
            self._test_custom_policy(use_uniform_kwargs)

    def _test_custom_policy(self, use_uniform_kwargs: bool):
        print(f"use_uniform_kwargs={use_uniform_kwargs}")
        model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            DEVICEInitMode.DEVICE_BEFORE,
            {},
        )

        if use_uniform_kwargs:

            def lambda_fn(module: nn.Module):
                if module is model.bn:
                    return True
                elif isinstance(
                    module, (TransformerEncoderLayer, TransformerDecoderLayer)
                ):
                    return True
                return False

        else:

            def lambda_fn(module: nn.Module):
                if module is model.bn:
                    return {"sharding_strategy": ShardingStrategy.NO_SHARD}
                elif isinstance(module, TransformerEncoderLayer):
                    return True
                elif isinstance(module, TransformerDecoderLayer):
                    return {
                        "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
                        "backward_prefetch": BackwardPrefetch.BACKWARD_POST,
                    }
                return False

        policy = CustomPolicy(lambda_fn)
        # Use a size-2 dummy PG to avoid clamping the sharding strategy to
        # `NO_SHARD` as for a size-1 PG
        process_group = DummyProcessGroup(rank=0, size=2)
        fp16_mp = MixedPrecision(param_dtype=torch.float16)
        fp32_mp = MixedPrecision()
        model = FSDP(
            model,
            process_group=process_group,
            auto_wrap_policy=policy,
            mixed_precision=fp16_mp,
        )
        encoder_layers = set(model.module.transformer.encoder.layers)
        decoder_layers = set(model.module.transformer.decoder.layers)
        bn = model.module.bn
        bn_strategy = (
            ShardingStrategy.FULL_SHARD
            if use_uniform_kwargs
            else ShardingStrategy.NO_SHARD
        )
        bn_prefetch = BackwardPrefetch.BACKWARD_PRE
        encoder_strategy = root_strategy = ShardingStrategy.FULL_SHARD
        encoder_prefetch = root_prefetch = BackwardPrefetch.BACKWARD_PRE
        decoder_strategy = (
            ShardingStrategy.FULL_SHARD
            if use_uniform_kwargs
            else ShardingStrategy.SHARD_GRAD_OP
        )
        decoder_prefetch = (
            BackwardPrefetch.BACKWARD_PRE
            if use_uniform_kwargs
            else BackwardPrefetch.BACKWARD_POST
        )
        for module in model.modules():
            if module is bn:
                self.assertTrue(isinstance(module, FSDP))
                self.assertEqual(module.sharding_strategy, bn_strategy)
                self.assertEqual(module.backward_prefetch, bn_prefetch)
                # We currently override batch norm modules to use fp32
                self.assertEqual(module.mixed_precision, fp32_mp)
            elif module in encoder_layers:
                self.assertTrue(isinstance(module, FSDP))
                self.assertEqual(module.sharding_strategy, encoder_strategy)
                self.assertEqual(module.backward_prefetch, encoder_prefetch)
                self.assertEqual(module.mixed_precision, fp16_mp)
            elif module in decoder_layers:
                self.assertTrue(isinstance(module, FSDP))
                self.assertEqual(module.sharding_strategy, decoder_strategy)
                self.assertEqual(module.backward_prefetch, decoder_prefetch)
                self.assertEqual(module.mixed_precision, fp16_mp)
            elif module is model:
                self.assertTrue(isinstance(module, FSDP))
                self.assertEqual(module.sharding_strategy, root_strategy)
                self.assertEqual(module.backward_prefetch, root_prefetch)
                self.assertEqual(module.mixed_precision, fp16_mp)
            else:
                self.assertFalse(isinstance(module, FSDP))

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_auto_wrap_api(self):
        """
        Test to ensure with auto wrap, we wrap child modules correctly based on the min_num_params.
        ``nn.Linear(5, 5)`` does not exceed the bucket size, but combined they do.
        """
        sequential = TestFSDPWrap.NestedSequentialModel.get_model(device=False)
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=40
        )
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )

        TestFSDPWrap.NestedSequentialModel.verify_model(self, model)

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_auto_wrap_preset_exclude_wrap(self):
        """
        Test to ensure excluded modules are not wrapped, regardless if the total param size is greater than the
        min_num_params. the size_based_auto_wrap_policy excludes wrapping for {nn.ModuleList, nn.ModuleDict}
        """
        sequential = nn.ModuleList([nn.Linear(5, 5), nn.Linear(5, 5)])
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=40
        )

        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )

        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model[0], nn.Linear))
        self.assertTrue(isinstance(model[1], nn.Linear))

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_auto_wrap_preset_exclude_wrap_include_children(self):
        """
        Test to ensure excluded modules are not wrapped, but children are if param size is greater than
        min_num_params
        """
        sequential = nn.ModuleList([nn.Linear(10, 10)])
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=40
        )
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )

        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model[0], FSDP))

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_auto_wrap_preset_force_leaf(self):
        """
        Test to ensure force-leaf modules are not wrapped, and children are not wrapped. The
        size_based_auto_wrap_policy forces leaf modules of type {nn.MultiheadAttention} to not be wrapped
        """
        sequential = nn.Sequential(nn.Linear(10, 10), nn.MultiheadAttention(100, 1))
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=40
        )
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )
        self.assertTrue(isinstance(model.module[0], FSDP))
        # Assert children of multihead attention are not wrapped
        self.assertTrue(isinstance(model.module[1], nn.MultiheadAttention))
        self.assertTrue(isinstance(model.module[1].out_proj, nn.Linear))

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_auto_wrap_preset_force_leaf_custom(self):
        """
        Test to ensure force-leaf modules are not wrapped.
        """
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=40,
            force_leaf_modules=size_based_auto_wrap_policy.FORCE_LEAF_MODULES.union(
                {nn.Linear}
            ),
        )
        sequential = nn.Sequential(
            nn.Linear(10, 10), nn.ModuleList([nn.Linear(10, 10)])
        )
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )
        # Model was wrapped in FSDP as no inner modules were wrapped.
        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model.module[0], nn.Linear))
        self.assertTrue(isinstance(model.module[1], nn.ModuleList))

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "Test Requires CUDA or XPU")
    @parametrize(
        "device_init_mode", [DEVICEInitMode.DEVICE_BEFORE, DEVICEInitMode.DEVICE_AFTER]
    )
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)],
    )
    @parametrize("use_device_id", [True, False])
    def test_auto_wrap_smoke_test(self, device_init_mode, cpu_offload, use_device_id):
        # CPU offload and CUDA after don't work together as expected.
        if (
            cpu_offload.offload_params
            and device_init_mode == DEVICEInitMode.DEVICE_AFTER
        ):
            return

        device = torch.device(device_type)
        torch.accelerator.set_device_index(0)
        device_id = (
            torch.device(device_type, torch.accelerator.current_device_index())
            if use_device_id
            else None
        )

        # Random port in case the next test run quickly, same port would cause conflict.
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        file_name = tempfile.NamedTemporaryFile(delete=False).name
        torch.distributed.init_process_group(
            backend=backend,
            init_method=f"{FILE_SCHEMA}_{file_name}",
            rank=0,
            world_size=1,
        )

        # NOTE: We move model to GPU after init with FSDP to simulate real use
        # cases where full model cannot be loaded onto GPU, but their shards can.
        device_after_init = device_init_mode == DEVICEInitMode.DEVICE_AFTER
        try:
            sequential = TestFSDPWrap.NestedSequentialModel.get_model(
                device=(not device_after_init)
            )
            my_auto_wrap_policy = functools.partial(
                size_based_auto_wrap_policy, min_num_params=40
            )
            model = FSDP(
                sequential,
                cpu_offload=cpu_offload,
                auto_wrap_policy=my_auto_wrap_policy,
                device_id=device_id,
            )
            TestFSDPWrap.NestedSequentialModel.verify_model(self, model)
            if device_after_init:
                model = model.to(device=device_type)
            input = torch.rand((1, 5), dtype=torch.float).to(device)
            output = model(input)
            loss = F.mse_loss(input, output)
            loss.backward()
        finally:
            torch.distributed.destroy_process_group()

        try:
            os.remove(file_name)
        except FileNotFoundError:
            pass

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    @parametrize("wrap_method", [WrapMethod.FSDP_CTOR, WrapMethod.WRAP_API])
    def test_always_wrap_with_ignored_modules(self, wrap_method: WrapMethod):
        sequential = TestFSDPWrap.NestedSequentialModel.get_model(device=False)
        ignored_modules = [sequential[1], sequential[2][0]]
        fsdp_kwargs = {
            "process_group": self.process_group,
            "auto_wrap_policy": always_wrap_policy,
            "ignored_modules": ignored_modules,
        }
        if wrap_method == WrapMethod.FSDP_CTOR:
            model = FSDP(sequential, **fsdp_kwargs)
        elif wrap_method == WrapMethod.WRAP_API:
            with enable_wrap(wrapper_cls=FSDP, **fsdp_kwargs):
                model = wrap(sequential)
        else:
            assert 0, f"Unsupported wrap method: {wrap_method}"
        # All non-ignored modules should be wrapped with FSDP
        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model.module[0], FSDP))
        self.assertTrue(isinstance(model.module[1], nn.Linear))
        self.assertTrue(isinstance(model.module[2], FSDP))
        self.assertTrue(isinstance(model.module[2].module[0], nn.Linear))
        self.assertTrue(isinstance(model.module[2].module[1], FSDP))

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    @parametrize("wrap_method", [WrapMethod.FSDP_CTOR, WrapMethod.WRAP_API])
    def test_auto_wrap_with_ignored_modules(self, wrap_method: WrapMethod):
        sequential = TestFSDPWrap.NestedSequentialModel.get_model(device=False)
        ignored_modules = [sequential[1], sequential[2][0]]
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy,
            min_num_params=40,
        )
        fsdp_kwargs = {
            "process_group": self.process_group,
            "auto_wrap_policy": my_auto_wrap_policy,
            "ignored_modules": ignored_modules,
        }
        if wrap_method == WrapMethod.FSDP_CTOR:
            model = FSDP(sequential, **fsdp_kwargs)
        elif wrap_method == WrapMethod.WRAP_API:
            with enable_wrap(wrapper_cls=FSDP, **fsdp_kwargs):
                model = wrap(sequential)
        else:
            assert 0, f"Unsupported wrap method: {wrap_method}"
        # Since the 2nd linear (`sequential[1]`) is ignored, the wrapping
        # policy does not exceed the parameter threshold before the inner
        # sequential (`sequential[2]`) anymore; hence, it flattens
        # `sequential[0]` and `sequential[2][0]` into `model` and leaves
        # `sequential[1]` and `sequential[2][1]` as-is since they are ignored
        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model.module[0], nn.Linear))
        self.assertTrue(isinstance(model.module[1], nn.Linear))
        self.assertTrue(isinstance(model.module[2], nn.Sequential))
        self.assertTrue(isinstance(model.module[2][0], nn.Linear))
        self.assertTrue(isinstance(model.module[2][1], nn.Linear))

    @unittest.skipIf(not TEST_MULTIGPU, "Requires at least 2 GPUs")
    def test_frozen_params(self):
        """
        Tests that mixing frozen/non-frozen parameters in an FSDP instance
        raises for ``use_orig_params=False`` and warns for ``True``.
        """
        module_classes = (LoraAttention, LoraMLP, LoraDecoder)
        module_wrap_policy = ModuleWrapPolicy(module_classes)

        def lambda_fn_uniform(module: nn.Module):
            return isinstance(module, module_classes)

        def lambda_fn_nonuniform(module: nn.Module):
            if isinstance(module, LoraAttention):
                return {"sharding_strategy": ShardingStrategy.SHARD_GRAD_OP}
            elif isinstance(module, module_classes):
                return True
            return False

        lambda_wrap_policy_uniform = CustomPolicy(lambda_fn_uniform)
        lambda_wrap_policy_nonuniform = CustomPolicy(lambda_fn_nonuniform)

        for use_orig_params, policy in itertools.product(
            [True, False],
            [
                module_wrap_policy,
                lambda_wrap_policy_uniform,
                lambda_wrap_policy_nonuniform,
            ],
        ):
            self._test_frozen_params(use_orig_params, policy)

    def _test_frozen_params(self, use_orig_params: bool, policy: _Policy):
        model = LoraModel().to(device=device_type)
        msg = "layers.0.attn has both parameters with requires_grad=True and False. "
        if use_orig_params:
            msg += "We do not recommend wrapping such modules"
            ctx = self.assertWarnsRegex(UserWarning, msg)
        else:
            msg += "FSDP does not support wrapping such modules when use_orig_params=False."
            ctx = self.assertRaisesRegex(ValueError, msg)
        with ctx:
            FSDP(
                model,
                process_group=self.process_group,
                auto_wrap_policy=policy,
                use_orig_params=use_orig_params,
            )


class TestWrapUtils(TestCase):
    def test_validate_frozen_params(self):
        """Tests the method ``_validate_frozen_params()``."""
        for use_orig_params in [True, False]:
            self._test_validate_frozen_params(use_orig_params)

    def _test_validate_frozen_params(self, use_orig_params: bool):
        model = LoraModel()
        # Wrap only LoRA modules
        modules_to_wrap = {
            module
            for module_name, module in model.named_modules()
            if "lora_A" in module_name or "lora_B" in module_name
        }
        _validate_frozen_params(model, modules_to_wrap, set(), use_orig_params)
        # Additionally wrap attention
        for module in model.modules():
            if isinstance(module, LoraAttention):
                modules_to_wrap.add(module)
        _validate_frozen_params(model, modules_to_wrap, set(), use_orig_params)
        # Additionally wrap decoders
        for module in model.modules():
            if isinstance(module, LoraDecoder):
                modules_to_wrap.add(module)
        _validate_frozen_params(model, modules_to_wrap, set(), use_orig_params)
        # Do not wrap the LoRA-A modules (meaning mixed frozen/non-frozen)
        for module_name, module in model.named_modules():
            if "lora_A" in module_name:
                modules_to_wrap.remove(module)
        regex = "layers.0.attn has both parameters with requires_grad=True and False."
        if use_orig_params:
            # Wrapping the attention manages all parameters except those from
            # the LoRA-B module, which is separately wrapped and all nonfrozen
            lorab_numel = sum(
                p.numel() for p in model.layers[0].attn.lora_B.parameters()
            )
            attn_frozen_param_numel = sum(
                p.numel()
                for p in model.layers[0].attn.parameters()
                if not p.requires_grad
            )
            attn_nonfrozen_param_numel = (
                sum(
                    p.numel()
                    for p in model.layers[0].attn.parameters()
                    if p.requires_grad
                )
                - lorab_numel
            )
            attn_total_param_numel = (
                attn_frozen_param_numel + attn_nonfrozen_param_numel
            )
            regex += (
                " We do not recommend wrapping such modules since the "
                r"gradient memory usage will be higher than expected \("
                f"{attn_total_param_numel} numel instead of {attn_nonfrozen_param_numel} numel "
                r"before sharding via reduce-scatter\). "
            )
        else:
            regex += " FSDP does not support wrapping such modules when use_orig_params=False. "
        regex += "If possible, wrap the frozen parameters with FSDP separately.\n"
        regex += (
            "The following parameters have requires_grad=True:\n"
            r"\['layers.0.attn.lora_A.weight'\]\n"
            "The following parameters have requires_grad=False:\n"
            r"\['layers.0.attn.q_proj.weight', 'layers.0.attn.k_proj.weight', "
            r"'layers.0.attn.v_proj.weight', 'layers.0.attn.o_proj.weight'\]"
        )
        if use_orig_params:
            ctx = self.assertWarnsRegex(UserWarning, regex)
        else:
            ctx = self.assertRaisesRegex(ValueError, regex)
        with ctx:
            _validate_frozen_params(model, modules_to_wrap, set(), use_orig_params)
        # Now ignore those LoRA-A modules' parameters
        ignored_params = set()
        for module_name, module in model.named_modules():
            if "lora_A" in module_name:
                ignored_params.update(module.parameters())
        _validate_frozen_params(model, modules_to_wrap, ignored_params, use_orig_params)


instantiate_parametrized_tests(TestFSDPWrap)
instantiate_parametrized_tests(TestAutoWrap)

if __name__ == "__main__":
    run_tests()
