# Owner(s): ["oncall: distributed"]

import functools
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
)
from torch.distributed.fsdp.wrap import (
    _FSDPPolicy,
    _or_policy,
    _wrap_module_cls_individually,
    always_wrap_policy,
    enable_wrap,
    ModuleWrapPolicy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    wrap,
)
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.modules.batchnorm import _BatchNorm
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    _maybe_cuda,
    CUDAInitMode,
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
    TestCase,
)


class BatchNormNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(10, 10, bias=False)
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm2d(10)
        self.bn3 = nn.BatchNorm3d(10)
        self.sync_bn = nn.SyncBatchNorm(10)


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
        def get_model(cuda=True):
            sequential = nn.Sequential(
                nn.Linear(5, 5),
                nn.Linear(5, 5),
                nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 5)),
            )
            if cuda:
                sequential = sequential.cuda()
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
        self, cuda_init_mode=CUDAInitMode.CUDA_BEFORE, nested=False
    ) -> FSDP:
        fn_self = self

        class MyModel(nn.Module):
            def __init__(self, nested):
                super().__init__()
                # TODO: test the various init modes.
                move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE
                # if nested=True, the FSDP module will be nested one layer deep
                # and we should pick that up.
                if nested:
                    self.lin1 = nn.Sequential(
                        _maybe_cuda(fn_self._get_linear(1, 1), move_to_cuda),
                        FSDP(_maybe_cuda(fn_self._get_linear(1, 1), move_to_cuda)),
                    )
                else:
                    self.lin1 = FSDP(
                        _maybe_cuda(fn_self._get_linear(1, 1), move_to_cuda)
                    )
                self.lin2 = FSDP(_maybe_cuda(fn_self._get_linear(1, 1), move_to_cuda))
                self.lin3 = FSDP(_maybe_cuda(fn_self._get_linear(1, 1), move_to_cuda))

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                return self.lin3(self.lin2(self.lin1(input)))

        model = MyModel(nested=nested)
        return model

    @skip_if_lt_x_gpu(2)
    @parametrize("nested", [True, False])
    @parametrize("cuda_init_mode", [CUDAInitMode.CUDA_AFTER, CUDAInitMode.CUDA_BEFORE])
    def test_error_already_wrapped(self, nested, cuda_init_mode):
        """
        Test that an error is raised if we attempt to wrap when submodules are
        already FSDP.
        """
        wrapped_fsdp = self._get_already_wrapped_fsdp(
            nested=nested, cuda_init_mode=cuda_init_mode
        )
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
            wrapped_fsdp = wrapped_fsdp.cuda()

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
            def __init__(self):
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
    @parametrize("cuda_init_mode", [CUDAInitMode.CUDA_AFTER, CUDAInitMode.CUDA_BEFORE])
    def test_main_wrap_api(
        self,
        cpu_offload: CPUOffload,
        backward_prefetch: BackwardPrefetch,
        forward_prefetch: bool,
        cuda_init_mode: CUDAInitMode,
    ):
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER and cpu_offload.offload_params:
            # they don't work together, expected
            return

        move_to_cuda = cuda_init_mode == CUDAInitMode.CUDA_BEFORE

        class Nested(nn.Module):
            def __init__(self):
                super().__init__()
                self.nested_lin = _maybe_cuda(nn.Linear(1, 1, bias=False), move_to_cuda)

            def forward(self, input):
                return self.nested_lin(input)

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = _maybe_cuda(nn.Linear(1, 1, bias=False), move_to_cuda)
                self.lin2 = _maybe_cuda(nn.Linear(1, 1, bias=False), move_to_cuda)
                self.lin3 = _maybe_cuda(nn.Linear(1, 1, bias=False), move_to_cuda)
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
        if cuda_init_mode == CUDAInitMode.CUDA_AFTER:
            wrapped_model = wrapped_model.cuda()

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
        inp = torch.ones(1).cuda()
        for _ in range(6):
            optim.zero_grad()
            loss = wrapped_model(inp).sum()
            loss.backward()
            optim.step()


class TestAutoWrap(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # For all the tests here, we use a fake group
        self.process_group = DummyProcessGroup(rank=0, size=1)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
    def test_wrap_disabled_outside_context(self):
        pg = self.process_group

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = wrap(nn.Linear(5, 5), process_group=pg)

        model = MyModel()
        with enable_wrap(wrapper_cls=FSDP, process_group=pg):
            model = wrap(model)

        self.assertTrue(isinstance(model, FSDP))
        self.assertFalse(isinstance(model.lin, FSDP))
        self.assertTrue(isinstance(model.lin, nn.Linear))

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
    def test_wrap_override_defaults(self):
        new_process_group = DummyProcessGroup(rank=0, size=2)
        with enable_wrap(wrapper_cls=FSDP, process_group=self.process_group):
            layer = wrap(nn.Linear(5, 5), process_group=new_process_group)
        self.assertTrue(isinstance(layer, FSDP))
        self.assertTrue(layer.process_group is new_process_group)
        self.assertEqual(layer.rank, 0)
        self.assertEqual(layer.world_size, 2)

    @unittest.skipIf(not torch.cuda.is_available(), "Test Requires CUDA")
    def test_always_wrap(self):
        """
        Test to ensure that if `always_wrap_policy` is
        passed into FSDP, all submodules are wrapped.
        """
        seq = TestFSDPWrap.NestedSequentialModel.get_model(cuda=True)
        model = FSDP(
            seq, process_group=self.process_group, auto_wrap_policy=always_wrap_policy
        )
        TestFSDPWrap.NestedSequentialModel.verify_model_all_wrapped(self, model)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
    def test_transformer_auto_wrap_policy(self):
        """Tests the ``transformer_auto_wrap_policy``."""
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerEncoderLayer, TransformerDecoderLayer},
        )
        self._test_transformer_wrapping(auto_wrap_policy)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
    def test_module_wrap_policy(self):
        """Tests the ``ModuleWrapPolicy``."""
        auto_wrap_policy = ModuleWrapPolicy(
            {TransformerEncoderLayer, TransformerDecoderLayer}
        )
        self._test_transformer_wrapping(auto_wrap_policy)

    def _test_transformer_wrapping(
        self, auto_wrap_policy: Union[Callable, _FSDPPolicy]
    ):
        fsdp_kwargs = {"auto_wrap_policy": auto_wrap_policy}
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
    def test_auto_wrap_api(self):
        """
        Test to ensure with auto wrap, we wrap child modules correctly based on the min_num_params.
        ``nn.Linear(5, 5)`` does not exceed the bucket size, but combined they do.
        """
        sequential = TestFSDPWrap.NestedSequentialModel.get_model(cuda=False)
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=40
        )
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy,
        )

        TestFSDPWrap.NestedSequentialModel.verify_model(self, model)

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
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

    @unittest.skipIf(not torch.cuda.is_available(), "Test Requires CUDA")
    @parametrize("cuda_init_mode", [CUDAInitMode.CUDA_BEFORE, CUDAInitMode.CUDA_AFTER])
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)],
    )
    @parametrize("use_device_id", [True, False])
    def test_auto_wrap_smoke_test(self, cuda_init_mode, cpu_offload, use_device_id):
        # CPU offload and CUDA after don't work together as expected.
        if cpu_offload.offload_params and cuda_init_mode == CUDAInitMode.CUDA_AFTER:
            return

        device = torch.device("cuda")
        torch.cuda.set_device(0)
        device_id = (
            torch.device("cuda", torch.cuda.current_device()) if use_device_id else None
        )

        # Random port in case the next test run quickly, same port would cause conflict.
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(find_free_port())

        file_name = tempfile.NamedTemporaryFile(delete=False).name
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"{FILE_SCHEMA}_{file_name}",
            rank=0,
            world_size=1,
        )

        # NOTE: We move model to CUDA after init with FSDP to simulate real use
        # cases where full model cannot be loaded onto GPU, but their shards can.
        cuda_after_init = cuda_init_mode == CUDAInitMode.CUDA_AFTER
        try:
            sequential = TestFSDPWrap.NestedSequentialModel.get_model(
                cuda=(not cuda_after_init)
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
            if cuda_after_init:
                model = model.cuda()
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
    @parametrize("wrap_method", [WrapMethod.FSDP_CTOR, WrapMethod.WRAP_API])
    def test_always_wrap_with_ignored_modules(self, wrap_method: WrapMethod):
        sequential = TestFSDPWrap.NestedSequentialModel.get_model(cuda=False)
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

    @unittest.skipIf(torch.cuda.device_count() < 2, "Requires at least 2 GPUs")
    @parametrize("wrap_method", [WrapMethod.FSDP_CTOR, WrapMethod.WRAP_API])
    def test_auto_wrap_with_ignored_modules(self, wrap_method: WrapMethod):
        sequential = TestFSDPWrap.NestedSequentialModel.get_model(cuda=False)
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


class LoraModel(nn.Module):
    """This is a toy LoRA decoder model."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 32)
        self.layers = nn.ModuleList([LoraDecoder() for _ in range(4)])
        self.norm = nn.LayerNorm(32)
        self.embed_tokens.weight.requires_grad_(False)
        self.norm.weight.requires_grad_(False)
        self.norm.bias.requires_grad_(False)


class LoraDecoder(nn.Module):
    def __init__(self):
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
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.proj1 = nn.Linear(32, 128, bias=False)
        self.proj2 = nn.Linear(128, 32, bias=False)
        self.proj1.weight.requires_grad_(False)
        self.proj2.weight.requires_grad_(False)


class TestWrapUtils(TestCase):
    def test_validate_frozen_params(self):
        """Tests the method ``_validate_frozen_params()``."""
        for strict in [True, False]:
            self._test_validate_frozen_params(strict)

    def _test_validate_frozen_params(self, strict: bool):
        model = LoraModel()
        # Wrap only LoRA modules
        modules_to_wrap = {
            module
            for module_name, module in model.named_modules()
            if "lora_A" in module_name or "lora_B" in module_name
        }
        _validate_frozen_params(model, modules_to_wrap, set(), strict)
        # Additionally wrap attention
        for module in model.modules():
            if isinstance(module, LoraAttention):
                modules_to_wrap.add(module)
        _validate_frozen_params(model, modules_to_wrap, set(), strict)
        # Additionally wrap decoders
        for module in model.modules():
            if isinstance(module, LoraDecoder):
                modules_to_wrap.add(module)
        _validate_frozen_params(model, modules_to_wrap, set(), strict)
        # Do not wrap the LoRA-A modules (meaning mixed frozen/non-frozen)
        for module_name, module in model.named_modules():
            if "lora_A" in module_name:
                modules_to_wrap.remove(module)
        regex = "layers.0.attn has both parameters with requires_grad=True and False."
        if strict:
            regex += " FSDP does not support wrapping such modules.\n"
        else:
            regex += (
                " FSDP does not recommend wrapping such modules since the "
                "gradient memory usage will be higher than expected.\n"
            )
        regex += (
            "The following parameters have requires_grad=True:\n"
            r"\['layers.0.attn.lora_A.weight'\]\n"
            "The following parameters have requires_grad=False:\n"
            r"\['layers.0.attn.q_proj.weight', 'layers.0.attn.k_proj.weight', "
            r"'layers.0.attn.v_proj.weight', 'layers.0.attn.o_proj.weight'\]"
        )
        if strict:
            ctx = self.assertRaisesRegex(ValueError, regex)
        else:
            ctx = self.assertWarnsRegex(UserWarning, regex)
        with ctx:
            _validate_frozen_params(model, modules_to_wrap, set(), strict)
        # Now ignore those LoRA-A modules' parameters
        ignored_params = set()
        for module_name, module in model.named_modules():
            if "lora_A" in module_name:
                for param in module.parameters():
                    ignored_params.add(param)
        _validate_frozen_params(model, modules_to_wrap, ignored_params, strict)


instantiate_parametrized_tests(TestFSDPWrap)
instantiate_parametrized_tests(TestAutoWrap)

if __name__ == "__main__":
    run_tests()
