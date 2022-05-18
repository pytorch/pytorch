# Owner(s): ["oncall: distributed"]

from enum import Enum, auto
import functools
import os
import tempfile
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    always_wrap_policy,
    default_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.testing._internal.common_distributed import (
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_fsdp import (
    DummyProcessGroup,
    FSDPTest,
    FSDPInitMode,
    _maybe_cuda,
)
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    run_tests,
    find_free_port,
    TestCase,
    parametrize,
    instantiate_parametrized_tests,
)

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
        self, fsdp_init_mode=FSDPInitMode.CUDA_BEFORE, nested=False
    ) -> FSDP:
        fn_self = self

        class MyModel(nn.Module):
            def __init__(self, nested):
                super().__init__()
                # TODO: test the various init modes.
                move_to_cuda = fsdp_init_mode == FSDPInitMode.CUDA_BEFORE
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
    @parametrize("fsdp_init_mode", [FSDPInitMode.CUDA_AFTER, FSDPInitMode.CUDA_BEFORE])
    def test_error_already_wrapped(self, nested, fsdp_init_mode):
        """
        Test that an error is raised if we attempt to wrap when submodules are
        already FSDP.
        """
        wrapped_fsdp = self._get_already_wrapped_fsdp(nested=nested, fsdp_init_mode=fsdp_init_mode)
        if fsdp_init_mode == FSDPInitMode.CUDA_AFTER:
            wrapped_fsdp = wrapped_fsdp.cuda()

        with self.assertRaisesRegex(ValueError, "to NOT be FullyShardedDataParallel"):
            mod = FSDP(wrapped_fsdp, auto_wrap_policy=default_auto_wrap_policy)

    @skip_if_lt_x_gpu(2)
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)]
    )
    @parametrize(
        "backward_prefetch",
        [BackwardPrefetch.BACKWARD_POST, BackwardPrefetch.BACKWARD_PRE]
    )
    @parametrize(
        "fsdp_init_mode",
        [FSDPInitMode.CUDA_AFTER, FSDPInitMode.CUDA_BEFORE]
    )
    def test_main_wrap_api(self, cpu_offload, backward_prefetch, fsdp_init_mode):

        if fsdp_init_mode == FSDPInitMode.CUDA_AFTER and cpu_offload.offload_params:
            # they don't work together, expected
            return

        move_to_cuda = fsdp_init_mode == FSDPInitMode.CUDA_BEFORE

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
                default_auto_wrap_policy,
                min_num_params=0,  # wrap all modules
            ),
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
        )
        if fsdp_init_mode == FSDPInitMode.CUDA_AFTER:
            wrapped_model = wrapped_model.cuda()

        modules_in_fsdp_graph_order = [
            wrapped_model.module.lin1,
            wrapped_model.module.lin2,
            wrapped_model.module.lin3,
            wrapped_model.module.lin4.module.nested_lin,
            wrapped_model.module.lin4,
            wrapped_model
        ]

        for module in modules_in_fsdp_graph_order:
            self.assertTrue(isinstance(module, FSDP))
            self._check_cpu_offload(module, cpu_offload)
            self._check_backward_prefetch(module, backward_prefetch)

        # Run model a few times for sanity check.
        optim = torch.optim.SGD(wrapped_model.parameters(), lr=1e-2, momentum=0.9)
        inp = torch.ones(1).cuda()
        for _ in range(6):
            optim.zero_grad()
            loss = wrapped_model(inp).sum()
            loss.backward()
            optim.step()

        # Since we ran with backward prefetch, verify backward prefetch related
        # data.
        for i, module in enumerate(modules_in_fsdp_graph_order):
            self.assertEqual(i, module._my_fsdp_idx_in_graph)
            self.assertTrue(
                module._fsdp_graph_order == modules_in_fsdp_graph_order
            )


class TestAutoWrap(TestCase):
    def setUp(self) -> None:
        super().setUp()
        # For all the tests here, we use a fake group
        self.process_group = DummyProcessGroup(rank=0, size=1)

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
                auto_wrap_policy=functools.partial(default_auto_wrap_policy, min_num_params=1)
            )
        self.assertTrue(isinstance(layer, FSDP))
        self.assertEqual(layer.rank, self.process_group.rank())
        self.assertEqual(layer.world_size, self.process_group.size())

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
        model = FSDP(seq, process_group=self.process_group, auto_wrap_policy=always_wrap_policy)
        TestFSDPWrap.NestedSequentialModel.verify_model_all_wrapped(self, model)

    def test_auto_wrap_api(self):
        """
        Test to ensure with auto wrap, we wrap child modules correctly based on the min_num_params.
        ``nn.Linear(5, 5)`` does not exceed the bucket size, but combined they do.
        """
        sequential = TestFSDPWrap.NestedSequentialModel.get_model(cuda=False)
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=40
        )
        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy
        )

        TestFSDPWrap.NestedSequentialModel.verify_model(self, model)


    def test_auto_wrap_preset_exclude_wrap(self):
        """
        Test to ensure excluded modules are not wrapped, regardless if the total param size is greater than the
        min_num_params. the default_auto_wrap_policy excludes wrapping for {nn.ModuleList, nn.ModuleDict}
        """
        sequential = nn.ModuleList([nn.Linear(5, 5), nn.Linear(5, 5)])
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=40
        )

        model = FSDP(
            sequential,
            process_group=self.process_group,
            auto_wrap_policy=my_auto_wrap_policy
        )

        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model[0], nn.Linear))
        self.assertTrue(isinstance(model[1], nn.Linear))

    def test_auto_wrap_preset_exclude_wrap_include_children(self):
        """
        Test to ensure excluded modules are not wrapped, but children are if param size is greater than
        min_num_params
        """
        sequential = nn.ModuleList([nn.Linear(10, 10)])
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=40
        )
        model = FSDP(sequential, process_group=self.process_group, auto_wrap_policy=my_auto_wrap_policy)

        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model[0], FSDP))

    def test_auto_wrap_preset_force_leaf(self):
        """
        Test to ensure force-leaf modules are not wrapped, and children are not wrapped. The
        default_auto_wrap_policy forces leaf modules of type {nn.MultiheadAttention} to not be wrapped
        """
        sequential = nn.Sequential(nn.Linear(10, 10), nn.MultiheadAttention(100, 1))
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=40
        )
        model = FSDP(sequential, process_group=self.process_group, auto_wrap_policy=my_auto_wrap_policy)
        self.assertTrue(isinstance(model.module[0], FSDP))
        # Assert children of multihead attention are not wrapped
        self.assertTrue(isinstance(model.module[1], nn.MultiheadAttention))
        self.assertTrue(isinstance(model.module[1].out_proj, nn.Linear))

    def test_auto_wrap_preset_force_leaf_custom(self):
        """
        Test to ensure force-leaf modules are not wrapped.
        """
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy,
            min_num_params=40,
            force_leaf_modules=default_auto_wrap_policy.FORCE_LEAF_MODULES.union(
                {nn.Linear}
            ),
        )
        sequential = nn.Sequential(
            nn.Linear(10, 10), nn.ModuleList([nn.Linear(10, 10)])
        )
        model = FSDP(sequential, process_group=self.process_group, auto_wrap_policy=my_auto_wrap_policy)
        # Model was wrapped in FSDP as no inner modules were wrapped.
        self.assertTrue(isinstance(model, FSDP))
        self.assertTrue(isinstance(model.module[0], nn.Linear))
        self.assertTrue(isinstance(model.module[1], nn.ModuleList))

    @unittest.skipIf(not torch.cuda.is_available(), "Test Requires CUDA")
    @parametrize("fsdp_init_mode", [FSDPInitMode.CUDA_BEFORE, FSDPInitMode.CUDA_AFTER])
    @parametrize(
        "cpu_offload",
        [CPUOffload(offload_params=False), CPUOffload(offload_params=True)]
    )
    def test_auto_wrap_smoke_test(self, fsdp_init_mode, cpu_offload):
        # CPU offload and CUDA after don't work together as expected.
        if (
            cpu_offload.offload_params and fsdp_init_mode == FSDPInitMode.CUDA_AFTER
        ):
            return

        device = torch.device("cuda")
        torch.cuda.set_device(0)

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
        cuda_after_init = fsdp_init_mode == FSDPInitMode.CUDA_AFTER
        try:
            sequential = TestFSDPWrap.NestedSequentialModel.get_model(cuda=(not cuda_after_init))
            my_auto_wrap_policy = functools.partial(
                default_auto_wrap_policy, min_num_params=40
            )
            model = FSDP(sequential, cpu_offload=cpu_offload, auto_wrap_policy=my_auto_wrap_policy)
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

    @parametrize("wrap_method", [WrapMethod.FSDP_CTOR, WrapMethod.WRAP_API])
    def test_auto_wrap_with_ignored_modules(self, wrap_method: WrapMethod):
        sequential = TestFSDPWrap.NestedSequentialModel.get_model(cuda=False)
        ignored_modules = [sequential[1], sequential[2][0]]
        my_auto_wrap_policy = functools.partial(
            default_auto_wrap_policy, min_num_params=40,
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


instantiate_parametrized_tests(TestFSDPWrap)
instantiate_parametrized_tests(TestAutoWrap)

if __name__ == "__main__":
    run_tests()
