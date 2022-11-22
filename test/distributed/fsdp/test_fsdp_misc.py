# Owner(s): ["oncall: distributed"]

import functools
import sys
from collections import namedtuple
from contextlib import suppress
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    CPUOffload,
    FlatParameter,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    always_wrap_policy,
    ModuleWrapPolicy,
    transformer_auto_wrap_policy,
)
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    _assert_module_states,
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    NestedWrappedModule,
    TransformerWithSharedParams,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
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


class TestFSDPMisc(FSDPTest):
    @property
    def world_size(self):
        return 2

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_namedtuple(self):
        # Ensure namedtuple support, preventing issues such as
        # https://github.com/pytorch/pytorch/issues/83053
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(100, 100)

            def forward(self, x):
                return x

        m = MyModule().cuda()
        m = FSDP(m)
        t = torch.ones(1, device="cuda", requires_grad=True)

        MyOutputType = namedtuple(
            "MyOutputType", ["a", "b", "c", "d"], defaults=(t, t, t, t)
        )

        inp = MyOutputType()
        out = m(inp)
        # Ensure hooks are registered
        for x in out:
            self.assertNotEqual([], list(x._backward_hooks.values()))

        # TODO: we should check backward() and param is resharded
        # as well, but this is blocked by
        # https://github.com/pytorch/pytorch/issues/83107 and
        # https://github.com/pytorch/pytorch/issues/83129

    @skip_if_lt_x_gpu(2)
    def test_fsdp_not_all_outputs_used_in_loss(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(4, 4)
                self.lin2 = nn.Linear(4, 4)

            def forward(self, x):
                a = self.lin1(x)
                b = self.lin2(x)
                return (a, b)

        def _check_resharded(fsdp_module):
            for handle in fsdp_module._handles:
                param = handle.flat_param
                if handle.uses_sharded_strategy:
                    full_param = param._full_param_padded
                    self.assertEqual(full_param.storage().size(), 0)

                self.assertEqual(param.data_ptr(), param._local_shard.data_ptr())

        def _check_equal(local, fsdp):
            with FSDP.summon_full_params(fsdp):
                for p1, p2 in zip(fsdp.parameters(), local.parameters()):
                    torch.testing.assert_close(p1, p2)

        for sharding_strategy in [
            ShardingStrategy.FULL_SHARD,
            ShardingStrategy.SHARD_GRAD_OP,
            ShardingStrategy.NO_SHARD,
        ]:
            with self.subTest(sharding_strategy=sharding_strategy):
                fsdp_ctor = functools.partial(FSDP, sharding_strategy=sharding_strategy)
                m = MyModule().cuda()
                m_local = deepcopy(m)
                local_m = m_local
                prev_params = [p.clone() for p in m_local.parameters()]

                m.lin1 = fsdp_ctor(m.lin1)
                m = fsdp_ctor(m)
                _check_equal(m_local, m)

                opt = torch.optim.SGD(m.parameters(), lr=1e-3)
                opt_local = torch.optim.SGD(local_m.parameters(), lr=1e-3)

                for i in range(6):
                    t = torch.ones(4, device="cuda")
                    a, b = m(t)
                    local_a, local_b = local_m(t)
                    if i < 2:
                        # use both params in loss computation. Later,
                        # b will go unused and we check grads are the
                        # same as local training.
                        loss = (a @ b).sum()
                        loss_local = (local_a @ local_b).sum()
                    else:
                        loss = a.sum()
                        loss_local = local_a.sum()

                    loss.backward()
                    loss_local.backward()
                    _check_resharded(m)
                    opt.step()
                    opt_local.step()
                    _check_equal(m_local, m)
                    # Ensure at least some change from previous params, otherwise
                    # above check would be vacuously true.
                    self.assertTrue(
                        any(
                            not torch.equal(p1, p2)
                            for p1, p2 in zip(prev_params, m_local.parameters())
                        )
                    )
                    prev_params = [p.clone() for p in local_m.parameters()]
                    opt.zero_grad()
                    opt_local.zero_grad()

                dist.barrier()

    @skip_if_lt_x_gpu(2)
    @parametrize("use_second_layer", [True, False])
    @parametrize("sharding_strategy", [ShardingStrategy.NO_SHARD, None])
    def test_fsdp_module_no_compute_grad(self, use_second_layer, sharding_strategy):
        # When use_second_layer=True, b is involved in forward computation but does
        # not receive grad in backward. Otherwise, b is not involved in forward
        # computation.
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(10, 10)
                self.b = nn.Linear(10, 10)

            def forward(self, x, y):
                out1 = self.a(x)
                if use_second_layer:
                    out2 = self.b(y)
                    return out1, out2
                else:
                    return out1

        fsdp = FSDP(
            MyModel().cuda(),
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=always_wrap_policy,
        )
        x = torch.randn(10, 10, device="cuda")
        y = torch.randn(10, 10, device="cuda")
        for i in range(4):
            if use_second_layer:
                a, b = fsdp(x, y)
            else:
                a = fsdp(x, y)
            loss = a.sum()
            loss.backward()

            # self.a receives grad, self.b does not
            a_grad = fsdp.module.a._handles[0].flat_param.grad
            b_grad = fsdp.module.b._handles[0].flat_param.grad
            self.assertIsNotNone(a_grad)
            self.assertIsNone(b_grad)

    @skip_if_lt_x_gpu(2)
    def test_device_id_auto_wrap(self):
        """Tests that ``auto_wrap_policy`` propagates ``device_id`` to all
        nested FSDP instances."""
        self.run_subtests(
            {"use_callable": [False, True]},
            self._test_device_id_auto_wrap,
        )

    def _test_device_id_auto_wrap(self, use_callable: bool):
        module_classes = {TransformerEncoderLayer, TransformerDecoderLayer}
        if use_callable:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=module_classes,
            )
        else:
            auto_wrap_policy = ModuleWrapPolicy(module_classes)
        fsdp_kwargs = {
            "auto_wrap_policy": auto_wrap_policy,
            "device_id": torch.cuda.current_device(),
        }
        fsdp_model = TransformerWithSharedParams.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            fsdp_kwargs,
        )
        for fsdp_module in FSDP.fsdp_modules(fsdp_model):
            self.assertEqual(
                fsdp_module.compute_device,
                torch.device("cuda", torch.cuda.current_device()),
            )

    @skip_if_lt_x_gpu(2)
    def test_fsdp_device_id_cpu_offload(self):
        """
        Ensures that even if device_id is specified but we have
        CPU offload, module is on CPU after init.
        """

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(10, 10)
                self.b = nn.Linear(10, 10)

            def forward(self, x):
                return self.b(self.a(x))

        model = MyModel()

        fsdp = FSDP(
            model,
            auto_wrap_policy=always_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            device_id=torch.cuda.current_device(),
        )

        cpu_device = torch.device("cpu")

        for fsdp_unit in FSDP.fsdp_modules(fsdp):
            # This FSDP unit may not directly manage
            # any parameters.
            if len(fsdp_unit.params) > 0:
                fsdp_param = fsdp_unit.params[0]
                self.assertEqual(fsdp_param.device, cpu_device)

    @skip_if_lt_x_gpu(2)
    @parametrize("use_index", [True, False])
    def test_fsdp_device_id(self, use_index):
        """
        Tests the FSDP ``device_id`` argument:
          - Wrapping a CPU module should move the module to the GPU matching
          ``device_id``
          - Wrapping a GPU module already on the GPU matching ``device_id``
          should not raise an error
          - Wrapping a GPU module already on GPU and passing a GPU device
          without specifying a device ID (i.e. ``torch.device("cuda")``) warns
        """
        dev_id = (
            torch.cuda.current_device()
            if use_index
            else torch.device("cuda", torch.cuda.current_device())
        )

        def _check_device_matches(module, device_id):
            """Checks that the ``FlatParameter``s in ``module`` have device
            matching ``device_id``."""
            devices = {
                p.device for p in module.parameters() if isinstance(p, FlatParameter)
            }
            assert len(devices) > 0
            self.assertEqual(1, len(devices))
            found_device = devices.pop()
            if use_index and not isinstance(device_id, torch.device):
                device = torch.device("cuda", device_id)
            else:
                device = device_id
            self.assertEqual(found_device, device)

        # Check that FSDP parameters are moved to `device_id` for a CPU module
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_NEVER,
            fsdp_kwargs={"device_id": dev_id},
        )
        _check_device_matches(nested_wrapped_module, dev_id)
        # Check that specifying `device_id` for a GPU module already on that
        # device does not raise an error
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_BEFORE,
            fsdp_kwargs={"device_id": dev_id},
        )
        _check_device_matches(nested_wrapped_module, dev_id)
        # Check that passing in `torch.device("cuda")` for a GPU module warns
        regex = "does not have an explicit index"
        context = self.assertWarnsRegex(
            expected_warning=UserWarning, expected_regex=regex
        )
        with context:
            nested_wrapped_module = NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_BEFORE,
                fsdp_kwargs={"device_id": torch.device("cuda")},
            )
        _check_device_matches(
            nested_wrapped_module, torch.device("cuda", torch.cuda.current_device())
        )

    @skip_if_lt_x_gpu(2)
    def test_module_device_mismatches_device_id(self):
        """Tests that specifying a ``device_id`` argument to FSDP for a GPU
        module that does not match the GPU device ID raises an error."""
        context = (
            self.assertRaisesRegex(ValueError, f"cuda:{self.rank} vs cuda:0")
            if self.rank != 0
            else suppress()
        )
        with context:
            NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                # Move wrapped modules to CUDA before wrapping with FSDP
                cuda_init_mode=CUDAInitMode.CUDA_BEFORE,
                # Should raise error since rank 1 is given `device_id=0` when
                # the model is on cuda:1
                fsdp_kwargs={"device_id": 0},
            )

    @skip_if_lt_x_gpu(2)
    def test_multi_device_not_supported(self):
        """Tests that wrapping a multi-device module (i.e. with submodules on
        both GPU and CPU) with FSDP raises an error."""

        class MultiDeviceModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(1, 1).cuda()
                self.b = nn.Linear(1, 1)

        with self.assertRaisesRegex(
            RuntimeError, "FSDP only supports single device modules"
        ):
            FSDP(MultiDeviceModule())

    @skip_if_lt_x_gpu(2)
    def test_no_params(self):
        """
        Test that device_id and cpu init work if module has no params
        (they are effective noops, but ensure FSDP does not assume module
        has parameters during init)
        """
        # Test CPU
        no_params = nn.ReLU()
        module = FSDP(no_params)
        # Test CUDA
        no_params = nn.ReLU().cuda()
        module = FSDP(no_params)
        # Test CPU + device_id
        no_params = nn.ReLU()
        module = FSDP(no_params, device_id=torch.cuda.current_device())
        # For modules with no params, wrong device_id will raise error about
        # inconsistency between compute_device and device_id, since compute_device
        # is computed as torch.cuda.current_device when there are no params.
        no_params = nn.ReLU().cuda()
        context = (
            (
                self.assertRaisesRegex(
                    ValueError, f"Inconsistent.*cuda:{self.rank} vs cuda:0"
                )
            )
            if self.rank != 0
            else suppress()
        )
        with context:
            module = FSDP(no_params, device_id=0)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_cpu_init_stays_on_cpu(self):
        """Tests that passing a CPU module to FSDP preserves that the wrapped
        module is on CPU after FSDP initialization, albeit after loging a
        warning, and that FSDP moves CPU input to GPU before the forward."""
        torch.cuda.set_device(self.rank)
        regex = "passed-in `module` is on CPU"
        context = self.assertWarnsRegex(
            expected_warning=UserWarning, expected_regex=regex
        )
        with context:
            nested_wrapped_module = NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.RECURSIVE,
                CUDAInitMode.CUDA_NEVER,
            )
            fsdp_model = FSDP(nested_wrapped_module, self.process_group)
        devices = {p.device for p in fsdp_model.parameters()}
        self.assertEqual(1, len(devices))
        self.assertEqual(torch.device("cpu"), devices.pop())
        fsdp_model = fsdp_model.cuda()
        # Ensure fwd + backward can be performed after moving to CUDA.
        # CPU input also tests that input is correctly moved to appropriate
        # CUDA device.
        inp = fsdp_model.module.get_input(device=torch.device("cpu"))
        fsdp_model(*inp).sum().backward()

    @skip_if_lt_x_gpu(2)
    def test_cpu_init_with_sync_module_states(self):
        """Tests that passing ``sync_module_states=True`` raises an error for
        a CPU module since the synchronization requires GPU communication,
        while additionally passing ``device_id`` does not raise an error."""
        nested_wrapped_module = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.RECURSIVE,
            CUDAInitMode.CUDA_NEVER,
        )
        with self.assertRaisesRegex(
            ValueError, "The module has CPU parameters when `sync_module_states=True`"
        ):
            FSDP(nested_wrapped_module, self.process_group, sync_module_states=True)

        # Specifying device_id with sync_module_states=True works.
        FSDP(
            nested_wrapped_module,
            self.process_group,
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
        )

    @skip_if_lt_x_gpu(2)
    def test_fsdp_same_model_across_ranks(self):
        """
        FSDP broadcasts model from rank 0 to ensure it starts off with the same
        values.
        """

        class MyModel(nn.Module):
            def __init__(self, rank):
                super().__init__()
                # Seed via rank to make model different across ranks
                torch.manual_seed(rank)
                torch.cuda.manual_seed(rank)
                self.lin = nn.Linear(10, 10, bias=False)
                self.register_buffer("buffer", torch.ones(1) * rank)

        m = MyModel(self.rank).cuda()
        _assert_module_states(
            m, process_group=self.process_group, assert_fn=self.assertNotEqual
        )
        # Passing sync_module_states into FSDP makes model the same during init.
        fsdp = FSDP(m, sync_module_states=True)
        with fsdp.summon_full_params(fsdp):
            _assert_module_states(
                fsdp, process_group=self.process_group, assert_fn=self.assertEqual
            )

        # sync_module_states also works with CPU module with device_id passed in
        m = MyModel(self.rank)
        _assert_module_states(
            m, process_group=self.process_group, assert_fn=self.assertNotEqual
        )
        # Passing sync_module_states into FSDP makes model the same during init.
        fsdp = FSDP(m, device_id=torch.cuda.current_device(), sync_module_states=True)
        with fsdp.summon_full_params(fsdp):
            _assert_module_states(
                fsdp, process_group=self.process_group, assert_fn=self.assertEqual
            )


instantiate_parametrized_tests(TestFSDPMisc)

if __name__ == "__main__":
    run_tests()
