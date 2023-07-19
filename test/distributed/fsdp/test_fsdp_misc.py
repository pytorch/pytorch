# Owner(s): ["oncall: distributed"]

import functools
import os
import sys
import warnings
from collections import namedtuple
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Tuple

import torch
import torch.distributed as dist
import torch.distributed.fsdp._traversal_utils as traversal_utils
import torch.nn as nn
from torch.distributed.fsdp import (
    CPUOffload,
    FlatParameter,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)
from torch.distributed.fsdp._runtime_utils import HOMOGENEOUS_ATTR_NAMES
from torch.distributed.fsdp.flat_param import _FSDP_USE_UNSAFE_SETATTR
from torch.distributed.fsdp.wrap import (
    always_wrap_policy,
    ModuleWrapPolicy,
    transformer_auto_wrap_policy,
)
from torch.distributed.optim import _apply_optimizer_in_backward
from torch.nn import TransformerDecoderLayer, TransformerEncoderLayer
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    _assert_module_states,
    CUDAInitMode,
    FSDPInitMode,
    FSDPTest,
    FSDPTestMultiThread,
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


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(2, 2)
        self.b = nn.Linear(2, 2)

    def forward(self, x, y):
        return self.b(self.a(x + y))


class TestFSDPMiscMultiProcess(FSDPTest):
    @property
    def world_size(self):
        return 2

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

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
    def test_fsdp_not_all_outputs_used_in_loss(self):
        self.run_subtests(
            {
                "sharding_strategy": [
                    ShardingStrategy.FULL_SHARD,
                    ShardingStrategy.SHARD_GRAD_OP,
                    ShardingStrategy.NO_SHARD,
                ]
            },
            self._test_fsdp_not_all_outputs_used_in_loss,
        )

    def _test_fsdp_not_all_outputs_used_in_loss(
        self, sharding_strategy: ShardingStrategy
    ):
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
    def test_fsdp_optim_overlap_no_use_orig_params_error(self):
        fsdp_overlap = FSDP(
            MyModel().cuda(),
            auto_wrap_policy=always_wrap_policy,
            use_orig_params=False,
        )
        optim_cls = torch.optim.SGD
        optim_kwargs = {"lr": 0.03}
        _apply_optimizer_in_backward(
            optimizer_class=optim_cls,
            params=fsdp_overlap.parameters(),
            optimizer_kwargs=optim_kwargs,
            register_hook=False,
        )

        inp = torch.randn(10, 10, device="cuda")
        with self.assertRaisesRegex(
            RuntimeError, "only supported with use_orig_params=True"
        ):
            fsdp_overlap(inp, inp)

    @skip_if_lt_x_gpu(2)
    def test_fsdp_optimizer_overlap(self):
        torch.manual_seed(0)
        for cpu_offload in [True, False]:
            offload = CPUOffload(offload_params=cpu_offload)
            model = MyModel().cuda()
            model_overlap = deepcopy(model)
            fsdp = FSDP(
                model.cuda(),
                auto_wrap_policy=always_wrap_policy,
                use_orig_params=True,
                cpu_offload=offload,
            )
            fsdp_overlap = FSDP(
                model_overlap.cuda(),
                auto_wrap_policy=always_wrap_policy,
                use_orig_params=True,
                cpu_offload=offload,
            )
            optim_cls = torch.optim.SGD
            optim_kwargs = {"lr": 0.03}
            _apply_optimizer_in_backward(
                optimizer_class=optim_cls,
                params=fsdp_overlap.parameters(),
                optimizer_kwargs=optim_kwargs,
                register_hook=False,
            )
            for p in fsdp_overlap.parameters():
                assert hasattr(p, "_in_backward_optimizers")
            optim = optim_cls(fsdp.parameters(), **optim_kwargs)

            # Verify params initially equal
            for p1, p2 in zip(fsdp.parameters(), fsdp_overlap.parameters()):
                self.assertEqual(p1, p2)

            with FSDP.summon_full_params(fsdp_overlap):
                fsdp_overlap_prev_params = [
                    (n, p.clone()) for n, p in fsdp_overlap.named_parameters()
                ]

            for i in range(6):
                inp = torch.randn(2, 2, device="cuda")
                with torch.no_grad():
                    inp_clone = inp.clone()
                fsdp(inp, inp).sum().backward()
                fsdp_overlap(inp_clone, inp_clone).sum().backward()

                optim.step()
                optim.zero_grad()

                # Overlapped optimizer FSDP module should have sharded_grad as None.
                for fsdp_unit in FSDP.fsdp_modules(fsdp_overlap):
                    handles = fsdp_unit._handles
                    for handle in handles:
                        handle_grad = handle.sharded_grad
                        self.assertEqual(
                            None,
                            handle_grad,
                            "Overlapped FSDP sharded_grad is not None!",
                        )

                # Note: FSDP without optimizer overlap won't set sharded_grad to None until the next
                # pre-forward since it needs to run FSDP specific logic that picks up that set_to_none=True
                # has been called (or that the gradients have been otherwise set to None)

                # Verify parameters are different than prev iteration
                with FSDP.summon_full_params(fsdp_overlap, with_grads=True):
                    for (n, p), (n_prev, p_prev) in zip(
                        fsdp_overlap.named_parameters(), fsdp_overlap_prev_params
                    ):
                        self.assertNotEqual(
                            p,
                            p_prev,
                            f"{n_prev} Params at iter {i} same as previous iter!",
                        )

                # Verify overlap and non overlapped are the same
                with FSDP.summon_full_params(fsdp_overlap):
                    with FSDP.summon_full_params(fsdp):
                        for (n_overlap, p_overlap), (n, p) in zip(
                            fsdp_overlap.named_parameters(), fsdp.named_parameters()
                        ):
                            self.assertEqual(n_overlap, n)
                            self.assertEqual(
                                p,
                                p_overlap,
                                f"Rank {self.rank}: Params not equal at iteration {i}: {n_overlap} - {p} vs {p_overlap}",
                            )
                            self.assertEqual(
                                None, p.grad, f"Expected param {n} grad to be None"
                            )
                            self.assertEqual(
                                None,
                                p_overlap.grad,
                                f"Expected param {n_overlap} grad to be None",
                            )

                    fsdp_overlap_prev_params = [
                        (n, p.clone()) for n, p in fsdp_overlap.named_parameters()
                    ]

    @skip_if_lt_x_gpu(2)
    def test_fsdp_cpu_init_stays_on_cpu(self):
        # Move me to MT test once warning logging and backward collective issue
        # is resolved.
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
        """
        Tests that passing ``sync_module_states=True`` raises an error for
        a CPU module since the synchronization requires GPU communication,
        while additionally passing ``device_id`` does not raise an error, even
        when the model has CPU buffers.
        """

        def init_nested_wrapped_module():
            return NestedWrappedModule.init(
                self.process_group,
                FSDPInitMode.NO_FSDP,
                CUDAInitMode.CUDA_NEVER,
            )

        with self.assertRaisesRegex(
            ValueError,
            "The module has CPU parameters or buffers when `sync_module_states=True`",
        ):
            FSDP(
                init_nested_wrapped_module(),
                self.process_group,
                sync_module_states=True,
            )

        # Check that `device_id` with `sync_module_states=True` works
        nested_wrapped_module = init_nested_wrapped_module()
        nested_wrapped_module.buf = nn.Buffer(
            torch.ones((2, 2), device="cpu") * self.rank
        )
        nested_wrapped_module.module[0].buf = nn.Buffer(
            torch.ones((3, 2), device="cpu") * self.rank
        )
        nested_wrapped_module = FSDP(
            nested_wrapped_module,
            self.process_group,
            auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
            device_id=torch.cuda.current_device(),
            sync_module_states=True,
        )
        # Each rank's buffers should be 0s since rank 0 is the source, and they
        # should be on GPU since we specified `device_id`
        self.assertEqual(
            nested_wrapped_module.buf.device,
            torch.device("cuda", torch.cuda.current_device()),
        )
        self.assertEqual(nested_wrapped_module.buf, torch.zeros((2, 2)))
        self.assertEqual(
            nested_wrapped_module.module.module[0].buf.device,
            torch.device("cuda", torch.cuda.current_device()),
        )
        self.assertEqual(
            nested_wrapped_module.module.module[0].buf, torch.zeros((3, 2))
        )


class TestFSDPMiscMultiThread(FSDPTestMultiThread):
    @property
    def world_size(self):
        return 2

    @property
    def process_group(self):
        return dist.distributed_c10d._get_default_group()

    @skip_if_lt_x_gpu(2)
    def test_fsdp_namedtuple(self):
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
        Tests FSDP when specifying both ``device_id`` and parameter CPU
        offloading.
        """
        self.run_subtests(
            {"use_orig_params": [False, True]},
            self._test_fsdp_device_id_cpu_offload,
        )

    def _test_fsdp_device_id_cpu_offload(self, use_orig_params: bool):
        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.Linear(10, 10),
                )
                self.lin = nn.Linear(10, 10)

            def forward(self, x):
                return self.lin(self.seq(x))

        model = MyModel()
        # Choose a wrapping policy such that there are (1) nested FSDP
        # instances and (2) the parent FSDP instance has managed parameters
        auto_wrap_policy = ModuleWrapPolicy({nn.Sequential})
        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            device_id=torch.cuda.current_device(),
            use_orig_params=use_orig_params,
        )
        cpu_device = torch.device("cpu")
        for handle in traversal_utils._get_fsdp_handles(fsdp_model):
            self.assertEqual(handle.flat_param.device, cpu_device)

    @skip_if_lt_x_gpu(2)
    def test_module_device_mismatches_device_id(self):
        """Tests that specifying a ``device_id`` argument to FSDP for a GPU
        module that does not match the GPU device ID raises an error."""
        # TODO: override FSDP MT Thread _run to set this instead of here for
        # every test.
        torch.cuda.set_device(self.rank)
        context = (
            self.assertRaisesRegex(ValueError, f"cuda:{self.rank} vs cuda:0")
            if self.rank != 0
            else nullcontext()
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
    def test_cpu_gpu_module(self):
        """Tests a CPU + GPU module supported if device_id is passed
        in, errors if device_id is not.
        """
        torch.cuda.set_device(self.rank)

        class CPUGPUModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.a = nn.Linear(1, 1).cuda()
                self.b = nn.Linear(1, 1)

        cpu_gpu = CPUGPUModule()
        fsdp = FSDP(cpu_gpu, device_id=torch.cuda.current_device())
        for param in fsdp.parameters():
            self.assertEqual(param.device, torch.device(torch.cuda.current_device()))

        # without device_id, we hit an error
        with self.assertRaisesRegex(RuntimeError, "please pass in device_id"):
            FSDP(CPUGPUModule())

    @skip_if_lt_x_gpu(2)
    def test_multigpu_module(self):
        """
        Module on multiple GPUs wrapped in FSDP should raise an error.
        """

        class MultiGPUModule(nn.Module):
            def __init__(self, rank):
                super().__init__()
                self.rank = rank
                self.a = nn.Linear(1, 1).cuda(self.rank)
                self.b = nn.Linear(1, 1).cuda((self.rank + 1) % dist.get_world_size())

        with self.assertRaisesRegex(
            RuntimeError, "FSDP only supports single device modules"
        ):
            FSDP(MultiGPUModule(self.rank))

    @skip_if_lt_x_gpu(2)
    def test_no_params(self):
        """
        Test that device_id and cpu init work if module has no params
        (they are effective noops, but ensure FSDP does not assume module
        has parameters during init)
        """
        # TODO: override FSDP MT Thread _run to set this instead of here for
        # every test.
        torch.cuda.set_device(self.rank)
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
            else nullcontext()
        )
        with context:
            FSDP(no_params, device_id=0)

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
                self.buffer = nn.Buffer(torch.ones(1) * rank)

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

    @skip_if_lt_x_gpu(2)
    def test_homogeneous_attributes(self):
        """
        Tests that passing heterogeneous values for attributes designated as
        homogeneous raises an error.
        """
        # Manually construct this list but verify against the global list of
        # homogeneous attribute names
        all_attr_name_and_values = [
            ("_use_orig_params", False, True),
            ("limit_all_gathers", False, True),
            ("_use_full_prec_in_eval", False, True),
        ]
        self.assertEqual(
            [
                attr_name_and_values[0]
                for attr_name_and_values in all_attr_name_and_values
            ],
            HOMOGENEOUS_ATTR_NAMES,
        )

        self.run_subtests(
            {"attr_name_and_values": all_attr_name_and_values},
            self._test_homogeneous_attributes,
        )

    def _test_homogeneous_attributes(self, attr_name_and_values: Tuple[str, Any, Any]):
        model = NestedWrappedModule.init(
            self.process_group,
            FSDPInitMode.NO_FSDP,
            CUDAInitMode.CUDA_BEFORE,
            {},
        )
        attr_name = attr_name_and_values[0]

        if "_use_full_prec_in_eval" == attr_name:
            model.module[1] = FSDP(model.module[1])
            os.environ["FSDP_USE_FULL_PREC_IN_EVAL"] = "1"
            fsdp_model = FSDP(model)
        else:
            fsdp_kwargs_inner = {attr_name.lstrip("_"): attr_name_and_values[1]}
            fsdp_kwargs_outer = {attr_name.lstrip("_"): attr_name_and_values[2]}
            model.module[1] = FSDP(model.module[1], **fsdp_kwargs_inner)
            fsdp_model = FSDP(model, **fsdp_kwargs_outer)

        # Run a forward to trigger lazy initialization and the error
        with self.assertRaisesRegex(
            ValueError, f"Expects one homogeneous value for {attr_name}"
        ):
            inp = fsdp_model.module.get_input(torch.device("cuda"))
            fsdp_model(*inp)


class TestFSDPMiscWorldSize1(FSDPTestMultiThread):
    @property
    def world_size(self) -> int:
        return 1

    @skip_if_lt_x_gpu(1)
    def test_world_size_1_sharding_strategy_warning(self):
        """
        Tests that FSDP issues a warning when it switches to using ``NO_SHARD``
        when the world size is 1.
        """
        warning_prefix = "FSDP is switching to use `NO_SHARD` instead of"
        # If the user already passes `NO_SHARD`, then there should not be a
        # warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # trigger all warnings
            FSDP(nn.Linear(3, 3).cuda(), sharding_strategy=ShardingStrategy.NO_SHARD)
            for warning in w:
                self.assertTrue(
                    warning.category != UserWarning
                    or not str(warning.message).startswith(warning_prefix)
                )

        # Check that a warning is issued
        warning_suffix = " since the world size is 1."
        # - Pass `FULL_SHARD` or `None`
        expected_regex_full_shard = (
            warning_prefix + " " + str(ShardingStrategy.FULL_SHARD) + warning_suffix
        )
        with self.assertWarnsRegex(UserWarning, expected_regex_full_shard):
            FSDP(nn.Linear(3, 3).cuda(), sharding_strategy=ShardingStrategy.FULL_SHARD)
        with self.assertWarnsRegex(UserWarning, expected_regex_full_shard):
            FSDP(nn.Linear(3, 3).cuda())
        # - Pass `SHARD_GRAD_OP`
        expected_regex_shard_grad_op = (
            warning_prefix + " " + str(ShardingStrategy.SHARD_GRAD_OP) + warning_suffix
        )
        with self.assertWarnsRegex(UserWarning, expected_regex_shard_grad_op):
            FSDP(
                nn.Linear(3, 3).cuda(), sharding_strategy=ShardingStrategy.SHARD_GRAD_OP
            )

    @skip_if_lt_x_gpu(1)
    def test_training_device_mismatch_errors(self):
        """
        Tests that, when training starts, if FSDP parameters are not on the
        expected device, then an informative error is raised. This applies for
        both no parameter CPU offloading and parameter CPU offloading.
        """
        # Incorrectly not moving from CPU -> GPU
        model = torch.nn.Linear(10, 10)
        fsdp_model = FSDP(model)
        inp = torch.randn((2, 10))
        with self.assertRaisesRegex(
            RuntimeError,
            "An FSDP-managed module unexpectedly has parameters on cpu. Make "
            "sure to move the module to cuda:0 before training.",
        ):
            fsdp_model(inp)

        # Incorrectly moving from CPU -> GPU
        model = torch.nn.Linear(10, 10)
        fsdp_model = FSDP(model, cpu_offload=CPUOffload(offload_params=True))
        fsdp_model.to(torch.device("cuda"))
        inp = torch.randn((2, 10))
        with self.assertRaisesRegex(
            RuntimeError,
            "An FSDP-managed module with parameter CPU offloading enabled has "
            "parameters on cuda:0. Make sure to not move the module from CPU "
            "when offloading parameters.",
        ):
            fsdp_model(inp)

    @skip_if_lt_x_gpu(2)
    def test_unsafe_setattr(self):
        """
        Tests that the environment variable for using unsafe setattr gates as
        expected.
        """
        self.run_subtests(
            {"use_orig_params": [False, True]},
            self._test_unsafe_setattr,
        )

    def _test_unsafe_setattr(self, use_orig_params: bool):
        called_setattr_override = False

        class SetattrLinear(nn.Module):
            def __init__(self, in_dim: int, out_dim: int, device: torch.device) -> None:
                super().__init__()
                self.weight = nn.Parameter(
                    torch.randn((in_dim, out_dim), device=device)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x @ self.weight

            def __setattr__(self, name: str, value: Any) -> None:
                nonlocal called_setattr_override
                called_setattr_override = True
                return super().__setattr__(name, value)

        # Construct FSDP module without changing any environment variables and
        # run forward, which triggers both unsharded and sharded view setting
        module = SetattrLinear(5, 5, torch.device("cuda"))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        inp = torch.randn((8, 5), device=torch.device("cuda"))
        called_setattr_override = False
        fsdp_module(inp)
        self.assertTrue(called_setattr_override)

        # Repeat with unsafe setattr explicitly enabled
        os.environ[_FSDP_USE_UNSAFE_SETATTR] = "1"
        module = SetattrLinear(5, 5, torch.device("cuda"))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        called_setattr_override = False
        fsdp_module(inp)
        self.assertFalse(called_setattr_override)

        # Repeat with unsafe setattr explicitly disabled
        os.environ[_FSDP_USE_UNSAFE_SETATTR] = "0"
        module = SetattrLinear(5, 5, torch.device("cuda"))
        fsdp_module = FSDP(module, use_orig_params=use_orig_params)
        called_setattr_override = False
        fsdp_module(inp)
        self.assertTrue(called_setattr_override)


instantiate_parametrized_tests(TestFSDPMiscMultiThread)
instantiate_parametrized_tests(TestFSDPMiscMultiProcess)

if __name__ == "__main__":
    run_tests()
