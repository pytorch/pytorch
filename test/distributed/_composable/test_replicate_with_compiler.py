# Owner(s): ["oncall: distributed"]

import contextlib
import functools
import os
import unittest
from copy import deepcopy
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch import _inductor as inductor, nn
from torch._C import FileCheck
from torch._dynamo import compiled_autograd
from torch._dynamo.utils import counters
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_triton_code
from torch.distributed._composable.replicate import replicate
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as ddp_default_hooks,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    RowwiseParallel,
)
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
    sm_is_or_higher_than,
)
from torch.testing._internal.common_utils import run_tests, skipIfRocm
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils.checkpoint import checkpoint


DIM = 2000


class Net(nn.Module):
    def __init__(self, checkpoint=False):
        super().__init__()
        self.fc1 = nn.Linear(DIM, DIM)
        self.fc2 = nn.Linear(DIM, DIM)
        self.fc3 = nn.Linear(DIM, DIM)
        self.fc4 = nn.Linear(DIM, DIM)
        self.use_checkpoint = checkpoint

    def forward(self, x):
        if self.use_checkpoint:
            _fc1 = checkpoint(self.fc1, x, use_reentrant=False)
        else:
            _fc1 = self.fc1(x)
        return self.fc4(self.fc3(self.fc2(_fc1)))


def compiler_fn(no_inductor=False):
    def _compiler_fn(gm):
        def inner_compiler(gm_, example_inputs_):
            if no_inductor:
                return gm_
            else:
                return inductor.compile(gm_, example_inputs_)

        gm = torch.compile(gm, fullgraph=True, backend=inner_compiler)
        return gm

    return _compiler_fn


class MultiProcessInductorTestCase(MultiProcessTestCase, InductorTestCase):
    """
    A version of MultiProcessTestCase that derives from the Inductor TestCase
    to handle isolation of the inductor cache dir.
    """


class ReplicateTest(MultiProcessInductorTestCase):
    # TODO: consider using all devices? The min(2, ...) here would limit the
    # test to always run on 2 GPUs only.
    @property
    def world_size(self) -> int:
        return min(2, torch.cuda.device_count())

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _test_compile(
        self,
        *,
        use_gpu: bool,
        no_sync: bool,
        setup_func: Optional[Callable] = None,
        no_inductor: bool = False,
        no_compile_forward: bool = False,
        checkpoint: bool = False,
    ):
        backend = "nccl" if use_gpu else "gloo"
        dist.init_process_group(
            backend=backend,
            rank=self.rank,
            world_size=self.world_size,
            store=dist.FileStore(self.file_name, self.world_size),
        )
        if use_gpu:
            torch.cuda.set_device(f"cuda:{self.rank}")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        torch._dynamo.config.optimize_ddp = (
            "python_reducer_without_compiled_forward"
            if no_compile_forward
            else "python_reducer"
        )
        torch.manual_seed(123)
        model = Net(checkpoint=checkpoint).to(device)
        input = torch.randn([1, DIM], device=device)

        compiled_replicate_model = replicate(deepcopy(model))
        if not no_compile_forward:
            compiled_replicate_model = torch.compile(
                compiled_replicate_model, fullgraph=False
            )
        compiled_replicate_optim = torch.optim.Adam(
            compiled_replicate_model.parameters()
        )
        compiled_ddp_model = DDP(deepcopy(model))
        if not no_compile_forward:
            compiled_ddp_model = torch.compile(compiled_ddp_model, fullgraph=True)
        compiled_ddp_optim = torch.optim.Adam(compiled_ddp_model.parameters())
        model = replicate(model)
        optim = torch.optim.Adam(model.parameters())

        if setup_func:
            setup_func(model, compiled_replicate_model, compiled_ddp_model)

        models = [model, compiled_replicate_model, compiled_ddp_model]
        optims = [optim, compiled_replicate_optim, compiled_ddp_optim]
        sync_contexts = [
            contextlib.nullcontext(),
            contextlib.nullcontext(),
            compiled_ddp_model.no_sync(),
        ]

        # Run multiple iterations so that we could test no_sync
        for i in range(2):
            # Setting a different random seed so that if the allreduces are not
            # executed correctly, the gradients won't be correct compared to the
            # eager DDP.
            torch.manual_seed(123 + self.rank + i)
            input = torch.randn([1, DIM], device=device)

            for model_idx in range(3):
                if no_sync and i % 2 == 0:
                    context = sync_contexts[model_idx]
                    if model_idx <= 1:
                        models[model_idx].set_requires_gradient_sync(False)
                else:
                    context = contextlib.nullcontext()
                    if model_idx <= 1:
                        models[model_idx].set_requires_gradient_sync(True)
                context = contextlib.nullcontext()

                with context:
                    bwd_context = (
                        contextlib.nullcontext()
                        if model_idx == 0
                        else compiled_autograd.enable(compiler_fn(no_inductor))
                    )
                    with bwd_context:
                        loss = models[model_idx](input).sum()
                        loss.backward()

            if not no_sync or i % 2 == 1:
                for p1, p2, p3 in zip(
                    model.parameters(),
                    compiled_replicate_model.parameters(),
                    compiled_ddp_model.parameters(),
                ):
                    self.assertEqual(p1.grad, p2.grad)
                    self.assertEqual(p1.grad, p3.grad)
                for optim in optims:
                    optim.step()
                    optim.zero_grad()

        self.assertEqual(
            tuple(model.parameters()), tuple(compiled_replicate_model.parameters())
        )
        self.assertEqual(
            tuple(model.parameters()), tuple(compiled_ddp_model.parameters())
        )

    def test_compile_cpu(self):
        # Test the coalesced_op with CPU.
        torch._inductor.config._fuse_ddp_communication_passes = [
            "fuse_ddp_with_coalesced_op",
            "schedule_comm_wait",
        ]
        self._test_compile(use_gpu=False, no_sync=False)

    def test_compile_cpu_no_sync(self):
        # Test the coalesced_op with CPU.
        torch._inductor.config._fuse_ddp_communication_passes = [
            "fuse_ddp_with_coalesced_op",
            "schedule_comm_wait",
        ]
        self._test_compile(use_gpu=False, no_sync=True)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm_multiprocess
    @skip_if_lt_x_gpu(2)
    @torch._inductor.config.patch(
        reorder_for_locality=False, reorder_for_peak_memory=False
    )
    def test_compile_gpu(self):
        self._test_compile(use_gpu=True, no_sync=False, checkpoint=False)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm_multiprocess
    @skip_if_lt_x_gpu(2)
    @torch._inductor.config.patch(
        reorder_for_locality=False, reorder_for_peak_memory=False
    )
    def test_compile_gpu_ac(self):
        self._test_compile(use_gpu=True, no_sync=False, checkpoint=True)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm_multiprocess
    @skip_if_lt_x_gpu(2)
    def test_compile_bf16(self):
        # Check device capability wrt bf16
        device = torch.device("cuda", self.rank % torch.cuda.device_count())
        if not sm_is_or_higher_than(device, 8, 0):
            self.skipTest("bf16 requires sm >= 8.0")

        def setup(model, compiled_replicate_model, compiled_ddp_model) -> None:
            model.register_comm_hook(None, ddp_default_hooks.bf16_compress_hook)
            compiled_m = compiled_replicate_model._orig_mod
            compiled_m.register_comm_hook(None, ddp_default_hooks.bf16_compress_hook)
            compiled_ddp_model.register_comm_hook(
                None, ddp_default_hooks.bf16_compress_hook
            )

        self._test_compile(use_gpu=True, no_sync=False, setup_func=setup)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm_multiprocess
    @skip_if_lt_x_gpu(2)
    def test_compile_fp16(self):
        def setup(model, compiled_replicate_model, compiled_ddp_model) -> None:
            model.register_comm_hook(None, ddp_default_hooks.fp16_compress_hook)
            compiled_m = compiled_replicate_model._orig_mod
            compiled_m.register_comm_hook(None, ddp_default_hooks.fp16_compress_hook)
            compiled_ddp_model.register_comm_hook(
                None, ddp_default_hooks.fp16_compress_hook
            )

        # TODO: figure out why we need to disable Inductor to avoid test errors.
        self._test_compile(
            use_gpu=True, no_sync=False, setup_func=setup, no_inductor=True
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm_multiprocess
    @skip_if_lt_x_gpu(2)
    def test_compile_backward_only(self):
        self._test_compile(use_gpu=True, no_sync=False, no_compile_forward=True)

    def _test_bucketing(self, init_process_group=True, loop=1):
        if init_process_group:
            dist.init_process_group(
                backend="gloo",
                rank=self.rank,
                world_size=self.world_size,
                store=dist.FileStore(self.file_name, self.world_size),
            )
        model = Net()
        input = torch.randn([1, DIM])
        torch._dynamo.config.optimize_ddp = "python_reducer"
        compiled_replicate_model = torch.compile(
            replicate(deepcopy(model)), fullgraph=False
        )

        def bwd(loss):
            with compiled_autograd.enable(compiler_fn()):
                loss.backward()

        for i in range(loop):
            loss = compiled_replicate_model(input).sum()
            if i != loop - 1:
                # Leave the last bwd for the run_and_get_triton_code.
                bwd(loss)

        code = run_and_get_triton_code(functools.partial(bwd, loss=loss))

        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        return code

    @torch._inductor.config.patch(
        _fuse_ddp_communication_passes=[
            "fuse_ddp_with_coalesced_op",
            "schedule_comm_wait",
        ]
    )
    # todo: This pass mucks things up since Inductor thinks its inference
    # and can apply this. Should turn off these passes in compiled autograd
    @torch._inductor.config.patch(
        reorder_for_locality=False,
        reorder_for_peak_memory=False,
        # The correctness of this test relies on the pointless permute ops
        # in the joint graph does not get eliminated..
        pattern_matcher=False,
    )
    def test_bucketing_coalesced_op(self):
        # Gradient is None
        code = self._test_bucketing()
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        fc = FileCheck()
        for i in range(3):
            fc.check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_coalesced_.default("
            )
        for i in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")

        fc.run(code)

        # Gradient is None
        code = self._test_bucketing(init_process_group=False, loop=2)
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        fc = FileCheck()
        for i in range(3):
            fc.check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_coalesced_.default("
            )
        for i in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")

        fc.run(code)

    @torch._inductor.config.patch(
        _fuse_ddp_communication_passes=[
            "fuse_ddp_with_concat_op",
            "schedule_comm_wait",
        ]
    )
    # todo: This pass mucks things up since Inductor thinks its inference
    # and can apply this. Should turn off these passes in compiled autograd
    @torch._inductor.config.patch(
        reorder_for_locality=False,
        reorder_for_peak_memory=False,
        # The correctness of this test relies on the pointless permute ops
        # in the joint graph does not get eliminated..
        pattern_matcher=False,
    )
    def test_bucketing_concat_op(self):
        # Gradient is None
        code = self._test_bucketing()
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        fc = FileCheck()
        for i in range(3):
            fc.check("aten.flatten.using_ints(").check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_.default("
            )
        for i in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")
        fc.run(code)

        # Gradient is not None
        code = self._test_bucketing(init_process_group=False, loop=2)
        self.assertEqual(counters["inductor"]["ddp_buckets"], 3)
        fc = FileCheck()
        for i in range(3):
            fc.check("aten.flatten.using_ints(").check("cpp_fused_").check(
                "torch.ops._c10d_functional.all_reduce_.default("
            )
        for i in range(3):
            fc.check("torch.ops._c10d_functional.wait_tensor.default")
        fc.run(code)


class DDP_TP_Test(InductorTestCase):
    def setUp(self):
        # Hmm, why a specific set_device call for rank 0?
        self.rank = 0
        self.world_size = 4
        torch.cuda.set_device("cuda:0")

        store = FakeStore()
        dist.init_process_group(
            backend="fake",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

    def tearDown(self):
        dist.destroy_process_group()

    @unittest.skip(
        "Temporarily disabled due to SymInt error: `unhashable type: non-nested SymInt`"
    )
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @skipIfRocm
    def test_ddp_tp(self):
        ref_model = Net()
        compiled_replicate_model = deepcopy(ref_model)
        mesh_2d = init_device_mesh(
            "cuda", (2, self.world_size // 2), mesh_dim_names=("dp", "tp")
        )
        tp_mesh = mesh_2d["tp"]
        dp_mesh = mesh_2d["dp"]
        parallelize_plan = {
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),
            "fc3": ColwiseParallel(),
            "fc4": RowwiseParallel(),
        }
        ref_model = parallelize_module(ref_model, tp_mesh, parallelize_plan)
        ref_model = replicate(ref_model, device_mesh=dp_mesh)
        compiled_replicate_model = parallelize_module(
            compiled_replicate_model, tp_mesh, parallelize_plan
        )
        compiled_replicate_model = replicate(
            compiled_replicate_model, device_mesh=dp_mesh
        )
        compiled_replicate_model = torch.compile(compiled_replicate_model)
        data = torch.randn([1, DIM])
        with compiled_autograd.enable(compiler_fn()):
            loss = compiled_replicate_model(data).sum()
            # TODO: We need "pre-dispatch tracing of backward graph" to make this work:
            # https://github.com/pytorch/pytorch/issues/127797#issuecomment-2291695474
            with self.assertRaisesRegex(
                AssertionError,
                "Expected ProxyTensor, got <class 'torch.distributed._tensor.api.DTensor'>",
            ):
                loss.backward()

        # ref_loss = ref_model(data).sum()
        # ref_loss.backward()
        # for p1, p2 in zip(
        #     ref_model.parameters(), compiled_replicate_model.parameters()
        # ):
        #     self.assertEqual(p1.grad, p2.grad)


if __name__ == "__main__":
    run_tests()
