# Owner(s): ["oncall: distributed"]

import contextlib
import os
import unittest
from copy import deepcopy
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch import _inductor as inductor, nn
from torch._dynamo import compiled_autograd
from torch.distributed._composable.replicate import replicate
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks as ddp_default_hooks,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
    skip_if_rocm,
)
from torch.testing._internal.common_utils import run_tests
from torch.utils._triton import has_triton
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


def compiler_fn(no_inductor):
    def _compiler_fn(gm):
        def inner_compiler(gm_, example_inputs_):
            if no_inductor:
                return gm_
            else:
                return inductor.compile(gm_, example_inputs_)

        gm = torch.compile(gm, fullgraph=True, backend=inner_compiler)
        return gm

    return _compiler_fn


class ReplicateTest(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 2

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
        model = Net().to(device)
        input = torch.randn([1, DIM], device=device)

        compiled_model = torch.compile(replicate(deepcopy(model)), fullgraph=True)
        compiled_optim = torch.optim.Adam(compiled_model.parameters())
        model = replicate(model)
        optim = torch.optim.Adam(model.parameters())

        if setup_func:
            setup_func(model, compiled_model)

        # Run multiple iterations so that we could test no_sync
        for i in range(2):
            # Setting a different random seed so that if the allreduces are not
            # executed correctly, the gradients won't be correct compared to the
            # eager DDP.
            torch.manual_seed(123 + self.rank + i)
            input = torch.randn([1, DIM], device=device)

            if no_sync and i % 2 == 0:
                context = replicate.state(model)._ddp.no_sync()
            else:
                context = contextlib.nullcontext()
            with context:
                loss = model(input).sum()
                loss.backward()

            compiled_m = getattr(compiled_model, "_orig_mod", compiled_model)
            if no_sync and i % 2 == 0:
                context = replicate.state(compiled_m)._ddp.no_sync()
            else:
                context = contextlib.nullcontext()
            with context:
                with compiled_autograd.enable(compiler_fn(no_inductor)):
                    compiled_loss = compiled_model(input).sum()
                    compiled_loss.backward()

            if not no_sync or i % 2 == 1:
                for p1, p2 in zip(model.parameters(), compiled_model.parameters()):
                    self.assertEqual(p1.grad, p2.grad)
                compiled_optim.step()
                # Right now we have to use `set_to_none=False`, otherwise
                # the backward will be recompiled every iteration.
                # With `set_to_none=False`, it will only be recompiled once.
                # https://github.com/pytorch/pytorch/issues/118435
                compiled_optim.zero_grad(set_to_none=False)
                optim.step()
                optim.zero_grad()

        self.assertEqual(tuple(model.parameters()), tuple(compiled_model.parameters()))

    def test_compile_cpu(self):
        self._test_compile(use_gpu=False, no_sync=False)

    def test_compile_cpu_no_sync(self):
        self._test_compile(use_gpu=False, no_sync=True)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm
    @skip_if_lt_x_gpu(2)
    def test_compile_gpu(self):
        self._test_compile(use_gpu=True, no_sync=False)

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm
    @skip_if_lt_x_gpu(2)
    def test_compile_bf16(self):
        def setup(model, compiled_model) -> None:
            replicate.state(model)._ddp.register_comm_hook(
                None, ddp_default_hooks.bf16_compress_hook
            )
            compiled_m = compiled_model._orig_mod
            replicate.state(compiled_m)._ddp.register_comm_hook(
                None, ddp_default_hooks.bf16_compress_hook
            )

        self._test_compile(
            use_gpu=True, no_sync=False, setup_func=setup, no_inductor=True
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm
    @skip_if_lt_x_gpu(2)
    def test_compile_fp16(self):
        def setup(model, compiled_model) -> None:
            replicate.state(model)._ddp.register_comm_hook(
                None, ddp_default_hooks.fp16_compress_hook
            )
            compiled_m = compiled_model._orig_mod
            replicate.state(compiled_m)._ddp.register_comm_hook(
                None, ddp_default_hooks.fp16_compress_hook
            )

        # TODO: figure out why we need to disable Inductor to avoid test errors.
        self._test_compile(
            use_gpu=True, no_sync=False, setup_func=setup, no_inductor=True
        )

    @unittest.skipIf(not has_triton(), "Inductor+gpu needs triton and recent GPU arch")
    @skip_if_rocm
    @skip_if_lt_x_gpu(2)
    def test_compile_backward_only(self):
        self._test_compile(use_gpu=True, no_sync=False, no_compile_forward=True)


if __name__ == "__main__":
    run_tests()
