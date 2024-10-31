# Owner(s): ["oncall: distributed"]

import contextlib
import unittest
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    CheckpointImpl,
    CheckpointWrapper,
    offload_wrapper,
    OffloadWrapper,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.checkpoint import checkpoint


_SAVED_PREFIX = "_saved_"
GRAD_FN_NEXT_FUNCTIONS = "next_functions"


class CheckpointWrapperTest(TestCase):
    def test_load_activation_checkpointed_module(self):
        lin = nn.Linear(10, 10, bias=False)
        lin = checkpoint_wrapper(
            lin,
            checkpoint_fn=checkpoint,
            # checkpoint kwargs
            use_reentrant=True,
            preserve_rng_state=False,
        )
        state_dict = deepcopy(lin.state_dict())
        # Load into non-checkpoint wrapped linear module
        lin_new = nn.Linear(10, 10, bias=False)
        lin_new.load_state_dict(state_dict)
        for p1, p2 in zip(lin.parameters(), lin_new.parameters()):
            self.assertEqual(p1, p2)
            self.assertTrue(torch.allclose(p1, p2))

        # Load non-checkpoint wrapped module into checkpoint wrapped one
        # Make params different
        for p in lin_new.parameters():
            with torch.no_grad():
                p.add_(0.5)

        state_dict = deepcopy(lin_new.state_dict())
        # Verify checkpoint wrapped linear can load unwrapped linear
        lin.load_state_dict(state_dict)
        for p1, p2 in zip(lin.parameters(), lin_new.parameters()):
            self.assertEqual(p1, p2)

    def test_checkpoint_wrapper_kwarg_support(self):
        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(10, 10)

            def forward(self, a, b, c=None, d=None, **kwargs):
                return (self.lin(a), self.lin(b), self.lin(c), self.lin(d))

        for wrapper in [
            partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT),
            partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            offload_wrapper,
        ]:
            with self.subTest(wrapper=wrapper):
                model = wrapper(MyModel())
                if wrapper == offload_wrapper:
                    self.assertTrue(isinstance(model, OffloadWrapper))
                else:
                    self.assertTrue(isinstance(model, CheckpointWrapper))
                # Verify kwargs can be passed in
                inp = torch.ones(4, 10, requires_grad=True)
                out = model(inp, inp, c=inp, d=inp, e=inp, f=inp)
                self.assertTrue(isinstance(out, tuple))
                self.assertEqual(4, len(out))
                # Without kwargs should have equivalent gradient requirements.
                out_no_kwarg = model(inp, inp, inp, inp)
                for t1, t2 in zip(out_no_kwarg, out):
                    self.assertEqual(t1, t2)
                    self.assertEqual(t1.requires_grad, t2.requires_grad)

        # Test model that enforces kwarg inputs
        class ModelEnforceKwarg(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(10, 10)

            def forward(self, *, a=None, b=None):
                return (self.lin(a), self.lin(b))

        model = checkpoint_wrapper(
            ModelEnforceKwarg(), checkpoint_impl=CheckpointImpl.REENTRANT
        )

        inp = torch.ones(4, 10, requires_grad=True)
        out = model(a=inp, b=inp)
        self.assertEqual(2, len(out))

    def test_checkpoint_wrapper_args_kwargs(self):
        """
        Tests that checkpoint_wrapper can pass down args / kwargs to configure
        torch.utils.checkpoint.
        """

        count = 0

        @contextlib.contextmanager
        def ctx_manager():
            nonlocal count
            count += 1
            yield

        def get_ctx_mgrs():
            return (ctx_manager(), ctx_manager())

        # kwargs test
        torch_utils_checkpoint = torch.utils.checkpoint.checkpoint
        m = checkpoint_wrapper(
            torch.nn.Linear(1, 1),
            checkpoint_fn=torch_utils_checkpoint,
            use_reentrant=False,
            context_fn=get_ctx_mgrs,
        )
        m(torch.randn(2, 1)).sum().backward()
        self.assertEqual(2, count)

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_checkpoint_wrapper_parity(self):
        """
        Tests that using checkpoint_wrapper or the functional
        torch.utils.checkpoint (with the same reentrant config)
        results in the same maximum memory usage, i.e. they are
        equivalent memory usage wise.
        """

        class Model(nn.Module):
            def __init__(
                self,
                n: int,
                use_cp: bool,
                use_wrapper: bool = False,
                use_reentrant: bool = True,
            ):
                super().__init__()
                self.layers = nn.ModuleList()
                self.n = n
                self.use_cp = use_cp
                self.use_wrapper = use_wrapper
                self.use_reentrant = use_reentrant
                wrp = partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.REENTRANT
                    if use_reentrant
                    else CheckpointImpl.NO_REENTRANT,
                )
                for i in range(self.n):
                    l = nn.Sequential(
                        nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 256)
                    )
                    use_checkpoint_wrapper = self.use_wrapper
                    if use_checkpoint_wrapper:
                        l = wrp(l)
                    self.layers.append(l)

            def forward(self, x):
                for i in range(self.n):
                    if self.use_wrapper or not self.use_cp:
                        x = self.layers[i](x)
                    else:
                        x = checkpoint(
                            self.layers[i], x, use_reentrant=self.use_reentrant
                        )
                return x

        def test(use_checkpointing, use_wrapper, use_reentrant):
            a = Model(
                8,
                use_checkpointing,
                use_wrapper=use_wrapper,
                use_reentrant=use_reentrant,
            ).cuda()
            x = torch.randn(10000, 256, requires_grad=True).cuda()
            torch.cuda.reset_peak_memory_stats()
            loss = a(x).sum()
            loss.backward()
            return torch.cuda.max_memory_allocated()

        functional_no_reentrant = test(
            use_checkpointing=True, use_wrapper=False, use_reentrant=False
        )
        wrapper_no_reentrant = test(
            use_checkpointing=False, use_wrapper=True, use_reentrant=False
        )
        self.assertEqual(functional_no_reentrant, wrapper_no_reentrant)

        functional_reentrant = test(
            use_checkpointing=True, use_wrapper=False, use_reentrant=True
        )
        wrapper_reentrant = test(
            use_checkpointing=False, use_wrapper=True, use_reentrant=True
        )
        self.assertEqual(functional_reentrant, wrapper_reentrant)

    def test_forward_missing_attributes(self):
        lin = nn.Linear(1, 1)
        m = nn.Sequential(lin, lin)
        wrapped = CheckpointWrapper(m)
        # Test indexing is forwarded
        self.assertEqual(wrapped[0], lin)
        # Test missing attributes are forwarded.
        m._foo = "bar"
        self.assertEqual(wrapped._foo, "bar")

    def test_apply_activation_checkpointing(self):
        """
        Ensures that `apply_activation_checkpointing` can be used
        to swap modules for their checkpoint-wrapped counterparts given
        a model.
        """

        class LinearWithBatchNorm(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lin = nn.Linear(10, 10)
                self.bn = nn.BatchNorm1d(10)
                self.nested_linear = nn.Sequential(nn.Linear(10, 10))

            def forward(self, x):
                return self.bn(self.nested_linear(self.lin(x)))

        class MyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.seq = nn.Sequential(
                    LinearWithBatchNorm(), LinearWithBatchNorm(), LinearWithBatchNorm()
                )

            def forward(self, x):
                return self.seq(x)

        def check_fn(l):
            return isinstance(l, nn.Linear)

        n_linear = None

        for i, wrapper in enumerate(
            [
                partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT),
                partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
                ),
                offload_wrapper,
            ]
        ):
            model = MyModel()
            if n_linear is None:
                n_linear = sum(
                    1 if isinstance(x, nn.Linear) else 0 for x in model.modules()
                )

            with self.subTest(wrapper=wrapper):
                if i != 0:
                    apply_activation_checkpointing(
                        model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn
                    )
                else:
                    apply_activation_checkpointing(
                        model,
                        checkpoint_wrapper_fn=wrapper,
                        auto_wrap_policy=ModuleWrapPolicy({nn.Linear}),
                    )
                n_linear_wrapped = sum(
                    1 if isinstance(x, nn.Linear) else 0 for x in model.modules()
                )
                n_checkpointed = sum(
                    1 if isinstance(x, (CheckpointWrapper, OffloadWrapper)) else 0
                    for x in model.modules()
                )
                self.assertEqual(n_checkpointed, n_linear_wrapped)
                self.assertEqual(n_linear, n_linear_wrapped)
                for j in range(3):
                    self.assertTrue(
                        isinstance(
                            model.seq[j].lin, (CheckpointWrapper, OffloadWrapper)
                        )
                    )
                    self.assertTrue(
                        isinstance(
                            model.seq[j].nested_linear[0],
                            (CheckpointWrapper, OffloadWrapper),
                        )
                    )

                inp = torch.randn(4, 10, requires_grad=True)
                for i in range(6):
                    # Kwarg input
                    loss = model(x=inp).sum()
                    self.assertTrue(loss.requires_grad)
                    loss.backward()
                    # ensure checkpointed part of model has gradients
                    for j in range(3):
                        weight_lin = model.seq[j].lin._checkpoint_wrapped_module.weight
                        bias_lin = model.seq[j].lin._checkpoint_wrapped_module.bias
                        weight_nested_lin = (
                            model.seq[j]
                            .nested_linear[0]
                            ._checkpoint_wrapped_module.weight
                        )
                        bias_nested_lin = (
                            model.seq[j]
                            .nested_linear[0]
                            ._checkpoint_wrapped_module.bias
                        )
                        for param in [
                            weight_lin,
                            bias_lin,
                            weight_nested_lin,
                            bias_nested_lin,
                        ]:
                            self.assertTrue(param.requires_grad)
                            self.assertFalse(param.grad is None)

    def test_fqn(self):
        lin = nn.Linear(10, 10, bias=False)
        lin = checkpoint_wrapper(lin)
        state_dict = lin.state_dict()
        for fqn, _ in lin.named_parameters():
            self.assertTrue(fqn in state_dict, msg=f"{fqn} not in state_dict.")

    @unittest.skipIf(not torch.cuda.is_available(), "Test requires CUDA")
    def test_checkpoint_wrapper_cpu_offload(self):
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10),
        ).cuda()

        # Patch saved_tensor_hooks to make the unpack keep the tensor on CPU for
        # testing, otherwise the tensor access during the DFS will cause orig
        # unpack to run, transferring the tensor back to GPU.
        def patched_init(saved_tensor_hook_obj, pack_hook, _):
            saved_tensor_hook_obj.pack_hook = pack_hook

            def testing_cpu_offload_unpack_hook(packed):
                _, tensor = packed
                return tensor

            saved_tensor_hook_obj.unpack_hook = testing_cpu_offload_unpack_hook

        orig_init = torch.autograd.graph.saved_tensors_hooks.__init__
        torch.autograd.graph.saved_tensors_hooks.__init__ = patched_init

        model = offload_wrapper(model)

        inp = torch.randn(3, 10, device="cuda")
        loss = model(inp).sum()

        # All autograd saved tensors should be offloaded to CPU.
        offload_verified = False

        def dfs(grad_fn):
            for e in dir(grad_fn):
                if not e.startswith(_SAVED_PREFIX):
                    continue

                saved = getattr(grad_fn, e)
                if isinstance(saved, torch.Tensor):
                    self.assertEqual(torch.device("cpu"), saved.device)
                    nonlocal offload_verified
                    offload_verified = True

            if hasattr(grad_fn, GRAD_FN_NEXT_FUNCTIONS):
                for next_grad_fn, _ in grad_fn.next_functions:
                    dfs(next_grad_fn)

        dfs(loss.grad_fn)

        self.assertTrue(offload_verified)

        torch.autograd.graph.saved_tensors_hooks.__init__ = orig_init


if __name__ == "__main__":
    run_tests()
