# Owner(s): ["oncall: distributed"]

from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing_wrapper,
    CheckpointWrapper,
    CheckpointImpl
)

from torch.utils.checkpoint import checkpoint

from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
)

import unittest

class CheckpointWrapperTest(TestCase):
    def setUp(self):
        super().setUp()

    def test_load_activation_checkpointed_module(self):
        lin = nn.Linear(10, 10, bias=False)
        lin = checkpoint_wrapper(lin)
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
                use_reentrant: bool = True
            ):
                super().__init__()
                self.layers = nn.ModuleList()
                self.n = n
                self.use_cp = use_cp
                self.use_wrapper = use_wrapper
                self.use_reentrant = use_reentrant
                wrp = partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.REENTRANT if use_reentrant else CheckpointImpl.NO_REENTRANT
                )
                for i in range(self.n):
                    l = nn.Sequential(nn.Linear(256, 256), nn.Linear(256, 256), nn.Linear(256, 256))
                    use_checkpoint_wrapper = self.use_wrapper
                    if use_checkpoint_wrapper:
                        l = wrp(l)
                    self.layers.append(l)

            def forward(self, x):
                for i in range(self.n):
                    if (
                        self.use_wrapper or
                        not self.use_cp
                    ):
                        x = self.layers[i](x)
                    else:
                        x = checkpoint(self.layers[i], x, use_reentrant=self.use_reentrant)
                return x

        def test(use_checkpointing, use_wrapper, use_reentrant):
            a = Model(8, use_checkpointing, use_wrapper=use_wrapper, use_reentrant=use_reentrant).cuda()
            x = torch.randn(10000, 256, requires_grad=True).cuda()
            torch.cuda.reset_peak_memory_stats()
            loss = a(x).sum()
            loss.backward()
            return torch.cuda.max_memory_allocated()

        functional_no_reentrant = test(use_checkpointing=True, use_wrapper=False, use_reentrant=False)
        wrapper_no_reentrant = test(use_checkpointing=False, use_wrapper=True, use_reentrant=False)
        self.assertEqual(functional_no_reentrant, wrapper_no_reentrant)

        functional_reentrant = test(use_checkpointing=True, use_wrapper=False, use_reentrant=True)
        wrapper_reentrant = test(use_checkpointing=False, use_wrapper=True, use_reentrant=True)
        self.assertEqual(functional_no_reentrant, wrapper_no_reentrant)

    def test_forward_missing_attributes(self):
        lin = nn.Linear(1, 1)
        m = nn.Sequential(lin, lin)
        wrapped = CheckpointWrapper(m)
        # Test indexing is forwarded
        self.assertEqual(wrapped[0], lin)
        # Test missing attributes are forwarded.
        m._foo = 'bar'
        self.assertEqual(wrapped._foo, 'bar')

    def test_apply_activation_checkpointing_wrapper(self):
        """
        Ensures that `apply_activation_checkpointing_wrapper` can be used
        to swap modules for their checkpoint-wrapped counterparts given
        a model.
        """
        class LinearWithBatchNorm(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(10, 10)
                self.bn = nn.BatchNorm1d(10)
                self.nested_linear = nn.Sequential(nn.Linear(10, 10))

            def forward(self, x):
                return self.bn(self.nested_linear(self.lin(x)))

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.seq = nn.Sequential(
                    LinearWithBatchNorm(), LinearWithBatchNorm(), LinearWithBatchNorm()
                )

            def forward(self, x):
                return self.seq(x)

        model = MyModel()
        n_linear = sum(1 if isinstance(x, nn.Linear) else 0 for x in model.modules())

        def check_fn(l):
            return isinstance(l, nn.Linear)

        apply_activation_checkpointing_wrapper(
            model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn
        )
        n_linear_wrapped = sum(1 if isinstance(x, nn.Linear) else 0 for x in model.modules())
        n_checkpointed = sum(1 if isinstance(x, CheckpointWrapper) else 0 for x in model.modules())
        self.assertEqual(n_checkpointed, n_linear_wrapped)
        self.assertEqual(n_linear, n_linear_wrapped)
        for j in range(3):
            self.assertTrue(isinstance(model.seq[j].lin, CheckpointWrapper))
            self.assertTrue(isinstance(model.seq[j].nested_linear[0], CheckpointWrapper))

        inp = torch.randn(4, 10, requires_grad=True)
        for i in range(6):
            loss = model(inp).sum()
            self.assertTrue(loss.requires_grad)
            loss.backward()
            # ensure checkpointed part of model has gradients
            for j in range(3):
                weight_lin = model.seq[j].lin._checkpoint_wrapped_module.weight
                bias_lin = model.seq[j].lin._checkpoint_wrapped_module.bias
                weight_nested_lin = model.seq[j].nested_linear[0]._checkpoint_wrapped_module.weight
                bias_nested_lin = model.seq[j].nested_linear[0]._checkpoint_wrapped_module.bias
                for param in [weight_lin, bias_lin, weight_nested_lin, bias_nested_lin]:
                    self.assertTrue(param.requires_grad)
                    self.assertFalse(param.grad is None)


if __name__ == "__main__":
    run_tests()
