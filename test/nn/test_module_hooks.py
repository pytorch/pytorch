# Owner(s): ["module: nn"]
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    skipIfTorchDynamo,
)

import torch
import torch.nn as nn

from functools import partial
from typing import List, Tuple


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq2(self.seq1(x))


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = Net()
        self.net2 = Net()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net2(self.net1(x))


def forward_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    inp: Tuple[torch.Tensor],
    out: torch.Tensor,
) -> None:
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(inp), 1)


def forward_pre_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    inp: Tuple[torch.Tensor],
) -> None:
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(inp), 1)


def full_backward_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    grad_input: Tuple[torch.Tensor],
    grad_output: Tuple[torch.Tensor],
) -> None:
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(grad_input), 1)
    self.assertEqual(len(grad_output), 1)


def full_backward_pre_hook(
    self: TestCase,
    fired_hooks: List[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    grad_input: Tuple[torch.Tensor],
) -> None:
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(grad_input), 1)


class TestModuleHooks(TestCase):

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_forward_hooks(self):
        fired_hooks: List[int] = []
        model = ToyModel()
        x = torch.randn(10, 10)
        hook = partial(forward_hook, self, fired_hooks, model.net1.seq2)
        model.net1.seq2.register_forward_hook(partial(hook, 0))
        model.net1.seq2.register_forward_hook(partial(hook, 1), prepend=True)
        model.net1.seq2.register_forward_hook(partial(hook, 2))
        model.net1.seq2.register_forward_hook(partial(hook, 3))
        model.net1.seq2.register_forward_hook(partial(hook, 4), prepend=True)
        expected = [4, 1, 0, 2, 3]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, expected)
        out.sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x).sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_forward_pre_hooks(self):
        fired_hooks: List[int] = []
        model = ToyModel()
        x = torch.randn(10, 10)
        hook = partial(forward_pre_hook, self, fired_hooks, model.net2.seq1)
        model.net2.seq1.register_forward_pre_hook(partial(hook, 0), prepend=True)
        model.net2.seq1.register_forward_pre_hook(partial(hook, 1))
        model.net2.seq1.register_forward_pre_hook(partial(hook, 2))
        model.net2.seq1.register_forward_pre_hook(partial(hook, 3))
        model.net2.seq1.register_forward_pre_hook(partial(hook, 4), prepend=True)
        expected = [4, 0, 1, 2, 3]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, expected)
        out.sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x).sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_full_backward_hooks(self):
        fired_hooks: List[int] = []
        model = ToyModel()
        x = torch.randn(10, 10)
        hook = partial(full_backward_hook, self, fired_hooks, model.net1)
        model.net1.register_full_backward_hook(partial(hook, 0))
        model.net1.register_full_backward_hook(partial(hook, 1))
        model.net1.register_full_backward_hook(partial(hook, 2))
        model.net1.register_full_backward_hook(partial(hook, 3), prepend=True)
        model.net1.register_full_backward_hook(partial(hook, 4), prepend=True)
        expected = [4, 3, 0, 1, 2]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, [])
        out.sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x).sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_full_backward_pre_hooks(self):
        fired_hooks: List[int] = []
        model = ToyModel()
        x = torch.randn(10, 10)
        hook = partial(full_backward_pre_hook, self, fired_hooks, model.net1)
        model.net1.register_full_backward_pre_hook(partial(hook, 0), prepend=True)
        model.net1.register_full_backward_pre_hook(partial(hook, 1), prepend=True)
        model.net1.register_full_backward_pre_hook(partial(hook, 2))
        model.net1.register_full_backward_pre_hook(partial(hook, 3))
        model.net1.register_full_backward_pre_hook(partial(hook, 4))
        expected = [1, 0, 2, 3, 4]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, [])
        out.sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x).sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

    @skipIfTorchDynamo("Dynamo does not yet capture hooks")
    def test_mixed_hooks(self):
        fired_hooks: List[int] = []
        model = ToyModel()
        x = torch.randn(10, 10)
        model.register_forward_pre_hook(partial(forward_pre_hook, self, fired_hooks, model, 0))
        model.register_forward_hook(partial(forward_hook, self, fired_hooks, model, 1))
        model.register_full_backward_pre_hook(partial(full_backward_pre_hook, self, fired_hooks, model, 2))
        model.register_full_backward_hook(partial(full_backward_hook, self, fired_hooks, model, 3))

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, [0, 1])
        out.sum().backward()
        self.assertEqual(fired_hooks, [0, 1, 2, 3])
        model(x).sum().backward()
        self.assertEqual(fired_hooks, [0, 1, 2, 3, 0, 1, 2, 3])


if __name__ == "__main__":
    run_tests()
