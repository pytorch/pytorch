# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Minimal standalone demo of InvocationContext + add_summary with torch.compile.
"""

import threading
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.utils._pytree as pytree


# --- MetricValue base + WeightedScalar ---


class MetricValue:
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        pytree.register_pytree_node(
            cls,
            lambda x: x.__flatten__(),
            cls.__unflatten__,
        )

    def value(self):
        raise NotImplementedError

    def merge_(self, other):
        raise NotImplementedError

    def validate_placements(self, name):
        pass

    def __flatten__(self):
        raise NotImplementedError

    @classmethod
    def __unflatten__(cls, leaves, ctx):
        raise NotImplementedError


class WeightedScalar(MetricValue):
    """Stores (sum, weight); value() returns sum/weight (the mean)."""

    def __init__(self, *, mean=None, sum=None, weight=1.0):
        if mean is not None and sum is not None:
            raise ValueError("Specify mean or sum, not both")
        w = weight if isinstance(weight, torch.Tensor) else torch.tensor(float(weight))
        if mean is not None:
            s = mean if isinstance(mean, torch.Tensor) else torch.tensor(float(mean))
            self._sum = s * w
        else:
            self._sum = sum if isinstance(sum, torch.Tensor) else torch.tensor(float(sum))
        self._weight = w

    def value(self):
        return self._sum / self._weight

    def merge_(self, other):
        self._sum = self._sum + other._sum
        self._weight = self._weight + other._weight
        return self

    def __flatten__(self):
        return [self._sum, self._weight], None

    @classmethod
    def __unflatten__(cls, leaves, ctx):
        obj = cls.__new__(cls)
        obj._sum, obj._weight = leaves
        return obj

    def __repr__(self):
        return f"WeightedScalar(value={self.value().item():.4f}, weight={self._weight.item():.1f})"


# --- InvocationContext ---


@dataclass
class _ContextStack(threading.local):
    stack: list = field(default_factory=list)


_global_context_stack = _ContextStack()


class InvocationContext:
    def __init__(self):
        self._summaries: dict[str, MetricValue] = {}
        self._step_state: dict[Any, Any] = {}

    def __enter__(self):
        _global_context_stack.stack.append(self)
        return self

    def __exit__(self, *args):
        _global_context_stack.stack.pop()

    def add_summary(self, name: str, value: MetricValue):
        if not isinstance(value, MetricValue):
            raise TypeError(f"Value must be a MetricValue, got {type(value)}")
        value.validate_placements(name)
        with torch.no_grad():
            if name in self._summaries:
                self._summaries[name].merge_(value)
            else:
                self._summaries[name] = value

    def summaries(self):
        return self._summaries.copy()

    def __flatten__(self):
        summary_leaves, summary_spec = pytree.tree_flatten(self._summaries)
        state_leaves, state_spec = pytree.tree_flatten(self._step_state)
        return (*summary_leaves, *state_leaves), (summary_spec, state_spec)

    @classmethod
    def __unflatten__(cls, leaves, ctx):
        obj = cls()
        summary_spec, state_spec = ctx
        n = summary_spec.num_leaves
        obj._summaries = pytree.tree_unflatten(leaves[:n], summary_spec)
        obj._step_state = pytree.tree_unflatten(leaves[n:], state_spec)
        return obj


pytree.register_pytree_node(
    InvocationContext,
    lambda x: x.__flatten__(),
    InvocationContext.__unflatten__,
)


def current_context() -> InvocationContext | None:
    if not _global_context_stack.stack:
        return None
    return _global_context_stack.stack[-1]


# --- Demo ---


class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 1)

    def forward(self, x):
        out = self.linear(x)
        loss = out.sum()
        ctx = current_context()
        if ctx is not None:
            ctx.add_summary(
                "loss", WeightedScalar(mean=loss.detach(), weight=x.shape[0])
            )
            ctx.add_summary(
                "output_norm", WeightedScalar(mean=out.detach().norm(), weight=1.0)
            )
        return loss


if __name__ == "__main__":
    # 1. Eager mode
    print("=== Eager mode ===")
    model = MyModel()
    context = InvocationContext()
    with context:
        x = torch.randn(8, 4)
        loss = model(x)
        loss.backward()
    for name, metric in context.summaries().items():
        print(f"  {name}: {metric}")

    # 2. torch.compile mode
    print("\n=== torch.compile mode ===")
    model2 = MyModel()
    compiled_model = torch.compile(model2, backend="aot_eager")
    context2 = InvocationContext()
    with context2:
        x = torch.randn(8, 4)
        loss = compiled_model(x)
        loss.backward()
    for name, metric in context2.summaries().items():
        print(f"  {name}: {metric}")

    # 3. Merge across micro-batches
    print("\n=== Merge across micro-batches ===")
    model3 = MyModel()
    context3 = InvocationContext()
    with context3:
        for i in range(3):
            x = torch.randn(4, 4)
            loss = model3(x)
            loss.backward()
    for name, metric in context3.summaries().items():
        print(f"  {name}: {metric}")

    # 4. Pytree roundtrip (what torch.compile uses internally)
    print("\n=== Pytree roundtrip ===")
    context4 = InvocationContext()
    context4.add_summary("test", WeightedScalar(mean=torch.tensor(3.14), weight=1.0))
    leaves, spec = pytree.tree_flatten(context4)
    print(f"  Flattened into {len(leaves)} leaves: {leaves}")
    reconstructed = pytree.tree_unflatten(leaves, spec)
    print(f"  Reconstructed summaries: {reconstructed.summaries()}")
