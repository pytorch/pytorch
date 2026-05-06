# Owner(s): ["module: autograd"]
"""Tests for autograd Node lifecycle management.

Verifies that autograd nodes, their Python wrappers, and associated objects
are properly freed — both by reference counting and by the GC cycle collector.

The test matrix covers:
  - Node type: C++ (MulBackward0 etc.) vs Python (custom Function)
  - Hook type: none, tensor hook, grad_fn hook, post_accumulate_grad hook
  - Hook captures grad_fn: creates a cycle requiring GC
  - Backward state: none, backward(), backward(retain_graph=True)
  - Shared grad_fn: one grad_fn with multiple output tensors
"""

import gc
import weakref
from enum import auto, Enum

import torch
from torch.autograd import Function
from torch.testing._internal.common_utils import run_tests, TestCase


class NodeType(Enum):
    CPP = auto()
    PYTHON = auto()


class HookType(Enum):
    NONE = auto()
    TENSOR_HOOK = auto()
    TENSOR_HOOK_CAPTURES_GRAD_FN = auto()
    GRAD_FN_HOOK = auto()
    GRAD_FN_HOOK_CAPTURES_GRAD_FN = auto()
    POST_ACCUMULATE_GRAD = auto()


class BackwardState(Enum):
    NONE = auto()
    BACKWARD = auto()
    RETAIN_GRAPH = auto()


class PythonId(Function):
    """Identity custom function."""

    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad


class PythonMul(Function):
    """Custom function that saves a tensor."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * 2

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * 2


class PythonExp(Function):
    """Custom function that saves its output."""

    @staticmethod
    def forward(ctx, x):
        y = x.exp()
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad):
        (y,) = ctx.saved_tensors
        return grad * y


class PythonSplit(Function):
    """Custom function with two outputs."""

    @staticmethod
    def forward(ctx, x):
        return x[:2].clone(), x[2:].clone()

    @staticmethod
    def backward(ctx, grad0, grad1):
        return torch.cat([grad0, grad1])


def make_output(node_type, a):
    """Create an output tensor with the given node type."""
    if node_type == NodeType.CPP:
        return a * 2
    else:
        return PythonMul.apply(a)


class Marker:
    """Weak-ref-able sentinel to track object lifetime."""


class TestNodeLifecycle(TestCase):
    """Single output, various node types / hooks / backward states."""

    def _run(self, node_type, hook_type, backward_state):
        a = torch.randn(4, requires_grad=True)
        b = make_output(node_type, a)
        marker = Marker()
        marker_ref = weakref.ref(marker)

        grad_fn = b.grad_fn

        if hook_type == HookType.TENSOR_HOOK:

            def hook(grad):
                marker

            b.register_hook(hook)
        elif hook_type == HookType.TENSOR_HOOK_CAPTURES_GRAD_FN:

            def hook(grad):
                grad_fn
                marker

            b.register_hook(hook)
        elif hook_type == HookType.GRAD_FN_HOOK:

            def hook(*args):
                marker

            b.grad_fn.register_hook(hook)
        elif hook_type == HookType.GRAD_FN_HOOK_CAPTURES_GRAD_FN:

            def hook(*args):
                grad_fn
                marker

            b.grad_fn.register_hook(hook)

        if backward_state == BackwardState.BACKWARD:
            b.sum().backward()
        elif backward_state == BackwardState.RETAIN_GRAPH:
            b.sum().backward(retain_graph=True)

        return a, b, marker, marker_ref, grad_fn

    def _check_freed_by_refcount(self, node_type, hook_type, backward_state):
        a, b, marker, marker_ref, grad_fn = self._run(
            node_type, hook_type, backward_state
        )
        del a, b, marker, grad_fn
        self.assertIsNone(
            marker_ref(),
            f"not freed by refcount: {node_type} {hook_type} {backward_state}",
        )

    def _check_freed_by_gc(self, node_type, hook_type, backward_state):
        a, b, marker, marker_ref, grad_fn = self._run(
            node_type, hook_type, backward_state
        )
        del a, b, marker, grad_fn
        gc.collect()
        self.assertIsNone(
            marker_ref(), f"not freed by GC: {node_type} {hook_type} {backward_state}"
        )

    # === No hooks ===

    def test_cpp_no_hook_no_backward_refcount(self):
        self._check_freed_by_refcount(NodeType.CPP, HookType.NONE, BackwardState.NONE)

    def test_cpp_no_hook_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.NONE, BackwardState.BACKWARD
        )

    def test_cpp_no_hook_retain_refcount(self):
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.NONE, BackwardState.RETAIN_GRAPH
        )

    def test_python_no_hook_no_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.NONE, BackwardState.NONE
        )

    def test_python_no_hook_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.NONE, BackwardState.BACKWARD
        )

    def test_python_no_hook_retain_refcount(self):
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.NONE, BackwardState.RETAIN_GRAPH
        )

    # === Tensor hooks (no cycle) ===

    def test_cpp_tensor_hook_no_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.TENSOR_HOOK, BackwardState.NONE
        )

    def test_cpp_tensor_hook_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.TENSOR_HOOK, BackwardState.BACKWARD
        )

    def test_python_tensor_hook_no_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.TENSOR_HOOK, BackwardState.NONE
        )

    def test_python_tensor_hook_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.TENSOR_HOOK, BackwardState.BACKWARD
        )

    # === Tensor hooks capturing grad_fn (cycle, needs GC) ===

    def test_cpp_tensor_hook_captures_no_backward_gc(self):
        self._check_freed_by_gc(
            NodeType.CPP, HookType.TENSOR_HOOK_CAPTURES_GRAD_FN, BackwardState.NONE
        )

    def test_cpp_tensor_hook_captures_backward_gc(self):
        self._check_freed_by_gc(
            NodeType.CPP, HookType.TENSOR_HOOK_CAPTURES_GRAD_FN, BackwardState.BACKWARD
        )

    def test_python_tensor_hook_captures_no_backward_gc(self):
        self._check_freed_by_gc(
            NodeType.PYTHON, HookType.TENSOR_HOOK_CAPTURES_GRAD_FN, BackwardState.NONE
        )

    def test_python_tensor_hook_captures_backward_gc(self):
        self._check_freed_by_gc(
            NodeType.PYTHON,
            HookType.TENSOR_HOOK_CAPTURES_GRAD_FN,
            BackwardState.BACKWARD,
        )

    # === grad_fn hooks (no cycle) ===

    def test_cpp_grad_fn_hook_no_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.GRAD_FN_HOOK, BackwardState.NONE
        )

    def test_cpp_grad_fn_hook_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.CPP, HookType.GRAD_FN_HOOK, BackwardState.BACKWARD
        )

    def test_python_grad_fn_hook_no_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.GRAD_FN_HOOK, BackwardState.NONE
        )

    def test_python_grad_fn_hook_backward_refcount(self):
        self._check_freed_by_refcount(
            NodeType.PYTHON, HookType.GRAD_FN_HOOK, BackwardState.BACKWARD
        )

    # === grad_fn hooks capturing grad_fn (cycle, needs GC) ===

    def test_cpp_grad_fn_hook_captures_no_backward_gc(self):
        self._check_freed_by_gc(
            NodeType.CPP, HookType.GRAD_FN_HOOK_CAPTURES_GRAD_FN, BackwardState.NONE
        )

    def test_cpp_grad_fn_hook_captures_backward_gc(self):
        self._check_freed_by_gc(
            NodeType.CPP, HookType.GRAD_FN_HOOK_CAPTURES_GRAD_FN, BackwardState.BACKWARD
        )

    def test_python_grad_fn_hook_captures_no_backward_gc(self):
        self._check_freed_by_gc(
            NodeType.PYTHON, HookType.GRAD_FN_HOOK_CAPTURES_GRAD_FN, BackwardState.NONE
        )

    def test_python_grad_fn_hook_captures_backward_gc(self):
        self._check_freed_by_gc(
            NodeType.PYTHON,
            HookType.GRAD_FN_HOOK_CAPTURES_GRAD_FN,
            BackwardState.BACKWARD,
        )


class TestLeafNodeLifecycle(TestCase):
    """Leaf-specific tests (AccumulateGrad)."""

    def test_accumulate_grad_freed_refcount(self):
        """AccumulateGrad releases variable when graph is freed."""
        a = torch.randn(2, requires_grad=True)
        b = a * 2
        self.assertEqual(a._use_count(), 2)
        del b
        self.assertEqual(a._use_count(), 1)

    def test_post_accumulate_grad_hook_freed(self):
        """Post-accumulate-grad hook freed when leaf is freed."""
        marker = Marker()
        ref = weakref.ref(marker)
        a = torch.randn(2, requires_grad=True)

        def hook(t):
            marker  # noqa: F821

        a.register_post_accumulate_grad_hook(hook)
        b = a * 2
        b.sum().backward()
        del a, b, marker, hook
        gc.collect()
        self.assertIsNone(ref())

    def test_accumulate_grad_freed_after_backward(self):
        """AccumulateGrad releases variable after backward."""
        a = torch.randn(2, requires_grad=True)
        b = a * 2
        self.assertEqual(a._use_count(), 2)
        b.sum().backward()
        del b
        self.assertEqual(a._use_count(), 1)

    def test_accumulate_grad_freed_retain_graph(self):
        """AccumulateGrad keeps variable during retain_graph, frees after."""
        a = torch.randn(2, requires_grad=True)
        b = (a * 2).sum()
        b.backward(retain_graph=True)
        self.assertEqual(a._use_count(), 2)
        del b
        self.assertEqual(a._use_count(), 1)


class TestSharedGradFn(TestCase):
    """Tests for grad_fn shared by multiple output tensors."""

    def test_cpp_shared_grad_fn_one_deleted(self):
        """Shared grad_fn stays alive when one output is deleted."""
        a = torch.randn(4, requires_grad=True)
        b0, b1 = (a * 2).split(2)
        self.assertEqual(a._use_count(), 2)
        del b0
        self.assertEqual(a._use_count(), 2)
        del b1
        self.assertEqual(a._use_count(), 1)

    def test_python_shared_grad_fn_one_deleted(self):
        """Python shared grad_fn stays alive when one output is deleted."""
        a = torch.randn(4, requires_grad=True)
        b0, b1 = PythonSplit.apply(a)
        ref = weakref.ref(b0.grad_fn)
        del b0
        self.assertIsNotNone(ref())
        del b1
        self.assertIsNone(ref())

    def test_cpp_shared_grad_fn_backward_one(self):
        """backward on one shared output, then free both."""
        a = torch.randn(4, requires_grad=True)
        b0, b1 = (a * 2).split(2)
        b0.sum().backward(retain_graph=True)
        del b0, b1
        self.assertEqual(a._use_count(), 1)

    def test_python_shared_grad_fn_backward_one(self):
        a = torch.randn(4, requires_grad=True)
        b0, b1 = PythonSplit.apply(a)
        ref = weakref.ref(b0.grad_fn)
        b0.sum().backward(retain_graph=True)
        del b0, b1
        self.assertIsNone(ref())

    def test_cpp_shared_grad_fn_hook_cycle_gc(self):
        """Hook capturing shared grad_fn creates a cycle, GC collects it."""
        marker = Marker()
        ref = weakref.ref(marker)

        a = torch.randn(4, requires_grad=True)
        b0, b1 = (a * 2).split(2)
        grad_fn = b0.grad_fn

        def hook(*args):
            grad_fn  # noqa: F821
            marker  # noqa: F821

        b0.grad_fn.register_hook(hook)
        del a, b0, b1, grad_fn, hook, marker
        gc.collect()
        self.assertIsNone(ref())

    def test_python_shared_grad_fn_hook_cycle_gc(self):
        """Hook capturing shared Python grad_fn, GC collects it."""
        marker = Marker()
        ref = weakref.ref(marker)

        a = torch.randn(4, requires_grad=True)
        b0, b1 = PythonSplit.apply(a)
        grad_fn = b0.grad_fn

        def hook(*args):
            grad_fn  # noqa: F821
            marker  # noqa: F821

        b0.grad_fn.register_hook(hook)
        del a, b0, b1, grad_fn, hook, marker
        gc.collect()
        self.assertIsNone(ref())


class TestChainLifecycle(TestCase):
    """Tests for chains of nodes."""

    def test_cpp_chain_freed_refcount(self):
        a = torch.randn(2, requires_grad=True)
        b = a * 2
        c = b + 1
        d = c * 3
        self.assertEqual(a._use_count(), 2)
        del b, c, d
        self.assertEqual(a._use_count(), 1)

    def test_python_chain_freed_refcount(self):
        a = torch.randn(2, requires_grad=True)
        b = PythonMul.apply(a)
        c = PythonId.apply(b)
        ref_b = weakref.ref(b.grad_fn)
        ref_c = weakref.ref(c.grad_fn)
        del b, c
        self.assertIsNone(ref_b())
        self.assertIsNone(ref_c())

    def test_mixed_chain_freed_refcount(self):
        a = torch.randn(2, requires_grad=True)
        b = a * 2  # C++
        c = PythonId.apply(b)  # Python
        d = c + 1  # C++
        ref_c = weakref.ref(c.grad_fn)
        del b, c, d
        self.assertIsNone(ref_c())

    def test_deep_chain_no_stack_overflow(self):
        x = torch.randn(2, requires_grad=True)
        for _ in range(2000):
            x = x + torch.randn(2)
        del x

    def test_deep_python_chain_no_stack_overflow(self):
        x = torch.randn(2, requires_grad=True)
        for _ in range(2000):
            x = PythonId.apply(x)
        del x


class TestSavedVariableLifecycle(TestCase):
    """Tests that saved tensors are freed at the right time.

    Uses b**2 for C++ tests (PowBackward0 saves self) and PythonMul for
    Python tests (save_for_backward saves input). Both use a non-leaf
    intermediate tracked via weakref.
    """

    def test_cpp_saved_freed_on_graph_free(self):
        a = torch.randn(2, requires_grad=True)
        b = a + 1  # non-leaf intermediate
        ref = weakref.ref(b)
        c = b**2  # PowBackward0 saves b
        del b
        self.assertIsNotNone(ref())
        del c
        self.assertIsNone(ref())

    def test_python_saved_freed_on_graph_free(self):
        a = torch.randn(2, requires_grad=True)
        b = a + 1
        ref = weakref.ref(b)
        c = PythonMul.apply(b)
        del b
        self.assertIsNotNone(ref())
        del c
        self.assertIsNone(ref())

    def test_cpp_saved_freed_after_backward(self):
        a = torch.randn(2, requires_grad=True)
        b = a + 1
        ref = weakref.ref(b)
        c = (b**2).sum()
        del b
        self.assertIsNotNone(ref())
        c.backward()
        self.assertIsNone(ref())

    def test_python_saved_freed_after_backward(self):
        a = torch.randn(2, requires_grad=True)
        b = a + 1
        ref = weakref.ref(b)
        c = PythonMul.apply(b)
        del b
        self.assertIsNotNone(ref())
        c.sum().backward()
        self.assertIsNone(ref())

    def test_cpp_saved_kept_with_retain_graph(self):
        a = torch.randn(2, requires_grad=True)
        b = a + 1
        ref = weakref.ref(b)
        c = (b**2).sum()
        del b
        c.backward(retain_graph=True)
        self.assertIsNotNone(ref())
        del c
        self.assertIsNone(ref())

    def test_python_saved_kept_with_retain_graph(self):
        a = torch.randn(2, requires_grad=True)
        b = a + 1
        ref = weakref.ref(b)
        c = PythonMul.apply(b)
        del b
        c.sum().backward(retain_graph=True)
        self.assertIsNotNone(ref())
        del c
        self.assertIsNone(ref())

    def test_cpp_saved_output_freed_on_graph_free(self):
        a = torch.randn(2, requires_grad=True)
        b = a.exp()  # ExpBackward0 saves its output (b)
        ref = weakref.ref(b)
        del b
        self.assertIsNone(ref())

    def test_python_saved_output_freed_on_graph_free(self):
        a = torch.randn(2, requires_grad=True)
        b = PythonExp.apply(a)
        ref = weakref.ref(b)
        del b
        self.assertIsNone(ref())

    def test_cpp_saved_output_freed_after_backward(self):
        a = torch.randn(2, requires_grad=True)
        b = a.exp()
        ref = weakref.ref(b)
        b.sum().backward()
        del b
        self.assertIsNone(ref())

    def test_python_saved_output_freed_after_backward(self):
        a = torch.randn(2, requires_grad=True)
        b = PythonExp.apply(a)
        ref = weakref.ref(b)
        b.sum().backward()
        del b
        self.assertIsNone(ref())


if __name__ == "__main__":
    run_tests()
