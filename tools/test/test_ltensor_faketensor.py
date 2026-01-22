"""Test LTensor behavior during FakeTensor tracing."""
import torch
import torch._dynamo as dynamo
from torch.utils._pytree import tree_map


class WrapperLTensor(torch.Tensor):
    """
    LTensor using _make_wrapper_subclass pattern (works with FakeTensor).

    This is the pattern used by FunctionalTensor, etc.
    """

    @staticmethod
    def __new__(cls, local_tensor, variant_dims):
        # Use _make_wrapper_subclass instead of as_subclass
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            local_tensor.shape,
            strides=local_tensor.stride(),
            storage_offset=local_tensor.storage_offset(),
            dtype=local_tensor.dtype,
            layout=local_tensor.layout,
            device=local_tensor.device,
            requires_grad=local_tensor.requires_grad,
        )
        r._local_tensor = local_tensor
        r._variant_dims = variant_dims
        return r

    def __repr__(self):
        return f"WrapperLTensor(variant_dims={self._variant_dims}, shape={self.shape})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """Required when using _make_wrapper_subclass."""
        func_name = func.__name__ if hasattr(func, '__name__') else str(func)
        print(f"  __torch_dispatch__: {func_name}")

        # Check types
        def check_type(t):
            if isinstance(t, torch.Tensor):
                tname = type(t).__name__
                print(f"    arg type: {tname}")

        tree_map(check_type, args)

        # Unwrap
        def unwrap(t):
            return t._local_tensor if isinstance(t, WrapperLTensor) else t

        unwrapped_args = tree_map(unwrap, args)
        kwargs = kwargs or {}
        unwrapped_kwargs = tree_map(unwrap, kwargs)

        # Call the actual op
        result = func(*unwrapped_args, **unwrapped_kwargs)

        # Wrap outputs
        def wrap(t):
            if isinstance(t, torch.Tensor) and not isinstance(t, WrapperLTensor):
                return WrapperLTensor(t, set())
            return t

        return tree_map(wrap, result)

    def __tensor_flatten__(self):
        return ["_local_tensor"], {"variant_dims": self._variant_dims}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        return WrapperLTensor(
            inner_tensors["_local_tensor"],
            metadata["variant_dims"],
        )


def test_compile_what_types():
    """Check what types are seen during torch.compile."""
    print("=" * 60)
    print("Test 1: What types are seen during torch.compile?")
    print("=" * 60)

    def fn(x):
        print(f"  fn sees x as: {type(x).__name__}")
        result = x * 2
        print(f"  fn result type: {type(result).__name__}")
        return result

    x = torch.randn(10)
    lt = WrapperLTensor(x, {"dp"})

    print("First call (tracing):")
    fn_compiled = torch.compile(fn, backend="eager")
    out1 = fn_compiled(lt)
    print(f"  Final output type: {type(out1).__name__}")

    print("\nSecond call (cached):")
    out2 = fn_compiled(lt)
    print(f"  Final output type: {type(out2).__name__}")
    print()


def test_inner_tensor_type():
    """Check the inner tensor type during tracing."""
    print("=" * 60)
    print("Test 2: What is _local_tensor during __torch_dispatch__?")
    print("=" * 60)

    class InspectingLTensor(WrapperLTensor):
        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            for arg in args:
                if isinstance(arg, InspectingLTensor):
                    inner = arg._local_tensor
                    print(f"  _local_tensor type: {type(inner).__name__}")
            return super().__torch_dispatch__(func, types, args, kwargs)

    def fn(x):
        return x * 2

    x = torch.randn(10)
    lt = InspectingLTensor(x, {"dp"})

    print("Compiled (tracing):")
    fn_compiled = torch.compile(fn, backend="eager")
    out = fn_compiled(lt)
    print(f"  Output type: {type(out).__name__}")
    print()


def test_graph_breaks():
    """Check for graph breaks."""
    print("=" * 60)
    print("Test 3: Graph breaks?")
    print("=" * 60)

    def fn(x):
        return x * 2 + 1

    x = torch.randn(10)
    lt = WrapperLTensor(x, {"dp"})

    explanation = dynamo.explain(fn)(lt)
    print(f"Graph break count: {explanation.graph_break_count}")
    if explanation.graph_break_count > 0:
        print(f"Break reasons: {explanation.break_reasons}")
    print()


if __name__ == "__main__":
    dynamo.reset()
    test_compile_what_types()

    dynamo.reset()
    test_inner_tensor_type()

    dynamo.reset()
    test_graph_breaks()
