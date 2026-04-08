# mypy: allow-untyped-defs
import contextlib
from pathlib import Path

import torch


_TORCHBIND_IMPLS_INITIALIZED = False

_TENSOR_QUEUE_GLOBAL_TEST: torch.ScriptObject | None = None


def init_torchbind_implementations():
    global _TORCHBIND_IMPLS_INITIALIZED
    global _TENSOR_QUEUE_GLOBAL_TEST
    if _TORCHBIND_IMPLS_INITIALIZED:
        return

    load_torchbind_test_lib()
    register_fake_operators()
    register_fake_classes()
    _TENSOR_QUEUE_GLOBAL_TEST = _empty_tensor_queue()
    _TORCHBIND_IMPLS_INITIALIZED = True


def _empty_tensor_queue() -> torch.ScriptObject:
    return torch.classes._TorchScriptTesting._TensorQueue(
        torch.empty(
            0,
        ).fill_(-1)
    )


# put these under a function because the corresponding library might not be loaded yet.
def register_fake_operators():
    @torch.library.register_fake("_TorchScriptTesting::takes_foo_python_meta")
    def fake_takes_foo(foo, z):
        return foo.add_tensor(z)

    @torch.library.register_fake("_TorchScriptTesting::queue_pop")
    def fake_queue_pop(tq):
        return tq.pop()

    @torch.library.register_fake("_TorchScriptTesting::queue_push")
    def fake_queue_push(tq, x):
        return tq.push(x)

    torch.library.register_autocast(
        "_TorchScriptTesting::queue_push", "cpu", torch.float32
    )
    torch.library.register_autocast(
        "_TorchScriptTesting::queue_push", "cuda", torch.float32
    )

    torch.library.register_autocast(
        "_TorchScriptTesting::queue_pop", "cpu", torch.float32
    )
    torch.library.register_autocast(
        "_TorchScriptTesting::queue_pop", "cuda", torch.float32
    )

    @torch.library.register_fake("_TorchScriptTesting::queue_size")
    def fake_queue_size(tq):
        return tq.size()

    def meta_takes_foo_list_return(foo, x):
        a = foo.add_tensor(x)
        b = foo.add_tensor(a)
        c = foo.add_tensor(b)
        return [a, b, c]

    def meta_takes_foo_tuple_return(foo, x):
        a = foo.add_tensor(x)
        b = foo.add_tensor(a)
        return (a, b)

    @torch.library.register_fake("_TorchScriptTesting::takes_foo_tensor_return")
    def meta_takes_foo_tensor_return(foo, x):
        # This implementation deliberately creates unbacked symint for testing
        ctx = torch.library.get_ctx()
        fake_shape = [ctx.new_dynamic_size() for _ in range(2)]
        return torch.empty(fake_shape, dtype=torch.int, device="cpu")

    torch.ops._TorchScriptTesting.takes_foo_list_return.default.py_impl(
        torch._C.DispatchKey.Meta
    )(meta_takes_foo_list_return)

    torch.ops._TorchScriptTesting.takes_foo_tuple_return.default.py_impl(
        torch._C.DispatchKey.Meta
    )(meta_takes_foo_tuple_return)

    torch.ops._TorchScriptTesting.takes_foo.default.py_impl(torch._C.DispatchKey.Meta)(
        # make signature match original cpp implementation to support kwargs
        lambda foo, x: foo.add_tensor(x)
    )


def register_fake_classes():
    # noqa: F841
    @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
    class FakeFoo:
        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y

        @classmethod
        def __obj_unflatten__(cls, flattend_foo):
            return cls(**dict(flattend_foo))

        def add_tensor(self, z):
            return (self.x + self.y) * z

    @torch._library.register_fake_class("_TorchScriptTesting::_ContainsTensor")
    class FakeContainsTensor:
        def __init__(self, t: torch.Tensor):
            self.t = t

        @classmethod
        def __obj_unflatten__(cls, flattend_foo):
            return cls(**dict(flattend_foo))

        def get(self):
            return self.t

    @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
    class FakeTensorQueue:
        def __init__(self, queue):
            self.queue = queue

        @classmethod
        def __obj_unflatten__(cls, flattened_ctx):
            return cls(**dict(flattened_ctx))

        def push(self, x):
            self.queue.append(x)

        def pop(self):
            if self.is_empty():
                return torch.empty([])
            return self.queue.pop(0)

        def size(self):
            return len(self.queue)

        def is_empty(self):
            return len(self.queue) == 0

        def float_size(self):
            return float(len(self.queue))

    @torch._library.register_fake_class("_TorchScriptTesting::_FlattenWithTensorOp")
    class FakeFlatten:
        def __init__(self, t):
            self.t = t

        def get(self):
            return self.t

        @classmethod
        def __obj_unflatten__(cls, flattened_ctx):
            return cls(**dict(flattened_ctx))


def load_torchbind_test_lib():
    import unittest

    from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
        find_library_location,
        IS_FBCODE,
        IS_MACOS,
        IS_SANDCASTLE,
        IS_WINDOWS,
    )

    if IS_MACOS:
        raise unittest.SkipTest("non-portable load_library call used in test")
    elif IS_SANDCASTLE or IS_FBCODE:
        lib_file_path = Path("//caffe2/test/cpp/jit:test_custom_class_registrations")
    elif IS_WINDOWS:
        lib_file_path = find_library_location("torchbind_test.dll")
    else:
        lib_file_path = find_library_location("libtorchbind_test.so")
    torch.ops.load_library(str(lib_file_path))


@contextlib.contextmanager
def _register_py_impl_temporarily(op_overload, key, fn):
    try:
        op_overload.py_impl(key)(fn)
        yield
    finally:
        del op_overload.py_kernels[key]
        op_overload._dispatch_cache.clear()
