import contextlib
from typing import Callable, cast, Generator, Mapping, Optional, Sequence, Type

import torch
import torch.fx
import torch.nn.functional as F
from torch import Tensor
from torch._guards import Source
from torch._ops import OpOverload
from torch._subclasses.meta_utils import MetaConverter
from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymbolicContext
from torch.fx.immutable_collections import immutable_dict
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode

pytree = torch.utils._pytree


@contextlib.contextmanager
def in_kernel_invocation_manager() -> Generator[None, None, None]:
    with torch._C._DisableTorchDispatch():
        with torch._C._PreserveDispatchKeyGuard():
            torch._C._set_meta_in_tls_dispatch_include(True)
            yield


class FakeSymTensor(torch.Tensor):
    def __new__(
        cls,
        elem: Tensor,
        device: torch.device,
        constant: Optional[Tensor] = None,
        real_tensor: Optional[Tensor] = None,
    ):
        self = Tensor._make_subclass(
            cls,
            elem,
            elem.requires_grad,
            dispatch_device=True,
            device_for_backend_keys=device,
        )
        self.real_tensor = real_tensor

        return self

    @classmethod
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # Avoid infinite recursion
        if func == torch.ops.prim.device.default:
            return torch.device("meta")

        with in_kernel_invocation_manager():
            return func(*args, **kwargs)


class FakeSymTensorMode(TorchDispatchMode):
    def __init__(self):
        super().__init__()

        self.shape_env = ShapeEnv()
        self.conv = MetaConverter(copy_data=False)

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Sequence[Type],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] = immutable_dict(),
    ) -> object:
        print(func)

        # Covert args if they are Tensors
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))
        args_new = []
        for arg in flat_args:
            if (
                isinstance(arg, torch.Tensor)
                and not isinstance(arg, FakeSymTensor)
                and not arg.device == torch.device("meta")
            ):
                args_new.append(from_real_tensor(arg, self.conv, self.shape_env))
            else:
                args_new.append(arg)
        args = args_new

        args, kwargs = pytree.tree_unflatten(args, args_spec)

        if handler := _DISPATCH_META_HANDLERS.get(func):
            return handler(args)

        from torch._decomp import decomposition_table

        if func in decomposition_table:
            return decomposition_table[func](*args, **kwargs)

        with in_kernel_invocation_manager():
            r = func(*args, **kwargs)
        return r


def from_real_tensor(
    t: Tensor,
    meta_converter: MetaConverter,
    shape_env: Optional[ShapeEnv] = None,
    make_constant: bool = False,
    *,
    source: Optional[Source] = None,
    symbolic_context: Optional[SymbolicContext] = None,
    trace: bool = True,
) -> FakeSymTensor:

    if type(t) is torch.nn.Parameter:
        assert not make_constant

    constant = t if make_constant else None

    def mk_fake_tensor(
        make_meta_t: Callable[[], object], device: torch.device
    ) -> FakeSymTensor:
        make_meta_t(),
        with no_dispatch():
            return FakeSymTensor(
                make_meta_t(),
                device,
                constant=constant,
                real_tensor=t,
            )

    out = meta_converter(
        t,
        shape_env=shape_env,
        callback=mk_fake_tensor,
        source=source,
        symbolic_context=symbolic_context,
        trace=trace,
    )
    return out


_DISPATCH_META_HANDLERS = {
    torch.ops.prim.device.default: lambda _: torch.device("meta"),
    torch.ops.aten.size.default: lambda args: tuple(
        int(s) for s in cast(Tensor, args[0]).size()
    ),
    torch.ops.aten.stride.default: lambda args: tuple(
        int(s) for s in cast(Tensor, args[0]).stride()
    ),
    torch.ops.aten.storage_offset.default: lambda args: int(
        cast(Tensor, args[0]).storage_offset()
    ),
}


class FakeSymTensorTest(TestCase):
    def setUp(self):
        self.shape_env = ShapeEnv()
        self.conv = MetaConverter(copy_data=False)

    def test_elementwise(self):
        with FakeSymTensorMode():
            # Create inputs
            a = from_real_tensor(torch.randn(5, 6), self.conv, self.shape_env)
            b = from_real_tensor(torch.randn(5, 6), self.conv, self.shape_env)

            # Perform op
            z = a + b

            # Verify
            self.assertIsInstance(z.shape[0], torch.SymInt)
            self.assertIsInstance(z.shape[1], torch.SymInt)
            self.assertEqual(z.shape, a.shape)

    def test_concat(self):
        with FakeSymTensorMode():
            # Create inputs
            a = from_real_tensor(torch.randn(5, 6), self.conv, self.shape_env)
            b = from_real_tensor(torch.randn(5, 6), self.conv, self.shape_env)

            # Perform op
            z = torch.concat([a, b], dim=1)

            # Verify
            self.assertIsInstance(z.shape[0], torch.SymInt)
            self.assertIsInstance(z.shape[1], torch.SymInt)
            self.assertEqual(z.shape[0], a.shape[0])
            self.assertEqual(z.shape[1], 2 * a.shape[1])

    def test_sum(self):
        with FakeSymTensorMode():
            # Create inputs
            a = from_real_tensor(torch.randn(5, 6), self.conv, self.shape_env)

            # Perform op
            z = torch.sum(a, dim=1)

            # Verify
            self.assertIsInstance(z.shape[0], torch.SymInt)
            self.assertEqual(z.shape[0], a.shape[0])

    def test_mm(self):
        with FakeSymTensorMode():
            # Create inputs
            a = from_real_tensor(torch.randn(5, 6), self.conv, self.shape_env)
            b = from_real_tensor(torch.randn(6, 8), self.conv, self.shape_env)

            # Perform op
            z = torch.ops.aten.mm.default(a, b)

            # Verify
            self.assertIsInstance(z.shape[0], torch.SymInt)
            self.assertIsInstance(z.shape[1], torch.SymInt)
            self.assertEqual(z.shape[0], a.shape[0])
            self.assertEqual(z.shape[1], b.shape[1])

    def test_view(self):
        with FakeSymTensorMode():
            # Create inputs
            a = from_real_tensor(torch.randn(5, 6, 7), self.conv, self.shape_env)

            # Perform op
            z = a.view(a.shape[0] * a.shape[1], a.shape[2])

            # Verify
            self.assertIsInstance(z.shape[0], torch.SymInt)
            self.assertIsInstance(z.shape[1], torch.SymInt)
            self.assertEqual(z.shape[0], a.shape[0] * a.shape[1])
            self.assertEqual(z.shape[1], a.shape[2])

    def test_embedding(self):
        with FakeSymTensorMode():
            # Create inputs
            a = from_real_tensor(
                torch.randn(5, 8).to(dtype=torch.int), self.conv, self.shape_env
            )

            # Perform op
            emb = torch.nn.Embedding(8, 32)
            z = emb(a)

            # Verify
            self.assertIsInstance(z.shape[0], torch.SymInt)
            self.assertIsInstance(z.shape[1], torch.SymInt)
            self.assertIsInstance(z.shape[2], torch.SymInt)


if __name__ == "__main__":
    run_tests()
