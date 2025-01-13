import contextlib
from typing import Callable, cast, Generator, Optional

import torch
import torch.fx
import torch.nn.functional as F
from torch import Tensor
from torch._C import _disabled_torch_function_impl
from torch._decomp import decomposition_table
from torch._guards import Source
from torch._subclasses.fake_impls import stride_incorrect_op
from torch._subclasses.meta_utils import MetaConverter
from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymbolicContext
from torch.utils._mode_utils import no_dispatch

pytree = torch.utils._pytree


@contextlib.contextmanager
def in_kernel_invocation_manager() -> Generator[None, None, None]:
    with torch._C._DisableTorchDispatch():
        with torch._C._PreserveDispatchKeyGuard():
            torch._C._set_meta_in_tls_dispatch_include(True)
            yield


class FakeSymbolicTensor(torch.Tensor):
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
        return self

    @classmethod
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        flat_args, args_spec = pytree.tree_flatten((args, kwargs))

        print(
            "Dispatching %s with args shapes %s"
            % (
                func,
                str([a.shape for a in flat_args if isinstance(a, FakeSymbolicTensor)]),
            )
        )

        if handler := _DISPATCH_META_HANDLERS.get(func):
            return handler(args)

        if func in decomposition_table:
            print("Decomposing %s" % func)
            out = decomposition_table[func](*args, **kwargs)
            print(out)
            return out

        if (
            "prims::" in func._schema.name
            and hasattr(func, "prim_meta_impl")
            and not stride_incorrect_op(func)
        ):
            return func.prim_meta_impl(*args, **kwargs)

        with in_kernel_invocation_manager():
            return func(*args, **kwargs)


def from_real_tensor(
    t: Tensor,
    meta_converter: MetaConverter,
    shape_env: Optional[ShapeEnv] = None,
    make_constant: bool = False,
    *,
    source: Optional[Source] = None,
    symbolic_context: Optional[SymbolicContext] = None,
    trace: bool = True,
) -> FakeSymbolicTensor:

    if type(t) is torch.nn.Parameter:
        assert not make_constant

    constant = t if make_constant else None

    def mk_fake_tensor(
        make_meta_t: Callable[[], object], device: torch.device
    ) -> FakeSymbolicTensor:
        make_meta_t(),
        with no_dispatch():
            return FakeSymbolicTensor(
                make_meta_t(),
                device,
                constant=constant,
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


shape_env = ShapeEnv()
conv = MetaConverter(copy_data=False)

a = from_real_tensor(torch.randn(5, 6), conv, shape_env)
b = from_real_tensor(torch.randn(5, 6), conv, shape_env)

z = a + b
print(z.shape)

z = torch.concat([a, b], dim=1)
print(z.shape)

z = torch.sum(z, dim=1)
print(z.shape)

a = from_real_tensor(torch.randn(5, 6), conv, shape_env)
b = from_real_tensor(torch.randn(6, 8), conv, shape_env)

z = torch.ops.aten.mm.default(a, b)
print(z.shape)

a = from_real_tensor(torch.randn(5, 6), conv, shape_env)
b = from_real_tensor(torch.randn(6, 8), conv, shape_env)

z = torch.ops.aten.mm.default(a, b)
print(z.shape)

a = from_real_tensor(torch.randn(5, 6, 7), conv, shape_env)
z = a.view(30, 7)
print(z.shape)

a = from_real_tensor(torch.randn(5, 6, 7), conv, shape_env)
b = from_real_tensor(torch.randn(7, 8), conv, shape_env)

# z = a.view(a.shape[0] * a.shape[1], 7)
# z = torch.ops.aten.mm(z, b)
# z = z.view(a.shape[0], a.shape[1], b.shape[1])
#
# print(z.shape)
