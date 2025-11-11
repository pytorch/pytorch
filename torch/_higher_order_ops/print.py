import builtins

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


class Print(HigherOrderOperator):
    """
    print(format_str, **kwargs) -> None

    This Higher Order Operator (HOP) provides a functional version of print for use in PyTorch graphs.
    It enables format printing with named arguments, e.g., torch._higher_order_ops.print("moo {x} {y}", x=1, y=2).

    This HOP enables printing without causing graph break.
    """

    def __init__(self) -> None:
        super().__init__("print")

    def __call__(self, format_str: str, **kwargs: object) -> None:
        assert isinstance(format_str, str)
        return super().__call__(format_str, **kwargs)

    # pyrefly: ignore [bad-override]
    def gen_schema(self, format_str: str, **kwargs: object) -> torch.FunctionSchema:
        from torch._higher_order_ops.schema import HopSchemaGenerator

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_output(None)
        schema_gen.add_schema_tree_spec(format_str, **kwargs)

        return schema_gen.gen_schema()


print = Print()


@print.py_impl(ProxyTorchDispatchMode)
# pyre-ignore
def print_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode, format_str: str, **kwargs: object
) -> None:
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)  # type: ignore[union-attr]  # noqa: F841
    mode.tracer.create_proxy("call_function", print, (format_str,), proxy_kwargs)


@print.py_impl(FakeTensorMode)
# pyre-ignore
def print_fake_tensor_mode(mode, format_str: str, **kwargs: object):
    return None


@print.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def print_impl(format_str: str, **kwargs: object) -> None:
    # Ensure all immutable_dict/list in kwargs are converted to regular dict/list
    map_types: dict[type, type] = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    new_kwargs = pytree.tree_map_only(
        tuple(map_types.keys()),
        lambda a: map_types[type(a)](a),
        kwargs,
        lambda a: isinstance(a, tuple(map_types.keys())),
    )
    #  Use built-in print to avoid recursion with the HOP print
    builtins.print(format_str.format(**new_kwargs))


print.fallthrough(torch._C.DispatchKey.AutogradCPU)
print.fallthrough(torch._C.DispatchKey.AutogradCUDA)


@print.py_functionalize_impl
def print_func(ctx, format_str: str, **kwargs: object):
    from torch._higher_order_ops.effects import handle_effects

    return handle_effects(
        ctx.mode._allow_token_discovery,
        ctx.mode._tokens,
        print,  # type: ignore[arg-type]
        (format_str,),
        kwargs,  # type: ignore[arg-type]
    )
