import builtins

import torch
import torch.utils._pytree as pytree
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode


class _PrintGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, format_str: str, *tensors: torch.Tensor):  # type: ignore[override]
        ctx.format_str = format_str
        return tensors

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor):  # type: ignore[override]
        # Use the print HOP (not builtins.print) so this gets properly traced
        # into the backward graph during AOT compilation.
        print("[backward] " + ctx.format_str, *grad_outputs)
        return (None,) + grad_outputs


class Print(HigherOrderOperator):
    """
    print(format_str, *args, **kwargs) -> None

    This Higher Order Operator (HOP) provides a functional version of print for use in PyTorch graphs.
    It supports the calling conventions of print(format_str.format(*args, **kwargs)):

    1. Format string with keyword arguments (named placeholders):
       torch._higher_order_ops.print("moo {x} {y}", x=1, y=2)
       Output: "moo 1 2"

    2. Format string with positional arguments (positional placeholders):
       torch._higher_order_ops.print("moo {} {}", 1, 2)
       Output: "moo 1 2"

    3. Mixed positional and keyword arguments:
       torch._higher_order_ops.print("moo {} {y}", 1, y=2)
       Output: "moo 1 2"

    4. DTensor support:
       DTensor args are unwrapped to local tensors via to_local() (no collective).
       Each rank prints its own local view, prefixed with [rank N].
       For the global view of a sharded tensor, call full_tensor() before
       passing to print.

       dt = DTensor.from_local(local_shard, mesh, [Shard(0)])
       torch._higher_order_ops.print("activations: {}", dt)
       # Output: [rank 0] activations: tensor([0., 1.])

    5. Gradient printing during backward (print_backward=True):
       x = torch._higher_order_ops.print("x: {}", x, print_backward=True)
       # Forward: prints "x: tensor([...])"
       # Backward: prints "[backward] x: tensor([...])" with gradient values
       # Returns the tensor args so they stay in the autograd graph.

    This HOP enables printing without causing graph break.
    """

    def __init__(self) -> None:
        super().__init__("print")

    def __call__(
        self,
        format_str: str,
        *args: object,
        print_backward: bool = False,
        **kwargs: object,
    ):  # type: ignore[override]
        if not isinstance(format_str, str):
            raise AssertionError(f"format_str must be a string, got {type(format_str)}")

        # Forward print via normal HOP dispatch
        # pyrefly: ignore [missing-attribute]
        super().__call__(format_str, *args, **kwargs)

        if not print_backward:
            return None

        # Collect tensor args, build a grad-only format string for backward
        tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
        tensor_kwargs = {k: v for k, v in kwargs.items() if isinstance(v, torch.Tensor)}

        all_tensors = list(tensor_args) + list(tensor_kwargs.values())
        if not all_tensors:
            return None

        # Build a format string for the gradient print using only tensor placeholders
        grad_fmt_parts = []
        for i, a in enumerate(args):
            if isinstance(a, torch.Tensor):
                grad_fmt_parts.append("{}")
        grad_kwarg_keys = [k for k, v in kwargs.items() if isinstance(v, torch.Tensor)]
        for k in grad_kwarg_keys:
            grad_fmt_parts.append(f"{k}=" + "{}")

        grad_format_str = (
            format_str if not grad_fmt_parts else "grad " + " ".join(grad_fmt_parts)
        )

        results = _PrintGradFunction.apply(grad_format_str, *all_tensors)

        if len(all_tensors) == 1:
            return results[0]
        return results

    # pyrefly: ignore [bad-override]
    def gen_schema(
        self, format_str: str, *args: object, **kwargs: object
    ) -> torch.FunctionSchema:
        from torch._higher_order_ops.schema import HopSchemaGenerator

        schema_gen = HopSchemaGenerator(self)
        schema_gen.add_arg("format_str", format_str[0])

        # Add each positional arg
        for i, value in enumerate(args):
            schema_gen.add_arg(f"arg{i}", value)

        # Add each kwarg as a keyword-only argument
        for key, value in kwargs.items():
            schema_gen.add_arg(key, value, kw_only=True)

        schema_gen.add_schema_tree_spec(format_str, *args, **kwargs)

        return schema_gen.gen_schema()


print = Print()


@print.py_impl(ProxyTorchDispatchMode)
# pyre-ignore
def print_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode, format_str: str, *args: object, **kwargs: object
) -> None:
    proxy_args = pytree.tree_map(mode.tracer.unwrap_proxy, args)  # type: ignore[union-attr]
    proxy_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, kwargs)  # type: ignore[union-attr]
    mode.tracer.create_proxy(
        "call_function", print, (format_str, *proxy_args), proxy_kwargs
    )


@print.py_impl(FakeTensorMode)
# pyre-ignore
def print_fake_tensor_mode(mode, format_str: str, *args: object, **kwargs: object):
    return None


@print.py_impl(torch._C.DispatchKey.CompositeExplicitAutograd)
# pyre-ignore
def print_impl(format_str: str, *args: object, **kwargs: object) -> None:
    # Ensure all immutable_dict/list in args and kwargs are converted to regular dict/list
    map_types: dict[type, type] = {
        torch.fx.immutable_collections.immutable_dict: dict,
        torch.fx.immutable_collections.immutable_list: list,
    }
    new_args, new_kwargs = pytree.tree_map_only(
        tuple(map_types.keys()),
        lambda a: map_types[type(a)](a),
        (args, kwargs),
        lambda a: isinstance(a, tuple(map_types.keys())),
    )
    #  Use built-in print to avoid recursion with the HOP print
    builtins.print(format_str.format(*new_args, **new_kwargs))


print.fallthrough(torch._C.DispatchKey.AutogradCPU)
print.fallthrough(torch._C.DispatchKey.AutogradCUDA)


def _register_dtensor_impl() -> None:
    from torch.distributed.tensor import DTensor

    @print.py_impl(DTensor)  # pyrefly: ignore [missing-attribute]
    # pyre-ignore
    def print_dtensor(format_str: str, *args: object, **kwargs: object) -> None:
        # Unwrap DTensors to local tensors via to_local() — no collective is
        # introduced so there is no OOM or performance risk.  Every rank prints
        # its own local view (including Replicate, where to_local() already
        # holds the full tensor).
        #
        # The output is prefixed with [rank N] so users can identify which rank
        # produced each line.
        #
        # If the user needs the global view of a sharded tensor, they can call
        # full_tensor() explicitly before passing it to print.
        import torch.distributed as dist

        local_args = pytree.tree_map_only(DTensor, DTensor.to_local, args)
        local_kwargs = pytree.tree_map_only(DTensor, DTensor.to_local, kwargs)
        if dist.is_initialized() and dist.get_world_size() > 1:
            format_str = f"[rank {dist.get_rank()}] {format_str}"
        print(  # pyrefly: ignore [no-matching-overload]
            format_str, *local_args, **local_kwargs
        )


@print.py_functionalize_impl
def print_func(ctx, format_str: str, *args: object, **kwargs: object):
    from torch._higher_order_ops.effects import handle_effects

    return handle_effects(
        ctx.mode._allow_token_discovery,
        ctx.mode._tokens,
        print,  # type: ignore[arg-type]
        (format_str, *args),
        kwargs,  # type: ignore[arg-type]
    )
