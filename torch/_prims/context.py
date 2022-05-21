import string
from typing import Callable, Sequence, Any, Dict
from itertools import chain
import functools


import torch
from torch.fx.graph import Graph, Node
import torch.overrides

from torch._prims.utils import TensorMeta, torch_function_passthrough
import torch._refs as refs

import torch._refs
import torch._refs.nn
import torch._refs.nn.functional
import torch._refs.special

import torch._prims


# TODO:  automap torch operations to references
# (need to throw a good assertion if the mapping doesn't exist)
_torch_to_reference_map = {
    torch.add: refs.add,
    # torch.div: refs.div,
    torch.mul: refs.mul,
    torch.ge: refs.ge,
    torch.gt: refs.gt,
    torch.le: refs.le,
    torch.lt: refs.lt,
}


class PrimContext(torch.overrides.TorchFunctionMode):
    """
    The prototype prim tracing context.

    Example usage:

    import torch._prims.utils as utils
    from torch._prims.context import PrimContext
    from torch._prims.executor import execute
    from torch.overrides import push_torch_function_mode

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    with push_torch_function_mode(PrimContext):
      meta_a = ctx.placeholder(utils.TensorMeta(a))
      meta_b = ctx.placeholder(utils.TensorMeta(b))
      result = torch.add(meta_a, meta_b)
      ctx.output(result)

    exc_result = execute(ctx, a, b)

    Currently this only acquires a trace of prims, and
    it does not account for control flow. As such,
    execute must be called with tensors that have the
    same metadata (dtype, device, shape...) as
    the tensors used to trace the operations.

    The tracing context's FX graph can be acquired
    using its graph attribute.
    """

    def __init__(self):
        self.graph = Graph()

        # Private attributes for generating names
        self._tensor_name_counter = 0
        self._dim_name_counter = 0
        self._shape_name_counter = 0
        self._lowercase = tuple(string.ascii_lowercase)
        self._uppercase = tuple(string.ascii_uppercase)

    @staticmethod
    def _create_name(idx, chars):
        name = ""
        while idx >= len(chars):
            name = chars[idx % len(chars)] + name
            idx = idx - len(chars)
        name = chars[idx] + name

        return name

    def _tensor_name(self):
        idx = self._tensor_name_counter
        self._tensor_name_counter = self._tensor_name_counter + 1

        return self._create_name(idx, self._lowercase)

    def _add_user(self, tm: TensorMeta, node: Node) -> None:
        assert tm.node is not None
        tm.node.users[node] = None

    def placeholder(self, a: Any):
        name = self._tensor_name()
        node = self.graph.placeholder(name)

        if isinstance(a, TensorMeta):
            if a.node is not None:
                raise ValueError("Attempting to reuse a TensorMeta in a new trace!")
            a.tname = name
            a.node = node

        return a

    def output(self, tm: TensorMeta):
        # TODO: allow other output types
        assert isinstance(tm, TensorMeta)

        node = self.graph.output(tm)
        self._add_user(tm, node)

    def __torch_function__(
        self,
        func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Dict = None,
    ):
        """
        Determines which function to call. The order of which
        function is called is determined by:

        - func's "meta" attribute, if it exists
        - if func is a torch operation, its corresponding reference
        - func
        """

        if kwargs is None:
            kwargs = {}

        if hasattr(func, "meta"):
            # TODO: add check that all args/kwargs are 'registered' properly
            # to this trace

            output = func.meta(*args, **kwargs)  # type: ignore[attr-defined]

            # Updates graph
            # TODO: handle outputs with multiple tensors
            # TODO: handle non-tensor outputs
            assert isinstance(output, TensorMeta)
            output_name = self._tensor_name()
            node = self.graph.create_node(
                "call_function", func, name=output_name, args=args, kwargs=kwargs
            )
            output.tname = output_name
            output.node = node

            # Marks uses
            for x in (
                x for x in chain(args, kwargs.values()) if isinstance(x, TensorMeta)
            ):
                self._add_user(x, node)

            return output

        # Remaps torch operations to their references
        if func in _torch_to_reference_map:
            fn = _torch_to_reference_map[func]
            with torch.overrides.enable_torch_function_mode(self, replace=self.inner):
                return fn(*args, **kwargs)  # type: ignore[operator]

        return func(*args, **kwargs)


@functools.lru_cache(None)
def torch_to_refs_map():
    """
    Mapping of torch API functions to torch._refs functions.
    E.g. torch_to_refs_map()[torch.add] == torch._refs.add
    """
    modules = [
        (torch, torch._refs),
        (torch.nn, torch._refs.nn),
        (torch.nn.functional, torch._refs.nn.functional),
        (torch.special, torch._refs.special),
    ]
    r = {}
    for mod_torch, mod_refs in modules:
        for s in mod_refs.__all__:
            r[mod_torch.__dict__.get(s)] = mod_refs.__dict__.get(s)
    return r


@functools.lru_cache(None)
def all_prims():
    """
    Set of all prim functions, e.g., torch._prims.add in all_prims()
    """
    return {torch._prims.__dict__.get(s) for s in torch._prims.__all__}


class TorchRefsMode(torch.overrides.TorchFunctionMode):
    """
    Switches the interpretation of torch.* functions and Tensor methods to
    use PrimTorch refs in torch._refs.  (Direct calls to _refs are unaffected.)

    >>> with TorchRefsMode.push():
    ...     torch.add(x, y)  # calls torch._refs.add(x, y)

    By default, this context manager will fall back on the torch.* if the
    ref does not exist; set strict=True to error if this occurs.
    """

    def __init__(self, strict=False):
        self.strict = strict

    def __torch_function__(
        self,
        orig_func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Dict = None,
    ):
        if kwargs is None:
            kwargs = {}
        # For primitive operations, run them as is without interception
        if orig_func in torch_function_passthrough or orig_func in all_prims():
            return orig_func(*args, **kwargs)
        mapping = torch_to_refs_map()
        func = mapping.get(orig_func, None)
        if func is not None:
            return func(*args, **kwargs)
        if self.strict:
            raise RuntimeError(
                f"no _refs support for {torch.overrides.resolve_name(orig_func)}"
            )
        return orig_func(*args, **kwargs)
