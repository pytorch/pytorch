import string
from typing import Callable, Sequence, Any, Dict
from itertools import chain

import torch
from torch.fx.graph import Graph, Node

from torch._prims.utils import TensorMeta
import torch._refs as refs


_torch_to_reference_map = {
    torch.add: refs.add,
}


class PrimContext(object):
    """
    The prototype prim tracing context.

    Example usage:

    import torch._prims.utils as utils
    from torch._prims.context import PrimContext
    from torch._prims.executor import execute

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    ctx = PrimContext()
    with ctx as tracing_ctx:
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

    def __enter__(self):
        self.old_ctx = TensorMeta.ctx
        TensorMeta.ctx = self

    def __exit__(self, type, value, traceback):
        TensorMeta.ctx = self.old_ctx

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

    def placeholder(self, tm: TensorMeta):
        # TODO: allow other input types
        assert isinstance(tm, TensorMeta)

        if tm.node is not None:
            raise ValueError("Attempting to reuse a TensorMeta in a new trace!")

        name = self._tensor_name()
        node = self.graph.placeholder(name)
        tm.name = name
        tm.node = node
        return tm

    def output(self, tm: TensorMeta):
        # TODO: allow other output types
        assert isinstance(tm, TensorMeta)

        node = self.graph.output(tm)
        self._add_user(tm, node)

    def handle_torch_function(
        self,
        func: Callable,
        types: Sequence,
        args: Sequence[Any],
        kwargs: Dict,
    ):
        """
        Determines which function to call. The order of which
        function is called is determined by:

        - func's "meta" attribute, if it exists
        - if func is a torch operation, its corresponding reference
        - func
        """

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
            output.name = output_name
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
            return fn(*args, **kwargs)

        return func(*args, **kwargs)
