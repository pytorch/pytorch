# mypy: allow-untyped-defs
from typing import Optional

import torch.fx
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import Node
from torch.fx._compatibility import compatibility
from torch.fx.experimental.proxy_tensor import py_sym_types, snapshot_fake
from torch.fx.node import map_aggregate
from torch.utils._ordered_set import OrderedSet


__all__ = ["FakeTensorProp"]


@compatibility(is_backward_compatible=False)
class FakeTensorProp(torch.fx.Interpreter):
    """
    Execute an FX graph Node-by-Node and record a fake tensor representing
    the metadata for the node.  Unlike ShapeProp, (1) this propagation
    is cheap--it does the propagation with meta tensors which do not actually
    store data, and (2) the fake tensors have much more fine grained information,
    e.g., they have accurate alias information that can be consulted by looking
    at the storages.

    Args:
         module (GraphModule): The module to be executed
         mode (Optional[FakeTensorMode]): The dispatch mode used to execute computation indicated by each FX Node.
    """

    def __init__(
        self, module: torch.fx.GraphModule, mode: Optional[FakeTensorMode] = None
    ):
        super().__init__(module)
        if mode is None:
            mode = FakeTensorMode()
        self._mode = mode
        mode.epoch += 1
        mode.reset_nt_tensor_id_counter()
        self.seen_subgraphs: OrderedSet[str] = OrderedSet()

    def run_node(self, n: Node):
        from torch.fx.experimental.symbolic_shapes import (
            compute_unbacked_bindings,
            rebind_unbacked,
        )

        if (
            n.op == "call_function"
            and n.target is torch.ops.higher_order.invoke_subgraph
            and n.args[1] not in self.seen_subgraphs
        ):
            # Prevent redundant fake tensor prop for invoke_subgraphs. Note that
            # there is also fake tensor caching for the entire subgraph. This
            # happens the next time we call `run_node` for the same subgraph,
            # which goes through super.run_node and caches the fake tensor prop.
            # Therefore, we are propagating fake tensor through the subgraphs
            # twice.
            assert isinstance(n.args[1], str)
            assert (
                isinstance(n.args[0], torch.fx.Node)
                and n.args[0].op == "get_attr"
                and isinstance(n.args[0].target, str)
            )
            self.seen_subgraphs.add(n.args[1])
            operands = n.args[2:]
            example_inputs = []
            for operand in operands:
                assert isinstance(operand, torch.fx.Node) and "val" in operand.meta
                example_inputs.append(operand.meta["val"])
            return FakeTensorProp(
                getattr(self.module, n.args[0].target), mode=self._mode
            ).propagate(*example_inputs)

        result = super().run_node(n)
        rebind_unbacked(self._mode.shape_env, n, result)

        def extract_val(obj):
            if isinstance(obj, FakeTensor):
                return snapshot_fake(obj)
            elif isinstance(obj, torch.Tensor):
                # TODO: How is it possible that we get a non fake tensor?  We
                # should be running under the mode...
                return snapshot_fake(self._mode.from_tensor(obj, static_shapes=True))
            elif isinstance(obj, py_sym_types):
                return obj
            else:
                return None

        meta = map_aggregate(result, extract_val)
        if meta is not None:
            n.meta["val"] = meta
            if (shape_env := self._mode.shape_env) and (
                symbol_to_path := compute_unbacked_bindings(shape_env, result)
            ):
                # pyrefly: ignore  # unbound-name
                n.meta["unbacked_bindings"] = symbol_to_path

        return result

    def propagate(self, *args):
        fake_args = [
            self._mode.from_tensor(a) if isinstance(a, torch.Tensor) else a
            for a in args
        ]
        return self.propagate_dont_convert_inputs(*fake_args)

    def propagate_dont_convert_inputs(self, *args):
        with self._mode:
            return super().run(*args)
