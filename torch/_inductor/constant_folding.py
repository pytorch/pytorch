import collections
from typing import Any, Callable, Optional

import torch
import torch.utils._pytree as pytree
from torch._inductor.freezing_utils import maybe_set_is_frozen_param
from torch.utils._ordered_set import OrderedSet


aten = torch.ops.aten

# We would like to split modules into two subgraphs for runtime weight updates to work correctly.
# The use case and more information could be found at:
# https://docs.google.com/document/d/1inZC-8KarJ6gKB7G9egmYLx1V_dKX_apxon0w4zPC0Q/edit?usp=sharing
META_TAG = "MODULE_TYPE"
MODULE_TAG = "_MAIN_MODULE"
CONST_MODULE_TAG = "_CONST_MODULE"


def replace_node_with_constant(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    constant: Optional[torch.Tensor] = None,
    name: Optional[str] = None,
) -> None:
    g = gm.graph

    if name:
        qualname = name
    else:
        if not hasattr(gm, "_frozen_param_count"):
            gm._frozen_param_count = 0  # type: ignore[assignment]
        i = gm._frozen_param_count

        while True:
            qualname = f"_frozen_param{i}"
            if not hasattr(gm, qualname):
                break
            i += 1  # type: ignore[assignment, operator]

        gm._frozen_param_count = i + 1  # type: ignore[assignment, operator]

    with g.inserting_before(node):
        if constant is not None:
            new_input_node = g.create_node("get_attr", qualname, (), {})
        else:
            # this is the case for lifted constants
            new_input_node = g.create_node("placeholder", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)
        new_input_node.name = node.name

    if constant is not None:
        # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
        gm.register_buffer(qualname, constant)
        setattr(gm, qualname, constant)
        # mark any constants created during freezing
        maybe_set_is_frozen_param(constant)


def is_const_source(
    node: torch.fx.Node, lifted_constant_names: Optional[list[str]]
) -> bool:
    return node.op == "get_attr" or node.name in (lifted_constant_names or ())


class ConstantFolder(torch.fx.Interpreter):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        skip_constructors: bool = False,
        lifted_constant_names: Optional[list[str]] = None,
        skip_folding_node_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
    ) -> None:
        super().__init__(gm)
        self.node_replacements: dict[torch.fx.Node, Any] = {}
        self.replaced_uses: dict[torch.fx.Node, int] = collections.Counter()
        self.unknown_value = object()
        self.skip_constructors: bool = skip_constructors

        # overwrite this to deallocate env values if their only remaining use
        # is the output
        self.user_to_last_uses = self.node_to_last_non_output_use()
        self.lifted_constant_names = lifted_constant_names
        self.deferred_value = object()
        self.skip_folding_node_fn = skip_folding_node_fn

    def _support_dynamic_shape(self) -> bool:
        # ConstantFolder not support dynamic shape now
        return False

    def _deduce_value(self, node: torch.fx.Node) -> Any:
        if self.lifted_constant_names is None:
            return super().run_node(node)
        # if lifted_constant_names is passed in, no concrete value is available
        # so we just check if all inputs have values
        if self.skip_folding_node_fn is not None and self.skip_folding_node_fn(node):
            return self.unknown_value
        flattened_node_inps = pytree.arg_tree_leaves(*node.args, **node.kwargs)
        for inp in flattened_node_inps:
            if (
                isinstance(inp, torch.fx.Node)
                and inp.name not in (self.lifted_constant_names or ())
                and self.env[inp] != self.deferred_value
            ):
                return self.unknown_value
        return self.deferred_value

    def is_impure(self, node: torch.fx.node.Node) -> bool:
        def is_woq_int8_pattern(node: torch.fx.node.Node) -> bool:
            return (
                node.target == torch.ops.prims.convert_element_type.default  # type: ignore[return-value]
                and isinstance(node.args[0], torch.fx.Node)
                and "val" in node.args[0].meta
                and node.args[0].meta["val"].dtype == torch.int8  # type: ignore[union-attr]
                and node.args[1] == torch.bfloat16
            )

        if (
            is_woq_int8_pattern(node)
            or (
                node.target == torch.ops.aten.permute.default
                and len(node.users) == 1
                and is_woq_int8_pattern(next(iter(node.users)))
            )
        ) and is_const_source(
            node.args[0],  # type: ignore[arg-type]
            self.lifted_constant_names,
        ):
            # Case 1: int8_weight -> dq -> bf16_weight
            # Case 2: int8_weight -> permute -> dq -> bf16_weight
            return True

        quant_registered = (
            getattr(torch.ops.quantized_decomposed, "dequantize_per_channel", None)
            is not None
        )
        if quant_registered and node.target in [
            torch.ops.quantized_decomposed.dequantize_per_channel.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
            torch.ops.quantized_decomposed.convert_element_type.no_fuse,
        ]:
            # For the pattern fp32_weight -> q -> dq
            # We only folding fp32_weight -> q
            # int8_weight and leave dq in graph to be fused
            return True
        return False

    def node_to_last_non_output_use(self) -> dict[torch.fx.Node, list[torch.fx.Node]]:
        last_non_output_use = collections.defaultdict(list)
        seen_uses = OrderedSet[torch.fx.Node]()
        output_node = next(iter(reversed(self.module.graph.nodes)))  # type: ignore[arg-type, union-attr]

        for node in reversed(self.module.graph.nodes):  # type: ignore[arg-type, union-attr]
            if node.target == "output":
                continue

            def add_use(inp: torch.fx.Node) -> None:
                if inp in seen_uses:
                    return

                seen_uses.add(inp)
                last_non_output_use[node].append(inp)

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, add_use, (node.args, node.kwargs))

            # if this node is only used in output, we want to gc it right away
            if len(node.users) == 1 and output_node in node.users:
                last_non_output_use[node].append(node)

        return last_non_output_use

    def run_node(self, node: torch.fx.Node) -> Any:
        if node.target == "output":
            # because we remove nodes from env on last non output use,
            # re-define them now or we'll get error in interpreter
            def set_env(arg: torch.fx.Node) -> None:
                self.env[arg] = self.unknown_value

            # In-place is fine since we don't mutate
            pytree.tree_map_only_(torch.fx.Node, set_env, node.args)
            return super().run_node(node)

        args, kwargs = self.fetch_args_kwargs_from_env(node)
        flattened_inputs = pytree.arg_tree_leaves(*args, **kwargs)

        # We need to do this weird thing because in cases where flattened_inputs
        # contains a ScriptObject, equality checking results in a type error if
        # the types are different.
        if any(
            type(self.unknown_value) == type(input_) and self.unknown_value == input_
            for input_ in flattened_inputs
        ):
            return self.unknown_value

        # TODO - fix errors with this
        if (
            node.op == "call_function"
            and node.target == aten._efficientzerotensor.default
        ):
            return self.unknown_value

        # TODO - constant folding triton kernel returns the inputs -- fix this
        if (
            node.op == "call_function"
            and node.name == "triton_kernel_wrapper_functional_proxy"
        ):
            return self.unknown_value

        # skip constructors, since inductor generates optimal code for them already
        # and turning into tensor would result in an additional global memory read
        # TODO - more complicated strategy
        if (
            self.skip_constructors
            and not is_const_source(node, self.lifted_constant_names)
            and not any(isinstance(e, torch.Tensor) for e in flattened_inputs)
        ):
            return self.unknown_value

        # All mutations should either be removed or on inputs which we did not make constant
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return self.unknown_value

        if node.op == "call_function" and isinstance(
            node.target, torch._ops.HigherOrderOperator
        ):
            return self.unknown_value

        out = self._deduce_value(node)

        if isinstance(out, torch._C.ScriptObject):
            return out

        if out == self.unknown_value:
            return self.unknown_value

        if not is_const_source(node, self.lifted_constant_names) and (
            isinstance(out, torch.Tensor) or out == self.deferred_value
        ):
            if out != self.deferred_value and out.device.type == "meta":
                return out

            if not self.insertable_tensor_check(out):
                return out

            if self.is_impure(node):
                return self.unknown_value

            self.add_node_replacement(node, out)

            flattened_node_inps = pytree.arg_tree_leaves(*node.args, **node.kwargs)

            for n in flattened_node_inps:
                if not isinstance(n, torch.fx.Node):
                    continue

                self.replaced_uses[n] += 1

            for to_delete in self.user_to_last_uses.get(node, []):
                if self.replaced_uses[to_delete] == len(to_delete.users):
                    self.node_replacements.pop(to_delete, None)

        return out

    def insertable_tensor_check(self, tensor: torch.Tensor) -> bool:
        return True

    def add_node_replacement(self, node: torch.fx.Node, tensor: torch.Tensor) -> None:
        self.node_replacements[node] = tensor

    def run(self) -> Any:  # type: ignore[override]
        env: dict[torch.fx.Node, Any] = {}
        self.insert_placerholder_values(env)
        return super().run(initial_env=env)

    def insert_placerholder_values(self, env: dict[torch.fx.Node, Any]) -> None:
        for n in self.module.graph.find_nodes(op="placeholder"):  # type: ignore[operator, union-attr]
            env[n] = self.unknown_value  # type: ignore[assignment]
        if self.lifted_constant_names is None:
            return
        for n in self.module.graph.nodes:  # type: ignore[union-attr]
            if n.name in (self.lifted_constant_names or ()):
                env[n] = self.deferred_value


def constant_fold(
    gm: torch.fx.GraphModule,
    constraint_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> None:
    with torch.utils._python_dispatch._disable_current_modes():
        cf = ConstantFolder(gm, skip_constructors=True)
        cf.run()

        for node, constant in cf.node_replacements.items():
            if constraint_fn is not None and not constraint_fn(node):
                continue
            replace_node_with_constant(gm, node, constant)

        erased_params = []
        for node in gm.graph.find_nodes(op="get_attr"):
            if len(node.users) == 0:
                if hasattr(gm, node.target):
                    delattr(gm, node.target)
                erased_params.append(node)

        for node in erased_params:
            gm.graph.erase_node(node)

        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()


def constant_graph_tag(
    gm: torch.fx.GraphModule,
    skip_constructors: bool = True,
    lifted_constant_names: Optional[list[str]] = None,
    skip_folding_node_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> None:
    with torch.utils._python_dispatch._disable_current_modes():
        cf = ConstantFolder(
            gm,
            skip_constructors=skip_constructors,
            lifted_constant_names=lifted_constant_names,
            skip_folding_node_fn=skip_folding_node_fn,
        )
        cf.run()

        for node in gm.graph.nodes:
            if skip_folding_node_fn is not None and skip_folding_node_fn(node):
                node.meta[META_TAG] = MODULE_TAG
                continue
            if (
                is_const_source(node, lifted_constant_names)
                or node in cf.node_replacements
                or node in cf.replaced_uses
            ):
                node.meta[META_TAG] = CONST_MODULE_TAG
            else:
                node.meta[META_TAG] = MODULE_TAG


def run_and_get_constant_graph(
    gm: torch.fx.GraphModule,
    skip_constructors: bool = True,
    lifted_constant_names: Optional[list[str]] = None,
    skip_folding_node_fn: Optional[Callable[[torch.fx.Node], bool]] = None,
) -> torch.fx.GraphModule:
    """
    Construct a GraphModule which corresponds to the part which could be
    constant folded in provided gm.
    """

    constant_graph_tag(
        gm, skip_constructors, lifted_constant_names, skip_folding_node_fn
    )

    def untag(node: torch.fx.Node) -> bool:
        used_to_fold = False
        for u in node.users:
            if u.meta[META_TAG] == CONST_MODULE_TAG:
                used_to_fold = True
                break
        if not used_to_fold:
            node.meta[META_TAG] = MODULE_TAG
        return used_to_fold

    # We rewrite the tags, if it's a constant being directly consumed, without
    # any folding opportunity, we keep it in main gm.
    for node in gm.graph.nodes:
        if node.op == "get_attr" or (node.name in (lifted_constant_names or ())):
            untag(node)

    new_graph = torch.fx.Graph()

    node_remapping: dict[torch.fx.Node, torch.fx.Node] = {}
    output_nodes = []
    for node in gm.graph.nodes:
        if node.meta[META_TAG] == MODULE_TAG:
            continue

        new_node = new_graph.node_copy(node, lambda x: node_remapping[x])
        node_remapping[node] = new_node

        for user in node.users:
            if user.meta[META_TAG] == MODULE_TAG:
                output_nodes.append(new_node)
                break

    new_graph.output(tuple(output_nodes))
    new_graph.lint()
    new_gm = torch.fx.GraphModule(gm, new_graph)

    return new_gm
