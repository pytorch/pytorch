from collections import Counter
from typing import Any, Callable, Dict, Optional

import torch
import torch.utils._pytree as pytree

aten = torch.ops.aten


def return_true(*args, **kwargs) -> bool:
    return True


def replace_node_with_constant(gm, node, constant):
    g = gm.graph

    if not hasattr(gm, "_frozen_param_count"):
        gm._frozen_param_count = 0

    i = gm._frozen_param_count

    while True:
        qualname = f"_frozen_param{i}"
        if not hasattr(gm, qualname):
            break
        i += 1

    gm._frozen_param_count = i + 1

    with g.inserting_before(node):
        new_input_node = g.create_node("get_attr", qualname, (), {})
        node.replace_all_uses_with(new_input_node)
        new_input_node.meta.update(node.meta)
        g.erase_node(node)

    # needed to suppress `does not reference an nn.Module, nn.Parameter, or buffer` warning
    gm.register_buffer(qualname, constant)
    setattr(gm, qualname, constant)


class ConstantFolder(torch.fx.Interpreter):
    def __init__(
        self,
        gm,
        skip_constructors=False,
        insertable_tensor_check: Optional[Callable[[torch.Tensor], bool]] = None,
    ):
        super().__init__(gm)
        self.node_replacements: Dict[torch.fx.Node, Any] = {}
        self.replaced_uses: Dict[torch.fx.Node, int] = Counter()
        self.unknown_value = object()
        self.skip_constructors = skip_constructors
        self.insertable_tensor_check = (
            insertable_tensor_check
            if insertable_tensor_check is not None
            else return_true
        )  # type: ignore[assignment]

    def is_impure(self, node: torch.fx.node.Node):
        if node.target == torch.ops.quantized_decomposed.dequantize_per_channel.default:
            # For the pattern fp32_weight -> quantized_decomposed.quantize_per_channel.default
            # -> quantized_decomposed.dequantize_per_channel.default
            # We only folding fp32_weight -> quantized_decomposed.quantize_per_channel.default into
            # int8_weight and leave quantized_decomposed.dequantize_per_channel.default in graph to be fused
            return True
        return False

    def run_node(self, node):
        aten = torch.ops.aten
        args, kwargs = self.fetch_args_kwargs_from_env(node)

        if node.target == "output":
            return super().run_node(node)

        flattened_inputs = pytree.tree_flatten((args, kwargs))[0]

        if self.unknown_value in flattened_inputs:
            return self.unknown_value

        # TODO - fix errors with this
        if (
            node.op == "call_function"
            and node.target == aten._efficientzerotensor.default
        ):
            return self.unknown_value

        # skip constructors, since inductor generates optimal code for them already
        # and turning into tensor would result in an additional global memory read
        # TODO - more complicated strategy
        if (
            self.skip_constructors
            and node.op != "get_attr"
            and not any(isinstance(e, torch.Tensor) for e in flattened_inputs)
        ):
            return self.unknown_value

        # All mutations should either be removed or on inputs which we did not make constant
        if (
            isinstance(node.target, torch._ops.OpOverload)
            and torch.Tag.nondeterministic_seeded in node.target.tags
        ):
            return self.unknown_value

        out = super().run_node(node)

        if node.op != "get_attr" and isinstance(out, torch.Tensor):
            if not self.insertable_tensor_check(out):  # type: ignore[operator]
                return out

            if self.is_impure(node):
                return self.unknown_value

            self.node_replacements[node] = out

            flattened_node_inps = pytree.tree_flatten((node.args, node.kwargs))[0]

            for n in flattened_node_inps:
                if not isinstance(n, torch.fx.Node):
                    continue

                self.replaced_uses[n] += 1

            for to_delete in self.user_to_last_uses.get(node, []):
                if self.replaced_uses[to_delete] == len(to_delete.users):
                    self.node_replacements.pop(to_delete, None)

        return out

    def run(self):
        env = {}
        for n in self.module.graph.nodes:
            if n.op == "placeholder":
                env[n] = self.unknown_value
        return super().run(initial_env=env)


@torch.utils._python_dispatch._disable_current_modes()
def constant_fold(gm):
    cf = ConstantFolder(gm, skip_constructors=True)
    cf.run()

    for node, constant in cf.node_replacements.items():
        replace_node_with_constant(gm, node, constant)

    erased_params = []
    for node in gm.graph.nodes:
        if node.op == "get_attr" and len(node.users) == 0:
            delattr(gm, node.target)
            erased_params.append(node)

    for node in erased_params:
        gm.graph.erase_node(node)

    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
