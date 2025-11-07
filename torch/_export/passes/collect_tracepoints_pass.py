# mypy: allow-untyped-defs
from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import torch
from torch.export.exported_program import ConstantArgument, TensorArgument
from torch.fx.passes.infra.pass_base import PassBase, PassResult


if TYPE_CHECKING:
    from torch.export.exported_program import ModuleCallSignature
    from torch.export.graph_signature import ExportGraphSignature


__all__ = ["CollectTracepointsPass"]


class CollectTracepointsPass(PassBase):
    """
    Performs constant folding and constant propagation.
    """

    def __init__(
        self, specs: dict[str, ModuleCallSignature], sig: ExportGraphSignature
    ) -> None:
        super().__init__()
        self.specs = specs
        self.sig = sig

    def call(self, gm: torch.fx.GraphModule) -> PassResult | None:
        def get_arg_spec(arg) -> TensorArgument | ConstantArgument:
            if isinstance(arg, torch.fx.Node):
                if isinstance(arg.meta.get("val"), torch.Tensor):
                    return TensorArgument(name=arg.name)
                else:
                    raise AssertionError(
                        "Symint input is not implemented yet for submodule call signature."
                    )
            else:
                return ConstantArgument(name="", value=arg)

        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            nn_module_stack = None
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target is torch.ops.higher_order._export_tracepoint:
                    kind = node.kwargs["kind"]
                    if kind == "module_call_outputs":
                        nn_module_stack = node.meta["nn_module_stack"]
                    elif kind == "module_call_inputs":
                        nn_module_stack = None
                    else:
                        raise AssertionError(f"Unknown tracepoint kind: {kind}")
                elif node.meta["nn_module_stack"] == nn_module_stack:
                    node.meta["nn_module_stack"].popitem()
                else:
                    nn_module_stack = None
            nn_module_stack = None
            for node in reversed(module.graph.nodes):
                if node.op != "call_function":
                    continue
                if node.target is torch.ops.higher_order._export_tracepoint:
                    kind = node.kwargs["kind"]
                    if kind == "module_call_inputs":
                        nn_module_stack = node.meta["nn_module_stack"]
                    elif kind == "module_call_outputs":
                        nn_module_stack = None
                    else:
                        raise AssertionError(f"Unknown tracepoint kind: {kind}")
                elif node.meta["nn_module_stack"] == nn_module_stack:
                    node.meta["nn_module_stack"].popitem()
                else:
                    nn_module_stack = None

        def copy_sig(sig) -> ModuleCallSignature:
            from torch.export.exported_program import ModuleCallSignature

            return ModuleCallSignature(
                inputs=[],
                outputs=[],
                in_spec=sig.in_spec,
                out_spec=sig.out_spec,
                forward_arg_names=None,
            )

        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target is torch.ops.higher_order._export_tracepoint:
                    # There's some subtlety worth noting. Here fqn corresponds to
                    # the call name, whereas path corresponds to the module name.
                    # They are not necessarily the same! When a submodule is shared
                    # through different aliases, there are as many _export_tracepoint
                    # markers as there are aliases, since the shared submodule is
                    # wrapped once for each alias.
                    path = node.kwargs["path"]
                    fqn, _ = next(reversed(node.meta["nn_module_stack"].values()))

                    module_key = next(reversed(node.meta["nn_module_stack"]))
                    if "@" in module_key:
                        suffix = module_key.split("@")[-1]
                        path = f"{path}@{suffix}"

                        call_fqn = f"{fqn}@{suffix}"
                        if call_fqn not in self.specs:
                            self.specs[call_fqn] = copy_sig(self.specs[fqn])
                        fqn = call_fqn

                    kind = node.kwargs["kind"]
                    for i, arg in enumerate(node.args):
                        # We only update the signature of the alias used to call
                        # the submodule. Otherwise the signatures of all aliases
                        # would get conflated; the inputs/outputs of every call
                        # would be recorded in every other call as well.
                        if fqn == path:
                            if kind == "module_call_inputs":
                                self.specs[path].inputs.append(get_arg_spec(arg))
                            elif kind == "module_call_outputs":
                                self.specs[path].outputs.append(get_arg_spec(arg))
                            else:
                                raise AssertionError(f"Unknown tracepoint kind: {kind}")
                        if isinstance(arg, torch.fx.Node):
                            for user in node.users:
                                assert user.op == "call_function"
                                assert user.target is operator.getitem
                                assert isinstance(user.args[1], int)
                                if user.args[1] == i:
                                    user.replace_all_uses_with(arg)
                                    self.sig.replace_all_uses(user.name, arg.name)
                                    break
                    users = list(node.users)
                    for user in users:
                        assert len(user.users) == 0
                        gm.graph.erase_node(user)
                    gm.graph.erase_node(node)
            return PassResult(gm, True)

        return None
