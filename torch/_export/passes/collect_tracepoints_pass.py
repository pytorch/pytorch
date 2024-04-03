import operator

import torch

from torch.export.exported_program import ConstantArgument, TensorArgument
from torch.fx.passes.infra.pass_base import PassBase, PassResult

__all__ = ["CollectTracepointsPass"]


class CollectTracepointsPass(PassBase):
    """
    Performs constant folding and constant propagation.
    """

    def __init__(self, specs, sig) -> None:
        super().__init__()
        self.specs = specs
        self.sig = sig

    def call(self, gm):
        def get_arg_spec(arg):
            if isinstance(arg, torch.fx.Node):
                if isinstance(arg.meta.get("val"), torch.Tensor):
                    return TensorArgument(name=arg.name)
                else:
                    raise AssertionError(
                        "Symint input is not implemented yet for submodule call signature."
                    )
            else:
                return ConstantArgument(value=arg)

        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target == torch.ops.higher_order._export_tracepoint:
                    for i, arg in enumerate(node.args):
                        kind = node.kwargs["kind"]
                        if kind == "module_call_inputs":
                            self.specs[node.kwargs["path"]].inputs.append(
                                get_arg_spec(arg)
                            )
                        elif kind == "module_call_outputs":
                            self.specs[node.kwargs["path"]].outputs.append(
                                get_arg_spec(arg)
                            )
                        else:
                            raise AssertionError(f"Unknown tracepoint kind: {kind}")
                        if isinstance(arg, torch.fx.Node):
                            for user in node.users:
                                assert user.op == "call_function"
                                assert user.target == operator.getitem
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
