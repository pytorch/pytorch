from typing import Tuple, Dict, Optional, List

import torch
from torch._export import export
from torch._export.pass_base import ExportPassBase
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor
from torch.fx.node import Target, Argument
from torch.library import Library
from torch.utils._pytree import tree_unflatten
import torch._export.exported_program as ep

lib = Library("aten", "FRAGMENT")
impl_lib = Library("aten", "IMPL")


class GraphModuleOpUpgrader:
    """Given model op set version number as well as compiler op set version number, do the following:
    1. retrieve upgraders from somewhere (TorchScript API or new API) and pass it into this upgrader.
    2. parse it and reorder for upgrading purpose.
    3. register old versions of operators as custom ops.
    4. prepare upgrader passes.

    In `upgrade()` API run these upgrader passes.

    An example of op_upgraders:
    {
        "aten::div__Scalar_0_3": (                              # versioned op name
            "div._Scalar(self: Tensor, other: Scalar)",         # old schema
            '''
            def div__Scalar_0_3(self: Tensor, other: number) -> Tensor:     # upgrader in literal string
              if (self.is_floating_point() or isinstance(other, float)):
                return self.true_divide_(other)
              return self.divide_(other, rounding_mode='trunc')
            ''',
        ),
    },
    """

    class UpgraderPass(ExportPassBase):
        def __init__(self, old_target: Target, new_target: Target):
            super().__init__()
            self.old_target = old_target
            self.new_target = new_target

        def call_operator(
                self,
                op,
                args: Tuple[Argument, ...],
                kwargs: Dict[str, Argument],
                meta: NodeMetadata,
        ) -> ProxyValue:
            if op == self.old_target:
                return super().call_operator(self.new_target, args, kwargs, meta)
            return super().call_operator(op, args, kwargs, meta)

    def __init__(
        self,
        compiler_opset_version: Optional[Dict[str, int]] = None,
        model_opset_version: Optional[Dict[str, int]] = None,
        op_upgraders: Optional[Dict[str, Tuple[str, str]]] = None,
        # TODO(larryliu): can add a new TS API: torch._C._get_upgraders_entry_map()
    ):
        self.compiler_opset_version = compiler_opset_version if compiler_opset_version else {}
        self.model_opset_version = model_opset_version if model_opset_version else {}
        # key is version number, value is a list of upgraders, keyed on op name
        self.upgraders: List[Tuple[str, str]] = self._parse_upgraders(op_upgraders)

        self.upgrader_passes: List[GraphModuleOpUpgrader.UpgraderPass] = self._populate_passes()

    def _parse_upgraders(self, op_upgraders: Optional[Dict[str, Tuple[str, str]]] = None) -> List[Tuple[str, str]]:
        """reorder op_upgraders by version number."""
        # TODO(larryliu0820): Add support for custom ops
        if not op_upgraders or "aten" not in self.model_opset_version or "aten" not in self.compiler_opset_version:
            return []
        model_ver = self.model_opset_version["aten"]
        curr_ver = self.compiler_opset_version["aten"]

        def get_target_version(versioned_upgrader_name: str) -> int:
            """div_Scalar_0_3 is the name of the upgrader, meaning it applies to opset version 0 to 3 and is
            upgrading to version 4."""
            return int(versioned_upgrader_name.split('_')[-1]) + 1

        versioned_upgraders: Dict[int, Tuple[str, str]] = {get_target_version(name): v for name, v in
                                                           op_upgraders.items()}
        target_upgraders: List[Tuple[str, str]] = []
        for ver in range(model_ver, curr_ver + 1):
            if ver in versioned_upgraders:
                target_upgraders.append(versioned_upgraders[ver])

        return target_upgraders

    def _populate_passes(self) -> List[UpgraderPass]:
        """Given a list of upgraders, loop through it from lower version to higher version and create passes for all
        upgraders. se torch.Library API to register old ops. Op name will be
        <name>_<valid_from_ver>_<valid_till_ver>. Register upgarders as CompositeImplicitAutograd kernels. For example:

        lib = Library("aten", "FRAGMENT")
        lib.define(old_schema)

        impl_lib = Library("aten", "IMPL")
        impl_lib.impl("div__Scalar_0_3", div__Scalar_0_3, "CompositeImplicitAutograd")
        """

        upgrader_passes = []

        def register_old_op(name: str, schema: str, impl_str: str):
            """Registers an old version operator using impl_name as old op name."""
            lib.define(schema)
            exec(impl_str)
            impl_lib.impl(name, locals()[name], "CompositeImplicitAutograd")

        for (schema, upgrader_str) in self.upgraders:
            upgrader_name = upgrader_str.split('(')[0].split(' ')[-1]
            op_name = schema.split('(')[0].split("::")[-1]
            schema = schema.replace(op_name, upgrader_name)
            try:
                register_old_op(name=upgrader_name, schema=schema, impl_str=upgrader_str)
            except RuntimeError as e:
                if "with the same name and overload name multiple times" in str(e):
                    print(f"Registering {upgrader_name} multiple times")
                else:
                    raise RuntimeError from e
            old_op_target = getattr(torch.ops.aten, upgrader_name).default
            op_name, overload_name = (op_name, "default") if "." not in op_name else tuple(op_name.split(".")[:2])
            new_op_target = getattr(getattr(torch.ops.aten, op_name), overload_name)
            # Note that the graph will have op names in the graph, but actually they are of old versions.
            upgrader_passes.append(
                GraphModuleOpUpgrader.UpgraderPass(old_target=new_op_target, new_target=old_op_target))

        return upgrader_passes

    def upgrade(self, exported_program: ep.ExportedProgram) -> ep.ExportedProgram:
        """Run each upgrader and then retrace to decompose it."""
        args = [n.meta["val"] for n in exported_program.graph.nodes if n.op == "placeholder"]
        args_real_tensors = [torch.ones(tuple(arg.size()), dtype=arg.dtype) if isinstance(arg, FakeTensor) else arg for
                             arg in args]
        inputs = tree_unflatten(args_real_tensors, exported_program.call_spec.in_spec)

        for _pass in self.upgrader_passes:
            upgraded_program = exported_program.transform(_pass)
            # NB: we have to retrace the graph_module instead of ep because of some failure.
            exported_program = export(upgraded_program.graph_module, inputs, [])
            exported_program.call_spec = upgraded_program.call_spec

        return exported_program
