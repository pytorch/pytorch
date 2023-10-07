import logging
from collections import defaultdict
from typing import Tuple, Dict, Optional, List

import torch
from torch._export import export
from torch._export.pass_base import _ExportPassBase
from torch._export.pass_infra.node_metadata import NodeMetadata
from torch._export.pass_infra.proxy_value import ProxyValue
from torch._subclasses import FakeTensor
from torch.fx.node import Target, Argument
from torch.library import Library
from torch.utils._pytree import tree_unflatten
import torch._export.exported_program as ep
import re

lib = Library("aten", "FRAGMENT")
impl_lib = Library("aten", "IMPL")

log = logging.getLogger(__name__)


def get_target_version(versioned_upgrader_name: str) -> int:
    """div_Scalar_0_3 is the name of the upgrader, meaning it applies to div.Scalar of version 0 to 3 and is
    upgrading to version 4."""
    if not re.match("^.*_[0-9]+_[0-9]+$", versioned_upgrader_name):
        raise RuntimeError(f"Upgrader name {versioned_upgrader_name} is invalid")

    return int(versioned_upgrader_name.split('_')[-1]) + 1


def get_upgraders() -> Dict[str, Tuple[str, str]]:
    """Getting upgraders entry map and operator version map and merge them into one dict."""
    upgraders = torch._C._get_upgraders_entry_map()
    op_version_map = torch._C._get_operator_version_map()
    output: Dict[str, Tuple[str, str]] = defaultdict(tuple)  # type: ignore[arg-type]
    for opname, entry_list in op_version_map.items():
        if not entry_list:
            raise RuntimeError(f"Op version map has an empty entry for opname {opname}")
        entry = entry_list[0]
        old_schema = entry.old_schema
        upgrader_name = entry.upgrader_name
        upgrader_str = upgraders.get(upgrader_name, None)
        if not upgrader_str:
            raise RuntimeError(f"Can't find upgrader for op {opname} and upgrader name {upgrader_name}")
        output[upgrader_name] = (old_schema, upgrader_str)
    return output


class GraphModuleOpUpgrader:
    """This upgrader is able to upgrade the old version of ops in a given GraphModule, if all upgraders are available.
    To use it, retrieve upgraders from somewhere (TorchScript API or new API) and pass it into this upgrader. In
    __init__() it does the following:
    1. parse the upgrader list and reorder for upgrading purpose.
    2. register old versions of operators as custom ops.
    3. prepare upgrader passes.

    In `upgrade()` API run these upgrader passes.

    An example of op_upgraders input:
    {
        "aten::div__Scalar_0_3": (                              # versioned op name
            "div._Scalar(self: Tensor, other: Scalar)",         # old schema
            '''
            def div__Scalar_0_3(self: torch.Tensor, other) -> torch.Tensor:     # upgrader in literal string
              if (self.is_floating_point() or isinstance(other, float)):
                return self.true_divide_(other)
              return self.divide_(other, rounding_mode='trunc')
            ''',
        ),
    },

    Note that we require the upgrader function to be runnable in Python (which is a stricter requirement than the
    original TorchScript upgrader).
    """

    class UpgraderPass(_ExportPassBase):
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
    ):
        self.op_upgraders: Dict[str, Tuple[str, str]] = get_upgraders() if not op_upgraders else op_upgraders
        self.compiler_opset_version = compiler_opset_version if compiler_opset_version else {}
        self.model_opset_version = model_opset_version if model_opset_version else {}
        self.upgrader_passes: List[GraphModuleOpUpgrader.UpgraderPass] = GraphModuleOpUpgrader._populate_passes(
            self._parse_upgraders(self.op_upgraders))

    def _parse_upgraders(self, op_upgraders: Optional[Dict[str, Tuple[str, str]]] = None) -> List[Tuple[str, str]]:
        """Reorder op_upgraders by version number, return an ordered list of tuples, containing old op schema as well
        as the upgrader function string literal."""
        # TODO(larryliu0820): Add support for custom ops
        op_namespace = "aten"
        if not op_upgraders or op_namespace not in self.model_opset_version or op_namespace not in self.compiler_opset_version:
            return []
        model_ver = self.model_opset_version[op_namespace]
        curr_ver = self.compiler_opset_version[op_namespace]

        # key is the target version. div__Scalar_0_3 should have a key of 4.
        versioned_upgraders: Dict[int, Tuple[str, str]] = {get_target_version(name): v for name, v in
                                                           op_upgraders.items()}
        target_upgraders: List[Tuple[str, str]] = []
        # we need all upgraders from model_ver + 1 to curr_ver, inclusively
        for ver in range(model_ver + 1, curr_ver + 1):
            if ver in versioned_upgraders:
                target_upgraders.append(versioned_upgraders[ver])
            else:
                # we may be able to get away with missing upgraders, if that operator is missing from given graph
                # module.
                log.warning("Missing an upgrader to upgrade to version {ver}.", extra={"ver": ver})

        return target_upgraders

    @staticmethod
    def _populate_passes(upgraders: List[Tuple[str, str]]) -> List[UpgraderPass]:
        """Given a list of upgraders, loop through it from lower version to higher version and create passes for all
        upgraders. se torch.Library API to register old ops. Op name will be
        <name>_<valid_from_ver>_<valid_till_ver>. Register upgarders as CompositeImplicitAutograd kernels. For example:

        lib = Library("aten", "FRAGMENT")
        lib.define(old_schema)

        impl_lib = Library("aten", "IMPL")
        impl_lib.impl("div__Scalar_0_3", div__Scalar_0_3, "CompositeImplicitAutograd")

        @:var upgraders: a list of tuples. The first element of the tuple is the old schema and the second is the
        upgrader function literal text.
        @:return upgrader passes, order matters
        """

        upgrader_passes = []

        def register_old_op(name: str, schema: str, impl_str: str):
            """Registers an old version operator using impl_name as old op name."""
            lib.define(schema)
            try:
                exec(impl_str)
            except Exception as e:
                raise RuntimeError(f"Invalid upgrader string: {impl_str}") from e
            impl_lib.impl(name, locals()[name], "CompositeImplicitAutograd")

        for (schema, upgrader_str) in upgraders:
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
            # for example, the operator instance of "aten::div" is torch.op.aten.div.default. We need to append the
            # "default" at the end.
            op_name, overload_name = (op_name, "default") if "." not in op_name else tuple(op_name.split(".")[:2])
            new_op_target = getattr(getattr(torch.ops.aten, op_name), overload_name)
            # Note that the graph will have op names in the graph, but actually they are of old versions.
            upgrader_passes.append(
                GraphModuleOpUpgrader.UpgraderPass(old_target=new_op_target, new_target=old_op_target))

        return upgrader_passes

    def upgrade(self, exported_program: ep.ExportedProgram) -> ep.ExportedProgram:
        """Run each upgrader pass and then retrace to decompose it. Each upgrader pass replaces the old version of
        operators with a custom operator. The custom operator contains a CompositeImplicitAutograd kernel (the
        upgrading function itself). After retrace, this custom operator will be decomposed into the ops used in the
        upgrader. After all passes are applied, the exported program will be upgraded to the target version."""
        if not self.upgrader_passes:
            return exported_program

        args = [n.meta.get("val", None) for n in exported_program.graph.nodes if n.op == "placeholder"]
        args_real_tensors = [torch.ones(tuple(arg.size()), dtype=arg.dtype) if isinstance(arg, FakeTensor) else arg for
                             arg in args]
        assert exported_program.call_spec.in_spec is not None
        inputs = tree_unflatten(args_real_tensors, exported_program.call_spec.in_spec)

        for _pass in self.upgrader_passes:
            upgraded_program = exported_program._transform(_pass)
            # NB: we have to retrace the graph_module instead of ep because of some failure.
            exported_program = export(upgraded_program.module(), inputs, {})
            exported_program._call_spec = upgraded_program.call_spec

        return exported_program
