# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
import argparse
import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Iterable, Any, List, Dict, Optional, Tuple
import logging

import torch
from torch.fx.passes.graph_manipulation import get_size_of_node
from torch.fx.node import map_arg
from torch.fx._compatibility import compatibility

from .operator_support import (
    get_node_target,
    OperatorSupportBase,
)
from .graph_drawer import FxGraphDrawer
from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
    FxNetAccFusionsFinder,
    CALLABLE_NODE_OPS,
    Tensors,
    NodeList,
    NodeSet,
    is_node_output_tensor,
)


__all__ = ['FxNetAccNodesFinder', 'FxNetSplitterInternalError', 'Subgraph', 'SplitResult', 'generate_inputs_for_submodules']
_LOGGER = logging.getLogger(__name__)

DEFAULT_MIN_ACC_MODULE_SIZE = 1
DEFAULT_SKIP_FUSION = False
DEFAULT_ALLOW_NON_TENSOR = False

class _SplitterSettingBase:
    def __init__(
        self,
        min_acc_module_size=DEFAULT_MIN_ACC_MODULE_SIZE,
        skip_fusion=DEFAULT_SKIP_FUSION,
        allow_non_tensor=DEFAULT_ALLOW_NON_TENSOR,
        max_acc_splits: int = -1,
    ):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--min-acc-module-size",
            "--min_acc_module_size",
            required=False,
            type=int,
            help="Minimum size limit of an accelerator subgraph.",
        )
        parser.add_argument(
            "--max-acc_splits",
            "--max-acc_splits",
            required=False,
            type=int,
            help="Enforce a maximum number of split subgraphs.",
        )
        parser.add_argument(
            "--skip-fusion",
            "--skip_fusion",
            default=False,
            action="store_true",
            help="If true then no fusion groups. Fusion group is used to "
            "enforce no non-tensor data flow between submodules. If we don't "
            "have this constrain, setting this to false is recommended as it "
            "can reduce overhead.",
        )
        parser.add_argument(
            "--allow-non-tensor",
            "--allow_non_tensor",
            default=False,
            action="store_true",
            help="For some backends non-tensor data flow between cpu and them "
            "are not allowed. Therefore, if a node supported by accelerator but "
            "it has non-tensor inputs or outputs to a cpu node we would want to "
            "consider it as a cpu node during splitting. However, for some backends "
            "we might not care about non-tensor data flow and we can set this option "
            "to true to disable the functionality that prevent non-tensor data flow.",
        )
        args, unknown = parser.parse_known_args()

        self.min_acc_module_size: int = args.min_acc_module_size if args.min_acc_module_size else min_acc_module_size
        self.skip_fusion: bool = args.skip_fusion if args.skip_fusion else skip_fusion
        self.allow_non_tensor: bool = args.allow_non_tensor if args.allow_non_tensor else allow_non_tensor
        self.max_acc_splits: int = max_acc_splits


@compatibility(is_backward_compatible=False)
class FxNetAccNodesFinder:
    """
    Finds a set of nodes that can be supported on ACC, excluding nodes that have non-tensor
    input/output to cpu nodes to prevent non-tensor data flow between backends and cpu.

    I.e. if we have a chain:

    ACC_NODE_1 -> ACC_NODE_2 -> ACC_NODE_3 -> CPU_NODE_1

    where every ACC node produces non-tensor output, then they all should be treated as CPU nodes.

    This behavior can be turned off by passing allow_non_tensor=True.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        operator_support: OperatorSupportBase,
        allow_non_tensor: bool,
    ):
        self.module = module
        self.operator_support = operator_support
        self.allow_non_tensor = allow_non_tensor
        self.acc_nodes: NodeSet = set()

    def reduce_acc_nodes_non_tensor_input_helper(
        self, cpu_worklist: NodeList
    ):
        """
        Transitively excludes nodes from ACC supported set.
        For every node in the worklist:
        - removes its downstream ACC nodes from ACC supported set,
        - if any downstream ACC node produces non-tensor output,
          then it gets added into the worklist.
        """
        while cpu_worklist:
            node = cpu_worklist.pop(0)

            for user in node.users:
                if user in self.acc_nodes:
                    self.acc_nodes.remove(user)
                    if not is_node_output_tensor(user):
                        cpu_worklist.append(user)

    def reduce_acc_nodes_non_tensor_input(self):
        """
        Excludes nodes from ACC supported set that have direct
        upstream CPU nodes that produce non-tensor outputs.
        """
        non_tensor_cpu_nodes: NodeList = []

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue
            if node in self.acc_nodes:
                continue
            if is_node_output_tensor(node):
                continue
            non_tensor_cpu_nodes.append(node)

        self.reduce_acc_nodes_non_tensor_input_helper(non_tensor_cpu_nodes)

    def reduce_acc_nodes_non_tensor_output(self):
        """
        Excludes nodes from ACC supported set that produce non-tensor
        outputs and have downstream CPU nodes.
        """
        while True:
            new_cpu_nodes: NodeList = []

            for acc_node in self.acc_nodes:
                if is_node_output_tensor(acc_node):
                    continue
                for user in acc_node.users:
                    if user not in self.acc_nodes:
                        new_cpu_nodes.append(acc_node)
                        break

            if not new_cpu_nodes:
                break

            for new_cpu_node in new_cpu_nodes:
                self.acc_nodes.remove(new_cpu_node)

            self.reduce_acc_nodes_non_tensor_input_helper(new_cpu_nodes)

    def __call__(self) -> NodeSet:
        submodules = dict(self.module.named_modules())
        self.acc_nodes = {
            n
            for n in self.module.graph.nodes
            if n.op in CALLABLE_NODE_OPS
            and self.operator_support.is_node_supported(submodules, n)
        }

        if not self.allow_non_tensor:
            self.reduce_acc_nodes_non_tensor_input()
            self.reduce_acc_nodes_non_tensor_output()

        return self.acc_nodes

@compatibility(is_backward_compatible=False)
class FxNetSplitterInternalError(Exception):
    pass

@compatibility(is_backward_compatible=False)
@dataclass
class Subgraph:
    is_acc: bool
    nodes: NodeList
    device_ordinal: Optional[int] = None

@compatibility(is_backward_compatible=False)
class SplitResult(NamedTuple):
    """
    Stores the results of the splitter.

    Attributes:
        split_module: root module after splitting.
        submodule_inputs: a dict that maps submodule name to its inputs.
        non_acc_submodule_prefix: the prefix for non acc submodules. For
            acc submodule the prefix is alwasy "_run_on_acc_".
    """

    split_module: torch.fx.GraphModule
    submodule_inputs: Dict[str, Any]
    non_acc_submodule_prefix: str


@compatibility(is_backward_compatible=False)
def generate_inputs_for_submodules(
    model: torch.nn.Module,
    inputs: Sequence[Any],
    target_submodules: Iterable[str],
    deepcopy: bool = False,
) -> Dict[str, Any]:
    """
    Generate inputs for targeting submdoules in the given model. Note that if two submodules refer to the same obj, this
    function doesn't work.

    Args:
        model: root model.
        inputs: inputs to the root model.
        target_submodules: submodules that we want to generate inputs for.

    Returns:
        A dict that maps from submodule name to its inputs.
    """

    handles = []
    results = {}
    submodule_to_names = {mod: name for name, mod in model.named_modules()}

    def pre_forward(module, module_inputs):
        results[submodule_to_names[module]] = copy.deepcopy(module_inputs) if deepcopy else module_inputs

    for name, mod in model.named_modules():
        if name in target_submodules:
            handles.append(mod.register_forward_pre_hook(pre_forward))

    def clean_up_handles():
        for h in handles:
            h.remove()

    try:
        with torch.no_grad():
            model(*inputs)
    except Exception as e:
        clean_up_handles()
        raise e

    clean_up_handles()
    return results


class _SplitterBase:
    """
    Splits a GraphModule into sub-GraphModules for execution on CPU or the accelerator.
    Output is a GraphModule with supported and unsupported operators grouped into as few sub-GraphModules as possible.
    Assumes that only "call_module", "call_function" and "call_method" from FX IR can potentially be executed on the accelerator.

    Given the following graph:
          ==> b ==>
        //         \\
       a             d
        \\         //
          ==> c ==>

    class SimpleModule(torch.nn.Module):
        def forward(self, a):
            b = torch.sin(a)
            c = torch.cos(a)
            d = b + c
            return d

    and providing "operator_support" that indicates that 'b' and 'c' can be executed on the accelerator,
    we will get the following split result:

    main:
    def forward(self, a):
        run_on_acc_0_0 = self._run_on_acc_0_0(a)
        getitem = run_on_acc_0_0[0]
        getitem_1 = run_on_acc_0_0[1]
        run_on_cpu_1_1 = self._run_on_cpu_1_1(getitem, getitem_1)
        return run_on_cpu_1_1

    _run_on_acc_0_0:
    def forward(self, a):
        sin_1 = torch.sin(a)
        cos_1 = torch.cos(a)
        return (sin_1, cos_1)

    _run_on_cpu_1_1:
    def forward(self, sin_1, cos_1):
        add_1 = sin_1 + cos_1
        return add_1
    """

    # PCIe bandwidth for the backend, default to 100 GB/s
    PCIe_BW = 100 * 2 ** 30

    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Sequence[Any],
        operator_support: OperatorSupportBase,
        settings: _SplitterSettingBase,
        non_acc_submodule_name: str = "_run_on_cpu_",
        return_tuple: bool = False,
    ):
        """
        Preprocesses graph before splitting:
        - finds nodes supported by ACC,
        - finds fusion groups for ACC nodes having non-tensor IO,
        - builds a graph of direct dependencies,
        - builds a map of fused nodes to their fusions.
        As a result we get self.acc_nodes, self.deps and self.fusions.
        """
        assert isinstance(module, torch.fx.GraphModule)

        self.module = module
        ShapeProp(self.module).propagate(*sample_input)

        self.settings = settings
        self.operator_support = operator_support
        self.sample_input = sample_input
        self.acc_nodes = FxNetAccNodesFinder(self.module, self.operator_support, self.settings.allow_non_tensor)()

        if self.settings.skip_fusion:
            self.fusions = {}
        else:
            self.fusions = FxNetAccFusionsFinder(module, self.acc_nodes)()

        # Modify deps to add more deps for fused nodes
        self.deps = self.find_deps()
        self.update_deps_for_fusions()

        self.non_acc_submodule_name = non_acc_submodule_name
        self._node_submodule_map: Dict[str, str] = {}
        self._return_tuple = return_tuple

        self.tags: List[str] = []

    # ===============================================================
    # Helpers for ctor and initial state
    # ===============================================================

    def get_node_submodule_map(self) -> Dict[str, str]:
        """ Returns a map from node name to submodule name, e.g.
            node: main_module_impl_impl_over_arch_unary_multiple_embedding
              _pooling_embedding_pooling_sparse_entity_equivalence_key
              _proxy_embedding_bag
            maps to submodule name of: _run_on_acc_1
        """
        return self._node_submodule_map

    def find_deps(self) -> Dict[torch.fx.Node, NodeSet]:
        """
        Builds a graph of node dependencies. Leaf nodes don't have any
        dependencies and the "output" node doesn't have nodes depending on it.

        Resulting graph has only direct dependencies, i.e. there are no
        transitive dependencies.
        """
        deps: Dict[torch.fx.Node, NodeSet] = defaultdict(set)
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            for user in node.users:
                if user.op != "output":
                    deps[user].add(node)
        return deps

    def update_deps_for_fusions(self):
        """
        Updates graph of dependencies so that:
        - nodes from the same fusion depend on the same set of outer nodes,
        - outer nodes depending on a fusion depend on all nodes in that fusion.
        """
        for node in self.fusions:
            fusion = self.fusions[node]
            for fused_neighbor in fusion:
                self.deps[node].update(self.deps[fused_neighbor] - fusion)

                for user in fused_neighbor.users:
                    if user not in fusion:
                        self.deps[user].add(node)

    # ===============================================================
    # Helpers for preview
    # ===============================================================

    def _lower_model_to_backend(
        self, mod: torch.fx.GraphModule, inputs: Tensors
    ) -> torch.nn.Module:
        """
        Lower the model to a backend.
        """

        return mod

    def _find_culprit(
        self, mod: torch.fx.GraphModule, inputs: Tensors
    ) -> str:
        """
        When an error occurs during lowering or running the lowered mod, we use this
        function to find culprits in the `mod` that causes the error.
        """

        return "Unable to find a culprit because _find_culprit() function is not implemented."

    def _draw_graph_based_on_node_support(
        self, mod: torch.fx.GraphModule, supported_nodes: NodeList
    ):
        color_map = {
            "default": "AliceBlue",
            "supported": "chartreuse1",
            "unsupported": "crimson",
        }

        class CustomDrawer(FxGraphDrawer):
            def _get_node_style(self, node):
                template = super()._get_node_style(node)
                if node in supported_nodes:
                    template["fillcolor"] = color_map["supported"]
                elif node.op in CALLABLE_NODE_OPS:
                    template["fillcolor"] = color_map["unsupported"]
                else:
                    template["fillcolor"] = color_map["default"]

                return template

        drawer = CustomDrawer(mod, "node_support", ignore_getattr=True)
        dot_graph = drawer.get_main_dot_graph()
        # pyre-fixme[16]: `pydot.Dot` has no attribute `write_raw`.
        dot_graph.write_raw("node_support.dot")

    def node_support_preview(self, dump_graph: bool = False):
        submodules = dict(self.module.named_modules())

        supported_nodes: NodeList = []
        supported_node_types = defaultdict(set)
        unsupported_node_types = defaultdict(set)

        def get_dtype(arg):
            tensor_meta = arg.meta.get("tensor_meta")
            return getattr(tensor_meta, "dtype", None)

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            target = get_node_target(submodules, node)

            # Store dtype of arg in node.args. If arg doesn't have dtype, i.e. not a tensor, we'll store None.
            arg_dtypes = [
                get_dtype(arg) if isinstance(arg, torch.fx.Node) else None
                for arg in node.args
            ]

            # Find last non-None element. If all elements are None, return max_len.
            last_index = len(arg_dtypes) - next(
                (
                    i
                    for i, dtype in enumerate(reversed(arg_dtypes))
                    if dtype is not None
                ),
                len(arg_dtypes),
            )

            # Strip None elements at the end.
            arg_dtypes_tuple = tuple(arg_dtypes[:last_index])
            kwarg_dtypes_tuple = tuple(
                (k, get_dtype(arg))
                for k, arg in node.kwargs.items()
                if isinstance(arg, torch.fx.Node)
            )

            if self.operator_support.is_node_supported(submodules, node):
                supported_nodes.append(node)
                supported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))
            else:
                unsupported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))

        if dump_graph:
            self._draw_graph_based_on_node_support(self.module, supported_nodes)

        reports = "\nSupported node types in the model:\n"
        for t, dtypes in supported_node_types.items():
            for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
                reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

        reports += "\nUnsupported node types in the model:\n"
        for t, dtypes in unsupported_node_types.items():
            for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
                reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

        print(reports)

        # Return reports for testing purpose
        return reports

    def split_preview(self, dump_graph: bool = False):
        reports = ""
        subgraphs = self.put_nodes_into_subgraphs()
        acc_subgraphs_num = len([g for g in subgraphs if g.is_acc])
        cpu_subgraphs_num = len(subgraphs) - acc_subgraphs_num
        reports += f"Before removing small acc subgraphs, total {len(subgraphs)} subgraphs are created:"
        reports += f" {acc_subgraphs_num} acc subgraphs and {cpu_subgraphs_num} cpu subgraphs.\n"

        subgraphs = self.remove_small_acc_subgraphs(subgraphs)
        acc_subgraphs_num = len([g for g in subgraphs if g.is_acc])
        cpu_subgraphs_num = len(subgraphs) - acc_subgraphs_num
        reports += f"After removing small acc subgraphs, total {len(subgraphs)} subgraphs are created:"
        reports += f" {acc_subgraphs_num} acc subgraphs and {cpu_subgraphs_num} cpu subgraphs.\n"

        for i, subgraph in enumerate(subgraphs):
            reports += f"_run_on_acc_{i}: " if subgraph.is_acc else f"{self.non_acc_submodule_name}{i}: "
            reports += f"{len(subgraph.nodes)} node(s)\n"

        self.tag(subgraphs)
        split_mod = self.split(remove_tag=True)
        split_mod.eval()

        if dump_graph:
            drawer = FxGraphDrawer(
                split_mod, "preview", ignore_getattr=True
            )
            dot_graphs = drawer.get_all_dot_graphs()
            for name, dot_graph in dot_graphs.items():
                # pyre-fixme[16]: `pydot.Dot` has no attribute `write_raw`.
                dot_graph.write_raw(f"{name}.dot")

        max_qps: float = self.PCIe_BW
        bottleneck_module = ""

        for node in split_mod.graph.nodes:
            if node.op == "call_module" and "acc" in node.target:
                reports += f"\nProcessing acc submodule {node.target}\n"

                submod = getattr(split_mod, node.target)

                def get_submod_inputs(main_mod, submod, example_inputs):
                    sub_inputs = None

                    def get_inputs(self, inputs):
                        nonlocal sub_inputs
                        sub_inputs = inputs

                    handle = submod.register_forward_pre_hook(get_inputs)
                    main_mod(*example_inputs)
                    handle.remove()
                    return sub_inputs

                submod_inputs = get_submod_inputs(
                    split_mod, submod, self.sample_input
                )
                ShapeProp(submod).propagate(*submod_inputs)

                total_input_bytes = 0
                total_output_bytes = 0

                reports += "Checking inputs...\n"
                for n in submod.graph.nodes:
                    if n.op == "placeholder":
                        if not is_node_output_tensor(n):
                            reports += f"Input {n.name} is not a tensor, this might cause problems during lowering!\n"
                        else:
                            total_input_bytes += get_size_of_node(submod, n)[0]
                    if n.op == "output":
                        output_node = n

                reports += "Checking outputs...\n"

                def get_bytes(node: torch.fx.Node):
                    nonlocal total_output_bytes
                    nonlocal reports
                    if not is_node_output_tensor(node):
                        reports += f"Output {node.name} is not a tensor, this might cause problems during lowering!\n"
                    else:
                        total_output_bytes += get_size_of_node(submod, node)[0]

                map_arg(output_node.args, get_bytes)  # type: ignore[possibly-undefined]
                qps = self.PCIe_BW / max(total_input_bytes, total_output_bytes)
                reports += f"Total input size in bytes is {total_input_bytes}, total output size in bytes is {total_output_bytes},"
                reports += f" theoretical max qps (bounds by PCIe bandwidth) for this submodule is {qps}.\n"

                if qps < max_qps:
                    max_qps = qps
                    bottleneck_module = node.target

                try:
                    lowered_submod = self._lower_model_to_backend(submod, submod_inputs)
                except RuntimeError:
                    reports += "Run into an error during lowering!\n"
                    reports += self._find_culprit(submod, submod_inputs)
                    continue

                try:
                    lowered_submod(*submod_inputs)
                except RuntimeError:
                    reports += "Run into an error during inference!\n"
                    reports += self._find_culprit(submod, submod_inputs)
                else:
                    reports += "Lowering and running succeed!\n"

        reports += f"\nTheoretical max qps (bounds by PCIe bandwidth) for this model is {max_qps},"
        reports += f" bottleneck is submodule {bottleneck_module}."
        print(reports)

        # return the reports for testing purposes
        return reports

    # ===============================================================
    # Helpers for extend_acc_subgraph() method
    # ===============================================================

    def find_reverse_deps(
        self, tag_id: Optional[int] = None
    ) -> Dict[torch.fx.Node, NodeSet]:
        """
        Builds reversed topological node dependencies, if tag_id is specified,
        we ignore nodes that are in later subgraph i.e. nodes have greater tag_id.
        """
        result: Dict[torch.fx.Node, NodeSet] = defaultdict(set)

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue

                if tag_id is None or (int(user.tag.split("_")[-1]) < tag_id):
                    result[node].add(user)

        return result

    def update_reverse_deps_for_fusions(
        self, deps: Dict[torch.fx.Node, NodeSet]
    ):
        processed_node = set()

        for node, fusion in self.fusions.items():
            if node in processed_node:
                continue

            new_dep = set()

            # Create a new dependency set which include all the
            # dependencies of the nodes in the fusion group
            for n in fusion:
                new_dep.update(deps[n])

            # Exclude nodes in the fusion
            new_dep.difference_update(fusion)

            # Update dependency
            for n in fusion:
                deps[n] = new_dep

                for arg in n.all_input_nodes:
                    if arg not in fusion:
                        deps[arg].update(fusion)

                processed_node.add(n)

    def find_parent_nodes_of_subgraph(self, tag: str) -> NodeSet:
        """
        Finds parent nodes of the `tag` subgraph.

        Traverse the inputs of nodes in the subgraph, if input doesn't belong to the subgraph
        and is not a placeholder, we consider it as the parent node of the subgraph.
        """
        parent_nodes = set()

        for node in self.module.graph.nodes:
            if node.op in CALLABLE_NODE_OPS and node.tag == tag:
                for arg in node.all_input_nodes:
                    if arg.op in CALLABLE_NODE_OPS and arg.tag != tag:
                        parent_nodes.add(arg)

        return parent_nodes

    def extend_acc_subgraph(self, tag: str):
        """
        Extend the acc subgraph with `tag` going the reversed topological direction.
        """
        # Dict that maps node to its users and ignore users that
        # are in the subgraph that has greater tag
        deps = self.find_reverse_deps(tag_id=int(tag.split("_")[-1]))
        self.update_reverse_deps_for_fusions(deps)

        # Parent nodes of the subgraph
        parent_nodes = self.find_parent_nodes_of_subgraph(tag)

        visited_nodes: NodeSet = set()

        while parent_nodes:
            node = None

            # Find a acc node that depends on visited nodes only
            for n in parent_nodes:
                if deps[n] <= visited_nodes and n in self.acc_nodes:
                    node = n
                    break

            if node is None:
                break

            # Put the node into `tag` subgraph
            node.tag = tag  # type: ignore[attr-defined]
            parent_nodes.remove(node)
            visited_nodes.add(node)

            # If node is in a fusion group, add all fusion buddies to parent nodes
            if node in self.fusions:
                for fusion_node in self.fusions[node]:
                    if fusion_node not in visited_nodes:
                        parent_nodes.add(fusion_node)

            # Add inputs of the node to parent nodes
            for arg in node.all_input_nodes:
                if arg.op in CALLABLE_NODE_OPS and arg not in visited_nodes:
                    parent_nodes.add(arg)

    # ===============================================================
    # Helpers for split() method
    # ===============================================================

    def starter_nodes(self) -> Tuple[NodeSet, NodeSet]:
        """
        Finds nodes that consume module inputs or get_attr nodes.
        """
        starter_cpu_nodes: NodeSet = set()
        starter_acc_nodes: NodeSet = set()
        for node in self.module.graph.nodes:
            if node.op not in {"placeholder", "get_attr"}:
                continue
            for user in node.users:
                if user in self.acc_nodes:
                    starter_acc_nodes.add(user)
                else:
                    starter_cpu_nodes.add(user)
        return starter_cpu_nodes, starter_acc_nodes

    def put_nodes_into_subgraphs(self) -> List[Subgraph]:
        # We start graph traversal from leaf nodes
        current_cpu_nodes, current_acc_nodes = self.starter_nodes()
        visited_nodes: NodeSet = set()

        # Determine which subgraph to start from based on which subgraph has
        # 0-dep node
        acc_subgraph: bool = not any(len(self.deps[n]) == 0 for n in current_cpu_nodes)

        current_subgraph_nodes: NodeList = []

        # Result accumulator
        subgraphs: List[Subgraph] = []
        while current_cpu_nodes or current_acc_nodes:
            # Find the first node that should belong to the current subgraph and has all dependencies resolved
            current_nodes = current_acc_nodes if acc_subgraph else current_cpu_nodes
            node = next(
                (n for n in current_nodes if self.deps[n] <= visited_nodes),
                None,
            )

            # If nothing was found, then it's time to flip the mode and start a new subgraph
            if node is None:
                if not current_subgraph_nodes:
                    raise FxNetSplitterInternalError("Subgraph can't be empty")

                subgraphs.append(
                    Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes)
                )
                acc_subgraph = not acc_subgraph
                current_subgraph_nodes = []
                continue

            current_nodes.remove(node)
            visited_nodes.add(node)
            current_subgraph_nodes.append(node)

            # Add fusion buddies
            if node in self.fusions:
                if node in self.acc_nodes:
                    current_acc_nodes.update(self.fusions[node] - visited_nodes)
                else:
                    current_cpu_nodes.update(self.fusions[node] - visited_nodes)

            # Put depending nodes into the queue
            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue

                # Add downstream nodes
                if user in self.acc_nodes:
                    current_acc_nodes.add(user)
                else:
                    current_cpu_nodes.add(user)

        # Check if the last subgraph was not created
        if current_subgraph_nodes:
            subgraphs.append(
                Subgraph(is_acc=acc_subgraph, nodes=current_subgraph_nodes)
            )

        if not subgraphs:
            raise FxNetSplitterInternalError("Couldn't create subgraphs")

        return subgraphs

    def remove_small_acc_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
        """
        This pass finds ACC submodules with less than specified size and merges
        them with adjacent CPU submodules.
        """
        result: List[Subgraph] = []
        for subgraph in subgraphs:
            if subgraph.is_acc:
                if len(subgraph.nodes) >= self.settings.min_acc_module_size:
                    result.append(subgraph)
                else:
                    print(
                        "Eliminating acc subgraph because it's smaller than the threshold: "
                        f"{len(subgraph.nodes)} < {self.settings.min_acc_module_size}"
                    )
                    if result:
                        result[-1].nodes.extend(subgraph.nodes)
                    else:
                        subgraph.is_acc = False
                        result.append(subgraph)
            else:
                if result and not result[-1].is_acc:
                    result[-1].nodes.extend(subgraph.nodes)
                else:
                    result.append(subgraph)
        return result

    def tag(self, subgraphs: List[Subgraph]):
        self.tags = []
        for subgraph in subgraphs:
            tag = f"_run_on_acc_{len(self.tags)}" if subgraph.is_acc else f"{self.non_acc_submodule_name}{len(self.tags)}"
            self.tags.append(tag)
            for node in subgraph.nodes:
                if hasattr(node, "tag"):
                    raise FxNetSplitterInternalError(f"Node {node} was already tagged")

                node.tag = tag  # type: ignore[attr-defined]
                self._node_submodule_map[node.name] = tag

    def split(self, remove_tag: bool = False) -> torch.fx.GraphModule:
        split_module = split_by_tags(self.module, self.tags, return_tuple=self._return_tuple)
        if remove_tag:
            for node in self.module.graph.nodes:
                if hasattr(node, "tag"):
                    del node.tag
        return split_module

    def __call__(self) -> torch.fx.GraphModule:
        subgraphs = self.put_nodes_into_subgraphs()
        subgraphs = self.remove_small_acc_subgraphs(subgraphs)
        acc_subgraphs_count = len([s for s in subgraphs if s.is_acc])
        non_acc_subgraphs_count = len(subgraphs) - acc_subgraphs_count
        print(f"Got {acc_subgraphs_count} acc subgraphs and {non_acc_subgraphs_count} non-acc subgraphs")
        self.tag(subgraphs)
        return self.split()

    def generate_split_results(self) -> SplitResult:
        split_module = self()
        submodule_names = []
        for name, mod in split_module.named_children():
            submodule_names.append(name)
        if (
            self.settings.max_acc_splits > 0
            and len(submodule_names) > self.settings.max_acc_splits
        ):
            raise ValueError(
                "Cannot fulfill max_acc_splits limit. "
                "This may cause split fragmentation and "
                "result in performance issues."
            )

        submodule_inputs = generate_inputs_for_submodules(split_module, self.sample_input, submodule_names)
        return SplitResult(split_module, submodule_inputs, self.non_acc_submodule_name)
