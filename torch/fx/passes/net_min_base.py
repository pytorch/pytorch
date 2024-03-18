import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.fx

from torch.fx._compatibility import compatibility
from torch.fx.node import map_arg

from .shape_prop import ShapeProp
from .split_utils import split_by_tags
from .tools_common import (
    CALLABLE_NODE_OPS,
    FxNetAccFusionsFinder,
    Names,
    NodeList,
    NodeSet,
    TensorOrTensors,
    Tensors,
)

__all__ = [
    "FxNetMinimizerBadModuleError",
    "FxNetMinimizerRunFuncError",
    "FxNetMinimizerResultMismatchError",
]

_LOGGER = logging.getLogger(__name__)


@compatibility(is_backward_compatible=False)
class FxNetMinimizerBadModuleError(Exception):
    """
    Raised if failed to split out a minimize module
    """

    pass


@compatibility(is_backward_compatible=False)
class FxNetMinimizerRunFuncError(Exception):
    """
    Raised if error occurs during run_a or run_b functions
    """

    pass


@compatibility(is_backward_compatible=False)
class FxNetMinimizerResultMismatchError(Exception):
    """
    Raised if comparing function thinks the results are mismatching.
    """

    pass


@dataclass
class _MinimizerSettingBase:
    """
    Args:
    `accumulate_error`: Instead of using a's input for both converted module to verify
    , use the previous outputs of each converted module as input to accumulate the
    errors.

    `traverse_method`: "sequential" or "binary" or "accumulate"
    Determine the way of traverse the nodes in FX module.

    `find_all`: Minimizer will go through the entire model and return all problematic nodes.

    `return_intermediate`: If true, when using `run_nodes()` function to run the
    model, intermediate results of all the ops will be returned as output.
    """

    accumulate_error: bool = False
    traverse_method: str = "sequential"
    find_all: bool = False
    return_intermediate: bool = False

    def __str__(self):
        settings_str = "FX Minimizer Settings:\n"

        for k, v in vars(self).items():
            settings_str += f"\t{k}: {v}\n"

        return settings_str


class _MinimizerBase:
    """
    This class is used to automatically find problematic nodes in a model. It takes a FX
    graphmodule and generate some submodules while traverse the graph. Then two functions
    `run_a` and `run_b` will be used to run the same submodule and a function `compare_fn`
    will be used to compare the results.

    Currently we provides two ways to traverse the graph and generate submodules.
        1. Sequential traversal: this will traverse the graph node by node and generate
           one submodule with one sigle node.
        2. Binary searching: this will do a binary search style traversal on the graph.

    For internal Users, a guide can be found here https://fb.quip.com/HDtuAgiKGfkP.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tensors,
        compare_fn: Callable[
            [TensorOrTensors, TensorOrTensors, Names], Tuple[float, bool]
        ],
        settings: _MinimizerSettingBase,
        module_exporter: Optional[
            Callable[
                [List[torch.Tensor], torch.fx.GraphModule, str],
                None
            ]
        ] = None,
    ):
        assert isinstance(module, torch.fx.GraphModule)

        self.module = module
        self.sample_input = sample_input
        self.compare_fn = compare_fn
        self.module_exporter = module_exporter
        self.settings = settings

        # Stores outputs of run_a function
        self.a_outputs: Dict[str, Any] = {}

        # Stores outputs of run_b function
        self.b_outputs: Dict[str, Any] = {}

        # Stores the results of compare_fn
        self.results: Dict[Any, Any] = {}

        # Stores the report for the runs
        self.reports: List[List[str]] = []

        # Current iteration
        self.iteration: int = 0

        callable_nodes = {
            node for node in self.module.graph.nodes if node.op in CALLABLE_NODE_OPS
        }
        ShapeProp(self.module).propagate(*self.sample_input)
        self.fusions = FxNetAccFusionsFinder(self.module, callable_nodes)()

        # Check if number of input in sample_input matches the number of placeholders
        placeholders = [
            node.name for node in self.module.graph.nodes if node.op == "placeholder"
        ]
        assert len(placeholders) == len(self.sample_input)

        # Store sample_input
        for i, name in enumerate(placeholders):
            self.a_outputs[name] = sample_input[i]
            self.b_outputs[name] = sample_input[i]

    def run_a(self, mod: torch.fx.GraphModule, inputs: Tensors) -> TensorOrTensors:
        """
        Run `mod` with `inputs` and generate output. The output will be compared with
        output of run_b().
        """
        raise RuntimeError("run_a() is not implemented.")

    def run_b(self, mod: torch.fx.GraphModule, inputs: Tensors) -> TensorOrTensors:
        """
        Run `mod` with `inputs` and generate output. The output will be compared with
        output of run_a().
        """
        raise RuntimeError("run_b() is not implemented.")

    def _store_outputs(
        self,
        a_result: TensorOrTensors,
        b_result: TensorOrTensors,
        submodule: torch.fx.GraphModule,
    ):
        """
        Store the outputs of self.run_a() and self.run_b() into self.a_outputs and
        self.b_outputs, so that we can use them when execute preceding nodes that
        use those outputs as inputs.

        Args:
            a_result: Output of self.run_a(). Could be a tensor or tensors.
            b_result: Output of self.run_b(). Could be a tensor or tensors.
            submodule: The module that generates a_result and b_result.
        """
        output_node = next(
            node for node in submodule.graph.nodes if node.op == "output"
        )

        # Only one output
        if isinstance(output_node.args[0], torch.fx.Node):
            self.a_outputs[output_node.args[0].name] = a_result
            self.b_outputs[output_node.args[0].name] = b_result
        # Multiple outputs
        else:
            for i, arg in enumerate(output_node.args[0]):
                self.a_outputs[arg.name] = a_result[i]
                self.b_outputs[arg.name] = b_result[i]

    def _get_submod_inputs(
        self, main_module: torch.fx.GraphModule, submod_path: str
    ) -> Tuple[Tensors, Tensors]:
        """
        Try get submodule inputs from stored outputs. If not found then use
        torch_glow.get_submod_inputs to get the inputs.

        If accumulate_error is False, use a_input for run_a() and run_b()
        otherwise use a_input for run_a and b_input for run_b.

        Args:
            main_module: Top-levlel fx module.
            submod_path: Path to the submodule we want to run and compare results.

        Returns:
            a_input: List of tensor(s) that will be used by run_a() as submodule inputs.
            b_input: List of tensor(s) that will be used by run_b() as submodule inputs.
        """
        a_input = []
        b_input = []
        submodule = getattr(main_module, submod_path)
        placeholders = [
            node.name for node in submodule.graph.nodes if node.op == "placeholder"
        ]

        # If all placeholder can be found in stored outputs, use stored
        # outputs as inputs. Otherwise, use `torch_glow.get_submod_inputs`
        # to get the inputs.
        if set(placeholders) <= self.a_outputs.keys():
            for name in placeholders:
                a_input.append(self.a_outputs[name])
                b_input.append(self.b_outputs[name])
        else:
            if self.settings.accumulate_error:
                print(f"Can't find previous stored outputs named {placeholders}!")

            def get_inputs(self: torch.nn.Module, inputs: Any):
                nonlocal a_input
                a_input = inputs

            # Use forward hook to get the inputs to the submodule
            handle = submodule.register_forward_pre_hook(get_inputs)
            main_module(*self.sample_input)
            handle.remove()

            b_input = a_input

        if not self.settings.accumulate_error:
            return a_input, a_input

        return a_input, b_input

    def _tag_nodes(self, selected_nodes: NodeSet):
        """
        Tag selected nodes with tag "minimize". Nodes with the same tags will
        be split to the same submodule afterwards.

        Args:
            selected_nodes: Nodes that we want to minimize. We will tag those nodes
                with "minimize", all preceding nodes with "main_0" and all following
                nodes with "main_1".
        """
        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            if node in selected_nodes:
                node.tag = "minimize"
            elif any(
                n.tag in {"minimize", "main_1"}
                for n in node.all_input_nodes
                if n.op in CALLABLE_NODE_OPS
            ):
                node.tag = "main_1"
            else:
                node.tag = "main_0"

    def _build_submodule(self, nodes: NodeSet) -> Tuple[torch.fx.GraphModule, str]:
        """
        Split self.module so that one submodule consists of `nodes` and only `nodes`.

        Args:
            nodes: Nodes that we want to include in the minimize submodule.

        Returns:
            split_module (torch.fx.GraphModule): the module after split.
            submodule_name (str): the name of the submodule that consists of `nodes`.
        """
        # Color provided nodes
        self._tag_nodes(nodes)

        # Split module based on coloring
        split_module = split_by_tags(self.module, ["main_0", "minimize", "main_1"])

        # Find submodule containing colored nodes
        submodule_name: str = ""
        for child_name, _ in split_module.named_children():
            # Skip submodules we're not interested in at the moment
            if "minimize" not in child_name:
                continue

            if submodule_name == "":
                submodule_name = child_name
            else:
                raise FxNetMinimizerBadModuleError(
                    f"Expected only one minimize submodule with nodes {nodes}"
                )

        if submodule_name == "":
            raise FxNetMinimizerBadModuleError(
                f"Minimize submodule was not found with nodes {nodes}"
            )

        return split_module, submodule_name

    def _run_and_compare(
        self, split_module: torch.fx.GraphModule, submod_name: str, output_names: Names
    ):
        """
        Run the submodule in `split_module` that has name `submod_name`
        using `self.run_a` and `self.run_b` and compare their results.

        Args:
            split_module: Main module that contains the minimize submodule.
            submod_name: Name of the minimize submodule.
            output_names: Names of the node we want to output. If None, we
                will use the original output.
        """
        submodule = getattr(split_module, submod_name)
        a_input, b_input = self._get_submod_inputs(split_module, submod_name)

        if len(self.reports) == 0:
            self.reports.append([])
            self.iteration = 1

        report = self.reports[self.iteration - 1]
        report.append("Run and compare ...")

        if output_names:
            output_nodes: NodeList = []
            for node in submodule.graph.nodes:
                if node.op == "output":
                    submodule.graph.erase_node(node)

                if node.name in output_names:
                    output_nodes.append(node)

            submodule.graph.output(
                output_nodes[0] if len(output_nodes) == 1 else tuple(output_nodes)
            )
            submodule.graph.lint()
            submodule.recompile()

        # Use name of args in output node as key to store comparison result
        for node in submodule.graph.nodes:
            if node.op == "output":
                result_key = map_arg(node.args, lambda x: x.name)

        try:
            a_result = self.run_a(submodule, a_input)
            b_result = self.run_b(submodule, b_input)
            self._store_outputs(a_result, b_result, submodule)
        except Exception as e:
            report.append(f"Exception raised when running {submod_name}: {e}")
            raise FxNetMinimizerRunFuncError(  # noqa: TRY200
                f"Exception raised when running {submod_name}: {e}"
            )

        # Compare results
        names: Names = output_names
        if output_names is None:
            names = [str(v) for v in result_key]  # type: ignore[possibly-undefined]

        numeric_result, bool_result = self.compare_fn(a_result, b_result, names)

        self.results[result_key] = numeric_result  # type: ignore[possibly-undefined]
        report.append(f"Numerical accuracy = {numeric_result}")
        if not bool_result:
            report.append(f"Result mismatch for {result_key}")
            if self.module_exporter:
                self.module_exporter(
                    List[torch.Tensor](a_input), submodule, str(result_key[0]) + "_cpu",
                )
                self.module_exporter(
                    List[torch.Tensor](b_input), submodule, str(result_key[0]) + "_acc",
                )
            raise FxNetMinimizerResultMismatchError(f"Result mismatch for {result_key}")

    def _binary_search_impl(
        self, all_nodes: NodeList, start_idx: int, end_idx: int
    ) -> NodeSet:
        """
        Recursive binary search implementation.
        """
        nodes: NodeList = all_nodes[start_idx:end_idx]

        report: List[str] = []
        self.reports.append(report)
        self.iteration += 1
        report.append(f"Binary search iteration {self.iteration}.")
        report.append(
            f"From node index {start_idx} to {end_idx-1}. "
            f"Size of the interested node list is {len(nodes)}"
        )

        cur_nodes: NodeSet = set(nodes)

        for node in nodes:
            if node in self.fusions:
                cur_nodes.update(self.fusions[node])

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, [])
        except (FxNetMinimizerRunFuncError, FxNetMinimizerResultMismatchError):

            if len(nodes) == 1:
                report.append(
                    f"This is the last node in the sub-module. "
                    f"Search in the current branch is successful with culprit = {cur_nodes}."
                )
                self.print_report(report)
                return cur_nodes

            report.append(
                "Proceed to split and lower the halves of the current "
                "sub-module individually."
            )
            self.print_report(report)

            mid = len(nodes) // 2
            culprits = self._binary_search_impl(all_nodes, start_idx, start_idx + mid)

            if len(culprits) != 0 and not self.settings.find_all:
                return culprits

            culprits = self._binary_search_impl(all_nodes, start_idx + mid, end_idx)

            if len(culprits) == 0:
                report.append(
                    f"Further split and lowering found no errors. "
                    f"Unable to minimize the submodule with list of nodes: {nodes}"
                )
                self.print_report(report)

            return culprits
        else:
            report.append("No discrepancy found.")
            self.print_report(report)
            return set()

    def _binary_traverse(self, nodes: NodeList) -> NodeSet:
        """
        Binary search on `nodes` for culprit.
        """
        return self._binary_search_impl(nodes, 0, len(nodes))

    def _sequential_traverse(self, nodes: NodeList) -> NodeSet:
        """
        Traverse `nodes` one by one and determine if any of them is a culprit.
        """
        culprits: NodeSet = set()

        for node in nodes:
            report: List[str] = []
            self.reports.append(report)
            self.iteration += 1
            report.append(f"Sequential traverse iteration {self.iteration}.")
            report.append(f"Visit node: {node.name}")

            _LOGGER.info("Visit node: %s", node.name)
            cur_nodes: NodeSet = {node}

            if node in self.fusions:
                cur_nodes = self.fusions[node]

            try:
                split_module, submod_name = self._build_submodule(cur_nodes)
                self._run_and_compare(split_module, submod_name, [node.name])
                self.print_report(report)
            except (FxNetMinimizerResultMismatchError):
                culprits.add(node)
                report.append(f"Found culprit from numeric error: {node}")
                self.print_report(report)
                if not self.settings.find_all:
                    return culprits
            except (FxNetMinimizerRunFuncError):
                culprits.update(cur_nodes)
                report.append(f"Found culprit from run error: {node}")
                self.print_report(report)
                if not self.settings.find_all:
                    return culprits

        return culprits

    def _defined_traverse(self, nodes: NodeList) -> NodeSet:
        """
        run user defined `nodes` and determine if it is a culprit.
        """
        culprits: NodeSet = set()

        first_node_name = nodes[0].name
        output_node_name = nodes[-1].name
        report = [f"Defined graph from {first_node_name} to {output_node_name}"]
        cur_nodes: NodeSet = set(nodes)
        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, [output_node_name])
            self.print_report(report)
        except (FxNetMinimizerResultMismatchError, FxNetMinimizerRunFuncError):
            report.append(f"Found culprit {cur_nodes}")
            self.print_report(report)
            return culprits

        return culprits

    def _accumulate_traverse(self, nodes: NodeList) -> NodeSet:
        culprits: NodeSet = set()
        nodes_to_run: NodeSet = set()

        # find_all is not supported for accumulate traversal because all the
        # ops run on NNPI. So we return after the first op that raises error.
        if self.settings.find_all:
            print("'Find All' mode is not supported in accumulate traversal.")
            return culprits

        for node in nodes:
            report: List[str] = []
            self.reports.append(report)
            self.iteration += 1
            report.append(f"Accumulate traverse iteration {self.iteration}.")

            nodes_to_run.add(node)

            node_name = node.name
            if node_name is not None and isinstance(node_name, tuple):
                node_name = node_name[0]
            assert node_name is not None and isinstance(
                node_name, str
            ), f"minimize: node_name: {node_name}"

            report.append(f"Add node: {node_name}")

            try:
                split_module, submod_name = self._build_submodule(nodes_to_run)
                self._run_and_compare(split_module, submod_name, [node_name])
                self.print_report(report)
            except (FxNetMinimizerResultMismatchError, FxNetMinimizerRunFuncError):
                culprits.add(node)
                report.append(f"Found culprit {node}")
                self.print_report(report)
                return culprits

        return culprits

    def _skip_traverse_impl(self, all_nodes: NodeList, start_idx: int, end_idx: int) -> NodeSet:
        """
        Skip certain nodes in graph based on settings
        """
        culprits: NodeSet = set()
        nodes: NodeList = all_nodes[start_idx:end_idx]

        report: List[str] = []
        self.reports.append(report)
        self.iteration += 1
        report.append(f" Nodes block {self.iteration}.")
        report.append(
            f"From node index {start_idx} to {end_idx-1}. "
            f"Size of the interested node list is {len(nodes)}"
        )

        cur_nodes: NodeSet = set(nodes)

        for node in nodes:
            if node in self.fusions:
                cur_nodes.update(self.fusions[node])

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, [])
        except (FxNetMinimizerResultMismatchError):
            culprits.update(cur_nodes)
            report.append(f"Found culprit from numeric error: {cur_nodes}")
            self.print_report(report)
            return culprits
        except (FxNetMinimizerRunFuncError):
            culprits.update(cur_nodes)
            report.append(f"Found culprit from run error: {node}")
            self.print_report(report)
            return culprits
        else:
            report.append("No discrepancy found.")
            self.print_report(report)
            return set()


    def _skip_traverse(self, all_nodes: NodeList, skip_nodes: List) -> NodeSet:
        """
        Skip certain nodes in graph based on settings
        """
        start_idx = 0
        num_nodes = len(all_nodes)
        idx = 0
        culprits = set()
        while idx < num_nodes:
            node = all_nodes[idx]
            if (node.name in skip_nodes):  # skip the node
                if idx > start_idx:
                    culprits = self._skip_traverse_impl(all_nodes, start_idx, idx)
                start_idx = idx + 1
            elif idx == num_nodes - 1 and start_idx <= idx:  # last node
                culprits = self._skip_traverse_impl(all_nodes, start_idx, idx + 1)
            idx += 1

        return culprits



    def _collect_nodes(self, start: Optional[str], end: Optional[str]) -> NodeList:
        """
        Collect nodes in the model that between nodes with name of `start` and `end`.
        These two nodes are also included.
        """
        nodes: NodeList = []
        add_node = start is None

        for node in self.module.graph.nodes:
            if node.op not in CALLABLE_NODE_OPS:
                continue

            if node.name == start:
                add_node = True

            if add_node:
                nodes.append(node)

            if node.name == end:
                break

        return nodes

    def run_nodes(self, start: Optional[str] = None, end: Optional[str] = None):
        """
        Run part of the model from `start` node to `end` node. If `start` is None
        then we start from the beginning of the model. If `end` is None then we
        stop at the end of the model.

        Args:
            start: The name of the node which is the first node of the submodule
                we want to run. If set to None, then we'll start with the first
                node of the model.
            end: The name of the node which is the last node of the submodule we
                want to run. If set to None, we'll end with the last node of the
                model.
        """
        nodes = self._collect_nodes(start, end)
        cur_nodes = set(nodes)

        for node in nodes:
            if node in self.fusions:
                cur_nodes.update(self.fusions[node])

        output_names = []
        if self.settings.return_intermediate:
            output_names = [node.name for node in nodes]

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, output_names)
        except (
            FxNetMinimizerRunFuncError,
            FxNetMinimizerResultMismatchError,
        ) as e:
            print(e)

    def print_report(self, report: List[str]):
        for i in range(len(report)):
            if i > 0:
                print(" . " + report[i])
            else:
                print(report[i])

    def print_reports(self):
        for report in self.reports:
            self.print_report(report)

    def minimize(
        self, start: Optional[str] = None, end: Optional[str] = None, skip_nodes: Optional[List] = None,
    ) -> NodeSet:
        """
        Minimizing the model from node with name `start` to node with name `end` base
        on self.settings. Find culprits that causes FxNetMinimizerRunFuncError or
        FxNetMinimizerResultMismatchError errors.

        Args:
            start: The name of the node where we want to start minimizing. If set
                to None, then we'll start with the first node of the model.
            end: The name of the node where we want to terminate minimizing. If
                set to None, we'll end with the last node of the model.

        Returns:
            nodes: A list of nodes that causes FxNetMinimizerRunFuncError or
                FxNetMinimizerResultMismatchError errors during minimizing.
        """

        print(self.settings)
        print(self.module.graph)

        nodes = self._collect_nodes(start, end)

        if self.settings.traverse_method == "sequential":
            return self._sequential_traverse(nodes)

        if self.settings.traverse_method == "binary":
            return self._binary_traverse(nodes)

        if self.settings.traverse_method == "accumulate":
            return self._accumulate_traverse(nodes)

        if self.settings.traverse_method == "skip":
            if (skip_nodes is None):
                raise RuntimeError("'skip_nodes' can't be None when 'traverse_method' is 'skip'.")
            return self._skip_traverse(nodes, skip_nodes)

        if self.settings.traverse_method == "defined":
            return self._defined_traverse(nodes)

        raise RuntimeError(f"Unknown traverse method {self.settings.traverse_method}!")
