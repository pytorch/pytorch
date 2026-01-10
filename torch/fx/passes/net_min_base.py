# mypy: allow-untyped-defs
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast, Optional

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


@compatibility(is_backward_compatible=False)
class FxNetMinimizerRunFuncError(Exception):
    """
    Raised if error occurs during run_a or run_b functions
    """


@compatibility(is_backward_compatible=False)
class FxNetMinimizerResultMismatchError(Exception):
    """
    Raised if comparing function thinks the results are mismatching.
    """


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

    `all_outputs`: If true, when using `_run_and_compare()` function,
    all the output nodes in the subgraph will be used for comparison.
    """

    accumulate_error: bool = False
    traverse_method: str = "sequential"
    find_all: bool = False
    return_intermediate: bool = False
    all_outputs: bool = False

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
           one submodule with one single node.
        2. Binary searching: this will do a binary search style traversal on the graph.

    For internal Users, a guide can be found here https://fb.quip.com/HDtuAgiKGfkP.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        sample_input: Tensors,
        compare_fn: Callable[
            [TensorOrTensors, TensorOrTensors, Names], tuple[float, bool]
        ],
        settings: _MinimizerSettingBase,
        module_exporter: Optional[
            Callable[[Tensors, torch.fx.GraphModule, str], None]
        ] = None,
        exclusion_fn: Optional[Callable[[NodeList, int, int], None]] = None,
    ):
        assert isinstance(module, torch.fx.GraphModule)

        self.module = module
        self.sample_input = sample_input
        self.compare_fn = compare_fn
        self.module_exporter = module_exporter
        self.settings = settings
        self.exclusion_fn = exclusion_fn

        # Stores outputs of run_a function
        self.a_outputs: dict[str, Any] = {}

        # Stores outputs of run_b function
        self.b_outputs: dict[str, Any] = {}

        # Stores the results of compare_fn
        self.results: dict[Any, Any] = {}

        # Stores the report for the runs
        self.reports: list[list[str]] = []

        # Current iteration
        self.iteration: int = 0

        callable_nodes = {
            node for node in self.module.graph.nodes if node.op in CALLABLE_NODE_OPS
        }
        self.run_shape_prop()
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

    def run_shape_prop(self) -> None:
        """
        Helper function to run shape propagation on module. Can be overridden by
        subclasses for custom shape propagation logic.
        """
        ShapeProp(self.module).propagate(*self.sample_input)

    def run_a(
        self, mod: torch.fx.GraphModule, inputs: Tensors, report_idx: int = -1
    ) -> TensorOrTensors:
        """
        Run `mod` with `inputs` and generate output. The output will be compared with
        output of run_b().
        """
        raise RuntimeError("run_a() is not implemented.")

    def run_b(
        self, mod: torch.fx.GraphModule, inputs: Tensors, report_idx: int = -1
    ) -> TensorOrTensors:
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
    ) -> tuple[Tensors, Tensors]:
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

    def _build_submodule(self, nodes: NodeSet) -> tuple[torch.fx.GraphModule, str]:
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
        for child_name, _ in split_module.named_children():  # type: ignore[union-attr]
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

        return split_module, submodule_name  # type: ignore[return-value]

    def _run_and_compare(
        self,
        split_module: torch.fx.GraphModule,
        submod_name: str,
        output_names: Names,
        report_idx: int = -1,
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

        report = self.reports[report_idx if report_idx >= 0 else self.iteration - 1]
        report.append("Run and compare ...")

        if output_names and not self.settings.all_outputs:
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
            a_result = self.run_a(submodule, a_input, report_idx)
            b_result = self.run_b(submodule, b_input, report_idx)
            self._store_outputs(a_result, b_result, submodule)
        except Exception as e:
            report.append(f"Exception raised when running {submod_name}: {e}")
            raise FxNetMinimizerRunFuncError(  # noqa: B904
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
            report.append(f"Result mismatch for {result_key}")  # type: ignore[possibly-undefined]
            if self.module_exporter:
                if isinstance(result_key, tuple):  # type: ignore[possibly-undefined]
                    # pyrefly: ignore [unbound-name]
                    result_key = result_key[-1]
                # If the result is still a tuple (happens in non-sequential mode),
                # we only use the first element as name.
                if isinstance(result_key, tuple):  # type: ignore[possibly-undefined]
                    # pyrefly: ignore [unbound-name]
                    result_key = str(result_key[0])
                # pyre-ignore[29]: not a function
                self.module_exporter(
                    a_input,
                    submodule,
                    # pyrefly: ignore [unbound-name]
                    result_key + "_cpu",
                )
                # pyre-ignore[29]: not a function
                self.module_exporter(
                    b_input,
                    submodule,
                    # pyrefly: ignore [unbound-name]
                    result_key + "_acc",
                )
            raise FxNetMinimizerResultMismatchError(f"Result mismatch for {result_key}")  # type: ignore[possibly-undefined]

    def _binary_search_impl(
        self, all_nodes: NodeList, start_idx: int, end_idx: int
    ) -> NodeSet:
        """
        Recursive binary search implementation.
        """
        culprits: NodeSet = set()
        nodes: NodeList = all_nodes[start_idx:end_idx]

        report: list[str] = []
        if self.exclusion_fn is not None:
            self.exclusion_fn(nodes, start_idx, end_idx)
            if len(nodes) == 0:
                report = ["All nodes are excluded by user"]
                self.reports.append(report)
                return culprits

        first_node_name = nodes[0].name
        output_node_name = nodes[-1].name
        self.iteration += 1
        self.reports.append(report)
        report.append(f"Binary search iteration {self.iteration}")
        report.append(
            f"From node index {start_idx}:{first_node_name} to {end_idx - 1}:{output_node_name}. "
            f"Size of the interested node list is {len(nodes)}"
        )
        cur_nodes: NodeSet = set(nodes)

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, [output_node_name])

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
            report: list[str] = []
            self.reports.append(report)
            self.iteration += 1
            report.append(f"Sequential traverse iteration {self.iteration}.")
            report.append(f"Visit node: {node.name}")

            _LOGGER.info("Visit node: %s", node.name)
            node_list: NodeList = [node]
            if self.exclusion_fn is not None:
                self.exclusion_fn(node_list, -1, -1)
                if len(node_list) == 0:
                    report.append(f"User exclusion : {node.name}")
                    self.print_report(report)
                    if not self.settings.find_all:
                        return culprits
                    else:
                        continue

            cur_nodes: NodeSet = {node}

            if node in self.fusions:
                cur_nodes = self.fusions[node]

            try:
                split_module, submod_name = self._build_submodule(cur_nodes)
                self._run_and_compare(split_module, submod_name, [node.name])
                self.print_report(report)
            except FxNetMinimizerResultMismatchError:
                culprits.add(node)
                report.append(f"Found culprit from numeric error: {node}")
                self.print_report(report)
                if not self.settings.find_all:
                    return culprits
            except FxNetMinimizerRunFuncError:
                culprits.update(cur_nodes)
                report.append(f"Found culprit from run error: {node}")
                self.print_report(report)
                if not self.settings.find_all:
                    return culprits

        return culprits

    def _block_traverse_impl(
        self, nodes: NodeList, start_idx: int, end_idx: int, find_last_node: bool
    ) -> Optional[int]:
        """
        Recursive block search implementation.
        find_last_node: If True, search for the last node which result in numerics difference
        if False: find first node in sorted node list
        """
        report: list[str] = []

        mid = (start_idx + end_idx) // 2
        cur_nodes_list: NodeList = nodes[: mid + 1] if find_last_node else nodes[mid:]

        if self.exclusion_fn:
            self.exclusion_fn(cur_nodes_list, -1, -1)

        cur_nodes = set(cur_nodes_list)

        first_node_name = cur_nodes_list[0].name
        last_node_name = cur_nodes_list[-1].name
        target_node_name = last_node_name if find_last_node else first_node_name

        self.iteration += 1
        self.reports.append(report)
        report.extend(
            [
                "=" * 30,
                f"Block search iteration {self.iteration}",
            ]
        )
        report.extend(
            [
                f"Search for {'last' if find_last_node else 'first'} node in culprits",
                f"From node index {start_idx}:{nodes[start_idx].name} to {end_idx}:{nodes[end_idx].name}. ",
                f"Subgraph constructed by {first_node_name} to {last_node_name}",
                f"Targeting node: {target_node_name}",
                f"Size of the interested node list is {end_idx - start_idx + 1}",
            ]
        )
        report_idx = len(self.reports) - 1

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(
                split_module, submod_name, [last_node_name], report_idx
            )
        except (FxNetMinimizerResultMismatchError, FxNetMinimizerRunFuncError):
            report.append(
                f"Culprits found from node {first_node_name} to {last_node_name}."
            )

            if start_idx == mid == end_idx:
                report.extend(
                    [
                        "This is the last node in the sub-module. ",
                        "Search in the current branch is successful with node :",
                        f"{start_idx}, node name: {nodes[start_idx].name}.",
                    ]
                )
                self.print_report(report)
                return start_idx

            report.append(
                "Proceed to split and lower the halves of the current "
                "sub-module individually."
            )
            self.print_report(report)

            if find_last_node:
                return self._block_traverse_impl(nodes, start_idx, mid, find_last_node)
            else:
                return self._block_traverse_impl(
                    nodes, mid + 1, end_idx, find_last_node
                )
        else:
            report.append(
                f"Culprits not found from node start to {mid}:{nodes[mid].name}."
            )

            if start_idx == mid == end_idx:
                # We did not find anything if the pointers have not moved
                if (start_idx == 0 and not find_last_node) or (
                    start_idx == len(nodes) - 1 and find_last_node
                ):
                    report.append(
                        f"At {'last' if find_last_node else 'first'} node, no culprits found."
                    )
                    self.print_report(report)
                    return None

                # Otherwise, we have converged on the border between discrepancy and valid
                return start_idx + (1 if find_last_node else -1)

            report.append(
                "Proceed to split and lower the halves of the current "
                "sub-module individually."
            )
            self.print_report(report)

            if find_last_node:
                return self._block_traverse_impl(
                    nodes, mid + 1, end_idx, find_last_node
                )
            else:
                return self._block_traverse_impl(nodes, start_idx, mid, find_last_node)

    def _block_traverse(
        self, nodes: NodeList, find_last_node: Optional[bool]
    ) -> NodeSet:
        """
        Traverse topologically sorted node list
        Find minimum block (start_idx, end_idx) which contains the culprit
        1st pass: search for end_idx by finding the last node in culprit block
        where Numerical accuracy (0, end_idx) > threshold
        2nd pass: search for start_idx by finding the first node in culprit block
        where Numerical accuracy (start_idx, end_idx) < threshold
        Form minimum block by (start_idx - 1, end_idx)
        """
        culprits: NodeSet = set()
        first_node_name = nodes[0].name
        last_node_name = nodes[-1].name
        last_node_report = [f"Block search from {first_node_name} to {last_node_name}"]
        last_node_report.append("*" * 50)
        self.reports.append(last_node_report)

        start_idx = 0
        end_idx = len(nodes) - 1

        final_start_idx: Optional[int] = start_idx
        final_end_idx: Optional[int] = end_idx

        run_both = find_last_node is None

        # step 1: find (0, end_idx) of culprit block
        if run_both or find_last_node:
            last_node_report.append("Start searching for last node in culprit")
            self.print_report(last_node_report)
            final_end_idx = self._block_traverse_impl(nodes, start_idx, end_idx, True)

            if final_end_idx is None:
                last_node_report.append("No culprits found")
                self.print_report(last_node_report)
                return culprits

            last_node_report.extend(
                [
                    "Finish Pass 1",
                    f"Find end_idx = {final_end_idx}:{nodes[final_end_idx].name}",
                ]
            )
            self.print_report(last_node_report)

        # step 2: reduce culprit block to (start_idx, end_idx)
        if run_both or not find_last_node:
            first_node_report = ["Start searching for first node in culprit"]
            self.print_report(first_node_report)
            final_start_idx = self._block_traverse_impl(
                nodes[0 : end_idx + 1], start_idx, final_end_idx or end_idx, False
            )

            if final_start_idx is None:
                last_node_report.append("No culprits found")
                self.print_report(last_node_report)
                return culprits

            first_node_report.append("*" * 50)
            self.reports.append(first_node_report)
            first_node_report.extend(
                [
                    "Finish Pass 2",
                    f"Find start_idx = {final_start_idx}:{nodes[final_start_idx].name}",
                ]
            )
            self.print_report(first_node_report)

        # step 3: form module with minimum culprits. These indexes are guaranteed to exist
        range_start, range_end = cast(int, final_start_idx), cast(int, final_end_idx)
        culprits.update(nodes[range_start : range_end + 1])
        result_report = [
            f"Finish searching, found minimum block ({nodes[range_start]},{nodes[range_end]})"
        ]
        self.reports.append(result_report)
        self.print_report(result_report)
        return culprits

    def _defined_traverse(self, nodes: NodeList) -> NodeSet:
        """
        run user defined `nodes` and determine if it is a culprit.
        """
        culprits: NodeSet = set()
        if self.exclusion_fn is not None:
            self.exclusion_fn(nodes, -1, -1)
        if len(nodes) == 0:
            report = ["All nodes are excluded by user"]
            self.reports.append(report)
            return culprits

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
            report: list[str] = []
            self.reports.append(report)
            self.iteration += 1
            report.append(f"Accumulate traverse iteration {self.iteration}.")

            nodes_to_run.add(node)

            node_name = node.name
            if node_name is not None and isinstance(node_name, tuple):
                node_name = node_name[0]
            assert node_name is not None and isinstance(node_name, str), (
                f"minimize: node_name: {node_name}"
            )

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

    def _skip_traverse_impl(
        self, all_nodes: NodeList, start_idx: int, end_idx: int
    ) -> NodeSet:
        """
        Skip certain nodes in graph based on settings
        """
        culprits: NodeSet = set()
        nodes: NodeList = all_nodes[start_idx:end_idx]
        cur_nodes: NodeSet = set(nodes)
        if self.exclusion_fn is not None:
            self.exclusion_fn(nodes, start_idx, end_idx)
            cur_nodes = set(nodes)
        else:
            for node in nodes:
                if node in self.fusions:
                    cur_nodes.update(self.fusions[node])
        report: list[str] = []
        self.reports.append(report)
        self.iteration += 1
        report.append(f" Nodes block {self.iteration}.")
        report.append(
            f"From node index {start_idx} to {end_idx - 1}. "
            f"Size of the interested node list is {len(nodes)}"
        )

        try:
            split_module, submod_name = self._build_submodule(cur_nodes)
            self._run_and_compare(split_module, submod_name, [])
        except FxNetMinimizerResultMismatchError:
            culprits.update(cur_nodes)
            report.append(f"Found culprit from numeric error: {cur_nodes}")
            self.print_report(report)
            return culprits
        except FxNetMinimizerRunFuncError:
            culprits.update(cur_nodes)
            report.append(f"Found culprit from run error: {cur_nodes}")
            self.print_report(report)
            return culprits
        else:
            report.append("No discrepancy found.")
            self.print_report(report)
            return set()

    def _skip_traverse(self, all_nodes: NodeList, skip_nodes: list) -> NodeSet:
        """
        Skip certain nodes in graph based on settings
        """
        start_idx = 0
        num_nodes = len(all_nodes)
        idx = 0
        culprits = set()
        while idx < num_nodes:
            node = all_nodes[idx]
            if node.name in skip_nodes:  # skip the node
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

    def print_report(self, report: list[str]):
        for i in range(len(report)):
            if i > 0:
                print(" . " + report[i])
            else:
                print(report[i])

    def print_reports(self):
        for report in self.reports:
            self.print_report(report)

    def minimize(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        skip_nodes: Optional[list] = None,
        find_last_node: Optional[bool] = None,
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
            skip_nodes: The names of nodes where we want to skip during minimizing.
                It'll create subgraphs without these skip nodes under the hood.
                Only applicable in mode "skip".
            find_last_node: True if only last_node of a culprits is needed in mode "block".
                False if only the first_node of a culprits is needed.
                Only applicable in mode "block".

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
            if skip_nodes is None:
                raise RuntimeError(
                    "'skip_nodes' can't be None when 'traverse_method' is 'skip'."
                )
            return self._skip_traverse(nodes, skip_nodes)

        if self.settings.traverse_method == "defined":
            return self._defined_traverse(nodes)

        if self.settings.traverse_method == "block":
            return self._block_traverse(nodes, find_last_node)

        raise RuntimeError(f"Unknown traverse method {self.settings.traverse_method}!")
