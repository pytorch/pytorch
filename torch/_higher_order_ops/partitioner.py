import logging
from typing import Any, Callable, Union

import torch
from torch._higher_order_ops.utils import create_bw_fn, materialize_as_graph


logger: logging.Logger = logging.getLogger(__name__)


def _find_hop_subgraph_outputs(gm: torch.fx.GraphModule) -> tuple[torch.fx.Node]:
    output_node_args = gm.graph.find_nodes(op="output")[0].args
    assert isinstance(output_node_args, tuple)
    return output_node_args[0]


def is_complex_expr(expr: Any) -> bool:
    return not expr.is_symbol


class HopPartitionedGraph:
    def __init__(
        self,
        fw_gm: torch.fx.GraphModule,
        bw_gm: torch.fx.GraphModule,
        n_fw_outputs: int,
        n_checkpoints: int,
    ):
        self.fw_gm = fw_gm
        self.bw_gm = bw_gm
        self.n_fw_outputs = n_fw_outputs
        self.n_checkpoints = n_checkpoints
        self._reorder_fw_output()
        self._check_partition_boundary()

    def _check_partition_boundary(self) -> None:
        """check partitioned graph is in valid state."""
        invalid_reasons = []
        fw_outputs = _find_hop_subgraph_outputs(self.fw_gm)
        for i, out in enumerate(fw_outputs):
            if "val" not in out.meta:
                invalid_reasons.append(f"fw_gm output[{i}] doesn't have a 'val' meta.")
            elif not isinstance(out.meta["val"], (torch.SymInt, torch.Tensor)):
                invalid_reasons.append(
                    f"fw_gm output[{i}] is of type {type(out.meta['val'])} but only SymInt or Tensor are allowed."
                )

            elif isinstance(out.meta["val"], torch.SymInt) and is_complex_expr(
                out.meta["val"].node.expr
            ):
                invalid_reasons.append(
                    f"fw_gm output[{i}] must be of type SymInt or Tensor but got {type(out.meta['val'])}"
                )

        if len(fw_outputs) != self.n_fw_outputs + self.n_checkpoints:
            invalid_reasons.append(
                f"len(fw_outputs) ({len(fw_outputs)}) != n_fw_outputs ({self.n_fw_outputs}) + n_checkpoints ({self.n_checkpoints})"
            )

        bw_phs = list(self.bw_gm.graph.find_nodes(op="placeholder"))

        if len(fw_outputs) != len(bw_phs):
            invalid_reasons.append(
                f"fw_gm's have {len(fw_outputs)} but backward takes {len(bw_phs)} inputs."
            )

        original_forward_outputs = fw_outputs[: self.n_fw_outputs]
        fw_intermediates = fw_outputs[self.n_fw_outputs :]

        bw_intermediates = bw_phs[: -self.n_fw_outputs]
        bw_grads = bw_phs[-self.n_fw_outputs :]

        def _match_size_or_expr(
            val1: Union[torch.SymInt, torch.Tensor],
            val2: Union[torch.SymInt, torch.Tensor],
        ) -> bool:
            if type(val1) != type(val2):
                return False

            if isinstance(val1, torch.SymInt) and isinstance(val2, torch.SymInt):
                return val1.node.expr == val2.node.expr
            elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                return val1.size() == val2.size()

            return False

        for fw, bw in zip(fw_intermediates, bw_intermediates):
            if fw.name != bw.name or not _match_size_or_expr(
                fw.meta["val"], bw.meta["val"]
            ):
                invalid_reasons.append("fw intermediates don't match bw intermediates")

        for fw_out, bw_grad in zip(original_forward_outputs, bw_grads):
            if not _match_size_or_expr(fw_out.meta["val"], bw_grad.meta["val"]):
                invalid_reasons.append("fw output don't match bw gradients")

        if len(invalid_reasons) > 0:
            newline = "\n"
            raise RuntimeError(
                "Invalid HopPartitionedGraph. Reasons:\n",
                f"{newline.join(invalid_reasons)}",
            )

    def _reorder_fw_output(self) -> None:
        """
        Before the pass, fw_gm returns (*fw_outputs, *intermediates1)
        and bw_gm takes (*intermediates2, *grad_fw_outputs) as input.
        intermediates1 and intermediates2 share the same node names but
        they might be in different order. E.g. this could happen if there
        are inputs that contain symints.

        To simplify downstream processing, this graph pass normalizes the output of fw_gm
        to be consistent with the bacwkard inputs:

        fw_gm:
          - input: fw_args
          - output: (*fw_outputs, *intermediates)

        bw_gm:
          - input: (*intermediates, *grad_fw_outputs)
          - output: grad_fw_args

        Example:

        def fw_gm(x, y, z):
           a, b, c = f(x), g(y), k(z)
           return a, b, c, f_tmp, g_tmp, k_tmp

        , where a, b, c are fw_outputs, f_tmp, g_tmp, k_tmp are intermediates

        The corresponding bw_gm has the following signature:

        def bw_gm(f_tmp, g_tmp, k_tmp, grad_a, grad_b, grac):
          return grad_x, grad_y, grad_z
        """
        # Initially the intermediates are the latter part of the fw outputs
        fw_gm_output_nodes = _find_hop_subgraph_outputs(self.fw_gm)
        fw_outputs_nodes = fw_gm_output_nodes[: self.n_fw_outputs]
        fw_intermediates_nodes = fw_gm_output_nodes[self.n_fw_outputs :]
        n_intermediates = len(fw_intermediates_nodes)
        if n_intermediates > 0:
            fw_intermediates_name_to_node = {n.name: n for n in fw_intermediates_nodes}

            # First n_intermediates placeholders
            bw_names: list[str] = [
                ph.name
                for ph in list(self.bw_gm.graph.find_nodes(op="placeholder"))[
                    :n_intermediates
                ]
            ]
            new_fw_outputs = list(fw_outputs_nodes) + [
                fw_intermediates_name_to_node[name] for name in bw_names
            ]

            output_node = self.fw_gm.graph.find_nodes(op="output")[0]
            output_node.args = (tuple(new_fw_outputs),)

            self.fw_gm.graph.lint()
            self.fw_gm.recompile()


class HopJointGraph:
    def __init__(
        self, joint_gm: torch.fx.GraphModule, n_primals: int, n_fw_outputs: int
    ):
        self.joint_gm = joint_gm
        self.n_primals = n_primals
        self.n_fw_outputs = n_fw_outputs
        self._rename_phs()
        self._remove_redundant_sym_size_ops()
        self._mark_complex_exprs_as_must_recompute()

    def _rename_phs(self) -> None:
        self.n_tangents = 0
        for i, ph in enumerate(self.joint_gm.graph.find_nodes(op="placeholder")):
            if i < self.n_primals:
                ph.target = f"primals_{i}"
                ph.name = f"primals_{i}"
            else:
                self.n_tangents += 1
                ph.target = f"tangents_{i - self.n_primals}"
                ph.name = f"tangents_{i - self.n_primals}"

        self.joint_gm.graph.lint()
        self.joint_gm.compile()

    def _remove_redundant_sym_size_ops(self) -> None:
        """
        Graph pass that deletes torch.ops.sym_size.int operators whose output is a
        corresponding placeholder that holds the same symbol, and replace all uses
        of their output to be directly accessing the placeholders.
        """
        placeholder_exprs = {}
        for node in self.joint_gm.graph.nodes:
            if (
                isinstance(node, torch.fx.Node)
                and node.op == "placeholder"
                and hasattr(node, "meta")
                and "val" in node.meta
            ):
                val = node.meta["val"]
                if isinstance(val, torch.SymInt):
                    placeholder_exprs[val.node.expr] = node

        nodes_to_remove = []
        for node in self.joint_gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.sym_size.int
        ):
            assert hasattr(node, "meta") and "val" in node.meta, node
            val = node.meta["val"]
            expr = val.node.expr
            if expr in placeholder_exprs:
                placeholder_node = placeholder_exprs[expr]
                node.replace_all_uses_with(placeholder_node)
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            self.joint_gm.graph.erase_node(node)

        self.joint_gm.graph.lint()
        self.joint_gm.recompile()

    def _mark_complex_exprs_as_must_recompute(self) -> None:
        """
        Graph pass that marks the recompute policy for nodes that have complex sympy expressions
        so that the graph boundary only contains tensors or basic symbols.
        """

        for n in (
            node for node in self.joint_gm.graph.nodes if node.op == "call_function"
        ):
            if "val" not in n.meta:
                continue
            val = n.meta["val"]
            if isinstance(val, torch.SymInt) and is_complex_expr(val.node.expr):
                assert n.meta.get("recompute", None) is None
                from torch._functorch.partitioners import CheckpointPolicy

                n.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE

        # Clean up and recompile
        self.joint_gm.graph.lint()
        self.joint_gm.recompile()

    def partition(self, partition_fn: Callable) -> HopPartitionedGraph:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "before min_cut_partition:\n%s",
                self.joint_gm.print_readable(print_output=False),
            )

        fw_gm, bw_gm = partition_fn(
            self.joint_gm, None, num_fwd_outputs=self.n_fw_outputs
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("after partition_fn:")
            logger.debug("fw_gm:\n%s", fw_gm.print_readable(print_output=False))
            logger.debug("bw_gm:\n%s", bw_gm.print_readable(print_output=False))

        n_checkpoints = len(_find_hop_subgraph_outputs(fw_gm)) - self.n_fw_outputs

        return HopPartitionedGraph(
            fw_gm,
            bw_gm,
            self.n_fw_outputs,
            n_checkpoints,
        )


class HopGraphMinCutPartitioner:
    @staticmethod
    def _create_joint_graph(
        fw_fn: Callable, fw_args: tuple[Union[torch.Tensor, torch.SymInt], ...]
    ) -> HopJointGraph:
        fw_gm = materialize_as_graph(fw_fn, fw_args, force_enable_grad=True)
        fw_gm_output_nodes = _find_hop_subgraph_outputs(fw_gm)

        assert all(
            isinstance(n, torch.fx.Node) and "val" in n.meta for n in fw_gm_output_nodes
        )
        fw_gm_output_vals = tuple(n.meta["val"] for n in fw_gm_output_nodes)  # type: ignore[arg-type]

        assert all(isinstance(val, torch.Tensor) for val in fw_gm_output_vals)
        example_grads = tuple(torch.zeros_like(val) for val in fw_gm_output_vals)

        joint_fn = create_bw_fn(fw_fn, fw_args, return_fw_outputs=True)
        # Need to first trace out the joint_fn with autograd info on
        # then functionalize the graph otherwise the grad information is lost for functional tensor
        joint_gm = materialize_as_graph(
            joint_fn, fw_args + example_grads, force_enable_grad=True
        )
        joint_gm_functionalized = materialize_as_graph(
            torch.func.functionalize(joint_gm, remove="mutations_and_views"),
            fw_args + example_grads,
        )

        return HopJointGraph(
            joint_gm_functionalized,
            len(fw_args),
            len(fw_gm_output_nodes),
        )

    @staticmethod
    def create(
        fw_fn: Callable, fw_args: tuple[Union[torch.Tensor, torch.SymInt], ...]
    ) -> HopPartitionedGraph:
        from torch._functorch.partitioners import min_cut_rematerialization_partition

        joint_graph: HopJointGraph = HopGraphMinCutPartitioner._create_joint_graph(
            fw_fn, fw_args
        )
        return joint_graph.partition(min_cut_rematerialization_partition)
