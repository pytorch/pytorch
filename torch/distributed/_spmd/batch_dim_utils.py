from typing import Callable, Dict, List, Set

import torch

import torch.fx as fx

import torch.utils._pytree as pytree

from torch import Tensor

from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from torch.distributed._tensor.ops.view_ops import (
    DimSpec,
    InputDim,
    ops as view_op_rules,
)
from torch.distributed._tensor.placement_types import _Partial, DTensorSpec

aten = torch.ops.aten


class BatchDimAnalyzer:
    """
    This class is used to analyze the batch dimension of each tensor/node in the
    graph. We need to know the batch dimension of each tensor/node so that we know
    exactly the sharding layout of intermediate tensors.

    We possibly should evaluate using symbolic shapes to track the batch dimension.
    We can experiment it later with dynamo integration (as dynamo have mark_dynamic
    API which allows marking batch dimension only) or try to use FakeTensorMode to
    mark the batch dimension. For now, let's just use the batch dimension of the first
    input tensor as the hint to track the batch dimension of all tensors/nodes in
    the graph.
    """

    def __init__(self, batch_dim: int = 0) -> None:
        self.batch_dim = batch_dim

        if batch_dim != 0:
            # TODO: see if this make sense or not
            raise RuntimeError("Data Parallel only supports batch dim on dimension 0!")

        self.batch_dim_map: Dict[fx.Node, int] = {}
        # batch dim size is used to track the batch dim size of the input tensor
        self.batch_dim_size = -1

        self.dim_rule_map: Dict[torch._ops.OpOverload, Callable[..., torch.Tensor]] = {
            aten.squeeze.default: torch.squeeze,
            aten.squeeze.dim: torch.squeeze,
            aten.view.default: Tensor.view,
            aten.reshape.default: torch.reshape,
            aten._unsafe_view.default: Tensor.view,
            aten.unsqueeze.default: torch.unsqueeze,
            aten.expand.default: Tensor.expand,
            aten.permute.default: torch.permute,
            aten.repeat.default: Tensor.repeat,
            aten.transpose.int: torch.transpose,
        }

    def init_batch_dim_size(self, batch_dim_size: int) -> None:
        """
        initialize batch dim size base on the first input batch size
        """
        if self.batch_dim_size != -1 and self.batch_dim_size != batch_dim_size:
            raise RuntimeError(
                f"batch dim size is already initialized! "
                f"Found new batch size: {batch_dim_size} not "
                f"matching existing batch dim size: {self.batch_dim_size}!"
            )
        self.batch_dim_size = batch_dim_size

    def set_batch_dim(self, node: fx.Node, batch_dim: int) -> None:
        self.batch_dim_map[node] = batch_dim

    def get_batch_dim(self, node: fx.Node) -> int:
        if node not in self.batch_dim_map:
            raise RuntimeError(f"batch dim analysis failed on node: {node}!")
        return self.batch_dim_map[node]

    def compute_batch_dim(self, node: fx.Node, full_reduction=False) -> int:
        """
        compute the batch dimension for the `node`
        """
        assert self.batch_dim_size != -1, "batch dim size is not initialized!"

        if node in self.batch_dim_map:
            # if batch dim already computed, simply return it
            return self.batch_dim_map[node]

        if node.target in self.dim_rule_map:
            view_op_rule = view_op_rules[self.dim_rule_map[node.target]]  # type: ignore[index]
            args_val = pytree.tree_map_only(fx.Node, lambda n: n.meta["val"], node.args)
            kwargs_val = pytree.tree_map_only(
                fx.Node, lambda n: n.meta["val"], node.kwargs
            )
            output_dim_rules = view_op_rule.dim_map(*args_val, **kwargs_val)

            def collect_input_dim(cmd: DimSpec, input_dims: Set[int]):
                if isinstance(cmd, InputDim):
                    input_dims.add(cmd.input_dim)
                for inp in cmd.inputs():
                    collect_input_dim(inp, input_dims)

            output_dim_to_input_dims: List[Set[int]] = []
            for inp in output_dim_rules:
                input_dims: Set[int] = set()
                collect_input_dim(inp, input_dims=input_dims)
                output_dim_to_input_dims.append(input_dims)

            operand = node.all_input_nodes[0]
            operand_batch_dim = self.get_batch_dim(operand)
            for output_dim, input_dims in enumerate(output_dim_to_input_dims):
                if operand_batch_dim in input_dims:
                    self.set_batch_dim(node, output_dim)
                    # update batch dim size before return
                    # this is because batch dim size might change during the middle
                    self.batch_dim_size = node.meta["val"].shape[output_dim]
                    return output_dim

        # if there's no hints from the output_dim_rules, we infer from output
        # shape to see if there's batch dim, and shard correspondingly
        node_val = node.meta["val"]
        if isinstance(node_val, (list, tuple)):
            shapes = [val.shape for val in node_val]
        else:
            shapes = [node_val.shape]

        # for reduction op that reduces over the sharded batch dim
        # we don't generate partial, but rather, we generate shard
        # This is because the intention of data parallel is to never
        # do full reduction across batch dimension, it would still
        # keep the reduction activation as sharded.
        full_reduction = False
        # loop through the dim size to find the output batch dim
        for shape in shapes:
            if len(shape) == 0:
                full_reduction = True

            for i, dim_size in enumerate(shape):
                if dim_size == self.batch_dim_size:
                    self.set_batch_dim(node, i)
                    return i

        operands = node.all_input_nodes
        if not operands:
            # if there's no operands, it must be factory ops and it's a tensor
            # generated for computation and should be marked as replicated
            self.set_batch_dim(node, -1)
            # -1 means replicated
            return -1
        else:
            # if there's operand we see the operand have batch dim, if operand
            # have batch dim but output does not, it's either a full reduction,
            # where we should stay sharded, or it's a reduction on batch dim only
            # where we should produce partial
            operand_batch_dim = -1
            for operand in operands:
                if operand in self.batch_dim_map:
                    operand_batch_dim = self.get_batch_dim(operand)
            # self.get_batch_dim(operands[0])
            if operand_batch_dim < 0:
                # if operand does not have batch dim, we also don't have batch dim
                self.set_batch_dim(node, operand_batch_dim)
                return operand_batch_dim
            elif full_reduction:
                self.set_batch_dim(node, operand_batch_dim)
                return operand_batch_dim
            else:
                # if operand have batch dim but output does not, it should
                # produce partial, we use -2 to indicate partial
                self.set_batch_dim(node, -2)
                return -2

    def compute_act_spec(self, node: fx.Node, mesh: DeviceMesh) -> DTensorSpec:
        """
        This function first compute the batch dimension for the current node,
        then generate the sharding spec that shards on the batch dimension.
        """
        node_batch_dim = self.compute_batch_dim(node)
        if node_batch_dim == -1:
            # indicate this activation is replicated
            act_spec = DTensorSpec(mesh=mesh, placements=[Replicate()])
        elif node_batch_dim == -2:
            # indicate this activation is partial
            act_spec = DTensorSpec(mesh=mesh, placements=[_Partial()])
        else:
            # indicate this activation is Shard
            act_spec = DTensorSpec(mesh=mesh, placements=[Shard(node_batch_dim)])

        return act_spec
