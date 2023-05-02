from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple

import math
import operator
import sympy

import torch.utils._pytree as pytree
import torch
import torch.fx
from torch.fx.passes.infra.pass_base import PassResult
from torch._export.pass_base import ExportPassBase, ProxyValue
from torch._export.graph_module import get_export_meta
from torch._export.pass_infra.node_metadata import NodeMetadata


__all__ = ["AddRuntimeAssertionsForConstraintsPass"]


ConstraintSpec = namedtuple("ConstraintSpec", ["constraint_dim", "min_val", "max_val"])


class AddRuntimeAssertionsForConstraintsPass(ExportPassBase):
    def __init__(self) -> None:
        super().__init__()
        self.current_gm: Optional[torch.fx.GraphModule] = None

    def _process_constraints(self, constraints) -> Dict[int, List[ConstraintSpec]]:
        constraints_id_to_constraint: Dict[int, List[ConstraintSpec]] = defaultdict(
            list
        )
        if constraints is None:
            return constraints_id_to_constraint

        # Convert simple sympy Integers into concrete int to
        # insert into graph
        def _convert_to_int(val):
            if val == sympy.oo:
                return math.inf
            if isinstance(val, sympy.Integer):
                return int(val)
            raise RuntimeError(
                "Export constraints cannot be non-integer expressions"
            )

        constraint_id_to_dim: Dict[int, Dict[int, List[Tuple[int, int]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for constraint in constraints:
            min_int_val = _convert_to_int(constraint["min"])
            max_int_val = _convert_to_int(constraint["max"])
            constraint_id_to_dim[constraint["t_id"]][constraint["dim"]].append(
                (min_int_val, max_int_val)
            )

        # Merge the constraints into a single list of constraints
        for t_id in constraint_id_to_dim:
            for dim, constraints in constraint_id_to_dim[t_id].items():
                min_vals = [x[0] for x in constraints]
                max_vals = [x[1] for x in constraints]
                min_val = sorted(min_vals, reverse=True)[0]
                max_val = sorted(max_vals, reverse=False)[0]

                assert min_val <= max_val

                constraints_id_to_constraint[t_id].append(
                    ConstraintSpec(constraint_dim=dim, min_val=min_val, max_val=max_val)
                )

        return constraints_id_to_constraint

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        # resets the counter
        self.input_tracker = 0
        self.current_gm = graph_module
        assert isinstance(self.current_gm, torch.fx.GraphModule)
        self.constraints = self._process_constraints(get_export_meta(self.current_gm).input_shape_constraints)
        self.example_inputs = pytree.tree_flatten(get_export_meta(self.current_gm).example_inputs)[0]
        return super().call(graph_module)

    def _insert_specialized_shapes_assert(self, arg, dims, current_inp):
        # we don't want to get shapes from meta as they will be symbolic
        shapes = current_inp.shape
        for dim in dims:
            assert_msg = (
                f"Input #{self.input_tracker}'s dimension #{dim} size is "
                f"specialized at {shapes[dim]}"
            )
            dim_node = self.call_operator(
                torch.ops.aten.sym_size,
                (arg, dim),
                {},
                NodeMetadata({}),
            )
            eq = self.call_operator(
                operator.eq,
                (dim_node, shapes[dim]),
                {},
                NodeMetadata({}),
            )
            tensor_eq = self.call_operator(
                torch.ops.aten.scalar_tensor.default,
                (eq,),
                {},
                NodeMetadata({}),
            )
            self.call_operator(
                torch.ops.aten._assert_async.msg,
                (tensor_eq, assert_msg),
                {},
                NodeMetadata({}),
            )

    def placeholder(self, name: str, arg, meta) -> ProxyValue:
        arg = super().placeholder(name, arg, meta)
        assert self.current_gm is not None
        current_inp = self.example_inputs[self.input_tracker]
        all_dims = set(range(current_inp.dim()))

        # If no dynamism is specified, we assume all dimensions are specialized
        if id(current_inp) not in self.constraints:
            self._insert_specialized_shapes_assert(arg, all_dims, current_inp)
            self.input_tracker += 1
            return arg

        constraints = self.constraints[id(current_inp)]

        constrained_dims = set()
        # Add runtime asserts for user specified constraints for each
        # individual dimensions (e.g not the relational constraints like
        # x[1] == x[0])
        for constraint in constraints:
            constrained_dims.add(constraint.constraint_dim)
            dim = self.call_operator(
                torch.ops.aten.sym_size,
                (arg, constraint.constraint_dim),
                {},
                NodeMetadata({}),
            )
            assert_msg = (
                f"Input #{self.input_tracker}'s dimension #{constraint.constraint_dim} size is "
                f"outside of specified dynamic range [{constraint.min_val}, {constraint.max_val}]"
            )

            # TODO (tmanlaibaatar) we are making an assumption that graph generated for
            # input dim N >=2 generalizes to N < 2. Ideally we should check that:
            # 1. if we can generalize to N < 2, not add any assertion saying N >= 2
            # 2. If we can't generalize to N < 2, add an assertion saying N >= 2
            # Above can be achieved via a seperate pass.
            if constraint.min_val > 2:
                ge = self.call_operator(
                    operator.ge,
                    (dim, constraint.min_val),
                    {},
                    NodeMetadata({}),
                )
                tensor_ge = self.call_operator(
                    torch.ops.aten.scalar_tensor.default,
                    (ge,),
                    {},
                    NodeMetadata({}),
                )
                self.call_operator(
                    torch.ops.aten._assert_async.msg,
                    (tensor_ge, assert_msg),
                    {},
                    NodeMetadata({}),
                )

            if constraint.max_val < math.inf:
                le = self.call_operator(
                    operator.le,
                    (dim, constraint.max_val),
                    {},
                    NodeMetadata({}),
                )
                tensor_le = self.call_operator(
                    torch.ops.aten.scalar_tensor.default,
                    (le,),
                    {},
                    NodeMetadata({}),
                )
                self.call_operator(
                    torch.ops.aten._assert_async.msg,
                    (tensor_le, assert_msg),
                    {},
                    NodeMetadata({}),
                )

        specialized_dims = all_dims - constrained_dims
        # Make all non-constrained dims to be static
        self._insert_specialized_shapes_assert(arg, specialized_dims, current_inp)

        # TODO Add relational constraints
        self.input_tracker += 1
        return arg
    # TODO implement adding inline constraints as assertion
