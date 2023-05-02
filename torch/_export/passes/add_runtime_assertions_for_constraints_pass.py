from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple, Optional

import math
import operator
import sympy

import torch.utils._pytree as pytree
import torch
import torch.fx
from torch._export.pass_base import ExportPassBase, ProxyValue
from torch._export.graph_module import get_export_meta
from torch._export.pass_infra.node_metadata import NodeMetadata


__all__ = ["AddRuntimeAssertionsForConstraintsPass"]


ConstraintSpec = namedtuple("ConstraintSpec", ["constraint_dim", "min_val", "max_val"])


class AddRuntimeAssertionsForConstraintsPass(ExportPassBase):
    def __init__(self) -> None:
        super().__init__()
        self.current_gm = None

    def _create_dummy_node_metadata(self, fake_mode, example_value: Optional[torch.Tensor] = None) -> NodeMetadata:
        val = example_value if example_value is not None else torch.empty(0)
        assert isinstance(val, torch.Tensor)

        dummy_meta_dict = {}
        dummy_meta_dict["val"] = fake_mode.from_tensor(val)
        dummy_meta_dict["stack_trace"] = ""
        dummy_meta_dict["nn_module_stack"] = {}
        dummy_meta_dict["debug_handle"] = ""
        return NodeMetadata(dummy_meta_dict)

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

    def call(self, graph_module: torch.fx.GraphModule) -> None:
        # resets the counter
        self.input_tracker = 0
        self.current_gm = graph_module
        self.constraints = self._process_constraints(get_export_meta(self.current_gm).input_shape_constraints)
        self.example_inputs = pytree.tree_flatten(get_export_meta(self.current_gm).example_inputs)[0]
        return super().call(graph_module)

    def _insert_specialized_shapes_assert(self, arg, dims, meta, current_inp):
        current_fake_mode = meta["val"].fake_mode
        # we don't want to get shapes from meta as they will be symbolic
        shapes = current_inp.shape
        for dim in dims:
            assert_msg = (
                f"Input #{self.input_tracker}'s dimension #{dim} size is "
                f"specialized at {shapes[dim]}"
            )
            dim_node = self._fx(
                "call_function",
                torch.ops.aten.sym_size,
                (arg, dim),
                {},
                self._create_dummy_node_metadata(current_fake_mode),
                self.interpreter,
            )
            eq = self._fx(
                "call_function",
                operator.eq,
                (dim_node, shapes[dim]),
                {},
                self._create_dummy_node_metadata(current_fake_mode),
                self.interpreter,
            )
            tensor_eq = self._fx(
                "call_function",
                torch.ops.aten.scalar_tensor.default,
                (eq,),
                {},
                self._create_dummy_node_metadata(current_fake_mode, torch.tensor(1)),
                self.interpreter,
            )
            self._fx(
                "call_function",
                torch.ops.aten._assert_async.msg,
                (tensor_eq, assert_msg),
                {},
                self._create_dummy_node_metadata(current_fake_mode),
                self.interpreter,
            )

    def placeholder(self, name: str, arg, meta) -> ProxyValue:
        arg = super().placeholder(name, arg, meta)
        assert self.current_gm is not None
        current_inp = self.example_inputs[self.input_tracker]
        current_fake_mode = meta["val"].fake_mode
        all_dims = set(range(current_inp.dim()))

        # If no dynamism is specified, we assume all dimensions are specialized
        if id(current_inp) not in self.constraints:
            self._insert_specialized_shapes_assert(arg, all_dims, meta, current_inp)
            self.input_tracker += 1
            return arg

        constraints = self.constraints[id(current_inp)]

        constrained_dims = set()
        # Add runtime asserts for user specified constraints for each
        # individual dimensions (e.g not the relational constraints like
        # x[1] == x[0])
        for constraint in constraints:
            constrained_dims.add(constraint.constraint_dim)
            dim = self._fx(
                "call_function",
                torch.ops.aten.sym_size,
                (arg, constraint.constraint_dim),
                {},
                self._create_dummy_node_metadata(current_fake_mode),
                self.interpreter,
            )
            assert_msg = (
                f"Input #{self.input_tracker}'s dimension #{constraint.constraint_dim} size is "
                f"outside of specified dynamic range [{constraint.min_val}, {constraint.max_val}]"
            )

            if constraint.min_val > 2:
                ge = self._fx(
                    "call_function",
                    operator.ge,
                    (dim, constraint.min_val),
                    {},
                    self._create_dummy_node_metadata(current_fake_mode),
                    self.interpreter,
                )
                tensor_ge = self._fx(
                    "call_function",
                    torch.ops.aten.scalar_tensor.default,
                    (ge,),
                    {},
                    self._create_dummy_node_metadata(current_fake_mode, torch.tensor(1)),
                    self.interpreter,
                )
                self._fx(
                    "call_function",
                    torch.ops.aten._assert_async.msg,
                    (tensor_ge, assert_msg),
                    {},
                    self._create_dummy_node_metadata(current_fake_mode),
                    self.interpreter,
                )

            if constraint.max_val < math.inf:
                le = self._fx(
                    "call_function",
                    operator.le,
                    (dim, constraint.max_val),
                    {},
                    self._create_dummy_node_metadata(current_fake_mode),
                    self.interpreter,
                )
                tensor_le = self._fx(
                    "call_function",
                    torch.ops.aten.scalar_tensor.default,
                    (le,),
                    {},
                    self._create_dummy_node_metadata(current_fake_mode, torch.tensor(1)),
                    self.interpreter
                )
                self._fx(
                    "call_function",
                    torch.ops.aten._assert_async.msg,
                    (tensor_le, assert_msg),
                    {},
                    self._create_dummy_node_metadata(current_fake_mode),
                    self.interpreter,
                )

        specialized_dims = all_dims - constrained_dims
        # Make all non-constrained dims to be static
        self._insert_specialized_shapes_assert(arg, specialized_dims, meta, current_inp)

        # TODO Add relational constraints
        self.input_tracker += 1
        return arg
    # TODO implement adding inline constraints as assertion
