import dataclasses
import weakref
from typing import Any, Callable, List, Tuple, Optional, Dict, Union

import sympy
from collections import namedtuple

import torch
import torch._dynamo
import torch.fx
from . import graph_module
from torch._decomp import core_aten_decompositions
from torch._dynamo.eval_frame import Constraint

import torch.utils._pytree as pytree
from torch._export.pass_base import PassType
from torch.fx.experimental.symbolic_shapes import (
    ConstraintViolationError,
    GuardOnDataDependentSymNode,
    StrictMinMaxConstraint,
)
from torch._dynamo.exc import UserError, UserErrorType
from torch.fx._compatibility import compatibility
from torch.fx.passes.pass_manager import PassManager
from torch.utils._sympy.value_ranges import ValueRanges, ValueRangeError

Value = Any

ExportGraphModule = torch.fx.GraphModule
EXPORT_METADATA = "_export_metadata_key"


# Note - [On Export Dynamic Dimension UX]
#
# After a lot of discussion, we have settled on a dynamic marking API
# for export that meets the following constraints:
# 1) Stateless
# 2) Safe for numerous .export calls within a single process
# 3) Simple to use
# 4) Can be extended to constraints easily
#
# While the underlying API is still torch._dynamo.mark_dynamic, we offer a higher
# level API that meets the constraints above.
#
# This API produces an object that is meant to be passed into torch._dynamo.export
# constraints field. See docs on torch._dynamo.export for more details.
#
# Note - The output type and structure here is NOT BC and NOT A CONTRACT, we reserve
# the right to change the output here at any time, and will do so as we extend the API.
#
# result = torch._dynamo.export(
#     my_model,
#     *sixtyfour_tensors,
#     constraints=[
#         # if you do only dynamic_dim, this is sugar for
#         # -Inf <= dynamic_dim(blah, 0) <= Inf; we don’t otherwise
#         # permit direct int->bool conversion
#         dynamic_dim(blah, 0),
#         # operator overloading because it makes it clear whether
#         # or not you’re inclusive-exclusive range or not
#         0 <= dynamic_dim(blah, 1) <= 100,
#         # NB: But we actually truncate ranges to be >= 2, because of
#         # 0/1 specialization
#     ]
# )
def dynamic_dim(t: torch.Tensor, index: int):
    return Constraint(
        weakref.ref(t),
        id(t),
        index,
        StrictMinMaxConstraint(
            vr=ValueRanges(lower=2, upper=sympy.oo), warn_only=False
        ),
    )


@dataclasses.dataclass
class ExportDynamoConfig:
    """
    Manage Export-specific configurations of Dynamo.
    TODO add tests to make sure the flags are not outdated
    """
    capture_scalar_outputs: bool = True
    capture_dynamic_output_shape_ops: bool = True
    guard_nn_modules: bool = True
    dynamic_shapes: bool = True
    specialize_int: bool = True
    allow_rnn: bool = True


DECOMP_TABLE = core_aten_decompositions()


def _export(
    f: Callable,
    args: Tuple[Value],
    constraints: Optional[List[Constraint]] = None,
) -> torch.fx.GraphModule:
    """
    Private API to export a single entry point or a free function. It is meant to be used
    inside top level torch.export.
    """
    if constraints is None:
        constraints = []

    with torch._dynamo.config.patch(dataclasses.asdict(ExportDynamoConfig())):  # type: ignore[attr-defined]
        try:
            gm, _ = torch._dynamo.export(
                f,
                *args,
                aten_graph=True,
                tracing_mode="symbolic",
                decomposition_table=DECOMP_TABLE,
                constraints=constraints,
                assume_static_by_default=True,
            )
        except (ConstraintViolationError, ValueRangeError) as e:
            raise UserError(UserErrorType.CONSTRAIN_VIOLATION, str(e))
        except GuardOnDataDependentSymNode as e:
            raise UserError(
                UserErrorType.ANTI_PATTERN,
                f"Consider annotating your code using constrain_as_*(). {str(e)}")

    flat_args, in_spec = pytree.tree_flatten(args)
    out_spec = (
        gm.graph._codegen.pytree_info.out_spec or pytree.tree_flatten(f(*args))[1]  # type: ignore[attr-defined]
    )
    # TODO: Track mutation
    mutation = None
    export_graph_module = graph_module.make_export_graph_module(
        gm, gm.graph, in_spec, out_spec, mutation, flat_args
    )
    return export_graph_module


# MultiMethodExportedProgram represents an exported program that contains
# multiple methods, all as valid entry points to the program.
#
# Internally, each method is represented as a separate ExportGraphModule.
# Methods (ExportGraphModule's) do not share anything with each other to
# ensure that each is self-contained. This is important because transformation
# passes can be local and do not need to concern themselves about other methods
# that exists on the same MultiMethodExportedProgram.
# TODO(gmagogsfm): Replace ExportedProgram with MultiMethodExportedProgram.


@compatibility(is_backward_compatible=False)
class MultiMethodExportedProgram:
    def __init__(self, gms: Dict[str, graph_module.ExportGraphModule]):
        # TODO(gmagogsfm): Support merging use case where user started by creating
        # an empty MultiMethodExportedProgram and then start adding more
        # graph modules to it.
        assert (
            len(gms) > 0
        ), "Expected at least 1 graph module in MultiMethodExportedProgram"
        self._method_to_graph_module = gms

    # Get the default method, which is either the only method contained
    # in this MultiMethodExportedProgram or the method named `forward`.
    # TODO(gmagogsfm):Throw when there is only a single non-forward method in the program
    def _get_default_method(self):
        if len(self._method_to_graph_module) == 1:
            return next(iter(self._method_to_graph_module.values()))
        elif "forward" in self._method_to_graph_module:
            return self._method_to_graph_module["forward"]
        else:
            return None

    def save(self) -> None:
        # TODO(gmagogsfm): Implement.
        raise NotImplementedError()

    def load(self) -> None:
        # TODO(gmagogsfm): Implement.
        raise NotImplementedError()

    def find_method(self, name: str) -> Optional[torch.nn.Module]:
        return self._method_to_graph_module.get(name)

    def merge(self, other: "MultiMethodExportedProgram"):
        for method_name, gm in other.methods().items():
            assert (
                method_name not in self._method_to_graph_module
            ), f"There already is a method named {method_name} in this program"
            self._method_to_graph_module[method_name] = gm

    def transform(self, *passes: PassType) -> "MultiMethodExportedProgram":
        pm = PassManager(list(passes))

        def apply_passes(
            gm: graph_module.ExportGraphModule,
        ) -> graph_module.ExportGraphModule:
            transformed = pm(gm).graph_module
            assert transformed is not None
            transformed.meta.update(gm.meta)
            return transformed

        method_name_to_transformed_graph_modules = {
            method_name: apply_passes(gm)
            for method_name, gm in self._method_to_graph_module.items()
        }
        return MultiMethodExportedProgram(method_name_to_transformed_graph_modules)

    def methods(self) -> Dict[str, graph_module.ExportGraphModule]:
        return self._method_to_graph_module

    def __call__(self, *args: Value, **kwargs: Value) -> Value:
        gm = self._get_default_method()

        assert (
            gm is not None
        ), """MultiMethodExportedProgram can not be called directly unless "
        "it only contains a single method or it contains a `forward` method. "
        "Please look up one of its methods first via "
        "`MultiMethodExportedProgram.find_method(method_name)`."""

        return gm(*args, **kwargs)

    def __repr__(self) -> str:
        # TODO(gmagogsfm): Implement.
        raise NotImplementedError()

    def __str__(self) -> str:
        # TODO(gmagogsfm): Implement a real one.
        return super().__str__()

    def access_property_of_default_method(self, property_name: str):
        default_module = self._get_default_method()
        assert (
            default_module is not None
        ), f"""Exported program contains more than one methods and none of them "
        "is named `forward`, it is impossible to identify the default method. "
        "please look up one of its methods first via `find_method(method_name)` "
        "to access property: {property_name}."""
        return getattr(default_module, property_name)

    @property
    def meta(self):
        return self.access_property_of_default_method("meta")

    @property
    def in_spec(self):
        return self.meta[graph_module.EXPORT_METADATA].in_spec

    @property
    def out_spec(self):
        return self.meta[graph_module.EXPORT_METADATA].out_spec

    @property
    def graph(self):
        return self.access_property_of_default_method("graph")

    @property
    def code(self):
        return self.access_property_of_default_method("code")

    @property
    def module(self):
        default_method = self._get_default_method()
        assert (
            default_method is not None
        ), """Exported program contains more than"
        " one methods and none of them is named `forward`,"
        " it is impossible to identify the default method "
        "to fetch GraphModule for."""
        return default_method

    # TODO(gmagogsfm): Implement custom __reduce__ to account for lost of
    # meta['val']


CompileSpec = namedtuple("CompileSpec", ["method_name", "callable", "args"])


@compatibility(is_backward_compatible=False)
def export(
    m: Union[torch.nn.Module, Callable[..., Any]],
    args: Union[Dict[str, Tuple[Value, ...]], Tuple[Value, ...]],
    constraints: Optional[List[Constraint]] = None,
):
    """
    capture_multiple traces either an nn.Module or just a callable with PyTorch
    operations inside and produce a single MultiMethodExportedProgram that
    can potentially have multiple entry points. When multiple entry points
    are traced, each of them is stored separately in the resulting
    MultiMethodExportedProgram without sharing state.

    Args:
        m: the `nn.Module` or callable to trace.

        args: Tracing example inputs.

        When `m` is an nn.Module, `args` can be
        1) A dictionary that maps names of method to their tracing example inputs.
        in this case, all specified methods will be captured.
        2) A tuple. In this case, only the `forward` method of `m` will be captured.
        It is equivalent to passing {"forward", tuple-type-args}

        When `m` is a non-Module callable, `args` must be a Tuple containing
        tracing example inputs.

        # TODO(gmagogsfm): Write tutorial on how to use them
        constraints: A list of constraints on the dynamic arguments specifying
        their possible range of their shapes

    Returns:
        A MultiMethodExportedProgram.

        if `m` is an nn.Module, returned program would have multiple
        captured methods, each corresponding to one entry in args dictionary.

        if `m` is a non-Module callable, returned program would have a single
        captured method named `forward`.

    Raises:
        AssertionError if given method name do not reference a valid method
        on the given nn.Module.
    """
    # Normalize m and args.
    compile_specs = []
    if isinstance(m, torch.nn.Module):
        if isinstance(args, tuple):
            compile_specs.append(CompileSpec("forward", m.forward, args))
        else:
            assert isinstance(
                args, dict
            ), f"Expected a tuple or Dict[str, tuple], got {type(args)}"
            for method_name, method_args in args.items():
                compile_specs.append(
                    CompileSpec(method_name, getattr(m, method_name), method_args)
                )
    else:
        # Reaching here means `m` is a non-Module callable.
        assert isinstance(
            m, Callable  # type: ignore[arg-type]
        ), f"Only nn.Module or callable allowed, got {type(m)}"
        assert isinstance(
            args, tuple
        ), f"When tracing a non-Module callable, `args` must be a tuple of tracing inputs, but got {type(args)}"
        compile_specs.append(CompileSpec("forward", m, args))

    method_name_to_graph_module: Dict[str, torch.fx.GraphModule] = {}
    for compile_spec in compile_specs:
        method_name_to_graph_module[compile_spec.method_name] = _export(
            compile_spec.callable, compile_spec.args, constraints
        )

    return MultiMethodExportedProgram(method_name_to_graph_module)
