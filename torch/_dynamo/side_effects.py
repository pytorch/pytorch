"""
Side effect tracking and management for TorchDynamo's compilation system.

This module provides infrastructure for tracking and managing side effects that occur
during symbolic execution, including:

- Tracking mutations to objects, attributes, and variables
- Managing context changes (cell variables, global namespace modifications)
- Handling aliasing and object identity preservation
- Managing stack frame state and local variable changes
- Tracking function calls with side effects

Key classes:
- SideEffects: Main container for tracking all side effects during execution
- MutableSideEffects: Specialization for mutable object tracking
- AttributeMutation/ValueMutation: Track specific types of mutations
- Various specialized side effect classes for different scenarios

The side effect system ensures that mutations performed during symbolic execution
are properly replayed during runtime, maintaining the correctness of compiled code
while enabling optimizations where safe.
"""

import collections
import contextlib
import inspect
import logging
import textwrap
import traceback
import types
import warnings
import weakref
from collections.abc import Generator, MutableMapping
from types import CellType
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.nn
from torch._dynamo.variables.misc import AutogradFunctionContextVariable
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import is_structseq_class

from . import config, graph_break_hints, utils, variables
from .bytecode_analysis import livevars_analysis
from .bytecode_transformation import (
    bytecode_from_template,
    create_call_function,
    create_call_method,
    create_instruction,
)
from .codegen import PyCodegen
from .exc import collapse_resume_frames, get_stack_above_dynamo, unimplemented
from .source import AttrSource, GlobalSource, LocalCellSource, Source, TempLocalSource
from .utils import (
    is_frozen_dataclass,
    is_namedtuple_cls,
    istype,
    nn_module_new,
    object_new,
)
from .variables.base import (
    AttributeMutation,
    AttributeMutationExisting,
    AttributeMutationNew,
    AttrMutationKind,
    is_side_effect_safe,
    ValueMutationExisting,
    ValueMutationNew,
    VariableTracker,
)
from .variables.user_defined import FrozenDataClassVariable


if TYPE_CHECKING:
    from torch._dynamo.output_graph import OutputGraph
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase
    from torch._dynamo.variables.lists import ListVariable


side_effects_log = torch._logging.getArtifactLogger(__name__, "side_effects")


def _manual_dict_setitem(
    dict_from: dict[Any, Any], dict_to: dict[Any, Any], mro_index: int
) -> None:
    # Carefully calls the dict or OrderedDict `clear` or `__setitem__`. We have
    # to be careful because we don't want to trigger the user defined object
    # setitem or clear. The mro_index is used to find the dict/OrderedDict from
    # the class mro.
    dict_class = type(dict_to).__mro__[mro_index]
    dict_class.clear(dict_to)  # type: ignore[attr-defined]
    for k, v in dict_from.items():
        dict_class.__setitem__(dict_to, k, v)  # type: ignore[index]


def _manual_list_update(list_from: list[Any], list_to: list[Any]) -> None:
    list.clear(list_to)
    list.extend(list_to, list_from)


class SideEffects:
    """
    Maintain records of mutations and provide methods to apply them during code generation.

    Handles tracking and applying side effects during PyTorch Dynamo compilation,
    maintaining Python semantics by managing mutations, attribute modifications,
    and other side effects that occur during program execution.

    Key responsibilities:
    - Tracks mutations to Python objects, lists, and dictionaries that need to be
    applied after an FX graph is run.
    - Manages attribute modifications and deletions
    - Handles tensor hooks and backward pass state
    - Tracks cell variable mutations and global variable changes
    - Ensures correct ordering and application of side effects after graph execution

    This ensures that optimized code behaves identically to the original Python code with
    respect to object mutations and other side effects.
    """

    id_to_variable: dict[int, VariableTracker]
    store_attr_mutations: dict[VariableTracker, dict[str, VariableTracker]]
    attr_mutation_kinds: dict[VariableTracker, dict[str, AttrMutationKind]]
    keepalive: list[Any]
    # Maps variable tracker to list of user stacks (StackSummary objects, formatted lazily)
    mutation_user_stacks: dict[VariableTracker, list[traceback.StackSummary]]

    def __init__(
        self,
        output_graph: "OutputGraph",
        id_to_variable: dict[int, VariableTracker] | None = None,
        store_attr_mutations: dict[VariableTracker, dict[str, VariableTracker]]
        | None = None,
        attr_mutation_kinds: dict[VariableTracker, dict[str, AttrMutationKind]]
        | None = None,
        mutation_user_stacks: dict[VariableTracker, list[traceback.StackSummary]]
        | None = None,
        keepalive: list[Any] | None = None,
        save_for_backward: list[
            tuple[AutogradFunctionContextVariable, list[VariableTracker]]
        ]
        | None = None,
        tensor_hooks: dict[
            int,
            tuple[
                "variables.TensorVariable",
                VariableTracker,
                "variables.RemovableHandleVariable",
                str,
            ],
        ]
        | None = None,
        mutated_dict_backing_ids: set[int] | None = None,
        dict_backing_mutation_versions: dict[int, int] | None = None,
    ) -> None:
        super().__init__()
        self.output_graph_weakref = weakref.ref(output_graph)
        self.id_to_variable = id_to_variable or {}
        self.store_attr_mutations = store_attr_mutations or {}
        self.attr_mutation_kinds = attr_mutation_kinds or {}
        self.mutation_user_stacks = mutation_user_stacks or {}
        self.keepalive = keepalive or []
        self.save_for_backward = save_for_backward or []
        self.tensor_hooks = tensor_hooks or {}
        # Used by MappingProxyVariable to graph break in case of any mutated
        # dict
        self._has_existing_dict_mutation = False
        self._mutated_dict_backing_ids = mutated_dict_backing_ids or set()
        self._dict_backing_mutation_versions = {}
        if dict_backing_mutation_versions is not None:
            self._dict_backing_mutation_versions.update(dict_backing_mutation_versions)
        else:
            for backing_id in self._mutated_dict_backing_ids:
                self._dict_backing_mutation_versions[backing_id] = 1
        # Track Compiled Autograd final callbacks that must be called at the end of Compiled Autograd backward graph.
        # Only applicable if this graph is created from Dynamo tracing in Compiled Autograd.
        self.ca_final_callbacks_var: ListVariable | None = None

        # Tracks VariableTracker objects whose mutations can be skipped.
        # For normal mutated variables, Dynamo generates code to replay/reconstruct
        # the mutations after graph execution. However, variables in this set have
        # their mutations ignored - the mutations happen during
        # execution but don't need to be replayed in the generated code.
        # Used for temporary mutations in contexts like torch.func.functional_call,
        # where module parameters/buffers are modified but later restored.
        self.ignore_mutation_on_these_variables: set[VariableTracker] = set()
        # Sources mutated during tracing: AttrSource for attribute
        # mutations, var.source for value mutations (list/dict/etc).
        self.mutated_sources: OrderedSet[Source] = OrderedSet()

        # Deferred side-effect checking for nullified attribute mutations.
        # Maps (vt_id, attr_name) → (original_value, current_value).
        # On validation, we check original == current.
        self.deferred_attr_mutations: dict[tuple[int, str], tuple[Any, Any]] = {}

    def ignore_mutations_on(self, var: VariableTracker) -> None:
        """Mutations to this variable will be executed but not tracked,
        typically used for temporary mutations that are later restored."""
        self.ignore_mutation_on_these_variables.add(var)

    def stop_ignoring_mutations_on(self, var: VariableTracker) -> None:
        """Remove a variable from the skip mutation set, restoring normal mutation tracking."""
        if var in self.ignore_mutation_on_these_variables:
            self.ignore_mutation_on_these_variables.remove(var)

    @contextlib.contextmanager
    def defer_side_effect_checks(self) -> Generator[None, None, None]:
        """Defer outer-scope attribute mutation checks until tracing completes.

        Context managers that flip-flop a flag (set on enter, restore on exit)
        produce no net side effect. Instead of failing immediately, we track
        original and current values, then validate they match after tracing.

        Note: this context only validates that mutations were nullified — it
        does NOT roll back store_attr_mutations. Callers must restore
        side_effects separately (e.g., via prev_side_effects pattern) to
        discard the mutations after the HOP.
        """
        saved = self.deferred_attr_mutations
        self.deferred_attr_mutations = {}
        try:
            yield
            self.validate_deferred_attr_mutations()
        finally:
            self.deferred_attr_mutations = saved

    def snapshot_attr_mutation(
        self, item: VariableTracker, name: str, value: VariableTracker
    ) -> bool:
        """Record an attribute mutation for deferred validation.

        Returns True if successfully deferred, False if we cannot read the
        original value (var_getattr raises NotImplementedError) or the
        original is not a python constant — caller should fall back to
        check_allowed_side_effect.
        """
        key = (id(item), name)
        if not value.is_python_constant():
            raise AssertionError(
                "value must be a python constant (guaranteed by caller store_attr)"
            )
        current = value.as_python_constant()
        if key in self.deferred_attr_mutations:
            original = self.deferred_attr_mutations[key][0]
        else:
            output_graph = self.output_graph_weakref()
            if output_graph is None:
                raise AssertionError("output_graph weakref is dead")
            tx = output_graph.current_tx
            try:
                original_vt = item.var_getattr(tx, name)  # type: ignore[arg-type]
            except NotImplementedError:
                return False
            if not original_vt.is_python_constant():
                return False
            original = original_vt.as_python_constant()
        self.deferred_attr_mutations[key] = (original, current)
        return True

    def validate_deferred_attr_mutations(self) -> None:
        """Check that all deferred attribute mutations were nullified."""
        for (_, name), (original, current) in self.deferred_attr_mutations.items():
            if original != current:
                unimplemented(
                    gb_type="HOP: Non-nullified side effect",
                    context=f"Attribute '{name}' was not restored to its original value",
                    explanation=f"Attribute '{name}' on an outer-scope object was "
                    f"changed from {original!r} to {current!r} inside a "
                    "higher-order op subgraph. Dynamo only supports mutations "
                    "that are undone before the subgraph exits (e.g., context "
                    "managers that save/restore a flag). If you intentionally "
                    "want this side effect, move the mutation outside of the "
                    "higher-order op.",
                    hints=[*graph_break_hints.FUNDAMENTAL],
                )

    def _capture_user_stack(self, key: VariableTracker) -> None:
        """Capture the current user stack from the instruction translator."""
        if config.side_effect_replay_policy == "silent":
            return
        if key not in self.mutation_user_stacks:
            self.mutation_user_stacks[key] = []
        self.mutation_user_stacks[key].append(
            torch._guards.TracingContext.extract_stack()
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SideEffects):
            raise AssertionError(f"Expected SideEffects, got {type(other)}")
        # NB: do NOT test keepalive
        return (
            self.id_to_variable == other.id_to_variable
            and self.store_attr_mutations == other.store_attr_mutations
            and self.attr_mutation_kinds == other.attr_mutation_kinds
            and self.save_for_backward == other.save_for_backward
            and self.tensor_hooks == other.tensor_hooks
            and self._mutated_dict_backing_ids == other._mutated_dict_backing_ids
            and self._dict_backing_mutation_versions
            == other._dict_backing_mutation_versions
        )

    def diff(self, other: "SideEffects") -> str | None:
        if self.id_to_variable != other.id_to_variable:
            sk_itv = self.id_to_variable.keys()
            ok_itv = other.id_to_variable.keys()
            if sk_itv != ok_itv:
                return f"id_to_variable keys: {sk_itv} != {ok_itv}"
            # Feel free to augment this with more fancy diffing logic
            # if needed for debugging
            return "id_to_variable: unknown diff"
        elif self.store_attr_mutations != other.store_attr_mutations:
            sk_sam = self.store_attr_mutations.keys()
            ok_sam = other.store_attr_mutations.keys()
            if sk_sam != ok_sam:
                return f"store_attr_mutations keys: {sk_sam} != {ok_sam}"
            return "store_attr_mutations: unknown diff"
        elif self.attr_mutation_kinds != other.attr_mutation_kinds:
            return "attr_mutation_kinds: unknown diff"
        elif self.save_for_backward != other.save_for_backward:
            return "save_for_backward"
        elif self.tensor_hooks != other.tensor_hooks:
            return "tensor_hooks"
        elif self._mutated_dict_backing_ids != other._mutated_dict_backing_ids:
            return "mutated_dict_backing_ids"
        elif (
            self._dict_backing_mutation_versions
            != other._dict_backing_mutation_versions
        ):
            return "dict_backing_mutation_versions"
        else:
            return None

    def clone(self) -> "SideEffects":
        """Create a shallow copy"""
        ref = self.output_graph_weakref()
        if ref is None:
            raise AssertionError("output_graph weakref is dead during clone")
        return self.__class__(
            output_graph=ref,
            id_to_variable=dict(self.id_to_variable),
            store_attr_mutations={
                k: dict(v) for k, v in self.store_attr_mutations.items()
            },
            attr_mutation_kinds={
                k: dict(v) for k, v in self.attr_mutation_kinds.items()
            },
            mutation_user_stacks=self.mutation_user_stacks,
            keepalive=list(self.keepalive),
            save_for_backward=self.save_for_backward,
            tensor_hooks=self.tensor_hooks,
            mutated_dict_backing_ids=set(self._mutated_dict_backing_ids),
            dict_backing_mutation_versions=dict(self._dict_backing_mutation_versions),
        )

    def __contains__(self, item: Any) -> bool:
        return id(item) in self.id_to_variable

    def __getitem__(self, item: Any) -> VariableTracker:
        return self.id_to_variable[id(item)]

    def should_allow_externally_visible_side_effects_in_subtracer(self) -> bool:
        output_graph = self.output_graph_weakref()
        return bool(
            output_graph
            and output_graph.current_tx.output.current_tracer.unsafe_allow_externally_visible_side_effects
        )

    def should_allow_side_effects_in_hop(self) -> bool:
        output_graph = self.output_graph_weakref()
        return bool(
            output_graph
            and output_graph.current_tx.output.current_tracer.allow_side_effects_in_hop
        )

    def is_reconstructing_generator(self) -> bool:
        output_graph = self.output_graph_weakref()

        return bool(
            output_graph
            and output_graph.current_tx.output.current_tracer.is_reconstructing_generator
        )

    def _maybe_record_side_effect(self, item: VariableTracker) -> None:
        """Record the first externally-visible side effect on the current tracer."""
        if item.mutation_type is not None and not is_side_effect_safe(
            item.mutation_type
        ):
            output_graph = self.output_graph_weakref()
            if output_graph:
                tracer = output_graph.current_tx.output.current_tracer
                if tracer.side_effect_stack is None:
                    tracer.side_effect_stack = (
                        torch._guards.TracingContext.extract_stack()
                    )

    def check_allowed_side_effect(self, item: VariableTracker) -> bool:
        from torch._dynamo.variables.misc import AutogradFunctionContextVariable

        # People do things like self.dim = dim inside autograd.Function.
        # These are benign.
        if isinstance(item, AutogradFunctionContextVariable):
            return True
        if self.should_allow_externally_visible_side_effects_in_subtracer():
            self._maybe_record_side_effect(item)
            return True
        if self.should_allow_side_effects_in_hop():
            self._maybe_record_side_effect(item)
            return True
        if self.is_reconstructing_generator():
            # This is missing the case where one mutates a tensor. See
            # test_generator.py::test_reconstruct_generator_tensor_mutation
            unimplemented(
                gb_type="Generator reconstruction with mutations",
                context=f"mutating object: {item}",
                explanation="Cannot reconstruct a generator with variable mutations. "
                "Dynamo needs to fully exhaust the generator, which may cause "
                "unintended variable modifications.",
                hints=[
                    "Remove mutations from the generator.",
                    *graph_break_hints.FUNDAMENTAL,
                ],
            )
        if item.mutation_type is None:
            raise AssertionError(
                f"mutation_type is None for {item} in check_allowed_side_effect"
            )
        if not is_side_effect_safe(item.mutation_type):
            unimplemented(
                gb_type="HOP: Unsafe side effect",
                context=f"Attempted to mutate {item}",
                explanation="Mutating a variable from outside the scope of this HOP is not supported.",
                hints=[
                    "If the HOP is activation checkpointing (torch.utils.checkpoint.checkpoint), this points to a "
                    "side effect in forward method. Eager activation checkpointing replays that side-effect while "
                    "recomputing the forward in the backward. If you are ok with side-effect not replayed in the "
                    "backward, try setting `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True`",
                ],
            )
        return False

    def store_attr(
        self,
        item: VariableTracker,
        name: str,
        value: VariableTracker,
        mutation_kind: AttrMutationKind = AttrMutationKind.GENERIC_SETATTR,
    ) -> None:
        if not self.is_attribute_mutation(item):
            raise AssertionError(
                f"Expected attribute mutation for {item} in store_attr"
            )
        backing_id = (
            self._dict_backing_id_for_attribute_mutation(item)
            if name != "__dict__"
            else None
        )
        self._check_live_dict_view_mutation(backing_id)
        # For constant attribute mutations on outer-scope objects, defer
        # the side-effect check and validate after tracing that the
        # mutation was nullified (value restored to original).
        deferred = False
        if (
            isinstance(item.mutation_type, AttributeMutationExisting)
            and not is_side_effect_safe(item.mutation_type)
            and not isinstance(item, AutogradFunctionContextVariable)
            and not self.should_allow_side_effects_in_hop()
            and not self.should_allow_externally_visible_side_effects_in_subtracer()
            and value.is_python_constant()
        ):
            deferred = self.snapshot_attr_mutation(item, name, value)
        if not deferred:
            self.check_allowed_side_effect(item)
        if item not in self.store_attr_mutations:
            self.store_attr_mutations[item] = {}
        if item not in self.attr_mutation_kinds:
            self.attr_mutation_kinds[item] = {}
        self.store_attr_mutations[item][name] = value
        self.attr_mutation_kinds[item][name] = mutation_kind
        # Capture user stack for this mutation
        self._capture_user_stack(item)
        item_source = getattr(item, "source", None)
        if item_source is not None:
            self.mutated_sources.add(AttrSource(item_source, name))
        if backing_id is not None:
            self._mark_dict_backing_id_mutated(backing_id)

    def store_instance_dict_attr(
        self, item: VariableTracker, name: str, value: VariableTracker
    ) -> None:
        self.store_attr(item, name, value, AttrMutationKind.INSTANCE_DICT)

    def get_attr_mutation_kind(
        self, item: VariableTracker, name: str
    ) -> AttrMutationKind:
        if name not in self.store_attr_mutations.get(item, ()):
            return AttrMutationKind.GENERIC_SETATTR
        if name not in self.attr_mutation_kinds.get(item, ()):
            raise AssertionError(f"Missing attribute mutation kind for {item}.{name}")
        return self.attr_mutation_kinds[item][name]

    def load_attr(
        self,
        item: VariableTracker,
        name: str,
        deleted_ok: bool = False,
        check: bool = False,
    ) -> VariableTracker:
        if check:
            if not self.is_attribute_mutation(item):
                raise AssertionError(
                    f"Expected attribute mutation for {item} in load_attr"
                )
        result = self.store_attr_mutations[item][name]
        if not deleted_ok and isinstance(result, variables.DeletedVariable):
            unimplemented(
                gb_type="Attempted to read a deleted variable",
                context=f"item: {item}, name: {name}",
                explanation="",
                hints=[*graph_break_hints.USER_ERROR],
            )
        return result

    def store_cell(self, cellvar: VariableTracker, value: VariableTracker) -> None:
        if cellvar.is_immutable():
            unimplemented(
                gb_type="Write to immutable cell",
                context=f"cellvar: {cellvar}, value: {value}",
                explanation="Dynamo doesn't support writing to immutable/sourceless cell variables.",
                hints=[*graph_break_hints.DIFFICULT],
            )
        if not isinstance(cellvar, variables.CellVariable):
            raise AssertionError(
                f"Expected CellVariable, got {type(cellvar)} in store_cell"
            )
        if not isinstance(value, variables.VariableTracker):
            raise AssertionError(
                f"Expected VariableTracker, got {type(value)} in store_cell"
            )
        self.store_attr(cellvar, "cell_contents", value)

    def load_cell(self, cellvar: VariableTracker) -> VariableTracker:
        if not isinstance(cellvar, variables.CellVariable):
            raise AssertionError(
                f"Expected CellVariable, got {type(cellvar)} in load_cell"
            )
        # Track the cell_contents source during subgraph tracing so that
        # mutations (e.g. nonlocal counter = 3) are detected by the reuse
        # mechanism via set intersection with mutated_sources.
        output_graph = self.output_graph_weakref()
        if output_graph:
            cell_source = getattr(cellvar, "source", None)
            if cell_source is not None:
                output_graph.current_tx.output.current_tracer.traced_sources.add(
                    AttrSource(cell_source, "cell_contents")
                )
        if self.has_pending_mutation_of_attr(cellvar, "cell_contents"):
            return self.load_attr(cellvar, "cell_contents", check=False)
        if cellvar.pre_existing_contents:
            return cellvar.pre_existing_contents
        unimplemented(
            gb_type="Read uninitialized cell",
            context=str(cellvar),
            explanation="Attempted to read a cell variable that has not been populated yet.",
            hints=[*graph_break_hints.USER_ERROR],
        )

    def load_global(self, gvar: VariableTracker, name: str) -> VariableTracker:
        if not isinstance(gvar, variables.VariableTracker):
            raise AssertionError(
                f"Expected VariableTracker, got {type(gvar)} in load_global"
            )
        return self.load_attr(gvar, name)

    def store_global(
        self, gvar: VariableTracker, name: str, value: VariableTracker
    ) -> None:
        if not isinstance(gvar, variables.VariableTracker):
            raise AssertionError(
                f"Expected VariableTracker for gvar, got {type(gvar)} in store_global"
            )
        if not isinstance(value, variables.VariableTracker):
            raise AssertionError(
                f"Expected VariableTracker for value, got {type(value)} in store_global"
            )
        self.store_attr(gvar, name, value)

    @staticmethod
    def cls_supports_mutation_side_effects(cls: type) -> bool:
        return inspect.getattr_static(cls, "__getattribute__", None) in (
            object.__getattribute__,
            dict.__getattribute__,
            set.__getattribute__,
            frozenset.__getattribute__,
            int.__getattribute__,
            str.__getattribute__,
            list.__getattribute__,
            tuple.__getattribute__,
            BaseException.__getattribute__,
        )

    def is_attribute_mutation(self, item: VariableTracker) -> bool:
        return isinstance(item.mutation_type, AttributeMutation)

    def has_pending_mutation(self, item: VariableTracker) -> bool:
        return self.is_attribute_mutation(item) and bool(
            self.store_attr_mutations.get(item)
        )

    def has_pending_mutation_of_attr(
        self,
        item: VariableTracker,
        name: str,
        mutation_kinds: AttrMutationKind | tuple[AttrMutationKind, ...] | None = None,
    ) -> bool:
        if not (
            self.is_attribute_mutation(item)
            and name in self.store_attr_mutations.get(item, ())
        ):
            return False
        if mutation_kinds is None:
            return True
        mutation_kind = self.get_attr_mutation_kind(item, name)
        if isinstance(mutation_kinds, AttrMutationKind):
            return mutation_kind is mutation_kinds
        return mutation_kind in mutation_kinds

    def is_modified(self, item: VariableTracker) -> bool:
        if item.is_immutable():
            return False
        if isinstance(item.mutation_type, (AttributeMutationNew, ValueMutationNew)):
            return True

        if isinstance(item, variables.UserDefinedObjectVariable):
            # Checks if the underlying dict or tuple vt has been modified
            return item in self.store_attr_mutations or item.is_base_vt_modified(self)

        if self.is_attribute_mutation(item):
            return item in self.store_attr_mutations
        if item.mutation_type is None:
            raise AssertionError(f"mutation_type is None for {item} in is_modified")
        return item.mutation_type.is_modified  # type: ignore[attr-defined]

    def _track_obj(
        self,
        item: Any,
        variable: VariableTracker,
        mutation_type_cls: type = ValueMutationExisting,
    ) -> VariableTracker:
        """Start tracking an existing or new variable for mutation"""
        if id(item) in self.id_to_variable:
            raise AssertionError(
                f"{variable} is already tracked for mutation. This could be "
                "because you are not using VariableBuilder to construct "
                "the variable tracker. "
                f"Source of new object: {variable.source}. "
                f"Source of previously tracked object: {self.id_to_variable[id(item)].source}."
            )

        variable.mutation_type = mutation_type_cls()
        self.id_to_variable[id(item)] = variable
        self.keepalive.append(item)
        return variable

    track_mutable = _track_obj

    def track_object_existing(
        self,
        item: Any,
        variable: VariableTracker,
    ) -> VariableTracker:
        # TODO: Modify this API so that we preserve type info of
        # variable
        return self._track_obj(
            item,
            variable,
            mutation_type_cls=AttributeMutationExisting,
        )

    def track_object_new(
        self,
        cls_source: Source | None,
        user_cls: Any,
        variable_cls: Any,
        options: dict[str, Any],
    ) -> VariableTracker:
        if user_cls is torch.autograd.function.FunctionCtx:
            with warnings.catch_warnings(record=True):
                obj = torch.autograd.Function()
        else:
            obj = object_new(user_cls)
        variable = variable_cls(
            obj,
            mutation_type=AttributeMutationNew(cls_source),
            **options,
        )
        self.id_to_variable[id(obj)] = variable
        self.keepalive.append(obj)
        return variable

    def get_variable_cls(self, user_cls: type) -> type:
        from torch.overrides import TorchFunctionMode

        from .variables.ctx_manager import GenericContextWrappingVariable
        from .variables.torch_function import TorchFunctionModeVariable
        from .variables.user_defined import is_forbidden_context_manager

        variable_cls: type[variables.UserDefinedObjectVariable] = (
            variables.UserDefinedObjectVariable
        )
        if issubclass(
            user_cls, TorchFunctionMode
        ) and TorchFunctionModeVariable.is_supported_torch_function_mode(user_cls):
            variable_cls = TorchFunctionModeVariable
        elif (
            hasattr(user_cls, "__enter__")
            and hasattr(user_cls, "__exit__")
            and not is_forbidden_context_manager(user_cls)
        ):
            variable_cls = GenericContextWrappingVariable
        elif issubclass(user_cls, torch.nn.Module):
            variable_cls = variables.UnspecializedNNModuleVariable
        elif issubclass(user_cls, collections.defaultdict):
            variable_cls = variables.DefaultDictVariable
        elif issubclass(user_cls, collections.OrderedDict):
            variable_cls = variables.OrderedDictVariable
        elif issubclass(user_cls, dict):
            variable_cls = variables.UserDefinedDictVariable
        elif issubclass(user_cls, (set, frozenset)):
            variable_cls = variables.UserDefinedSetVariable
        elif issubclass(user_cls, tuple):
            if is_namedtuple_cls(user_cls):
                variable_cls = variables.UserDefinedTupleVariable.get_vt_cls(user_cls)
            else:
                variable_cls = variables.UserDefinedTupleVariable
        elif issubclass(user_cls, list):
            variable_cls = variables.UserDefinedListVariable
        elif issubclass(user_cls, MutableMapping):
            variable_cls = variables.MutableMappingVariable
        elif is_frozen_dataclass(user_cls):
            variable_cls = FrozenDataClassVariable
        elif issubclass(user_cls, BaseException):
            variable_cls = variables.UserDefinedExceptionObjectVariable
        elif issubclass(
            user_cls,
            variables.user_defined._CONSTANT_BASE_TYPES,
        ):
            variable_cls = variables.UserDefinedConstantVariable
        elif variables.InspectVariable.is_matching_class(user_cls):
            variable_cls = variables.InspectVariable
        if not issubclass(variable_cls, variables.UserDefinedObjectVariable):
            raise AssertionError(
                f"Expected subclass of UserDefinedObjectVariable, got {variable_cls}"
            )
        return variable_cls

    def get_example_value(
        self,
        base_cls_vt: VariableTracker,
        cls_vt: VariableTracker,
        init_args: list[VariableTracker],
    ) -> Any:
        user_cls = cls_vt.value  # type: ignore[attr-defined]
        if issubclass(user_cls, torch.nn.Module):
            # TODO(anijain2305) - Is it possible to remove this specialization?
            obj = nn_module_new(user_cls)
        else:
            if isinstance(base_cls_vt, variables.BuiltinVariable):
                base_cls = base_cls_vt.fn
            elif isinstance(base_cls_vt, variables.DictBuiltinVariable):
                base_cls = dict
            elif isinstance(base_cls_vt, variables.ListBuiltinVariable):
                base_cls = list
            elif isinstance(base_cls_vt, variables.UserDefinedClassVariable):
                base_cls = base_cls_vt.value
            else:
                raise RuntimeError(f"Unexpected base_cls_vt {base_cls_vt}")

            if not variables.UserDefinedClassVariable.is_supported_new_method(
                base_cls.__new__
            ):
                raise AssertionError(f"Unsupported __new__ method for {base_cls}")
            if is_structseq_class(user_cls):
                # Structseq tp_new requires a sequence argument and rejects
                # tuple.__new__, so create a dummy with None placeholders.
                obj = user_cls([None] * user_cls.n_fields)
            elif init_args and issubclass(
                user_cls,
                variables.user_defined._CONSTANT_BASE_TYPES,
            ):
                example_args = [arg.as_python_constant() for arg in init_args]
                try:
                    obj = base_cls.__new__(  # pyrefly: ignore[bad-specialization]
                        user_cls, *example_args
                    )
                except Exception:
                    # __new__ can raise (e.g., exceeding int str digit limits).
                    # Fall back to creating without args — the example value is
                    # only used for tracing, not for correctness.
                    obj = base_cls.__new__(  # pyrefly: ignore[bad-specialization]
                        user_cls
                    )
            else:
                try:
                    obj = base_cls.__new__(user_cls)
                except TypeError as exc:
                    # Backstop for direct construction paths that bypass the
                    # UserDefinedClassVariable object.__new__ preflight.
                    unimplemented(
                        gb_type="Unsupported user-defined object construction during side-effect tracking",
                        context=f"class={user_cls}, base={base_cls}, error={exc}",
                        explanation=(
                            "Dynamo could not construct an example object for "
                            "side-effect replay using the class __new__ method."
                        ),
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )
        return obj

    def track_new_user_defined_object(
        self,
        base_cls_vt: VariableTracker,
        cls_vt: VariableTracker,
        init_args: list[VariableTracker],
        *,
        tx: "InstructionTranslatorBase | None" = None,
    ) -> VariableTracker:
        """
        Creates a UserDefinedObjectVariable (or its subclass) variable tracker
        and mark it for attribute mutation tracking.

        Also records the variable trackers to call __new__ method on
        reconstruction. Roughly, the reconstruction looks like this
            base_cls_vt.__new__(user_cls, *init_args)
        """
        cls_source = cls_vt.source
        user_cls = cls_vt.value  # type: ignore[attr-defined]
        variable_cls = self.get_variable_cls(user_cls)
        obj = self.get_example_value(base_cls_vt, cls_vt, init_args)

        kwargs: dict[str, Any] = {}
        if tx is not None:
            kwargs["tx"] = tx
        variable = variable_cls(
            obj,
            cls_source=cls_vt.source,
            base_cls_vt=base_cls_vt,
            init_args=init_args,
            mutation_type=AttributeMutationNew(cls_source),
            **kwargs,
        )
        self.id_to_variable[id(obj)] = variable
        self.keepalive.append(obj)
        return variable

    def track_cell_new(
        self,
    ) -> VariableTracker:
        obj = object()
        variable = variables.CellVariable(
            mutation_type=AttributeMutationNew(),
        )
        self.id_to_variable[id(obj)] = variable
        self.keepalive.append(obj)
        return variable

    def track_cell_existing(
        self, source: Source | None, cell: CellType, contents: VariableTracker
    ) -> VariableTracker:
        variable = variables.CellVariable(
            # We don't support mutation to cell without source because we need
            # source to properly codegen the mutations.
            mutation_type=None if source is None else AttributeMutationExisting(),
            pre_existing_contents=contents,
            source=source,
        )
        self.id_to_variable[id(cell)] = variable
        self.keepalive.append(cell)
        return variable

    def track_global_existing(self, source: Source, item: Any) -> VariableTracker:
        variable = variables.NewGlobalVariable(
            mutation_type=AttributeMutationExisting(),
            source=source,
        )
        self.id_to_variable[id(item)] = variable
        self.keepalive.append(item)
        return variable

    def track_save_for_backward(
        self, ctx: VariableTracker, args: list[VariableTracker]
    ) -> None:
        if not isinstance(ctx, variables.AutogradFunctionContextVariable):
            raise AssertionError(
                f"Expected AutogradFunctionContextVariable, got {type(ctx)}"
            )
        self.save_for_backward.append((ctx, args))

    def track_runahead_tensor_and_symvar_side_effects(
        self, other: "SideEffects"
    ) -> None:
        # In higher order ops we want to keep track of tensors seen in the
        # speculate_subgraph so that we don't lift them again as a new input in
        # other speculate_subgraph or in the root tracer.
        for other_item in other.keepalive:
            other_id = id(other_item)
            other_variable = other.id_to_variable[other_id]
            if other_id not in self.id_to_variable and isinstance(
                other_variable, (variables.TensorVariable, variables.SymNodeVariable)
            ):
                self.track_object_existing(other_item, other_variable)

    def prune_dead_object_new(self, tx: "InstructionTranslatorBase") -> None:
        # Avoid VT cycles from e.g., recursive function.
        visited: set[VariableTracker] = set()
        live_new_objects: set[VariableTracker] = set()

        def visit(var: VariableTracker) -> None:
            if var in visited:
                return
            visited.add(var)
            # Object may have been mutated, store this mutation.
            if isinstance(var.mutation_type, AttributeMutationNew):
                live_new_objects.add(var)
            # It's possible that we have mutated the value of this variable
            # to be another one. The new value is in store_attr_mutations.
            # Also recurse through the new value to detect alive AttributeMutationNew.
            if var in self.store_attr_mutations:
                VariableTracker.visit(
                    visit,  # noqa: F821
                    self.store_attr_mutations[var],
                )

        def is_live(var: VariableTracker) -> bool:
            if isinstance(var.mutation_type, AttributeMutationNew):
                return var in live_new_objects
            return True

        pre_existing_vars = [
            var
            for var in self.id_to_variable.values()
            if not isinstance(var.mutation_type, AttributeMutationNew)
        ]

        # The only live side effects come from returns (tx.stack), any intermediates
        # during a graph break (tx.symbolic_locals), and mutation on pre-existing variables.
        # Recursively visit Variables and see if any of them have been mutated.
        init_live_vars = []
        # gather stack/symbolic_locals for all tx's up the chain
        cur_tx: InstructionTranslatorBase | None = tx
        while cur_tx is not None:
            init_live_vars.extend([cur_tx.stack, cur_tx.symbolic_locals])
            if cur_tx.parent is not None:
                # for non-root tx'es, also keep the cells/freevars alive so they get codegen'd properly
                # TODO see if we could prune dead cells - cell pruning information needs to be forwarded
                # to the resume function creation as well.
                if cur_tx.post_prune_cell_and_freevars is None:
                    raise AssertionError(
                        "post_prune_cell_and_freevars is None for non-root tx"
                    )
                init_live_vars.append(cur_tx.post_prune_cell_and_freevars)
            cur_tx = cur_tx.parent
        VariableTracker.visit(
            visit,
            # TODO track from all possible sources.
            init_live_vars
            + [
                pre_existing_vars,
                tx.output.backward_state,
                self.tensor_hooks,
            ],
        )
        # Manually release the self-referential function, which indirectly
        # captures certain `VariableTracker` and affects parts of PT test/logic
        # that are sensitive to when certain objects get released.
        del visit

        # NB: cell variable handling.is tricky.
        # cell variables must stay alive if any NestedUserFunctionVariable
        # are live. "visit"-ing the NestedUserFunctionVariable visits
        # the .closures field, from which we will see if we need to keep
        # any mutations to cell variables alive.

        self.id_to_variable = {
            k: v for k, v in self.id_to_variable.items() if is_live(v)
        }
        self.store_attr_mutations = {
            k: v for k, v in self.store_attr_mutations.items() if is_live(k)
        }
        self.attr_mutation_kinds = {
            k: v for k, v in self.attr_mutation_kinds.items() if is_live(k)
        }

    def mutation(self, var: VariableTracker) -> None:
        if var in self.ignore_mutation_on_these_variables:
            return

        backing_id = self._dict_backing_id_for_value_mutation(var)
        self._check_live_dict_view_mutation(backing_id)
        self.check_allowed_side_effect(var)
        # Capture user stack for this mutation
        self._capture_user_stack(var)

        if isinstance(var.mutation_type, ValueMutationExisting):
            var.mutation_type.is_modified = True
        if var.source is not None:
            self.mutated_sources.add(var.source)
        if var.source and isinstance(var, variables.ConstDictVariable):
            self._has_existing_dict_mutation = True
        if backing_id is not None:
            self._mark_dict_backing_id_mutated(backing_id)

    def has_existing_dict_mutation(self) -> bool:
        return self._has_existing_dict_mutation

    def has_mutated_dict_backing_id(self, backing_id: int | None) -> bool:
        return backing_id is not None and backing_id in self._mutated_dict_backing_ids

    def has_mutated_dict_backing_id_since(
        self, backing_id: int | None, mutation_version: int
    ) -> bool:
        return (
            backing_id is not None
            and self.dict_backing_mutation_version(backing_id) > mutation_version
        )

    def dict_backing_mutation_version(self, backing_id: int | None) -> int:
        if backing_id is None:
            return 0
        return self._dict_backing_mutation_versions.get(backing_id, 0)

    def _mark_dict_backing_id_mutated(self, backing_id: int) -> None:
        self._mutated_dict_backing_ids.add(backing_id)
        self._dict_backing_mutation_versions[backing_id] = (
            self._dict_backing_mutation_versions.get(backing_id, 0) + 1
        )

    def dict_backing_id_for_variable(self, var: VariableTracker) -> int | None:
        return self._dict_backing_id_for_value_mutation(var)

    def _dict_backing_id(self, value: Any) -> int | None:
        backing_dict = utils.get_underlying_dict(value)
        if backing_dict is None:
            return None
        return id(backing_dict)

    def _tracked_object_id_for_var(self, var: VariableTracker) -> int | None:
        for object_id, tracked_var in self.id_to_variable.items():
            if tracked_var is var or getattr(tracked_var, "_base_vt", None) is var:
                return object_id
        return None

    def _dict_backing_id_for_value_mutation(self, var: VariableTracker) -> int | None:
        if isinstance(var, variables.ConstDictVariable):
            return self._tracked_object_id_for_var(var)
        return None

    def _dict_backing_id_for_attribute_mutation(
        self, item: VariableTracker
    ) -> int | None:
        if isinstance(item, variables.UserDefinedClassVariable):
            return self._dict_backing_id(item.value.__dict__)

        if isinstance(item, variables.UserDefinedObjectVariable):
            value = item.get_real_python_backed_value()
            if value is not variables.base.NO_SUCH_SUBOBJ:
                try:
                    instance_dict = object.__getattribute__(value, "__dict__")
                except AttributeError:
                    return None
                return self._dict_backing_id(instance_dict)

        return None

    def _check_live_dict_view_mutation(self, backing_id: int | None) -> None:
        if backing_id is None or not self._is_dict_view_backing_id_live(backing_id):
            return

        unimplemented(
            gb_type="Dictionary mutation when a dict view is live",
            context=f"Backing dict id: {backing_id}",
            explanation=(
                "Dynamo cannot safely trace a dictionary mutation while a live "
                "dict view that aliases the same dictionary may be observed later."
            ),
            hints=graph_break_hints.SUPPORTABLE,
        )

    def _is_dict_view_backing_id_live(self, backing_id: int) -> bool:
        output_graph = self.output_graph_weakref()
        if output_graph is None:
            return False

        live_backing_ids: set[int] = set()
        seen_real_values: set[int] = set()

        def collect_from_real_value(value: Any) -> None:
            obj_id = id(value)
            if obj_id in seen_real_values:
                return
            seen_real_values.add(obj_id)

            if isinstance(
                value, (utils.dict_keys, utils.dict_values, utils.dict_items)
            ):
                view_backing_id = self._dict_backing_id(value)
                if view_backing_id is not None:
                    live_backing_ids.add(view_backing_id)
                return

            if isinstance(value, (str, bytes, int, float, bool, type(None))):
                return

            if isinstance(value, dict):
                for key, val in value.items():
                    collect_from_real_value(key)
                    collect_from_real_value(val)
                return

            if isinstance(value, (list, tuple, set, frozenset, collections.deque)):
                for item in value:
                    collect_from_real_value(item)
                return

            tracked_var = self.id_to_variable.get(id(value))
            if isinstance(
                tracked_var,
                (
                    variables.UserDefinedObjectVariable,
                    variables.UserDefinedClassVariable,
                    variables.UserFunctionVariable,
                    variables.PythonModuleVariable,
                ),
            ):
                collect_from_python_backed_vt(tracked_var)
                return

            if isinstance(value, CellType):
                tracked_cell = self.id_to_variable.get(id(value))
                if isinstance(tracked_cell, variables.CellVariable):
                    visit_vt(self.load_cell(tracked_cell))
                    return
                try:
                    collect_from_real_value(value.cell_contents)
                except ValueError:
                    pass
                return

            if isinstance(value, types.FunctionType):
                collect_from_real_value(value.__defaults__)
                collect_from_real_value(value.__kwdefaults__)
                collect_from_real_value(value.__dict__)
                collect_from_real_value(value.__closure__)
                return

            if isinstance(value, types.ModuleType):
                if value is torch or value.__name__.startswith("torch."):
                    return
                collect_from_real_value(vars(value))
                return

            if isinstance(value, type):
                for cls in value.__mro__:
                    collect_from_real_value(dict(cls.__dict__))
                return

            try:
                instance_dict = object.__getattribute__(value, "__dict__")
            except AttributeError:
                instance_dict = None
            if instance_dict is not None:
                collect_from_real_value(instance_dict)

            for cls in type(value).__mro__:
                slots = cls.__dict__.get("__slots__")
                if slots is None:
                    continue
                if isinstance(slots, str):
                    slots = (slots,)
                for slot in slots:
                    if slot in ("__dict__", "__weakref__"):
                        continue
                    try:
                        collect_from_real_value(object.__getattribute__(value, slot))
                    except AttributeError:
                        pass

        def collect_from_vt(var: VariableTracker) -> None:
            if isinstance(var, variables.DictViewVariable):
                view_backing_id = getattr(var, "backing_dict_id", None)
                if view_backing_id is not None:
                    live_backing_ids.add(view_backing_id)
            elif isinstance(var, variables.DictKeySetVariable):
                view_backing_id = getattr(var, "backing_dict_id", None)
                if view_backing_id is not None:
                    live_backing_ids.add(view_backing_id)

            if isinstance(
                var,
                (
                    variables.UserDefinedObjectVariable,
                    variables.UserDefinedClassVariable,
                    variables.UserFunctionVariable,
                    variables.PythonModuleVariable,
                ),
            ):
                collect_from_python_backed_vt(var)

        def collect_dict_values(
            values: dict[Any, Any], skip_names: set[str] | None = None
        ) -> None:
            for key, val in values.items():
                if (
                    skip_names is not None
                    and isinstance(key, str)
                    and key in skip_names
                ):
                    continue
                collect_from_real_value(key)
                collect_from_real_value(val)

        def collect_instance_slots(value: Any, skip_names: set[str]) -> None:
            for cls in type(value).__mro__:
                slots = cls.__dict__.get("__slots__")
                if slots is None:
                    continue
                if isinstance(slots, str):
                    slots = (slots,)
                for slot in slots:
                    if slot in ("__dict__", "__weakref__") or slot in skip_names:
                        continue
                    try:
                        collect_from_real_value(object.__getattribute__(value, slot))
                    except AttributeError:
                        pass

        def collect_class_dicts(value: type, skip_names: set[str]) -> None:
            for cls in value.__mro__:
                collect_dict_values(
                    dict(cls.__dict__),
                    skip_names if cls is value else None,
                )

        def collect_from_python_backed_vt(var: VariableTracker) -> None:
            value = var.get_real_python_backed_value()
            if value is variables.base.NO_SUCH_SUBOBJ:
                return

            pending_names = set(self.store_attr_mutations.get(var, ()))

            if isinstance(var, variables.UserFunctionVariable):
                fn = cast(types.FunctionType, value)
                collect_from_real_value(fn.__defaults__)
                collect_from_real_value(fn.__kwdefaults__)
                collect_dict_values(fn.__dict__, pending_names)
                collect_from_real_value(fn.__closure__)
                return

            if isinstance(var, variables.PythonModuleVariable):
                module = cast(types.ModuleType, value)
                if module is torch or module.__name__.startswith("torch."):
                    return
                collect_dict_values(vars(module), pending_names)
                return

            if isinstance(var, variables.UserDefinedClassVariable):
                collect_class_dicts(cast(type, value), pending_names)
                return

            if isinstance(var, variables.UserDefinedObjectVariable):
                if getattr(var, "dict_vt", None) is None:
                    try:
                        instance_dict = object.__getattribute__(value, "__dict__")
                    except AttributeError:
                        instance_dict = None
                    if instance_dict is not None:
                        collect_dict_values(instance_dict, pending_names)
                collect_instance_slots(value, pending_names)
                collect_class_dicts(type(value), set())

        def visit_vt(value: Any, cache: dict[int, Any] | None = None) -> None:
            if cache is None:
                cache = {}
            value_id = id(value)
            if value_id in cache:
                return
            cache[value_id] = value

            if isinstance(value, VariableTracker):
                value = value.unwrap()
                collect_from_vt(value)
                value = value.unwrap()
                if isinstance(value, variables.CellVariable):
                    collect_from_vt(self.load_cell(value).unwrap())
                    return

                nonvars = value._nonvar_fields
                for key, subvalue in value.__dict__.items():
                    if key not in nonvars:
                        visit_vt(subvalue, cache)
                if value in self.store_attr_mutations:
                    visit_vt(self.store_attr_mutations[value], cache)
            elif istype(value, (list, tuple)):
                for subvalue in value:
                    visit_vt(subvalue, cache)
            elif istype(value, (dict, collections.OrderedDict)):
                for subvalue in value.values():
                    visit_vt(subvalue, cache)

        def vt_matches_real_value(var: VariableTracker, value: Any) -> bool:
            try:
                return var.get_real_python_backed_value() is value
            except NotImplementedError:
                pass
            try:
                return var.as_python_constant() is value
            except NotImplementedError:
                return False

        tx = output_graph.current_tx
        while tx is not None:
            live_locals = (
                livevars_analysis(tx.instructions, tx.current_instruction)
                if tx.instruction_pointer is not None
                else set(tx.symbolic_locals)
            )
            visit_vt(tx.stack)
            for name, local_vt in tx.symbolic_locals.items():
                if name not in live_locals:
                    continue
                visit_vt(local_vt)

            for name, value in tx.f_locals.items():
                if name in tx.cell_and_freevars() or name not in live_locals:
                    continue
                if name in tx.symbolic_locals and vt_matches_real_value(
                    tx.symbolic_locals[name].unwrap(), value
                ):
                    collect_from_real_value(value)

            tx = tx.parent

        return backing_id in live_backing_ids

    def _get_modified_vars(self) -> list[VariableTracker]:
        return [var for var in self.id_to_variable.values() if self.is_modified(var)]

    def codegen_save_tempvars(self, cg: PyCodegen) -> None:
        # We must codegen modified VT to their source by default, so that
        # mutation and aliasing are properly accounted for.
        #
        # Since newly constructed objects don't have a source, we manually
        # codegen their construction and store them to a newly assigned local
        # source. Note that `ValueMutationNew` isn't tracked by SideEffects.
        for var in self._get_modified_vars():
            if not isinstance(var.mutation_type, AttributeMutationNew):
                if var.source is None:
                    raise AssertionError(
                        f"Expected source for modified var {var} "
                        "with non-new mutation type"
                    )
                continue

            # Namedtuples/structseqs with no pending mutations should skip
            # codegen_save_tempvars so that restore_stack handles them. In
            # export, restore_stack uses value_from_source=False which makes
            # child tensors become graph outputs. If we processed them here,
            # add_cache would assign a TempLocalSource and restore_stack would
            # load from cache with value_from_source=True, hiding the tensors
            # from export.
            if isinstance(
                var,
                (variables.NamedTupleVariable, variables.StructSequenceVariable),
            ) and not self.has_pending_mutation(var):
                continue

            if isinstance(var, variables.CellVariable):
                # Cells created in the root frame are created either by
                # `MAKE_CELL` or by them being in `co_cellvars`, so we only emit
                # `make_cell` for the non-root-frame cells here.
                # TODO generalize this so we never need to call `make_cell`.
                if var.local_name is None:
                    cg.add_push_null(
                        lambda: cg.load_import_from(utils.__name__, "make_cell")
                    )
                    cg.extend_output(create_call_function(0, False))
                    cg.add_cache(var)
                    var.source = TempLocalSource(cg.tempvars[var])  # type: ignore[attr-defined]
                elif var.source is None:
                    var.source = LocalCellSource(var.local_name)
            elif var.is_tensor():
                # NOTE: for historical reasons we never assigned local sources
                # to newly constructed tensor object, so we keep it that way.
                # They are always loaded from output of the fx graph, so one can
                # think of it as having a "OutputGraphSource" for codegen
                # purposes.
                #
                # However, tensor subclass objects are different, because the
                # reconstruction logic in `PyCodegen` loads the data tensor from
                # graph output and then calls `as_subclass`, meaning we must
                # assign a source to it to ensure we only reconstruct one
                # subclass instance.
                if isinstance(
                    var, variables.torch_function.TensorWithTFOverrideVariable
                ):
                    # Don't codegen from temp source assigned from the 1st pass.
                    cg(var, allow_cache=False)
                    cg.add_cache(var)
                    # `add_cache` generates STORE and consumes TOS, but we never
                    # cleared it. TODO move this call into `add_cache`
                    cg.clear_tos()
                    var.source = TempLocalSource(cg.tempvars[var])
            elif isinstance(var, variables.AutogradFunctionContextVariable):
                unimplemented(
                    gb_type="AutogradFunctionContextVariable escaped Dynamo-traced region",
                    context="",
                    explanation="We cannot reconstruct a torch.autograd.Function's context object.",
                    hints=[],
                )
            else:
                # Reconstruct the bytecode for
                # base_cls.__new__(user_cls, *args)
                if isinstance(var, variables.UserDefinedObjectVariable):

                    def load_new_method() -> None:
                        # pyrefly: ignore [missing-attribute]
                        if var.base_cls_vt is None:
                            raise AssertionError(
                                "base_cls_vt is None in load_new_method"
                            )
                        cg(var.base_cls_vt)  # type: ignore[attr-defined]
                        cg.extend_output([cg.create_load_attr("__new__")])

                    cg.add_push_null(load_new_method)
                else:
                    cg.add_push_null(
                        lambda: cg.load_import_from(utils.__name__, "object_new")
                    )
                if var.mutation_type.cls_source is None:
                    unimplemented(
                        gb_type="Reconstruct user defined class without a source",
                        context=f"Class: {var.python_type().__name__}",
                        explanation=(
                            f"Cannot reconstruct an instance of {var.python_type().__name__} "
                            "that escapes the compiled region. This happens when the class is "
                            "defined dynamically (e.g., inside the compiled function) and the "
                            "class object itself has no source that can be reconstructed. "
                            "To fix this, move the class definition outside the compiled function "
                            "or prevent the object from escaping the compiled region."
                        ),
                        hints=[*graph_break_hints.SUPPORTABLE],
                    )
                cg(var.mutation_type.cls_source)

                # Generate the args to the __new__ method
                for arg in var.init_args:  # type: ignore[attr-defined]
                    cg(arg)

                # Call the __new__ method
                cg.extend_output(create_call_function(1 + len(var.init_args), False))  # type: ignore[attr-defined]

                cg.add_cache(var)
                var.source = TempLocalSource(cg.tempvars[var])

                # For frozen dataclasses, we must emit object.__setattr__
                # immediately after __new__ — before any other code can
                # access the object.  The suffix-based codegen in
                # codegen_update_mutated runs too late: if intervening code
                # calls __repr__ (e.g. f-strings), the attributes won't be
                # set yet.
                if (
                    isinstance(var, variables.FrozenDataClassVariable)
                    and var in self.store_attr_mutations
                ):
                    for name, value in self.store_attr_mutations[var].items():
                        cg.load_import_from("builtins", "object")
                        cg.load_method("__setattr__")
                        cg(var.source)
                        cg(variables.ConstantVariable(name))
                        cg(value)
                        cg.extend_output(
                            [*create_call_method(3), create_instruction("POP_TOP")]
                        )

        for ctx, args in self.save_for_backward:
            cg(ctx.source)
            cg.load_method("save_for_backward")
            for arg in args:
                cg(arg)
            cg.extend_output(
                [
                    *create_call_method(len(args)),
                    create_instruction("POP_TOP"),
                ]
            )

    def register_hook(
        self,
        tensor: "variables.TensorVariable",
        hook: VariableTracker,
        handle: "variables.RemovableHandleVariable",
        name: str,
    ) -> None:
        if not tensor.is_tensor():
            raise AssertionError(
                f"Expected tensor variable, got {type(tensor)} in register_hook"
            )
        if not isinstance(hook, variables.VariableTracker):
            raise AssertionError(f"Expected VariableTracker for hook, got {type(hook)}")
        if not isinstance(handle, variables.RemovableHandleVariable):
            raise AssertionError(
                f"Expected RemovableHandleVariable, got {type(handle)}"
            )
        if not handle.is_mutable():
            raise AssertionError("handle must be mutable in register_hook")
        if not hasattr(torch.Tensor, name):
            raise AssertionError(f"torch.Tensor has no attribute '{name}'")
        idx = len(self.tensor_hooks.keys())
        # duplicate index possible because of self.remove_hook()
        while idx in self.tensor_hooks:
            idx += 1
        self.tensor_hooks[idx] = (tensor, hook, handle, name)
        if handle.idx:
            raise AssertionError(f"handle.idx should be falsy, got {handle.idx}")
        handle.idx = idx

    def remove_hook(self, idx: int) -> None:
        del self.tensor_hooks[idx]

    def codegen_hooks(self, cg: PyCodegen) -> None:
        for (
            tensor,
            hook,
            handle,
            name,
        ) in self.tensor_hooks.values():
            # Note: [On tensor.register_hook]
            #
            # register_hook on a tensor, AKA backward hooks, have slightly nuanced differences in how they are implemented
            # when it comes to hooks on objects with sources (inputs, params) vs objects without sources (intermediaries).
            #
            # For tensors with a source, we bypass direct inclusion of register_hook calls in the graph.
            # Instead, these are tracked and stashed as a global variable, enabling their association with tensors in
            # the residuals. During dynamo's frame creation, these hooks are invoked seamlessly on known reconstructible/fetch-able
            # tensors. Because a source indicates knowledge of this object outside the torch compile region, and
            # because we are running residuals firmly before .backward() can be run, it is sound to invoke
            # `register_hook` on a known tensor.
            #
            # For tensors without a source, we support a limited subset of hooks. Global functions only, and
            # compiled_autograd must be enabled or we will graph break.
            #
            # Handling the Handle: When a user retains the register_hook result in a handle, we intercept the
            # STORE_FAST operation to record the user-designated local variable name. This ensures the reconstructed
            # bytecode retains this name. If no handle is defined, we simply pop the generated value to keep the
            # stack intact.
            #
            # Dynamo Tensor Hooks Workflow:
            # - Functions passed to register_hook are lifted globally.
            # - For tensors with sources:
            #   - In the "side_effects" phase of codegen, we iterate over tensors with hooks to:
            #     - Generate the tensor.
            #     - Issue a register_hook call on the tensor, linking to the globally stored function.
            #     - Incorporate a handle if one was established in the eager phase.
            #  - For tensors without sources:
            #    - We don't generate any instructions for registering a hook.
            #    - Handles from intermediary hooks are NYI.
            #    - We produce a call function that utilizes the trace_wrapped higher order op, closing over it.
            #    - We then manually insert the call function above into the graph.
            # - The handle's exact user-specified name, "user_code_variable_name", is discerned and associated during STORE_FAST.
            if not tensor.source:
                raise AssertionError(
                    "Hooks on non input tensors NYI - should not get here"
                )

            def gen_fn() -> None:
                cg(tensor)
                cg.extend_output([cg.create_load_attr(name)])

            cg.add_push_null(gen_fn)
            cg(hook)
            cg.extend_output(create_call_function(1, False))

            # Adding the handle to the cache means RemovableHandleVariable().reconstruct() will
            # be associated with the return value of register_hook().  This consumes the top of stack.
            cg.add_cache(handle)

    def get_ca_final_callbacks_var(self) -> "variables.ListVariable":
        from .variables.base import ValueMutationNew

        if self.ca_final_callbacks_var is None:
            self.ca_final_callbacks_var = variables.ListVariable(
                [], mutation_type=ValueMutationNew()
            )

        return self.ca_final_callbacks_var

    def _format_side_effect_message(self, var: VariableTracker) -> str:
        """Format a side effect log message with user stack."""
        if config.side_effect_replay_policy == "silent":
            raise AssertionError(
                "_format_side_effect_message should not be called "
                "when side_effect_replay_policy is 'silent'"
            )
        locations = self.mutation_user_stacks.get(var, [])
        description = f"Mutating object of type {var.python_type_name()}"
        source_info = " (no source)"
        if var.source is not None:
            if isinstance(var.source, TempLocalSource):
                source_info = " (source: created in torch.compile region)"
            elif isinstance(var, variables.CellVariable) and var.local_name is not None:
                source_info = f" (source: {var.local_name})"
            elif isinstance(
                var, variables.torch_function.TorchFunctionModeStackVariable
            ):
                source_info = " (source: torch function mode stack mutation)"
            else:
                # NOTE: NotImplementedError from var.source.name is a bug and must be fixed!
                source_info = f" (source name: {var.source.name})"

        if locations:
            # Format and dedupe stacks using tuple representation for efficiency
            seen = set()
            unique_formatted_stacks: list[str] = []
            stack_above_dynamo = collapse_resume_frames(get_stack_above_dynamo())
            for stack in locations:
                # Use tuple of frame info for fast deduplication
                # Include position info (colno, end_lineno, end_colno) to distinguish
                # multiple mutations on the same line (when available in Python 3.11+)
                stack_tuple = tuple(
                    (
                        f.filename,
                        f.lineno,
                        f.name,
                        f.line,
                        getattr(f, "colno", None),
                        getattr(f, "end_lineno", None),
                        getattr(f, "end_colno", None),
                    )
                    for f in stack
                )
                if stack_tuple not in seen:
                    seen.add(stack_tuple)
                    stack_augmented = collapse_resume_frames(stack_above_dynamo + stack)
                    unique_formatted_stacks.append(
                        "".join(traceback.format_list(stack_augmented))
                    )
            formatted_lines: str = "\n********\n\n".join(unique_formatted_stacks)
            log_str = f"{description}{source_info}\n\n{textwrap.indent(formatted_lines, '    ')}"
        else:
            log_str = (
                f"{description}{source_info} (unable to find user stacks for mutations)"
            )

        return log_str

    def _emit_side_effect_messages(self, side_effect_messages: list[str]) -> None:
        if not side_effect_messages:
            return

        for msg in side_effect_messages:
            side_effects_log.debug(msg)

        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "dynamo_side_effects",
                "encoding": "string",
            },
            payload_fn=lambda: "\n\n========================================\n\n".join(
                side_effect_messages
            ),
        )

    def codegen_update_mutated(
        self, cg: PyCodegen, log_side_effects: bool = False
    ) -> None:
        side_effect_messages: list[str] = []

        # NOTE: should only be called once per VT - only if a side effect actually gets codegen'd!
        def _maybe_log_side_effect(var: VariableTracker) -> None:
            if config.side_effect_replay_policy != "silent" and log_side_effects:
                msg = self._format_side_effect_message(var)
                side_effect_messages.append(msg)

        suffixes = []
        for var in self._get_modified_vars():
            # When replay_side_effects=False, only update variables with TempLocalSource
            if not config.replay_side_effects and not isinstance(
                var.source, TempLocalSource
            ):
                continue
            if isinstance(var, variables.ListVariable):
                # old[:] = new
                cg(var, allow_cache=False)  # Don't codegen via source
                cg(var.source)  # type: ignore[attr-defined]
                cg.extend_output(
                    [
                        cg.create_load_const(None),
                        cg.create_load_const(None),
                        create_instruction("BUILD_SLICE", arg=2),
                    ]
                )
                suffixes.append([create_instruction("STORE_SUBSCR")])
                _maybe_log_side_effect(var)
            elif isinstance(var, variables.lists.DequeVariable):
                # For limited maxlen, the order of operations matter for side
                # effect, but we currently don't track the order, so no support.
                if not var.maxlen.is_constant_none():
                    unimplemented(
                        gb_type="Side effect on existing deque with limited maxlen",
                        context="",
                        explanation="This is not supported.",
                        hints=[
                            "Don't use a deque with `maxlen` specified.",
                        ],
                    )

                # old.extend(new), this runs last
                cg(var.source)
                cg.load_method("extend")
                cg(var, allow_cache=False)  # Don't codegen via source
                suffixes.append(
                    [
                        *create_call_method(1),
                        create_instruction("POP_TOP"),
                    ]
                )

                # old.clear(), this runs first
                cg(var.source)
                cg.load_method("clear")
                suffixes.append(
                    [
                        *create_call_method(0),
                        create_instruction("POP_TOP"),
                    ]
                )
                _maybe_log_side_effect(var)

            elif isinstance(var, (variables.ConstDictVariable, variables.SetVariable)):
                # Reconstruct works as follow:
                # (1) Skip codegen if there are no new items
                # (2) codegen(...) each pair of key/value
                # (3) create a new dictionary with the pairs of key/values above
                # (4) clear the original dictionary
                #   + only if a key was removed from the input dict
                # (5) update the original dictionary with the dict created in (2)

                if var.has_new_items():
                    cg(var.source)  # type: ignore[attr-defined]
                    cg.load_method("update")
                    cg(var, allow_cache=False)  # Don't codegen via source

                    if var.should_reconstruct_all:
                        cg(var.source)  # type: ignore[attr-defined]
                        cg.load_method("clear")

                    suffixes.append(
                        [
                            *create_call_method(1),  # update
                            create_instruction("POP_TOP"),
                        ]
                    )

                    if var.should_reconstruct_all:
                        # clear will appear before "update" as the suffixes are
                        # applied in reverse order.
                        suffixes.append(
                            [
                                *create_call_method(0),  # clear
                                create_instruction("POP_TOP"),
                            ]
                        )
                    _maybe_log_side_effect(var)

            elif isinstance(
                var, variables.torch_function.TorchFunctionModeStackVariable
            ):
                cg.add_push_null(
                    lambda: cg.load_import_from(
                        utils.__name__, "set_torch_function_mode_stack"
                    )
                )

                cg.foreach(var.symbolic_stack)
                cg.append_output(
                    create_instruction("BUILD_LIST", arg=len(var.symbolic_stack))
                )
                cg.call_function(1, False)
                cg.append_output(create_instruction("POP_TOP"))
                _maybe_log_side_effect(var)

            elif isinstance(var, variables.CellVariable) and var.local_name is not None:
                # Emit more readable and performant bytecode.
                # TODO generalize this for cells created during inlining.
                if var in self.store_attr_mutations:
                    contents_var = self.load_cell(var)
                    cg(contents_var)
                    suffixes.append([cg.create_store_deref(var.local_name)])
                    _maybe_log_side_effect(var)

            elif self.is_attribute_mutation(var):
                # FrozenDataClassVariable attributes were emitted in
                # codegen_save_tempvars right after __new__. Skip here to
                # avoid double-emitting.
                if isinstance(var.mutation_type, AttributeMutationNew) and isinstance(
                    var, variables.FrozenDataClassVariable
                ):
                    continue

                if (
                    isinstance(
                        var,
                        variables.UserDefinedDictVariable,
                    )
                    and self.is_modified(
                        var._base_vt  # pyrefly: ignore[bad-argument-type]
                    )
                    and var._base_vt.has_new_items(  # pyrefly: ignore[union-attr,missing-attribute]
                    )
                ):
                    # Do dict related update manually here. The store_attr
                    # mutations will be applied later.
                    varname_map = {}
                    for name in _manual_dict_setitem.__code__.co_varnames:
                        varname_map[name] = cg.tx.output.new_var()

                    try:
                        mro_index = type(var.value).__mro__.index(
                            collections.OrderedDict
                        )
                    except ValueError:
                        mro_index = type(var.value).__mro__.index(dict)

                    cg.extend_output(
                        [
                            create_instruction("LOAD_CONST", argval=mro_index),
                            create_instruction(
                                "STORE_FAST", argval=varname_map["mro_index"]
                            ),
                        ]
                    )

                    cg(var.source)  # type: ignore[attr-defined]
                    cg.extend_output(
                        [
                            create_instruction(
                                "STORE_FAST", argval=varname_map["dict_to"]
                            )
                        ]
                    )

                    # Reconstruct all items — _manual_dict_setitem clears
                    # dict_to first, so we need every key/value, not just
                    # the ones that differ from original_items.
                    var._base_vt.should_reconstruct_all = True  # type: ignore[union-attr]
                    cg(var._base_vt, allow_cache=False)  # Don't codegen via source
                    cg.extend_output(
                        [
                            create_instruction(
                                "STORE_FAST", argval=varname_map["dict_from"]
                            )
                        ]
                    )

                    dict_update_insts = bytecode_from_template(
                        _manual_dict_setitem, varname_map=varname_map
                    )

                    suffixes.append(
                        [
                            *dict_update_insts,
                            create_instruction("POP_TOP"),
                        ]
                    )
                    _maybe_log_side_effect(
                        var._base_vt  # pyrefly: ignore[bad-argument-type]
                    )
                elif isinstance(
                    var,
                    variables.UserDefinedListVariable,
                ) and self.is_modified(
                    var._base_vt  # pyrefly: ignore[bad-argument-type]
                ):
                    # Update the list to the updated items. Be careful in
                    # calling the list methods and not the overridden methods.
                    varname_map = {}
                    for name in _manual_list_update.__code__.co_varnames:
                        varname_map[name] = cg.tx.output.new_var()

                    cg(var.source)  # type: ignore[attr-defined]
                    cg.extend_output(
                        [
                            create_instruction(
                                "STORE_FAST", argval=varname_map["list_to"]
                            )
                        ]
                    )

                    cg(var._base_vt, allow_cache=False)  # Don't codegen via source
                    cg.extend_output(
                        [
                            create_instruction(
                                "STORE_FAST", argval=varname_map["list_from"]
                            )
                        ]
                    )

                    list_update_insts = bytecode_from_template(
                        _manual_list_update, varname_map=varname_map
                    )

                    suffixes.append(
                        [
                            *list_update_insts,
                            create_instruction("POP_TOP"),
                        ]
                    )
                    _maybe_log_side_effect(
                        var._base_vt  # pyrefly: ignore[bad-argument-type]
                    )

                # Applying mutations involves two steps: 1) Push all
                # reconstructed objects onto the stack.  2) Call STORE_ATTR to
                # apply the mutations.
                #
                # Dynamo must ensure that mutations are applied in the same
                # order as in the original program. Therefore, two reverse
                # operations occur below.
                #
                # The first reverse operation concerns `suffixes`. We apply
                # suffixes in reverse order due to the way Python handles the
                # stack. In Step 1, we push all reconstructed objects onto the
                # stack, but the item at the top of the stack refers to the last
                # attribute in the mutation order. If not fixed, this will apply
                # the mutations of attributes in the reverse order.  To account
                # for this reversal, we iterate through the mutable attributes
                # in reverse order.
                side_effect_occurred = False
                for name, value in reversed(
                    self.store_attr_mutations.get(var, {}).items()
                ):
                    mutation_kind = self.get_attr_mutation_kind(var, name)
                    if isinstance(var, variables.NewGlobalVariable):
                        cg.tx.output.update_co_names(name)
                        cg(value)
                        if not isinstance(var.source, GlobalSource):  # type: ignore[attr-defined]
                            raise AssertionError(
                                f"Expected GlobalSource for NewGlobalVariable, "
                                f"got {type(var.source)}"  # type: ignore[attr-defined]
                            )
                        suffixes.append(
                            [create_instruction("STORE_GLOBAL", argval=name)]
                        )
                        side_effect_occurred = True
                    elif isinstance(value, variables.DeletedVariable):
                        if (
                            isinstance(var, variables.UserDefinedObjectVariable)
                            and mutation_kind is AttrMutationKind.INSTANCE_DICT
                        ):
                            original_dict = getattr(
                                getattr(var, "value", None), "__dict__", {}
                            )
                            # If the key only existed in the traced instance
                            # dict, the add/delete sequence is a replay no-op.
                            if name in original_dict:
                                cg.add_push_null(
                                    lambda: cg.load_import_from(
                                        utils.__name__,
                                        "object_delattr_ignore_descriptor",
                                    )
                                )
                                cg(var.source)  # type: ignore[attr-defined]
                                cg(variables.ConstantVariable(name))
                                suffixes.append(
                                    [
                                        *create_call_function(2, False),
                                        create_instruction("POP_TOP"),
                                    ]
                                )
                                side_effect_occurred = True
                        # GENERIC_SETATTR deletions on UDOV fall through to the
                        # normal DELETE_ATTR path below so descriptor semantics
                        # are preserved during replay.
                        elif isinstance(
                            var.mutation_type, AttributeMutationExisting
                        ) and hasattr(getattr(var, "value", None), name):
                            cg.tx.output.update_co_names(name)
                            cg(var.source)
                            suffixes.append(
                                [create_instruction("DELETE_ATTR", argval=name)]
                            )
                            side_effect_occurred = True
                    elif (
                        isinstance(var, variables.UserDefinedObjectVariable)
                        and mutation_kind is AttrMutationKind.INSTANCE_DICT
                    ):
                        cg.add_push_null(
                            lambda: cg.load_import_from(
                                utils.__name__, "object_setattr_ignore_descriptor"
                            )
                        )
                        cg(var.source)  # type: ignore[attr-defined]
                        cg(variables.ConstantVariable(name))
                        cg(value)
                        suffixes.append(
                            [
                                *create_call_function(3, False),
                                create_instruction("POP_TOP"),
                            ]
                        )
                        side_effect_occurred = True
                    elif (
                        isinstance(var, variables.UserDefinedObjectVariable)
                        and var.needs_slow_setattr()
                    ):
                        # __setattr__ is defined on this object, so call object.__setattr__ directly
                        cg.load_import_from("builtins", "object")
                        cg.load_method("__setattr__")
                        cg(var.source)  # type: ignore[attr-defined]
                        cg(variables.ConstantVariable(name))
                        cg(value)
                        suffixes.append(
                            [*create_call_method(3), create_instruction("POP_TOP")]
                        )
                        side_effect_occurred = True
                    else:
                        cg.tx.output.update_co_names(name)
                        cg(value)
                        cg(var)
                        suffixes.append([create_instruction("STORE_ATTR", argval=name)])
                        side_effect_occurred = True

                if side_effect_occurred:
                    _maybe_log_side_effect(var)
            elif isinstance(var, variables.ListIteratorVariable):
                for _ in range(var.index):
                    cg.add_push_null(
                        lambda: cg.load_import_from(utils.__name__, "iter_next")
                    )
                    cg(var.source)  # type: ignore[attr-defined]
                    cg.call_function(1, False)
                    cg.pop_top()
                _maybe_log_side_effect(var)
            elif isinstance(var, variables.CountIteratorVariable):
                for _ in range(var.advance_count):
                    cg.add_push_null(
                        lambda: cg.load_import_from(utils.__name__, "iter_next")
                    )
                    cg(var.source)  # type: ignore[attr-defined]
                    cg.call_function(1, False)
                    cg.pop_top()
                _maybe_log_side_effect(var)
            elif isinstance(var, variables.RandomVariable):
                # set correct random seed state
                def gen_fn() -> None:
                    cg(var.source)  # type: ignore[attr-defined]
                    cg.load_attr("setstate")

                cg.add_push_null(gen_fn)
                cg(var.wrap_state(var.random.getstate()))

                suffixes.append(
                    [
                        *create_call_function(1, False),  # setstate
                        create_instruction("POP_TOP"),
                    ]
                )
                _maybe_log_side_effect(var)
            else:
                raise AssertionError(type(var))

        # do all the actual mutations at the very end to handle dependencies
        for suffix in reversed(suffixes):
            cg.extend_output(suffix)

        # Send batched structured trace for all side effects in this compilation
        if log_side_effects and side_effect_messages:
            self._emit_side_effect_messages(side_effect_messages)

    def log_side_effects_summary(self) -> None:
        if config.side_effect_replay_policy == "silent":
            return
        if not side_effects_log.isEnabledFor(logging.DEBUG):
            return
        for var in self._get_modified_vars():
            msg = self._format_side_effect_message(var)
            side_effects_log.debug(msg)

    def is_empty(self) -> bool:
        return not (
            any(map(self.is_modified, self.id_to_variable.values()))
            or self.tensor_hooks
            or self.save_for_backward
            or self.tensor_hooks
        )

    def clear(self) -> None:
        self.keepalive.clear()
        self.id_to_variable.clear()


@contextlib.contextmanager
def allow_side_effects_in_hop(
    tx: "InstructionTranslatorBase",
) -> Generator[None, None, None]:
    """Context manager to temporarily allow side effects with extra outputs.

    This is used for special cases (like FSDP functions) that need to perform
    side effects even when the general policy is to disallow them.
    """
    orig_val = tx.output.current_tracer.allow_side_effects_in_hop
    try:
        tx.output.current_tracer.allow_side_effects_in_hop = True
        yield
    finally:
        tx.output.current_tracer.allow_side_effects_in_hop = orig_val


@contextlib.contextmanager
def allow_externally_visible_side_effects_in_subtracer(
    tx: "InstructionTranslatorBase",
) -> Generator[None, None, None]:
    orig_val = tx.output.current_tracer.unsafe_allow_externally_visible_side_effects
    try:
        tx.output.current_tracer.unsafe_allow_externally_visible_side_effects = True
        tx.output.current_tracer.traced_with_externally_visible_side_effects = True
        yield
    finally:
        tx.output.current_tracer.unsafe_allow_externally_visible_side_effects = orig_val


@contextlib.contextmanager
def disallow_side_effects_in_generator(
    tx: "InstructionTranslatorBase",
) -> Generator[None, None, None]:
    orig_val = tx.output.current_tracer.is_reconstructing_generator
    try:
        tx.output.current_tracer.is_reconstructing_generator = True
        yield
    finally:
        tx.output.current_tracer.is_reconstructing_generator = orig_val
