"""
This module provides utilities for generating Python bytecode in PyTorch's Dynamo system.
It includes functionality for:
- Constructing bytecode sequences for Python operations
- Managing stack operations and variable tracking
- Handling graph outputs and their conversions
- Supporting different Python versions (3.11+, 3.12+, 3.13+)
- Converting high-level operations to low-level bytecode instructions
- Managing constant loading and attribute access
- Supporting function creation and closure handling
"""

import collections
import dataclasses
import re
import sys
import types
from collections import Counter, deque
from collections.abc import Callable, Iterable
from typing import Any, TYPE_CHECKING, Union

import torch.nn
from torch.utils._ordered_set import OrderedSet

from . import config, graph_break_hints, utils
from .bytecode_transformation import (
    add_push_null,
    add_push_null_call_function_ex,
    create_binary_subscr,
    create_build_tuple,
    create_call_function,
    create_call_function_ex,
    create_call_method,
    create_dup_top,
    create_instruction,
    create_load_const,
    create_load_method,
    create_rot_n,
    Instruction,
)
from .exc import unimplemented
from .source import (
    AttrSource,
    ChainedSource,
    DictGetItemSource,
    Source,
    TempLocalSource,
)
from .utils import is_safe_constant, rot_n_helper
from .variables.base import ValueMutationExisting, VariableTracker
from .variables.functions import (
    ContextlibContextManagerLocalGeneratorObjectVariable,
    LocalGeneratorObjectVariable,
)
from .variables.lazy import ComputedLazyConstantVariable
from .variables.nn_module import NNModuleVariable
from .variables.script_object import CustomClassObjectVariable
from .variables.tensor import (
    NumpyNdarrayVariable,
    SymNodeVariable,
    TensorVariable,
    UnspecializedPythonVariable,
)
from .variables.torch_function import TensorWithTFOverrideVariable


if TYPE_CHECKING:
    from torch._dynamo.variables.builder import GraphArg

    from .symbolic_convert import InstructionTranslatorBase


@dataclasses.dataclass
class GraphOutputEntry:
    index: int
    variable: VariableTracker


class PyCodegen:
    """
    Helper class uses for constructing Python bytecode
    """

    def __init__(
        self,
        tx: "InstructionTranslatorBase",
        root: torch.nn.Module | None = None,
        graph_output_var: str | None = None,
        tempvars: dict[VariableTracker | Source, Any] | None = None,
        overridden_sources: dict[Source, Source] | None = None,
    ) -> None:
        self.root = root
        self.top_of_stack: VariableTracker | Source | None = None
        self.uses: Counter[VariableTracker | Source] = collections.Counter()
        self.graph_outputs: dict[int, GraphOutputEntry] = {}
        self._output: list[Instruction] = []
        # This determines which VariableTracker/Source should be stored as
        # locals, and maps the VariableTracker/Source to the local variable
        # name. Note that it could map to None initially, in which case we'll
        # overwrite it to map to real temporary names via `add_cache`.
        self.tempvars: dict[VariableTracker | Source, Any] = tempvars or {}
        self.tx = tx
        self.graph_output_var = graph_output_var
        self.code_options = self.tx.output.code_options
        self.cell_and_freevars = self.tx.cell_and_freevars
        self.new_var = self.tx.output.new_var
        self.value_from_source: bool = True
        # This serves as a way for codegen to use a different source; we need
        # this because sometimes we can't easily modify the original source
        # without affecting other components, e.g., guards.
        self.overridden_sources: dict[Source, Source] = overridden_sources or {}
        self.direct_source_uses: OrderedSet[Source] = OrderedSet()
        self._source_reconstruct_depth = 0
        self.pycodes = []

    def add_pycode(self, pycode: str, *args):
        if not config.generate_pycode:
            return
        for a in args:
            if isinstance(a, VariableTracker):
                a.realize()
        self.pycodes.append(pycode.format(*[a.reconstruct_pycode(self) for a in args]))

    def restore_stack(
        self, stack_values: list[Any], *, value_from_source: bool = True
    ) -> None:
        prev = self.value_from_source
        self.value_from_source &= value_from_source
        try:
            self.foreach(stack_values)
            # `restore_stack` is called once per codegen pass (e.g. pass1 for
            # use tracking, pass2 for actual emission). Reset the stack counter
            # so each pass produces the same `__stackN` names.
            self.tx.reset_pycode_varname_counter("stack")
            for v in stack_values:
                self.add_pycode(f"{self.tx.new_pycode_varname('stack')} = {{}}", v)
        finally:
            self.value_from_source = prev

    def graph_output_vars(self) -> list[VariableTracker]:
        return [x.variable for x in self.graph_outputs.values()]

    def call_reconstruct(
        self, value: Union[VariableTracker, Source, "GraphArg"]
    ) -> None:
        res = value.reconstruct(self)
        if res is not None:
            raise AssertionError(f"reconstruct!=None {value}")

    def add_push_null(
        self, gen_fn: Callable[[], None], call_function_ex: bool = False
    ) -> None:
        """
        `gen_fn` generates instructions via PyCodegen methods
        that push a single callable to the stack.

        `add_push_null` pushes a NULL to the stack before or after the
        instructions generated by `gen_fn`, depending on Python version.

        Will attempt to use the NULL push bit for instructions
        with such bits (LOAD_GLOBAL 3.11+, LOAD_ATTR 3.12+, LOAD_SUPER_ATTR).
        """
        old_len = len(self._output)
        if sys.version_info < (3, 13):
            # gen_fn may DUP_TOP instead if TOS is not cleared.
            # Will cause problems since NULL will be pushed right
            # before the generated instructions in <= 3.12
            self.clear_tos()
        gen_fn()
        # inplace modify self._output
        added_insts = self._output[old_len:]
        del self._output[old_len:]
        if call_function_ex:
            self._output.extend(add_push_null_call_function_ex(added_insts))
        else:
            self._output.extend(add_push_null(added_insts))
        if sys.version_info >= (3, 13):
            # NULL will be at top of stack
            self.clear_tos()

    def __call__(
        self, value: VariableTracker | Source | None, allow_cache: bool = True
    ) -> None:
        """
        Generate code such that top-of-stack (TOS) is set to value.

        `allow_cache` controls the behavior in the following manner. `value` can
        either be a VariableTracker or a Source.

        If `value` is a `Source`, `allow_cache` must be True (invariant asserted
        below). If the source was reconstructed earlier, we will reuse the
        generated code by loading from top of stack or tempvars.

        If `value` is a `VariableTracker`, we have the following cases:

        1) `allow_cache=True`
            a) If the value.source is not None, we will emit the code based on
            `value.source` to handle aliasing.
            b) If value.source is None (example reconstructing a local list
            returned by the compiled function), we will reconstruct the variable
            tracker (w/o any source) to emit bytecode that generates a new
            python object.

            In both cases of value.source being None or not, if the value was
            reconstructed earlier, we will reuse the generated code by loading from
            top of stack or tempvars.

        2) `allow_cache=False` - This is a special case (allow_cache defaults to
        True).
            a) If the value.source is not None, we reconstruct the variable
            tracker and emit a new python object. You might wonder what about
            aliasing? The place where we use this config also has the followup
            code where the original python object is assigned to this new python
            value to handle aliasing (check side_effects.py and search for
            allow_cache=False).

            b) If value.source is None, this is not allowed

        Notable effects:
        1. `self.top_of_stack` will be set to `value`, if we don't codegen
           `value` based on source.
        2. `self.uses[value]` will increment, unless (a). we codegen via
            `top_of_stack` or cached `tempvars`, or (b). `value` has special VT
            types like `NNModuleVariable`, etc.
        """
        if value is None:
            raise AssertionError("value must not be None")
        if isinstance(value, Source):
            # If the source needs to be overridden, use the new one.
            source = self.overridden_sources.get(value, value)
            if self._source_reconstruct_depth == 0:
                self.direct_source_uses.add(source)
            if allow_cache is not True:
                raise AssertionError("allow_cache must be True for Source")
            if self.top_of_stack is value:
                self._output.append(create_dup_top())
                return

            if self.tempvars.get(source) is not None:
                self._output.append(self.create_load(self.tempvars[source]))
                self.top_of_stack = source
                return

            self.uses[source] += 1
            try:
                self._source_reconstruct_depth += 1
                self.call_reconstruct(source)
            except NotImplementedError:
                unimplemented(
                    gb_type="Reconstruction failure: source.reconstruct not implemented",
                    context=str(source),
                    explanation=f"Dynamo has no bytecode reconstruction implemented for {type(source)} variable {source}.",
                    hints=[*graph_break_hints.DYNAMO_BUG],
                )
            finally:
                self._source_reconstruct_depth -= 1
            if source in self.tempvars:
                self._output.append(create_dup_top())
                self.add_cache(source)
            self.top_of_stack = source

            return

        if not isinstance(value, VariableTracker):
            raise AssertionError(f"expected VariableTracker, got {type(value)}")
        output = self._output
        graph_outputs = self.graph_outputs

        if allow_cache:
            if self.top_of_stack is value:
                output.append(create_dup_top())
                return

            if self.tempvars.get(value) is not None:
                output.append(self.create_load(self.tempvars[value]))
                self.top_of_stack = value
                return

        if value.is_realized() and isinstance(
            value, ContextlibContextManagerLocalGeneratorObjectVariable
        ):
            unimplemented(
                gb_type="reconstructing @contextmanager object",
                context=f"object: {value}",
                explanation="Returning a @contextmanager object from a compiled function is not supported.",
                hints=[
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # Dynamo normally prefers codegen from source to account for aliasing.
        if (
            value.source is not None
            and allow_cache
            and not (
                value.is_realized() and isinstance(value, LocalGeneratorObjectVariable)
            )
        ):
            # There's a corner case for export: for instance, if the computation
            # graph is just identity on an input tensor, Dynamo would just emit
            # a `LOAD_FAST` from the input source, rather than generating an
            # identity FX graph.
            #
            # However, export wants to maximize graph capture; in the case
            # above, export _wants to_ obtain an identity FX graph (despite it
            # appears unnecessarily expensive for `torch.compile`), so we have
            # the following option to override Dynamo's preference for codegen
            # from source. Moreover, this option applies recursively, for cases
            # like input tensor being returned in a new dictionary.
            #
            # And why the `ValueMutationExisting` check? Not sure, so leaving it
            # to keep the old behavior, as when `value_from_source` was
            # introduced. TODO sort out the invariants among side effect,
            # codegen and export.
            if (
                isinstance(value.mutation_type, ValueMutationExisting)
                or self.value_from_source
            ):
                return self(value.source)

        if isinstance(value, ComputedLazyConstantVariable) and not value.is_realized():
            # Recompute from the operands at runtime instead of burning in the value
            self.uses[value] += 1
            self.call_reconstruct(value)
            if allow_cache and value in self.tempvars:
                self._output.append(create_dup_top())
                self.add_cache(value)
        elif value.is_python_constant() and is_safe_constant(
            value.as_python_constant()
        ):
            output.append(self.create_load_const(value.as_python_constant()))
        elif isinstance(value, TensorWithTFOverrideVariable):
            graph_outputs_key = self.add_graph_output(value)

            self.add_push_null(
                lambda: self.load_import_from(utils.__name__, "to_subclass")
            )
            self.load_graph_output(graph_outputs[graph_outputs_key].index)
            output.append(
                self.create_load_global(
                    value.global_mangled_class_name(self.tx),  # type: ignore[arg-type]
                    add=True,
                )
            )
            output.extend(create_call_function(2, False))
        elif (
            isinstance(value, SymNodeVariable)
            and value.python_type() is float
            and not self.tx.export
        ):
            # This is a little unusual; force the output convention to be a
            # Tensor here.  Don't do this for export because this is
            # apparently load bearing for export tests (but I am a bit
            # doubtful it actually works in the real world)
            # NB: It works to add_graph_output on a computed expression
            # as_tensor here, because we memoize as_tensor calls on
            # SymNodeVariable!
            graph_outputs_key = self.add_graph_output(
                value.as_tensor(self.tx, torch.float64)
            )

            def gen_fn() -> None:
                self.load_graph_output(graph_outputs[graph_outputs_key].index)
                output.append(self.create_load_attr("item"))

            self.add_push_null(gen_fn)
            output.extend(create_call_function(0, False))
        elif isinstance(
            value,
            (
                TensorVariable,
                SymNodeVariable,
                UnspecializedPythonVariable,
                NumpyNdarrayVariable,
                CustomClassObjectVariable,
            ),
        ):
            graph_outputs_key = self.add_graph_output(value)

            if isinstance(value, NumpyNdarrayVariable):
                self.add_push_null(
                    lambda: self.load_import_from(utils.__name__, "to_numpy_helper")
                )
                self.load_graph_output(graph_outputs[graph_outputs_key].index)
                output.extend(create_call_function(1, False))
            elif isinstance(value, UnspecializedPythonVariable) and value.need_unwrap:

                def gen_fn() -> None:
                    self.load_graph_output(graph_outputs[graph_outputs_key].index)
                    output.append(self.create_load_attr("item"))

                self.add_push_null(gen_fn)
                output.extend(create_call_function(0, False))
            else:
                self.load_graph_output(graph_outputs[graph_outputs_key].index)
        elif isinstance(value, NNModuleVariable):
            parts = value.module_key.split(".")
            if parts[0] in self.code_options["co_varnames"]:
                output.append(self.create_load(parts[0]))
                parts = parts[1:]
            else:
                if self.root is None:
                    raise AssertionError("self.root must not be None")
                output.append(self.create_load_const_unchecked(self.root))
            for part in parts:
                output.append(self.create_load_attr(part))
        else:
            self.uses[value] += 1
            try:
                self.call_reconstruct(value)
            except NotImplementedError as e:
                unimplemented(
                    gb_type="Reconstruction failure",
                    context=str(value),
                    explanation=f"Dynamo has no bytecode reconstruction implemented for sourceless variable {value}.",
                    hints=[
                        "If Dynamo is attempting to trace a return statement and your code is attempting to return a variable "
                        "that Dynamo cannot reconstruct, then remove it from the return statement.",
                        *graph_break_hints.CAUSED_BY_EARLIER_GRAPH_BREAK,
                        "Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have "
                        "reconstruction rules may be fundamentally unreconstructable.",
                    ],
                    from_exc=e,
                )
            if allow_cache and value in self.tempvars:
                self._output.append(create_dup_top())
                self.add_cache(value)

        self.top_of_stack = value

    def add_graph_output(self, value: VariableTracker) -> int:
        graph_outputs_key = id(value.as_proxy())
        if graph_outputs_key not in self.graph_outputs:
            self.graph_outputs[graph_outputs_key] = GraphOutputEntry(
                len(self.graph_outputs), value
            )
        return graph_outputs_key

    def load_graph_output(self, index: int) -> None:
        output = self._output
        if self.graph_output_var is None:
            raise AssertionError("graph_output_var must not be None")
        output.append(self.create_load(self.graph_output_var))
        output.append(self.create_load_const(index))
        output.append(self.create_binary_subscr())

    def add_cache(self, value: VariableTracker | Source) -> None:
        var = self.new_var()
        self.tempvars[value] = var
        self._output.append(self.create_store(var))

    def clear_tempvars(self) -> None:
        for key, var in list(self.tempvars.items()):
            if var is not None:
                self._output.append(self.create_delete(var))
            del self.tempvars[key]
        self.top_of_stack = None

    def foreach(self, items: Iterable[VariableTracker | Source]) -> None:
        for i in items:
            self(i)

    def create_binary_subscr(self) -> Instruction:
        return create_binary_subscr()

    def setup_globally_cached(self, name: str, value: Any) -> list[Instruction]:
        """Store value in a new global"""
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        f_globals = self.tx.f_globals
        if name in f_globals:
            if id(f_globals[name]) != id(value):
                raise AssertionError(
                    f"f_globals[{name!r}] already exists with a different identity"
                )
        else:
            f_globals[name] = value
        return [self.create_load_global(name, add=True)]

    def clear_tos(self) -> None:
        self.top_of_stack = None

    def append_output(self, inst: Instruction) -> None:
        if not isinstance(inst, Instruction):
            raise AssertionError(f"expected Instruction, got {type(inst)}")
        self._output.append(inst)
        self.clear_tos()

    def extend_output(self, insts: list[Instruction]) -> None:
        if not all(isinstance(x, Instruction) for x in insts):
            raise AssertionError("all elements of insts must be Instruction instances")
        self._output.extend(insts)
        self.clear_tos()

    def get_instructions(self) -> list[Instruction]:
        return self._output

    def get_pycode(self) -> list[str] | None:
        if not config.generate_pycode:
            return None
        return self.pycodes

    def create_load(self, name: str) -> Instruction:
        if name not in self.code_options["co_varnames"]:
            raise AssertionError(f"{name} missing")
        return create_instruction("LOAD_FAST", argval=name)

    def create_load_closure(self, name: str) -> Instruction:
        if name not in self.cell_and_freevars():
            raise AssertionError(f"{name!r} not in cell_and_freevars")
        inst_name = "LOAD_FAST" if sys.version_info >= (3, 13) else "LOAD_CLOSURE"
        return create_instruction(inst_name, argval=name)

    def create_load_deref(self, name: str) -> Instruction:
        if name not in self.cell_and_freevars():
            raise AssertionError(f"{name!r} not in cell_and_freevars")
        return create_instruction("LOAD_DEREF", argval=name)

    def create_store(self, name: str) -> Instruction:
        if name not in self.code_options["co_varnames"]:
            raise AssertionError(f"{name} missing")
        return create_instruction("STORE_FAST", argval=name)

    def create_store_deref(self, name: str) -> Instruction:
        if name not in self.cell_and_freevars():
            raise AssertionError(f"{name!r} not in cell_and_freevars")
        return create_instruction("STORE_DEREF", argval=name)

    def create_load_global(self, name: str, add: bool = False) -> Instruction:
        if add:
            self.tx.output.update_co_names(name)
        if name not in self.code_options["co_names"]:
            raise AssertionError(f"{name} not in co_names")
        return create_instruction("LOAD_GLOBAL", argval=name)

    def create_load_const(self, value: Any) -> Instruction:
        return create_load_const(value)

    def create_load_const_unchecked(self, value: Any) -> Instruction:
        return create_load_const(value, checked=False)

    def load_method(self, name: str) -> None:
        self.tx.output.update_co_names(name)
        self.append_output(create_load_method(name))

    def call_method(self, nargs: int) -> None:
        self.extend_output(create_call_method(nargs))

    def create_list_append(self) -> list[Instruction]:
        # Append TOS to the list at TOS-1, leaving the list on the stack
        # (same stack effect as LIST_APPEND with arg=1).
        #
        # The bare LIST_APPEND opcode does not lock the list and so requires
        # the target be uniquely owned (refcnt == 1) on free-threaded builds.
        # Dynamo can't enforce this, so instead use LIST_EXTEND, which does
        # lock
        return [
            create_instruction("BUILD_LIST", arg=1),
            create_instruction("LIST_EXTEND", arg=1),
        ]

    def create_load_attr(self, name: str) -> Instruction:
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)
        return create_instruction("LOAD_ATTR", argval=name)

    def load_attr(self, name: str) -> None:
        self.append_output(self.create_load_attr(name))

    def create_load_attrs(self, names: str) -> list[Instruction]:
        return [self.create_load_attr(name) for name in names.split(".")]

    def create_store_attr(self, name: str) -> Instruction:
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] += (name,)
        return create_instruction("STORE_ATTR", argval=name)

    def store_attr(self, name: str) -> None:
        self.append_output(self.create_store_attr(name))

    def load_function_name(
        self, fn_name: str, push_null: bool, num_on_stack: int = 0
    ) -> list[Instruction]:
        """Load the global fn_name on the stack num_on_stack down"""
        output = []
        if push_null and sys.version_info >= (3, 11):
            output.extend(add_push_null(self.create_load_global(fn_name, add=True)))
            if num_on_stack > 0:
                output.extend(
                    [
                        *self.rot_n(num_on_stack + 2),
                        *self.rot_n(num_on_stack + 2),
                    ]
                )
        else:
            output.extend(
                [
                    self.create_load_global(fn_name, add=True),
                    *self.rot_n(num_on_stack + 1),
                ]
            )
        return output

    def rot_n(self, n: int) -> list[Instruction]:
        try:
            return create_rot_n(n)
        except AttributeError:
            # desired rotate bytecode doesn't exist, generate equivalent bytecode
            return [
                create_build_tuple(n),
                self.create_load_const_unchecked(rot_n_helper(n)),
                *create_rot_n(2),
                *create_call_function_ex(False, False),
                create_instruction("UNPACK_SEQUENCE", arg=n),
            ]

    def pop_null(self) -> list[Instruction]:
        # POP_TOP doesn't work for null, so we pop nulls by pushing in a
        # nop function, calling it (which consumes the null), and popping the result.
        if sys.version_info < (3, 11):
            raise AssertionError("pop_null requires Python 3.11+")
        return [
            self.create_load_const_unchecked(lambda: None),
            # 3.13 swapped NULL and callable
            *(
                (create_instruction("SWAP", arg=2),)
                if sys.version_info >= (3, 13)
                else ()
            ),
            *create_call_function(0, False),
            create_instruction("POP_TOP"),
        ]

    def pop_top(self) -> None:
        self.append_output(create_instruction("POP_TOP"))

    def call_function(self, nargs: int, push_null: bool) -> None:
        self.extend_output(create_call_function(nargs, push_null=push_null))

    def dup_top(self) -> None:
        self.append_output(create_dup_top())

    def store(self, varname: str) -> None:
        self.append_output(self.create_store(varname))

    def load_deref(self, varname: str) -> None:
        self.append_output(self.create_load_deref(varname))

    def make_function_with_closure(
        self,
        fn_name: str,
        code: types.CodeType,
    ) -> None:
        """Creates a closure with code object `code`.

        Expects the TOS to be the tuple of cells to use for this closure.
        TOS will be popped to create the closure.
        Args:
            - fn_name: name of the function
            - code: code object of the function
                (does not include the tuple of cells on the TOS)
        """
        output = self._output

        output.append(self.create_load_const(code))
        if sys.version_info < (3, 11):
            output.append(self.create_load_const(fn_name))
        if sys.version_info >= (3, 13):
            output.extend(
                [
                    create_instruction("MAKE_FUNCTION"),
                    create_instruction("SET_FUNCTION_ATTRIBUTE", arg=0x08),
                ]
            )
        else:
            output.append(create_instruction("MAKE_FUNCTION", arg=0x08))

        self.clear_tos()

    def create_load_python_module(self, mod: types.ModuleType) -> Instruction:
        """
        Generate a LOAD_GLOBAL instruction to fetch a given python module.
        """
        output = self.tx.output
        global_scope = output.global_scope
        name = re.sub(r"^.*[.]", "", mod.__name__)
        if global_scope.get(name, None) is mod:
            return self.create_load_global(name, add=True)
        prefix = f"___module_{name}"
        global_name = self.tx.output.install_global_by_id(prefix, mod)
        return self.create_load_global(global_name, add=True)

    def mark_source_temp(self, source: Source) -> None:
        """
        Mark a source as a temp variable, so that it can be reused.
        """
        if source not in self.tempvars:
            self.tempvars[source] = None

    def make_resume_arg_snapshot(
        self, name: str, value: VariableTracker | Source
    ) -> None:
        self.add_push_null(
            lambda: self.load_import_from(
                "torch._dynamo.resume_execution", "_make_resume_arg_snapshot"
            )
        )
        self(value)
        self.extend_output(create_call_function(1, False))
        self.append_output(self.create_store(name))

    def replay_fast_local_events(
        self,
        events: list[tuple[str | None, VariableTracker | None]] | None,
        resume_arg_paths_by_event: list[set[tuple[Any, ...]]] | None = None,
        initial_values: dict[str, VariableTracker] | None = None,
    ) -> dict[str, Source]:
        expected_ids_by_owner: dict[str, Source] = {}
        owner_names = {
            path[0]
            for paths in (resume_arg_paths_by_event or ())
            for path in paths
            if len(path) >= 2 and isinstance(path[0], str)
        }
        for owner_name in sorted(owner_names):
            if owner_name not in self.code_options["co_varnames"]:
                continue
            self.add_push_null(
                lambda: self.load_import_from(
                    "torch._dynamo.resume_execution",
                    "_snapshot_resume_arg_identities",
                )
            )
            self(self.tx._resume_arg_owner_source(owner_name))
            self.extend_output(create_call_function(1, False))
            identity_name = self.new_var("resume_arg_ids")
            self.append_output(self.create_store(identity_name))
            expected_ids_by_owner[owner_name] = TempLocalSource(identity_name)

        for name, value in (initial_values or {}).items():
            if name in self.code_options["co_varnames"]:
                self(value)
                self.append_output(self.create_store(name))
        events = events or []
        resume_arg_paths_by_event = resume_arg_paths_by_event or [set() for _ in events]
        if len(events) != len(resume_arg_paths_by_event):
            raise AssertionError("fast-local events and cleanup paths must align")
        resume_args_varname = self.tx._boxed_resume_arg_name()
        resume_args = (
            self.tx.f_locals.get(resume_args_varname)
            if resume_args_varname is not None
            else None
        )

        def refresh_frame_locals_if_needed() -> None:
            if not isinstance(resume_args, list):
                return
            if (
                resume_args_varname is None
                or resume_args_varname not in self.code_options["co_varnames"]
            ):
                return
            self.add_push_null(
                lambda: self.load_import_from(
                    "torch._dynamo.resume_execution",
                    "_refresh_frame_locals_if_resume_carrier_mutated",
                )
            )
            self(self.tx._resume_arg_owner_source(resume_args_varname))
            self.append_output(self.create_load_const(len(resume_args)))
            self.extend_output(create_call_function(2, False))
            self.pop_top()

        # Replay only the fast-local opcodes.  In Python 3.12 and earlier, an
        # already-materialized frame.f_locals dict intentionally remains stale
        # until CPython refreshes it on the next property access.  Updating the
        # dict here would release references earlier than eager execution.
        for (name, value), paths in zip(events, resume_arg_paths_by_event):
            if name is None:
                self.clear_resume_arg_paths(
                    paths, expected_ids_by_owner=expected_ids_by_owner
                )
                continue
            if name not in self.code_options["co_varnames"]:
                raise AssertionError(f"fast local {name!r} is not in co_varnames")
            if value is None:
                self.append_output(self.create_delete(name))
                self.add_pycode(f"del {name}")
            else:
                self(value)
                self.add_pycode(f"{name} = {{}}", value)
                self.append_output(self.create_store(name))
            self.clear_resume_arg_paths(
                paths, expected_ids_by_owner=expected_ids_by_owner
            )
            refresh_frame_locals_if_needed()
        return expected_ids_by_owner

    def delete_resume_arg_identity_snapshots(
        self, expected_ids_by_owner: dict[str, Source]
    ) -> None:
        for source in expected_ids_by_owner.values():
            if not isinstance(source, TempLocalSource):
                raise AssertionError("resume identity snapshot must be a temp local")
            self.append_output(self.create_delete(source.local_name))

    def clear_resume_arg_paths(
        self,
        paths: set[tuple[Any, ...]],
        *,
        preserve_observable_lifetime: bool = False,
        expected_ids_by_owner: dict[str, Source] | None = None,
    ) -> None:
        for owner_name, root_index in sorted(paths, key=repr, reverse=True):
            if owner_name not in self.code_options["co_varnames"]:
                raise AssertionError(f"resume carrier {owner_name!r} is not a local")
            self.add_push_null(
                lambda: self.load_import_from(
                    "torch._dynamo.resume_execution",
                    (
                        "_clear_resume_arg_path_if_unchanged"
                        if expected_ids_by_owner and owner_name in expected_ids_by_owner
                        else (
                            "_maybe_clear_resume_arg_path"
                            if preserve_observable_lifetime
                            else "_clear_resume_arg_path"
                        )
                    ),
                )
            )
            self(self.tx._resume_arg_owner_source(owner_name))
            self.append_output(self.create_load_const((root_index,)))
            if expected_ids_by_owner and owner_name in expected_ids_by_owner:
                self(expected_ids_by_owner[owner_name])
                self.append_output(self.create_load_const(preserve_observable_lifetime))
                self.extend_output(create_call_function(4, False))
            else:
                self.extend_output(create_call_function(2, False))
            self.pop_top()

    def make_call_generated_code(
        self,
        fn_name: str,
        graph_input_names_to_delete: set[str] | None = None,
        graph_input_names_to_clear: set[str] | None = None,
        resume_arg_indexes_to_clear: set[int] | None = None,
        resume_arg_locals_to_materialize: dict[str, VariableTracker | Source]
        | None = None,
        resume_arg_paths_to_clear: set[tuple[Any, ...]] | None = None,
        unconditional_resume_arg_paths_to_clear: set[tuple[Any, ...]] | None = None,
        fast_local_events: list[tuple[str | None, VariableTracker | None]]
        | None = None,
        fast_local_event_paths: list[set[tuple[Any, ...]]] | None = None,
        fast_local_initial_values: dict[str, VariableTracker] | None = None,
        fast_local_source_overrides: dict[Source, Source] | None = None,
        boxed_call: bool = False,
    ) -> None:
        """Call the generated code function stored in fn_name"""
        self.extend_output(self.load_function_name(fn_name, True))

        graphargs = self.tx.output.graphargs
        graph_input_names_to_delete = {
            name
            for name in graph_input_names_to_delete or set()
            if name in self.code_options["co_varnames"]
        }
        graph_input_names_to_clear = {
            name
            for name in graph_input_names_to_clear or set()
            if name in self.code_options["co_varnames"]
        }
        resume_args_varname = self.tx._boxed_resume_arg_name()
        if resume_args_varname is not None:
            graph_input_names_to_clear.discard(resume_args_varname)

        def extract_nested_sources(source: Source) -> list[Source]:
            nested_sources: list[Source] = []
            if isinstance(source, ChainedSource):
                nested_sources.append(source.base)
            if isinstance(source, DictGetItemSource) and isinstance(
                source.index, Source
            ):
                nested_sources.append(source.index)
            return nested_sources

        def collect_temp_sources(sources: deque[Source], codegen: PyCodegen) -> None:
            seen_sources: OrderedSet[Source] = OrderedSet()
            while sources:
                current_source = sources.popleft()
                if current_source in seen_sources:
                    # This source is used at least twice, so it can be reused
                    codegen.mark_source_temp(current_source)
                    # Don't trace source further. This prevents us from marking too
                    # many nodes as temp sources.
                    continue
                seen_sources.add(current_source)
                sources.extend(extract_nested_sources(current_source))

        # Collect all the sources that are used more than once, so that we can
        # generate tmp variables in the generated pre-graph bytecode. This
        # essentially implements CSE.
        collect_temp_sources(
            deque([arg.source for arg in graphargs if arg.source is not None]), self
        )

        cm_var = None
        if config.record_runtime_overhead:
            # Record the pregraph bytecode start
            self.add_push_null(
                lambda: self.load_import_from(
                    utils.__name__, "record_pregraph_bytecode_enter"
                )
            )
            self.extend_output(create_call_function(0, False))
            cm_var = self.new_var()
            self.store(cm_var)

        arg_varnames = []
        for i, arg in enumerate(graphargs):
            arg_varname = self.tx.new_pycode_varname("arg")
            arg_varnames.append(arg_varname)
            if arg.pass_arg_as_tensor:
                self.add_push_null(
                    lambda: self.extend_output(
                        [
                            self.create_load_python_module(torch),
                            self.create_load_attr("_as_tensor_fullprec"),
                        ]
                    )
                )
                self.call_reconstruct(arg)
                self.extend_output(create_call_function(1, False))
                self.add_pycode(f"{arg_varname} = torch._as_tensor_fullprec({{}})", arg)
            else:
                self.call_reconstruct(arg)
                self.add_pycode(f"{arg_varname} = {{}}", arg)
        if config.record_runtime_overhead:
            # Record the pregraph bytecode end
            self.add_push_null(
                lambda: self.load_import_from(
                    utils.__name__, "record_pregraph_bytecode_exit"
                )
            )
            if cm_var is None:
                raise AssertionError("cm_var must not be None")
            self.extend_output([self.create_load(cm_var)])
            self.extend_output(create_call_function(1, False))
            self.pop_top()

        self.clear_tempvars()

        for name, value in (resume_arg_locals_to_materialize or {}).items():
            self.make_resume_arg_snapshot(name, value)

        # STORE_FAST installs its replacement before decrefing the prior value.
        # Preserve that ordering when carrier cleanup can run user finalizers.
        prior_overridden_sources = self.overridden_sources
        if fast_local_source_overrides:
            self.overridden_sources = {
                **prior_overridden_sources,
                **fast_local_source_overrides,
            }
        try:
            expected_ids_by_owner = self.replay_fast_local_events(
                fast_local_events,
                fast_local_event_paths,
                fast_local_initial_values,
            )
        finally:
            self.overridden_sources = prior_overridden_sources

        for name in sorted(graph_input_names_to_clear):
            self.append_output(self.create_load(name))
            self.load_method("clear")
            self.call_method(0)
            self.pop_top()
        if resume_arg_paths_to_clear:
            unconditional_resume_arg_paths_to_clear = (
                unconditional_resume_arg_paths_to_clear or set()
            )
            self.clear_resume_arg_paths(
                unconditional_resume_arg_paths_to_clear,
                expected_ids_by_owner=expected_ids_by_owner,
            )
            self.clear_resume_arg_paths(
                resume_arg_paths_to_clear - unconditional_resume_arg_paths_to_clear,
                preserve_observable_lifetime=True,
                expected_ids_by_owner=expected_ids_by_owner,
            )
        self.delete_resume_arg_identity_snapshots(expected_ids_by_owner)
        if resume_args_varname is not None and resume_arg_indexes_to_clear:
            resume_args = self.tx.f_locals.get(resume_args_varname)
            if isinstance(resume_args, list):
                for idx in sorted(resume_arg_indexes_to_clear, reverse=True):
                    self.add_push_null(
                        lambda: self.load_import_from(
                            "torch._dynamo.resume_execution",
                            "_maybe_clear_tensor_resume_arg",
                        )
                    )
                    self.append_output(self.create_load(resume_args_varname))
                    self.append_output(self.create_load_const(idx))
                    self.extend_output(create_call_function(2, False))
                    self.pop_top()
        graph_input_names_to_delete |= graph_input_names_to_clear
        for name in sorted(graph_input_names_to_delete):
            self.append_output(self.create_delete(name))
        if graph_input_names_to_delete:
            self.add_pycode("del " + ", ".join(sorted(graph_input_names_to_delete)))

        if boxed_call:
            self.append_output(create_instruction("BUILD_LIST", arg=len(graphargs)))
            nargs = 1
        else:
            nargs = len(graphargs)

        self.extend_output(create_call_function(nargs, False))
        graph_call_args = (
            f"[{', '.join(arg_varnames)}]" if boxed_call else ", ".join(arg_varnames)
        )
        self.add_pycode(
            f"__graph_out = {fn_name}({graph_call_args})",
        )

    def create_import_name(self, module_name: str) -> Instruction:
        return create_instruction("IMPORT_NAME", argval=module_name)

    def load_import_from(self, module_name: str, object_name: str) -> None:
        source = AttrSource(self.tx.import_source(module_name), object_name)
        # Note: This approach is somewhat aggressive because typically, a source is marked
        # as a tempvar only when it is used more than once. In this case, we're marking it
        # as a tempvar without performing that analysis. However, this is a simple solution,
        # and in many cases, load imports are reused multiple times.
        self.mark_source_temp(source)
        self(source)

    def create_call_function_kw(
        self, nargs: int, kw_names: Iterable[str], push_null: bool
    ) -> list[Instruction]:
        if sys.version_info >= (3, 13):
            output = create_call_function(nargs, push_null)
            if output[-1].opname != "CALL":
                raise AssertionError(
                    f"expected last instruction to be CALL, got {output[-1].opname}"
                )
            output.insert(-1, self.create_load_const(kw_names))
            output[-1] = create_instruction("CALL_KW", arg=nargs)
            return output
        elif sys.version_info >= (3, 11):
            output = create_call_function(nargs, push_null)
            if sys.version_info >= (3, 12):
                idx = -1
                expected_inst = "CALL"
            else:
                idx = -2
                expected_inst = "PRECALL"
            if output[idx].opname != expected_inst:
                raise AssertionError(
                    f"expected instruction at index {idx} to be {expected_inst}, "
                    f"got {output[idx].opname}"
                )
            kw_names_inst = create_instruction("KW_NAMES", argval=kw_names)
            output.insert(idx, kw_names_inst)
            return output
        return [
            self.create_load_const(kw_names),
            create_instruction("CALL_FUNCTION_KW", arg=nargs),
        ]

    def create_delete(self, value: object) -> Instruction:
        return create_instruction("DELETE_FAST", argval=value)
