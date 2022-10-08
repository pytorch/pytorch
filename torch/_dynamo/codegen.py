import collections
import dataclasses
import re
import sys
import types
from typing import List

import torch.nn

from .bytecode_transformation import create_instruction, Instruction
from .exc import unimplemented
from .source import AttrSource, Source
from .utils import is_safe_constant, istype, rot_n_helper
from .variables.base import VariableTracker
from .variables.nn_module import NNModuleVariable
from .variables.tensor import (
    TensorVariable,
    TensorWithTFOverrideVariable,
    UnspecializedNumpyVariable,
    UnspecializedPythonVariable,
)


@dataclasses.dataclass
class GraphOutputEntry:
    index: int
    variable: VariableTracker

    def merge(self, other: VariableTracker):
        # merge in any extra guards
        self.variable = self.variable.add_options(other)


class PyCodegen(object):
    """
    Helper class uses for constructing Python bytecode
    """

    def __init__(
        self,
        tx=None,
        root: torch.nn.Module = None,
        graph_output_var: str = None,
        tempvars=None,
    ):
        self.root = root
        self.top_of_stack = None
        self.uses = collections.Counter()
        self.graph_outputs = collections.OrderedDict()
        self._output: List[Instruction] = []
        self.tempvars = tempvars or {}
        self.tx = tx
        self.graph_output_var = graph_output_var
        self.code_options = self.tx.output.code_options
        self.cell_and_freevars = self.tx.cell_and_freevars
        self.new_var = self.tx.output.new_var

    def graph_output_vars(self):
        return [x.variable for x in self.graph_outputs.values()]

    def __call__(self, value, allow_cache=True):
        """Generate code such that top-of-stack (TOS) is set to value"""
        if isinstance(value, Source):
            self._output.extend(value.reconstruct(self))
            self.clear_tos()
            return

        self.tx.output.guards.update(value.guards)

        assert isinstance(value, VariableTracker)
        output = self._output
        graph_outputs = self.graph_outputs

        if self.top_of_stack is value:
            output.append(create_instruction("DUP_TOP"))
            return

        if allow_cache:
            if value.mutable_local and value.mutable_local in self.tempvars:
                output.append(self.create_load(self.tempvars[value.mutable_local]))
                self.top_of_stack = value
                return
            if self.tempvars.get(value) is not None:
                output.append(self.create_load(self.tempvars[value]))
                self.top_of_stack = value
                return

        if value.source is not None and allow_cache:
            output.extend(value.source.reconstruct(self))
        elif value.is_python_constant() and is_safe_constant(
            value.as_python_constant()
        ):
            output.append(self.create_load_const(value.as_python_constant()))
        elif isinstance(
            value,
            (
                TensorVariable,
                TensorWithTFOverrideVariable,
                UnspecializedNumpyVariable,
                UnspecializedPythonVariable,
            ),
        ):
            if isinstance(value, TensorWithTFOverrideVariable):
                # unwrap back to tensor
                value = value.tensor_variable
            graph_outputs_key = id(value.proxy)
            if graph_outputs_key not in graph_outputs:
                graph_outputs[graph_outputs_key] = GraphOutputEntry(
                    len(graph_outputs), value
                )
            else:
                graph_outputs[graph_outputs_key].merge(value)

            output.append(self.create_load(self.graph_output_var))
            output.append(
                self._create_load_const(graph_outputs[graph_outputs_key].index)
            )
            output.append(create_instruction("BINARY_SUBSCR"))

            if isinstance(value, UnspecializedNumpyVariable):
                unspec_var = self.tx.output.new_var("unspec")
                raw_type = type(value.raw_value)
                output.extend(
                    [
                        self.create_load_attr("item"),
                        create_instruction("CALL_FUNCTION", 0),
                        self.create_store(unspec_var),
                        self.create_load_const(raw_type),
                        self.create_load(unspec_var),
                        create_instruction("CALL_FUNCTION", 1),
                    ]
                )
            if isinstance(value, UnspecializedPythonVariable) and value.need_unwrap:
                output.extend(
                    [
                        self.create_load_attr("item"),
                        create_instruction("CALL_FUNCTION", 0),
                    ]
                )
        elif isinstance(value, NNModuleVariable):
            parts = value.module_key.split(".")
            if parts[0] in self.code_options["co_varnames"]:
                output.append(self.create_load(parts[0]))
                parts = parts[1:]
            else:
                assert self.root is not None
                output.append(self.create_load_output(self.root))
            for part in parts:
                output.append(self.create_load_attr(part))
        else:
            self.uses[value] += 1
            try:
                output.extend(value.reconstruct(self))
            except NotImplementedError:
                unimplemented(f"reconstruct: {value}")
            if allow_cache and value in self.tempvars:
                self._output.append(create_instruction("DUP_TOP"))
                self.add_cache(value)

        self.top_of_stack = value

    def add_cache(self, value):
        var = self.new_var()
        self.tempvars[value] = var
        if value.mutable_local:
            self.tempvars[value.mutable_local] = var
        self._output.append(self.create_store(var))

    def foreach(self, items):
        for i in items:
            self(i)

    def setup_globally_cached(self, name, value):
        """Store value in a new global"""
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", name)
        f_globals = self.tx.f_globals
        if name in f_globals:
            assert id(f_globals[name]) == id(value)
        else:
            f_globals[name] = value
        return [self.create_load_global(name, add=True)]

    def clear_tos(self):
        self.top_of_stack = None

    def append_output(self, inst):
        assert isinstance(inst, Instruction)
        self._output.append(inst)
        self.clear_tos()

    def extend_output(self, insts):
        assert all(isinstance(x, Instruction) for x in insts)
        self._output.extend(insts)
        self.clear_tos()

    def get_instructions(self):
        return self._output

    def create_load(self, name):
        if name in self.cell_and_freevars():
            return create_instruction(
                "LOAD_DEREF", self.cell_and_freevars().index(name), name
            )
        assert name in self.code_options["co_varnames"], f"{name} missing"
        return create_instruction(
            "LOAD_FAST", self.code_options["co_varnames"].index(name), name
        )

    def create_load_closure(self, name):
        assert name in self.cell_and_freevars()
        return create_instruction(
            "LOAD_CLOSURE", self.cell_and_freevars().index(name), name
        )

    def create_store(self, name):
        if name in self.cell_and_freevars():
            return create_instruction(
                "STORE_DEREF", self.cell_and_freevars().index(name), name
            )
        assert name in self.code_options["co_varnames"]
        return create_instruction(
            "STORE_FAST", self.code_options["co_varnames"].index(name), name
        )

    def create_load_global(self, name, add=False):
        if add:
            self.tx.output.update_co_names(name)
        assert name in self.code_options["co_names"], f"{name} not in co_names"
        return create_instruction(
            "LOAD_GLOBAL", self.code_options["co_names"].index(name), name
        )

    def create_load_const(self, value):
        assert is_safe_constant(value), f"unsafe constant {value}"
        return self._create_load_const(value)

    @staticmethod
    def get_const_index(code_options, value):
        co_consts = code_options["co_consts"]
        assert istype(co_consts, tuple)
        index = None
        for i, v in enumerate(co_consts):
            if type(v) is type(value) and v == value:
                index = i
                break
        if index is None:
            index = len(co_consts)
            co_consts = co_consts + (value,)
            code_options["co_consts"] = co_consts
        return index

    def _create_load_const(self, value):
        index = self.get_const_index(self.code_options, value)
        return create_instruction("LOAD_CONST", index, value)

    create_load_output = _create_load_const

    def create_load_attr(self, name):
        if name not in self.code_options["co_names"]:
            self.code_options["co_names"] = self.code_options["co_names"] + (name,)
        return create_instruction(
            "LOAD_ATTR", self.code_options["co_names"].index(name), name
        )

    def create_load_attrs(self, names):
        return [self.create_load_attr(name) for name in names.split(".")]

    def load_function_name(self, fn_name, num_on_stack=0):
        """Load the global fn_name on the stack num_on_stack down"""
        return [self.create_load_global(fn_name, add=True)] + self.rot_n(
            num_on_stack + 1
        )

    def rot_n(self, n):
        if n == 0 or n == 1:
            return []
        elif n == 2:
            return [create_instruction("ROT_TWO")]
        elif n == 3:
            return [create_instruction("ROT_THREE")]
        elif n == 4 and sys.version_info >= (3, 8):
            return [create_instruction("ROT_FOUR")]
        elif sys.version_info >= (3, 10):
            return [create_instruction("ROT_N", n)]
        else:
            return [
                create_instruction("BUILD_TUPLE", n),
                self._create_load_const(rot_n_helper(n)),
                create_instruction("ROT_TWO"),
                create_instruction("CALL_FUNCTION_EX", 0),
                create_instruction("UNPACK_SEQUENCE", n),
            ]

    def make_function_with_closure(
        self, fn_name: str, code: types.CodeType, num_on_stack=0
    ):
        freevars = code.co_freevars
        assert freevars
        output = self._output
        for var in freevars:
            assert var in self.cell_and_freevars()
            output.append(
                create_instruction(
                    "LOAD_CLOSURE", self.cell_and_freevars().index(var), var
                )
            )
        output.append(create_instruction("BUILD_TUPLE", len(freevars)))
        output.append(self.create_load_const(code))
        output.append(self.create_load_const(fn_name))
        output.append(create_instruction("MAKE_FUNCTION", 0x08))
        output.extend(self.rot_n(num_on_stack + 1))
        self.clear_tos()

    def create_load_python_module(self, mod):
        """
        Generate a LOAD_GLOBAL instruction to fetch a given python module.
        """
        root_globals = self.tx.output.root_globals
        name = re.sub(r"^.*[.]", "", mod.__name__)
        if root_globals.get(name, None) is mod:
            return self.create_load_global(name, add=True)
        mangled_name = f"___module_{name}_{id(mod)}"
        if mangled_name not in root_globals:
            self.tx.output.install_global(mangled_name, mod)
        return self.create_load_global(mangled_name, add=True)

    def make_call_generated_code(self, fn_name: str) -> List[Instruction]:
        """Call the generated code function stored in fn_name"""
        self.extend_output(self.load_function_name(fn_name))

        graphargs = self.tx.output.graphargs
        for arg in graphargs:
            if arg.is_unspecialized:
                self.extend_output(
                    [
                        self.create_load_python_module(torch),
                        self.create_load_attr("tensor"),
                    ]
                )
                self.extend_output(arg.load(self))
                self.extend_output(
                    [
                        create_instruction("CALL_FUNCTION", 1),
                    ]
                )
            else:
                self.extend_output(arg.load(self))

        self.append_output(create_instruction("CALL_FUNCTION", len(graphargs)))

    def load_import_from(self, module_name, object_name):
        self.extend_output(
            AttrSource(self.tx.import_source(module_name), object_name).reconstruct(
                self
            )
        )

    def create_begin_finally(self):
        if sys.version_info < (3, 8):
            return self.create_load_const(None)
        else:
            return create_instruction("BEGIN_FINALLY")
