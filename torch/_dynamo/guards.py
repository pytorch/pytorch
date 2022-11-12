import collections
import dataclasses
import enum
import logging
import math
import os
import re
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

import sympy

import torch
from torch.fx.experimental.symbolic_shapes import FloorDiv

from . import config, convert_frame, mutation_guard
from .eval_frame import set_guard_error_hook, set_guard_fail_hook
from .exc import unimplemented
from .utils import (
    dict_const_keys,
    dict_param_key_ids,
    guard_failures,
    istype,
    orig_code_map,
    rename_implicit,
    tuple_iterator_getitem,
    tuple_iterator_len,
)

log = logging.getLogger(__name__)
TensorGuards = torch._C._dynamo.guards.TensorGuards
check_obj_id = torch._C._dynamo.guards.check_obj_id
check_type_id = torch._C._dynamo.guards.check_type_id


CLOSURE_VARS = collections.OrderedDict(
    [
        ("___check_type_id", check_type_id),
        ("___check_obj_id", check_obj_id),
        ("___is_grad_enabled", torch.is_grad_enabled),
        ("___odict_getitem", collections.OrderedDict.__getitem__),
        ("___dict_param_key_ids", dict_param_key_ids),
        ("___dict_const_keys", dict_const_keys),
        ("___tuple_iterator_len", tuple_iterator_len),
        ("___tuple_iterator_getitem", tuple_iterator_getitem),
        ("__math_isnan", math.isnan),
        ("inf", float("inf")),
    ]
)


class GuardSource(enum.Enum):
    LOCAL = 0
    GLOBAL = 1
    LOCAL_NN_MODULE = 2
    GLOBAL_NN_MODULE = 3
    CONSTANT = 4

    def select(self, locals_, globals_):
        if self in (GuardSource.LOCAL, GuardSource.LOCAL_NN_MODULE):
            return locals_
        if self in (GuardSource.GLOBAL, GuardSource.GLOBAL_NN_MODULE):
            return globals_
        raise NotImplementedError()

    def is_nn_module(self):
        return self in (GuardSource.GLOBAL_NN_MODULE, GuardSource.LOCAL_NN_MODULE)

    def is_local(self):
        return self in (GuardSource.LOCAL, GuardSource.LOCAL_NN_MODULE)


@dataclasses.dataclass
class Guard:
    name: str
    source: GuardSource
    create_fn: Callable
    is_volatile: bool = False

    # Export only. These values are written to at time of guard check_fn creation.
    guard_types: Optional[List[str]] = None
    code_list: Optional[List[str]] = None
    obj_weakref: Optional[Any] = None
    guarded_class_weakref: Optional[type] = None

    def __hash__(self):
        return hash((self.name, self.source, id(self.create_fn)))

    def sort_key(self):
        return (
            self.source.value if self.source else -1,
            len(self.name),
            self.name,
            self.create_fn.__code__.co_firstlineno,
        )

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()

    def __str__(self):
        s = f"""
            {self.source.name.lower() if self.source else ""} {repr(self.name)} {self.create_fn.__name__}
            {{
                'guard_types': {self.guard_types},
                'code': {self.code_list},
                'obj_weakref': {self.obj_weakref}
                'guarded_class': {self.guarded_class_weakref}
            }}
            """
        return s

    def create(self, local_builder: "GuardBuilder", global_builder: "GuardBuilder"):
        return self.create_fn(self.source.select(local_builder, global_builder), self)

    def is_nn_module(self):
        return self.source.is_nn_module()

    def is_local(self):
        return self.source.is_local()

    def set_export_info(self, guard_type, guarded_class, code_list, obj_weakref):
        if not self.guard_types:
            self.guard_types = list()

        self.guard_types.append(guard_type)

        assert self.guarded_class_weakref in (
            guarded_class,
            None,
        ), "Guarded class id must be identical, or None"
        self.guarded_class_weakref = guarded_class

        if not self.code_list:
            self.code_list = code_list
        else:
            self.code_list.extend(code_list)

        assert self.obj_weakref in (
            obj_weakref,
            None,
        ), "Guarded object must be identical, or None"
        self.obj_weakref = obj_weakref


def strip_function_call(name):
    """
    "___odict_getitem(a, 1)" => "a"
    """
    m = re.search(r"([a-z0-9_]+)\(([^(),]+)[^()]*\)", name)
    if m and m.group(1) != "slice":
        return strip_function_call(m.group(2))
    return strip_getattr_getitem(name)


def strip_getattr_getitem(name):
    """
    "a[1]" => "a"
    "a.foo" => "a"
    """
    return re.split(r"[.\[]", name)[0]


class GuardBuilder:
    def __init__(
        self, id_ref: Callable, scope: Dict[str, Any], guarded_code, renames=True
    ):
        self.id_ref = id_ref
        if scope:
            if renames:
                scope = {rename_implicit(k): v for k, v in scope.items()}
        else:
            scope = dict()
        self.scope = scope
        self.argnames: List[str] = []
        # Code is python expression strings generated for each guard
        self.code: List[str] = []
        self.tensor_check_names = []
        self.tensor_check_ids = {}
        self.tensor_check_examples = []
        self.guarded_code = guarded_code

    def get(self, name: str):
        return eval(name, self.scope, CLOSURE_VARS)

    def arg_ref(self, guard: Guard):
        if isinstance(guard, str):
            name = guard
        else:
            name = guard.name
        base = strip_getattr_getitem(strip_function_call(name))
        if base not in self.argnames:
            if re.match(r"^\d+$", base):
                log.warning(f"invalid var name: {guard}")
            self.argnames.append(base)

        return name

    def TYPE_MATCH(self, guard: Guard):
        # ___check_type_id is same as `id(type(x)) == y`
        t = type(self.get(guard.name))
        obj_id = self.id_ref(t)
        code = f"___check_type_id({self.arg_ref(guard)}, {obj_id})"
        self._produce_guard_code(guard, [code])

    def ID_MATCH(self, guard: Guard):
        # ___check_obj_id is same as `id(x) == y`
        m = re.match(r"^type\((.+)\)$", guard.name)
        if m:
            # optional optimization to produce cleaner/faster guard code
            return self.TYPE_MATCH(Guard(m.group(1), guard.source, None))

        code = f"___check_obj_id({self.arg_ref(guard)}, {self.id_ref(self.get(guard.name))})"
        self._produce_guard_code(guard, [code])

    def NAME_MATCH(self, guard: Guard):
        obj = self.get(guard.name)
        code = f"{self.arg_ref(guard)}.__name__ == {obj.__name__}"
        self._produce_guard_code(guard, [code])

    def HASATTR(self, guard: Guard):
        m = re.match(r"^(.*)[.]([a-zA-Z0-9_]+)$", guard.name)
        assert m, f"invalid hasattr check {guard.name}"
        base, attr = m.group(1, 2)
        ref = self.arg_ref(base)
        val = hasattr(self.get(base), attr)
        code = None
        if val:
            code = f"hasattr({ref}, {attr!r})"
        else:
            code = f"not hasattr({ref}, {attr!r})"

        self._produce_guard_code(guard, [code], provided_guarded_object=self.get(base))

    def EQUALS_MATCH(self, guard: Guard):
        ref = self.arg_ref(guard)
        val = self.get(guard.name)
        t = type(val)
        assert istype(
            val,
            (
                int,
                float,
                bool,
                type(None),
                str,
                type,
                list,
                tuple,
                set,
                slice,
                frozenset,
                range,
                torch.Size,
                torch.device,
                torch.dtype,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ), t.__name__
        if istype(val, (torch.device, torch.dtype)):
            # TODO(jansel): is this slow? perhaps optimize it
            code = f"str({ref}) == {str(val)!r}"
            self._produce_guard_code(guard, [code])
            return

        # Special case for nan because float("nan") == float("nan") evaluates to False
        if istype(val, float) and math.isnan(val):
            code = list()
            code.append(f"___check_type_id({ref}, {self.id_ref(t)})")
            code.append(f"__math_isnan({ref})")
            self._produce_guard_code(guard, code)
            return

        # Add type check to prevent equality check between tensor and non-tensor.
        code = list()
        if istype(val, (list, tuple)):
            self.LIST_LENGTH(guard)

            for idx, elem in enumerate(val):
                code.append(
                    f"___check_type_id({ref}[{idx}], {self.id_ref(type(elem))})"
                )

        elif not istype(val, torch.Size):
            code.append(f"___check_type_id({ref}, {self.id_ref(t)})")

        if istype(val, torch.Size):
            val = tuple(val)

        code.append(f"{ref} == {val!r}")
        self._produce_guard_code(guard, code)

    def CONSTANT_MATCH(self, guard: Guard):
        val = self.get(guard.name)
        if istype(val, (bool, type(None))):
            self.ID_MATCH(guard)
        else:
            self.EQUALS_MATCH(guard)

    def NN_MODULE(self, guard: Guard):
        self.ID_MATCH(guard)
        ref = self.arg_ref(guard)
        val = self.get(guard.name)

        def setup_guard():
            assert istype(val.training, bool)
            self.code.append(f"{ref}.training == {val.training}")

        if hasattr(val, "training"):
            # There are cases where a monkeypatched object has a guard made between __new__ and __init__
            setup_guard()
        else:
            unimplemented(f"Guard setup for uninitialized class {type(val)}")

    def FUNCTION_MATCH(self, guard: Guard):
        """things like torch.add and user defined functions"""
        if guard.is_local():
            return self.ID_MATCH(guard)

    def BUILTIN_MATCH(self, guard: Guard):
        return self.FUNCTION_MATCH(guard)

    def PYMODULE_MATCH(self, guard: Guard):
        return self.FUNCTION_MATCH(guard)

    def LIST_LENGTH(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        code = list()
        code.append(f"___check_type_id({ref}, {self.id_ref(t)})")
        code.append(f"len({ref}) == {len(value)}")

        self._produce_guard_code(guard, code)

    def TUPLE_ITERATOR_LEN(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        code = list()
        code.append(f"___check_type_id({ref}, {self.id_ref(t)})")
        code.append(f"___tuple_iterator_len({ref}) == {tuple_iterator_len(value)}")

        self._produce_guard_code(guard, code)

    def DICT_KEYS(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        code = list()
        code.append(f"___check_type_id({ref}, {self.id_ref(t)})")
        param_key_ids = set(dict_param_key_ids(value))
        const_keys = set(dict_const_keys(value))
        if param_key_ids:
            code.append(f"___dict_param_key_ids({ref}) == {param_key_ids!r}")
            code.append(f"___dict_const_keys({ref}) == {const_keys!r}")
        else:
            code.append(f"set({ref}.keys()) == {const_keys!r}")

        self._produce_guard_code(guard, code)

    def WEAKREF_ALIVE(self, guard):
        self._produce_guard_code(guard, [f"{self.arg_ref(guard)} is not None"])

    def NN_MODULE_PARAM_NAMES(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)
        keys = {k for k, v in value.named_parameters()}

        code = list()
        code.append(f"___check_type_id({ref}, {self.id_ref(t)})")
        code.append(f"{{k for k, v in {ref}.named_parameters()}} == {keys!r}")

        self._produce_guard_code(guard, code)

    def ODICT_KEYS(self, guard):
        """OrderedDict keys match"""
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        code = list()
        code.append(f"___check_type_id({ref}, {self.id_ref(t)})")
        code.append(f"str({ref}.keys()) == {str(value.keys())!r}")

        self._produce_guard_code(guard, code)

    def OBJECT_MUTATION(self, guard: Guard):
        mutation_guard.watch(self.get(guard.name), self.guarded_code)

    def GRAD_MODE(self, guard: Guard):
        """Guard on the initial grad state"""
        assert guard.name == ""
        assert guard.source is GuardSource.GLOBAL
        code = None
        if convert_frame.initial_grad_state:
            code = "___is_grad_enabled()"
        else:
            code = "not ___is_grad_enabled()"
        self._produce_guard_code(guard, [code])

    # This is a bit of a crutch for export case for symbolic shape guards.
    # SYMBOL_MATCH is only ever, and must only ever, be used for setting this value on
    # the create_fn field for tracking guards in export.
    @staticmethod
    def SYMBOL_MATCH():
        pass

    def TENSOR_MATCH(self, guard: Guard):
        if guard.is_nn_module():
            self.ID_MATCH(guard)
        else:
            value = self.get(guard.name)
            tensor_name = self.arg_ref(guard)
            self.tensor_check_names.append(tensor_name)
            self.tensor_check_examples.append(value)

            # STOP - DO NOT USE id_ref FOR TENSORS - TENSOR INVALIDATION RULES DIFFER
            self.tensor_check_ids[tensor_name] = id(value)

            # Note: Guard code produced for tensor_match is a little different.
            # We accumulate tensor names, then do a single install of `___check_tensors`.
            # See _guards.cpp and TensorGuard for more information.
            # TODO(voz): Add tensor matching code to export
            # Note: this is a bit of a special case, and so does not use _produce_guard_code
            guard.set_export_info(
                "TENSOR_MATCH",
                weakref.ref(type(value)),
                None,
                weakref.ref(value),
            )

    # A util that appends guarded code, or, in the case of export, adds data onto guards
    def _produce_guard_code(self, guard, code_list, provided_guarded_object=None):
        caller = currentframe().f_back
        func_name = getframeinfo(caller)[2]
        # We use func_name for export, so might as well get a nice defensive check out of it
        assert func_name in dir(
            self.__class__
        ), f"_produce_guard_code must be called from inside GuardedCode. Called from {func_name}"

        self.code.extend(code_list)

        # Not all guards have names, some can be installed globally (see asserts on HAS_GRAD)
        if provided_guarded_object is None:
            name_valid = guard.name is not None and guard.name != ""

            guarded_object = self.get(guard.name) if name_valid else None
        else:
            guarded_object = provided_guarded_object

        guarded_object_type = (
            weakref.ref(type(guarded_object)) if guarded_object is not None else None
        )
        obj_ref = None
        if hasattr(guarded_object.__class__, "__weakref__"):
            obj_ref = weakref.ref(guarded_object)

        guard.set_export_info(
            func_name,
            guarded_object_type,
            code_list,
            obj_ref,
        )


@dataclasses.dataclass
class GuardedCode:
    code: types.CodeType
    check_fn: Callable


from sympy.printing.str import StrPrinter


@dataclasses.dataclass
class TensorReference(object):
    """
    TensorReference objects are entirely optional. They are created to give us hints
    into where the symbolic shape came from.

    ref_id: The id of the tensor
    kind: A string tracking where in the tensor this value came from ("size","stride", etc)
    idx: An index in the structure

    NOTE - A symbolic shape coming from tensor at id 12345's shape dim 2, would be
    TensorReference(ref_id=12345, kind="size", idx=2)
    """

    ref_id: Optional[int] = None
    kind: Optional[str] = None
    idx: Optional[int] = None
    # Note - this is untyped because of TypeError: '_SpecialForm' object does not support item assignment
    # But it is a Optional[Union["sympy.Expr", int]]
    expr: Optional[object] = None  # Populated after association

    def __hash__(self):
        return hash((self.ref_id, self.kind, self.idx))


class DynamoGuardPrinter(StrPrinter):
    @staticmethod
    def tensor_ref_as_str(tensor_ref, id_to_name_map):
        if tensor_ref.kind in ("size", "stride"):
            return f"{id_to_name_map[tensor_ref.ref_id]}.{tensor_ref.kind}()[{tensor_ref.idx}]"
        return f"{id_to_name_map[tensor_ref.ref_id]}.{tensor_ref.kind}()"

    def __init__(
        self, expr_to_tensor_ref, id_to_name_map, shape_env, intermediary_symbols
    ):
        super().__init__()
        self.expr_to_tensor_ref = expr_to_tensor_ref
        self.id_to_name_map = id_to_name_map
        self.shape_env = shape_env
        self.intermediary_symbols = intermediary_symbols

    def _print_Symbol(self, expr) -> str:
        assert isinstance(expr, sympy.Symbol)
        if expr == 0:
            return "0"
        if expr == 1:
            return "1"
        expr_found = expr in (self.expr_to_tensor_ref) or (
            expr in self.intermediary_symbols
        )
        if not expr_found:
            if config.dynamic_shapes_ignore_assert:
                return f"{self.shape_env.var_to_val[expr]}"
        assert expr_found, f"Failed to find {expr}"
        refs = self.expr_to_tensor_ref[expr]
        if len(refs) == 0:
            return super()._print_Symbol(expr)
        tensor_ref = next(
            iter(refs)
        )  # Any is fine here, because we install equality guards later
        return DynamoGuardPrinter.tensor_ref_as_str(tensor_ref, self.id_to_name_map)


# NB: Naively, you'd expect this to only be a function that produces
# the callable that consistutes the guard.  However, there is some
# delicate handling for invalidating this check function when the
# locals/globals get invalidated, so there's some extra state
# we have to hold in this manager class.
#
# TODO: this object has reference cycle with itself, via check_fn which
# references back to CheckFunction via ___guarded_code in closure_vars.
# Ideally, there shouldn't be any ref cycle so that guards are
# promptly disposed of.
class CheckFunctionManager:
    def __init__(
        self,
        output_graph=None,
        guards: Optional[Set[Guard]] = None,
        f_locals: Optional[Dict] = None,
        f_globals: Optional[Dict] = None,
    ):
        self.valid = True
        self._weakrefs = []
        self._seen_ids = set()
        self.output_graph = output_graph

        # Note: right overrides left
        def combine_scopes(left, right):
            if left is None:
                return right

            if right is None:
                return left

            return {**left, **right}

        local_builder = GuardBuilder(
            self.id_ref, combine_scopes(f_globals, f_locals), self, renames=True
        )
        global_builder = GuardBuilder(self.id_ref, f_globals, self, renames=False)
        for guard in sorted(guards or [], key=Guard.sort_key):
            if not config.guard_nn_modules and guard.is_nn_module():
                continue
            guard.create(local_builder, global_builder)
        self.check_fn = self.compile_check_fn(local_builder, global_builder, guards)
        self._seen_ids.clear()

    """
    This is a complex bit of logic. The outline here is brief. For a line by line breakdown, see
    the code comments below.

    The role of this function is to take the current state of symbolic shape guards, tensor ids in the
    CURRENT dynamo frame, and tensor names (dynamo's frame agnostic tensor reference mechanism, see TensorCheck and
    guards.cpp for more info) - and produce executable python expressions for addition to our guarded code components
    that make their way into check_fn.

    We DO NOT create guards based on ids. The IDs act as a lookup for the following mapping:

    dynamo: tensor_name <> tensor_id
    shape_env: tensor_id <> shape_expr

    This allows us to then create a tensor_name <> shape_expr association for the current frames guards.
    """

    def _parse_symbolic_shape_expressions(self, tensor_check_names, tensor_check_ids):
        # Pre join output
        finished_expressions = []

        # A mapping of tensor_ids to tensor names
        id_to_name_map = {}

        # We should not have a shape env, or guards if we are not in config.dynamic shapes
        # But check it anyway.
        if not config.dynamic_shapes:
            return None

        expr_to_tensor_ref = {}
        guard_printer = DynamoGuardPrinter(
            expr_to_tensor_ref,
            id_to_name_map,
            self.output_graph.shape_env,
            self.output_graph.intermediary_symbols,
        )

        # tensor_check_names is the primary tensor association mechanism in dynamo.
        # All other guards installations are driven off of it, so these ones will too.
        for name in tensor_check_names:
            tensor_id = tensor_check_ids[name]
            id_to_name_map[tensor_id] = name

            if tensor_id in self.output_graph.tensor_id_to_sym_shape_ref:
                # If we made it here, this tensor_id is relevant to dynamo guard installation
                # AND was found in the shape_env
                tensor_ref_set = self.output_graph.tensor_id_to_sym_shape_ref[tensor_id]
                for tensor_ref in tensor_ref_set:
                    obj_expr = tensor_ref.expr
                    if obj_expr not in expr_to_tensor_ref:
                        expr_to_tensor_ref[obj_expr] = {}
                    expr_to_tensor_ref[obj_expr][tensor_ref] = ""

        guard_expression = self.output_graph.shape_env.get_guard_expr()
        expr_as_str = guard_printer.doprint(guard_expression)
        # We may get into a state where symbolic shape keys (all should be found in replacements)
        # Have not been removed from the expression. This is a serious enough error state that we need to assert.
        for key in self.output_graph.shape_env.var_to_val.keys():
            assert str(key) not in expr_as_str, f"Unknown shape symbol {key}. "
        finished_expressions.append(expr_as_str)

        for expr in expr_to_tensor_ref.keys():
            tensor_refs = expr_to_tensor_ref[expr].keys()
            equality_candidates = [
                DynamoGuardPrinter.tensor_ref_as_str(x, id_to_name_map)
                for x in tensor_refs
            ]

            if len(equality_candidates) > 1:
                equality_expr = " == ".join(equality_candidates)
                finished_expressions.append(equality_expr)

        # Redundant with code_parts, but allows us to wrap it with parens nicely.
        if len(finished_expressions) == 0:
            return None

        expression = " and ".join(finished_expressions)
        return f"({expression})"

    def compile_check_fn(self, local_builder, global_builder, guards_out):
        assert not (set(local_builder.argnames) & set(global_builder.argnames))
        # see parallel handling of ".0" / "___implicit0" in _eval_frame.c
        args = [a for a in local_builder.scope.keys() if a == "___implicit0"]
        args += [a for a in local_builder.argnames if a != "___implicit0"]
        args += ["**___kwargs_ignored"]
        args = ",".join(args)

        code_parts = (
            ["___guarded_code.valid"] + local_builder.code + global_builder.code
        )
        # TODO(whc) maybe only the 'check_tensors' one is ambiguous? if so we can be less general..
        verbose_code_parts = (
            ["___guarded_code.valid"] + local_builder.code + global_builder.code
        )

        tensor_check_names = (
            local_builder.tensor_check_names + global_builder.tensor_check_names
        )

        tensor_check_ids = local_builder.tensor_check_ids.copy()
        tensor_check_ids.update(global_builder.tensor_check_ids)

        check_tensors_fn = None
        check_tensors_verbose_fn = None
        if tensor_check_names:
            symbolic_shape_expression = self._parse_symbolic_shape_expressions(
                tensor_check_names, tensor_check_ids
            )
            tensor_check_examples = (
                local_builder.tensor_check_examples
                + global_builder.tensor_check_examples
            )
            tensor_guards = TensorGuards(
                *tensor_check_examples, dynamic_shapes=config.dynamic_shapes
            )
            check_tensors_fn = tensor_guards.check
            check_tensors_verbose_fn = tensor_guards.check_verbose
            code_parts.append(f"___check_tensors({', '.join(tensor_check_names)})")
            verbose_args = ", ".join(
                tensor_check_names + ["tensor_check_names=tensor_check_names"]
            )
            verbose_code_parts.append(f"___check_tensors_verbose({verbose_args})")
            if symbolic_shape_expression:
                code_parts.append(symbolic_shape_expression)
                verbose_code_parts.append(symbolic_shape_expression)
                guards_out.add(
                    Guard(
                        name="symbolic_shape_expression",
                        source=None,
                        create_fn=GuardBuilder.SYMBOL_MATCH,
                        code_list=symbolic_shape_expression,
                    )
                )

        def direct_equality(a, b):
            return a == b

        def direct_negation(a, b):
            return not direct_equality(a, b)

        code = " and ".join(unique(code_parts))
        closure_vars = collections.OrderedDict(
            [
                ("___guarded_code", self),
                ("___check_tensors", check_tensors_fn),
                ("___check_tensors_verbose", check_tensors_verbose_fn),
                ("tensor_check_names", tensor_check_names),
                ("floor", math.floor),
                ("ceiling", math.ceil),
                ("Eq", direct_equality),
                ("Ne", direct_negation),
                ("Mod", sympy.Mod),
                ("FloorDiv", FloorDiv),
            ]
        )
        closure_vars.update(CLOSURE_VARS)
        py_code = f"""\
def ___make_guard_fn({','.join(closure_vars.keys())}):
    return (lambda {args}: {code})
"""
        if os.environ.get("TORCHDYNAMO_PRINT_GUARDS", None) == "1":
            print("GUARDS", code)
        set_guard_fail_hook(guard_fail_hook)
        out = dict()
        try:
            exec(py_code, global_builder.scope, out)
        except:
            logging.error(f"Code that failed to compile:\n{py_code}")
            raise
        guard_fn = out["___make_guard_fn"](*closure_vars.values())
        guard_fn.closure_vars = closure_vars
        # TODO(whc) maybe '.code_parts' was only kept around for the guard callback? so we don't need both
        guard_fn.code_parts = code_parts
        guard_fn.verbose_code_parts = verbose_code_parts
        guard_fn.global_scope = global_builder.scope
        return guard_fn

    def invalidate(self, ref):
        # A weakref is no longer valid, self.check_fn should return false
        self.valid = False

    def id_ref(self, obj):
        """add a weakref, return the id"""
        try:
            if id(obj) not in self._seen_ids:
                self._weakrefs.append(weakref.ref(obj, self.invalidate))
                self._seen_ids.add(id(obj))
        except TypeError:
            pass  # cannot weakref bool object
        return id(obj)


def guard_fail_hook(
    guard_fn: Callable, code: types.CodeType, f_locals: Dict[str, Any], last: bool
):
    """
    called whenever a guard fails.
    """
    if not last:
        return
    scope = {rename_implicit(k): v for k, v in f_locals.items()}
    scope.update(guard_fn.closure_vars)
    reasons = []
    for part in guard_fn.verbose_code_parts:
        fail_reason = eval(part, guard_fn.global_scope, scope)
        # TODO(whc) hacky for now as not every 'part' in guard_fn.verbose_code_parts
        # is updated to return a string explaining the failure.
        if isinstance(fail_reason, str):
            reasons.append(fail_reason)
            break
        elif isinstance(fail_reason, bool) and not fail_reason:
            reasons.append(part)
            break
    guard_failures[orig_code_map[code]].append(reasons)


def guard_error_hook(
    guard_fn: Callable, code: types.CodeType, f_locals: Dict[str, Any], last: bool
):
    print(
        f"ERROR RUNNING GUARDS {code.co_name} {code.co_filename}:{code.co_firstlineno}"
    )
    print(" ", " and\n  ".join(guard_fn.code_parts))


set_guard_error_hook(guard_error_hook)


def unique(seq):
    seen = set()
    for x in seq:
        if x not in seen:
            yield x
            seen.add(x)
