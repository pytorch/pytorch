import builtins
import collections
import logging
import math
import os
import re
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from weakref import ReferenceType

import torch

from torch._guards import (
    DuplicateInputs,
    Guard,
    GuardBuilderBase,
    GuardEnvExpr,
    GuardSource,
    Source,
)
from torch.fx.experimental.symbolic_shapes import SYMPY_INTERP

from . import config, convert_frame, mutation_guard
from .eval_frame import set_guard_error_hook, set_guard_fail_hook
from .exc import unimplemented
from .types import GuardedCode, GuardFail, GuardFn  # noqa: F401
from .utils import (
    dict_const_keys,
    dict_const_keys_repr,
    dict_param_key_ids,
    guard_failures,
    HAS_NUMPY,
    istype,
    np,
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


class GuardBuilder(GuardBuilderBase):
    def __init__(
        self,
        id_ref: Callable[[Type[object]], str],
        source_ref: Callable[[Source], str],
        scope: Optional[Dict[str, object]],
        check_fn_manager: "CheckFunctionManager",
        renames=True,
    ):
        self.id_ref = id_ref
        self.source_ref = source_ref
        if scope:
            if renames:
                scope = {rename_implicit(k): v for k, v in scope.items()}
        else:
            scope = dict()
        self.scope: Dict[str, object] = scope
        self.scope["__builtins__"] = builtins.__dict__.copy()
        for (
            name,
            package_module,
        ) in torch.package.package_importer._package_imported_modules.items():
            name = name.replace(">", "_").replace("<", "_").replace(".", "_dot_")
            # Write the package module into the scope so that we can import it
            self.scope["__builtins__"][name] = package_module  # type: ignore[index]
            # Write the demangled name to the scope so that we can use it
            self.scope[name] = package_module

        self.argnames: List[str] = []
        # Code is python expression strings generated for each guard
        self.code: List[str] = []
        # shape_env_code is only used by local_builder and is used for
        # shape env code.  This exists only because we need to make sure
        # shape env guards get run after tensor match guards (since the
        # tensor match guards make sure we actually have tensors)
        self.shape_env_code: List[str] = []

        # Most of the time, we generate Python code in a guard to directly
        # check various properties.  However, tensors are a bit special;
        # it is too slow to check their properties one-by-one in Python.
        # Instead, there is a C++ function TensorGuards.check which takes
        # all of the tensor arguments and checks them all against compile-time
        # examples entirely in C++.  Thus, every time we process a
        # TENSOR_MATCH guard, we just add another entry to
        # tensor_check_names/tensor_check_examples, saying "for this local,
        # check it against this example", and it all ends up getting
        # swept up into a single call to ___check_tensors.  Invariant:
        # len(tensor_check_names) == len(tensor_check_examples).
        self.tensor_check_names: List[str] = []
        self.tensor_check_examples: List[torch.Tensor] = []

        self.tensor_check_ids: Dict[str, int] = {}
        self.check_fn_manager: CheckFunctionManager = check_fn_manager

    # Warning: use this with care!  This lets you access what the current
    # value of the value you are guarding on is.  You probably don't want
    # to actually durably save this value though (because it's specific
    # to this frame!)  Instead, you should be reading out some property
    # (like its type) which is what you permanently install into the
    # guard code.
    def get(self, name: str) -> Any:
        return eval(name, self.scope, CLOSURE_VARS)

    # Registers the usage of the source name referenced by the
    # string (or stored in the Guard) as being guarded upon.  It's important
    # to call this before generating some code that makes use of 'guard',
    # because without this call, we won't actually bind the variable
    # you reference in the actual guard closure (oops!)
    def arg_ref(self, guard: Union[str, Guard]) -> str:
        name: str
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
            return self.TYPE_MATCH(
                Guard(m.group(1), guard.source, GuardBuilder.TYPE_MATCH)
            )

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
        np_types = (
            (
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.float16,
                np.float32,
                np.float64,
            )
            if HAS_NUMPY
            else ()
        )
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
            )
            + np_types,
        ), t.__name__

        if istype(val, (torch.device, torch.dtype)):
            # TODO(jansel): is this slow? perhaps optimize it
            code = [f"str({ref}) == {str(val)!r}"]
            self._produce_guard_code(guard, code)
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
        const_keys_repr = dict_const_keys_repr(const_keys)
        if param_key_ids:
            code.append(f"___dict_param_key_ids({ref}) == {param_key_ids!r}")
            code.append(f"___dict_const_keys({ref}) == {const_keys_repr}")
        else:
            code.append(f"set({ref}.keys()) == {const_keys_repr}")

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
        mutation_guard.watch(self.get(guard.name), self.check_fn_manager)

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

    def SHAPE_ENV(self, guard: Guard):
        # Let's handle ShapeEnv guards.  To do this, we will resolve
        # shape variables to sources from tracked_fakes.  This must happen after
        # tensor checks.
        assert guard.name == ""
        output_graph = self.check_fn_manager.output_graph
        # NB: self.output_graph can be None in the debug_nops tests
        fs = output_graph.tracked_fakes
        guards = output_graph.shape_env.produce_guards(
            [a.fake for a in fs],
            [a.source for a in fs],
            source_ref=self.source_ref,
        )
        for shape_guard in guards:
            self._produce_guard_code(guard, [shape_guard], shape_env=True)

    def TENSOR_MATCH(self, guard: Guard):
        if guard.is_nn_module():
            self.ID_MATCH(guard)
        else:
            value = self.get(guard.name)
            assert isinstance(value, torch.Tensor)
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
    def _produce_guard_code(
        self, guard, code_list, provided_guarded_object=None, shape_env=False
    ):
        # WARNING: It is important that cur_frame/caller do NOT stay in
        # the current frame, because they will keep things live longer
        # than they should.  See TestMisc.test_release_module_memory
        cur_frame = currentframe()
        assert cur_frame is not None
        caller = cur_frame.f_back
        del cur_frame
        assert caller is not None
        func_name = getframeinfo(caller)[2]
        del caller
        # We use func_name for export, so might as well get a nice defensive check out of it
        assert func_name in dir(
            self.__class__
        ), f"_produce_guard_code must be called from inside GuardedCode. Called from {func_name}"

        if shape_env:
            self.shape_env_code.extend(code_list)
        else:
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
        f_locals: Optional[Dict[str, object]] = None,
        f_globals: Optional[Dict[str, object]] = None,
        guard_fail_fn: Optional[Callable[[Tuple[str, str]], None]] = None,
    ):
        guards = output_graph.guards if output_graph else None
        self.valid = True
        self._weakrefs: List["ReferenceType[object]"] = []
        self._seen_ids: Set[int] = set()
        self.output_graph = output_graph

        # Note: right overrides left
        def combine_scopes(left, right):
            if left is None:
                return right

            if right is None:
                return left

            return {**left, **right}

        def source_ref(source):
            guard_source = source.guard_source()
            if guard_source is GuardSource.CONSTANT:
                # No need to track constants
                return source.name()
            builder = guard_source.select(w_local(), w_global())
            assert builder is not None
            return builder.arg_ref(source.name())

        local_builder = GuardBuilder(
            self.id_ref,
            source_ref,
            combine_scopes(f_globals, f_locals),
            self,
            renames=True,
        )
        global_builder = GuardBuilder(
            self.id_ref, source_ref, f_globals, self, renames=False
        )
        # source_ref can cause a cycle, make sure we break it with weakref
        w_local = weakref.ref(local_builder)
        w_global = weakref.ref(global_builder)
        for guard in sorted(guards or [], key=Guard.sort_key):
            if (
                not config.guard_nn_modules
                and guard.is_nn_module()
                # Default func args must be guarded on.
                # TODO: we could make use of 'DefaultsSource' and offer a .guard.is_defaults() API
                and "__defaults__" not in guard.name
                and "__kwdefaults__" not in guard.name
            ):
                continue
            guard.create(local_builder, global_builder)
        self.check_fn = self.compile_check_fn(
            local_builder, global_builder, guards, guard_fail_fn
        )
        self._seen_ids.clear()

    def compile_check_fn(
        self, local_builder, global_builder, guards_out, guard_fail_fn
    ):
        assert not (set(local_builder.argnames) & set(global_builder.argnames))
        # see parallel handling of ".0" / "___implicit0" in _eval_frame.c
        largs = [a for a in local_builder.scope.keys() if a == "___implicit0"]
        largs += [a for a in local_builder.argnames if a != "___implicit0"]
        largs += ["**___kwargs_ignored"]
        args = ",".join(largs)

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

        aotautograd_guards: List[GuardEnvExpr] = (
            self.output_graph.tracing_context.guards_context.aotautograd_guards
            if self.output_graph
            else []
        )
        for guard in aotautograd_guards:
            if isinstance(guard, DuplicateInputs):
                pos_a = self.output_graph.pos_to_arg[guard.input_pos_a]
                pos_b = self.output_graph.pos_to_arg[guard.input_pos_b]
                assert (
                    pos_b >= 0 and pos_a >= 0
                ), "Deduped args out of bounds, cannot be negative"

                assert self.output_graph.graphargs[
                    pos_a
                ].is_tensor, "Deduped arg must be a tensor"
                assert self.output_graph.graphargs[
                    pos_b
                ].is_tensor, "Deduped arg must be a tensor"
                code_part = f"{self.output_graph.graphargs[pos_a].source.name()} is {self.output_graph.graphargs[pos_b].source.name()}"  # noqa: B950
                code_parts.append(code_part)
                verbose_code_parts.append(code_part)
            else:
                raise RuntimeError(f"Unknown GuardEnvExpr: {guard}")

        code_parts.extend(local_builder.shape_env_code)
        verbose_code_parts.extend(local_builder.shape_env_code)
        assert not global_builder.shape_env_code

        code = " and ".join(unique(code_parts))
        closure_vars = collections.OrderedDict(
            [
                ("___guarded_code", self),
                ("___check_tensors", check_tensors_fn),
                ("___check_tensors_verbose", check_tensors_verbose_fn),
                ("tensor_check_names", tensor_check_names),
            ]
            + list(SYMPY_INTERP.items())
        )
        closure_vars.update(CLOSURE_VARS)
        py_code = f"""\
def ___make_guard_fn({','.join(closure_vars.keys())}):
    return lambda {args}: {code}
"""
        if os.environ.get("TORCHDYNAMO_PRINT_GUARDS", None) == "1":
            print("GUARDS", code)
        set_guard_fail_hook(guard_fail_hook)
        out: Dict[str, Any] = dict()
        # print("RUNNING PY CODE", py_code)
        exec(py_code, global_builder.scope, out)
        guard_fn = out["___make_guard_fn"](*closure_vars.values())
        guard_fn.closure_vars = closure_vars
        # TODO(whc) maybe '.code_parts' was only kept around for the guard callback? so we don't need both
        guard_fn.args = largs
        guard_fn.code_parts = code_parts
        guard_fn.verbose_code_parts = verbose_code_parts
        guard_fn.global_scope = global_builder.scope
        guard_fn.guard_fail_fn = guard_fail_fn
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
    guard_fn: GuardFn, code: types.CodeType, f_locals: Dict[str, object], last: bool
) -> None:
    """
    called whenever a guard fails.
    """
    if not guard_fn.guard_fail_fn and not last:
        return
    scope = {rename_implicit(k): v for k, v in f_locals.items()}
    scope.update(guard_fn.closure_vars)
    reason = None
    for part in guard_fn.verbose_code_parts:
        fail_reason = eval(part, guard_fn.global_scope, scope)
        # TODO(whc) hacky for now as not every 'part' in guard_fn.verbose_code_parts
        # is updated to return a string explaining the failure.
        if isinstance(fail_reason, str):
            reason = fail_reason
            break
        elif isinstance(fail_reason, bool) and not fail_reason:
            reason = part
            break
    try:
        if guard_fn.guard_fail_fn is not None:
            guard_fn.guard_fail_fn(
                GuardFail(reason or "unknown reason", orig_code_map[code])
            )
    except Exception as e:
        log.error(
            "Failure in guard_fail_fn callback - raising here will cause a NULL Error on guard eval",
            exc_info=True,
        )

    if last:
        guard_failures[orig_code_map[code]].append(reason)


def guard_error_hook(
    guard_fn: GuardFn, code: types.CodeType, f_locals: Dict[str, object], last: bool
):
    print(
        f"ERROR RUNNING GUARDS {code.co_name} {code.co_filename}:{code.co_firstlineno}"
    )
    # TODO: If we passed in the exception here, we could get a precise
    # column number of which subexpression failed.  But that would also
    # require us to have the TRUE code that was eval'ed, not a shoddy
    # reconstruction (like is done here)
    print("lambda " + ", ".join(guard_fn.args) + ":")
    print(" ", " and\n  ".join(guard_fn.code_parts))


set_guard_error_hook(guard_error_hook)


def unique(seq):
    seen = set()
    for x in seq:
        if x not in seen:
            yield x
            seen.add(x)
