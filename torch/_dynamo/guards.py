import ast
import builtins
import collections
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import logging
import math
import os
import re
import sys
import types
import weakref
from inspect import currentframe, getframeinfo
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from weakref import ReferenceType

import torch
import torch.utils._device
from torch._dynamo.source import (
    is_from_local_source,
    TensorProperty,
    TensorPropertySource,
)

from torch._guards import (
    DuplicateInputs,
    Guard,
    GuardBuilderBase,
    GuardEnvExpr,
    GuardSource,
    Source,
)
from torch.fx.experimental.symbolic_shapes import (
    EqualityConstraint,
    is_concrete_int,
    SYMPY_INTERP,
)

from torch.utils._traceback import format_frame, report_compile_source_on_error
from torch.utils.weak import TensorWeakRef, WeakIdRef

from . import config, convert_frame, mutation_guard
from .eval_frame import set_guard_error_hook, set_guard_fail_hook
from .exc import unimplemented
from .source import LocalSource, TypeSource
from .types import GuardedCode, GuardFail, GuardFn  # noqa: F401
from .utils import (
    dict_const_keys,
    dict_const_keys_repr,
    dict_param_key_ids,
    guard_failures,
    is_guard_failure_reporting_enabled,
    istype,
    np,
    orig_code_map,
    tensor_always_has_static_shape,
    tuple_iterator_getitem,
    tuple_iterator_len,
)

log = logging.getLogger(__name__)
guards_log = torch._logging.getArtifactLogger(__name__, "guards")
verbose_guards_log = torch._logging.getArtifactLogger(__name__, "verbose_guards")

TensorGuards = torch._C._dynamo.guards.TensorGuards
check_obj_id = torch._C._dynamo.guards.check_obj_id
check_type_id = torch._C._dynamo.guards.check_type_id


# For user stack printing
@functools.lru_cache(None)
def uninteresting_files():
    import torch._dynamo.external_utils

    mods = [
        torch._dynamo.external_utils,
    ]
    return {inspect.getfile(m) for m in mods}


CLOSURE_VARS = collections.OrderedDict(
    [
        ("___check_type_id", check_type_id),
        ("___check_obj_id", check_obj_id),
        ("___is_grad_enabled", torch.is_grad_enabled),
        (
            "___are_deterministic_algorithms_enabled",
            torch.are_deterministic_algorithms_enabled,
        ),
        ("___is_torch_function_enabled", torch._C._is_torch_function_enabled),
        ("___odict_getitem", collections.OrderedDict.__getitem__),
        ("___dict_param_key_ids", dict_param_key_ids),
        ("___dict_const_keys", dict_const_keys),
        ("___tuple_iterator_len", tuple_iterator_len),
        ("___tuple_iterator_getitem", tuple_iterator_getitem),
        ("__math_isnan", math.isnan),
        ("inf", float("inf")),
        ("__load_module", lambda name: importlib.import_module(name)),
        ("utils_device", torch.utils._device),
        ("device", torch.device),
        ("__as_tensor", torch.as_tensor),
    ]
)

if sys.version_info[:2] <= (3, 8):
    # [Note: Python Version <= 3.8]
    # This branch should be dropped when we drop support for Python 3.8.
    # Reason: 'ast.unparse' function was introduced in Python 3.9.

    try:
        import astunparse  # type: ignore[import]

        def _ast_unparse(node: ast.AST) -> str:
            return astunparse.unparse(node).replace("\n", "")

        HAS_UNPARSE_FUNCTIONS = True
    except ImportError:
        HAS_UNPARSE_FUNCTIONS = False
        pass
else:
    HAS_UNPARSE_FUNCTIONS = True

    def _ast_unparse(node: ast.AST) -> str:
        return ast.unparse(node).replace("\n", "")


def strip_function_call(name):
    """
    "___odict_getitem(a, 1)" => "a"
    "a.layers[slice(2)][0]._xyz" ==> "a"
    "getattr(a.layers[slice(2)][0]._abc, '0')" ==> "a"
    "getattr(getattr(a.x[3], '0'), '3')" ==> "a"
    "a.layers[slice(None, -1, None)][0]._xyz" ==> "a"
    """
    # recursively find valid object name in fuction
    valid_name = re.compile("[A-Za-z_].*")
    curr = ""
    for char in name:
        if char in " (":
            curr = ""
        elif char in "),[]":
            if curr and curr != "None" and valid_name.match(curr):
                return strip_function_call(curr)
        else:
            curr += char

    return strip_getattr_getitem(name)


def strip_getattr_getitem(name):
    """
    "a[1]" => "a"
    "a.foo" => "a"
    """
    return re.split(r"[.\[]", name)[0]


# The ready to eval generated code (possibly multiple parts) for a guard, plus
# the original guard object that created it for provenance
@dataclasses.dataclass
class GuardCodeList:
    code_list: List[str]
    guard: Guard


class GuardBuilder(GuardBuilderBase):
    def __init__(
        self,
        id_ref: Callable[[Type[object]], str],
        source_ref: Callable[[Source], str],
        lookup_weakrefs: Callable[[Type[object]], WeakIdRef],
        user_scope: Optional[Dict[str, object]],
        check_fn_manager: "CheckFunctionManager",
        *,
        local: bool,
    ):
        self.local = local
        self.id_ref = id_ref
        self.source_ref = source_ref
        self.lookup_weakrefs = lookup_weakrefs
        if user_scope:
            scope = {"L" if local else "G": user_scope}
        else:
            scope = {"L" if local else "G": dict()}
        self.scope: Dict[str, Dict[str, object]] = scope
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
        self.code: List[GuardCodeList] = []
        # shape_env_code is only used by local_builder and is used for
        # shape env code.  This exists only because we need to make sure
        # shape env guards get run after tensor match guards (since the
        # tensor match guards make sure we actually have tensors)
        self.shape_env_code: List[GuardCodeList] = []

        # [Note - On Eager Tensor Guards]
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
        # TODO: something here
        self.tensor_check_names: List[str] = []
        self.tensor_check_examples: List[torch.Tensor] = []
        self.tensor_check_guards: List[Guard] = []

        self.check_fn_manager: CheckFunctionManager = check_fn_manager
        self.id_matched_objs: Dict[str, WeakIdRef] = {}

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
            if re.match(r"[a-zA-Z0-9_]+", base):
                if re.match(r"^\d+$", base):
                    log.warning("invalid var name: %s", guard)
                self.argnames.append(base)

        return name

    def TYPE_MATCH(self, guard: Guard):
        # ___check_type_id is same as `id(type(x)) == y`
        t = type(self.get(guard.name))
        obj_id = self.id_ref(t)
        code = f"___check_type_id({self.arg_ref(guard)}, {obj_id})"
        self._produce_guard_code(guard, [code])

    def BOOL_FALSE(self, guard: Guard):
        # Guard on the runtime value being 'False',
        # can be faster than seemingly equivalent checks like DICT_KEYS for empty dict
        #
        # WARNING: this guard is not safe to use generally.  It only works if the runtime
        # value is of a type that supports bool(), and some types e.g. Tensor do not.
        # Only use this guard in cases you can guarantee the runtime type will be friendly.
        # (e.g. Specialized NNModule with mutation protection via setattr)
        #
        # Why not simply check the runtime type inside this guard?  It's slow enough to defeat
        # the purpose of using this guard, which itself is supposed to be a faster alternative
        # to DICT_KEYS.
        ref = self.arg_ref(guard)
        code = f"not {ref}"
        self._produce_guard_code(guard, [code])

    def ID_MATCH(self, guard: Guard):
        # ___check_obj_id is same as `id(x) == y`
        if isinstance(guard.originating_source, TypeSource):
            # optional optimization to produce cleaner/faster guard code
            return self.TYPE_MATCH(
                Guard(
                    guard.originating_source.base, guard.source, GuardBuilder.TYPE_MATCH
                )
            )

        code = f"___check_obj_id({self.arg_ref(guard)}, {self.id_ref(self.get(guard.name))})"
        self._produce_guard_code(guard, [code])

    def NAME_MATCH(self, guard: Guard):
        obj = self.get(guard.name)
        code = f"{self.arg_ref(guard)}.__name__ == '{obj.__name__}'"
        self._produce_guard_code(guard, [code])

    def DATA_PTR_MATCH(self, guard: Guard):
        obj = self.get(guard.name)
        code = f"{self.arg_ref(guard)}.data_ptr() == {obj.data_ptr()}"
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
        ok_types = (
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
            *np_types,
        )
        if istype(val, dict):
            assert all(
                istype(x, ok_types) for x in itertools.chain(val.keys(), val.values())
            )
        else:
            assert istype(
                val,
                ok_types,
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

        code = list()

        # If matching equality against list/tuple, we must also check that
        # the internal types match.  (TODO: what about nested lists?)
        if istype(val, (list, tuple)):
            # NB: LIST_LENGTH takes care of the outer __check_type_id test
            self.LIST_LENGTH(guard)

            for idx, elem in enumerate(val):
                code.append(
                    f"___check_type_id({ref}[{idx}], {self.id_ref(type(elem))})"
                )
        else:
            # Add type check to prevent equality check between tensor and non-tensor.
            code.append(f"___check_type_id({ref}, {self.id_ref(t)})")

        if istype(val, torch.Size):
            val = tuple(val)

        # TODO: It feels like it would be better to just implement our own
        # equality test in C that handles all of the necessary type checking
        # and NaN tests
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

        # Keep track of ID_MATCH'd nn module objects. This will be used to
        # modify the cache size logic
        if self.local and isinstance(guard.originating_source, LocalSource):
            local_name = guard.originating_source.local_name
            if local_name in self.scope["L"]:
                weak_id = self.lookup_weakrefs(val)
                assert (
                    weak_id is not None
                ), "ID_MATCH is not called on the NN_MODULE guard"
                self.id_matched_objs[local_name] = weak_id

        def setup_guard():
            assert istype(val.training, bool)
            # TODO: Why doesn't this use produce_guard_code?
            self.code.append(
                GuardCodeList([f"{ref}.training == {val.training}"], guard)
            )

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

    # TODO(voz): Deduplicate w/ AOTAutograd dupe input guards
    def DUPLICATE_INPUT(self, guard, source_b):
        ref_a = self.arg_ref(guard)
        ref_b = self.arg_ref(source_b.name())

        code = [f"{ref_b} is {ref_a}"]
        self._produce_guard_code(guard, code)

    def DICT_KEYS(self, guard):
        ref = self.arg_ref(guard)
        value = self.get(guard.name)
        t = type(value)

        code = list()
        code.append(f"___check_type_id({ref}, {self.id_ref(t)})")
        param_key_ids = set(dict_param_key_ids(value))
        const_keys = set(dict_const_keys(value))
        const_keys_repr = dict_const_keys_repr(const_keys, local=self.local)
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

    def DETERMINISTIC_ALGORITHMS(self, guard: Guard):
        """Guard on the initial determinism algorithms state"""
        assert guard.source is GuardSource.GLOBAL
        code = None
        if convert_frame.initial_deterministic_algorithms_state:
            code = "___are_deterministic_algorithms_enabled()"
        else:
            code = "not ___are_deterministic_algorithms_enabled()"
        self._produce_guard_code(guard, [code])

    def TORCH_FUNCTION_STATE(self, guard: Guard):
        assert guard.source is GuardSource.GLOBAL
        code = None
        if convert_frame.initial_torch_function_state:
            code = "___is_torch_function_enabled()"
        else:
            code = "not ___is_torch_function_enabled()"
        self._produce_guard_code(guard, [code])

    def DEFAULT_DEVICE(self, guard: Guard):
        """Guard on CURRENT_DEVICE per torch.utils._device"""
        assert guard.source is GuardSource.GLOBAL
        import torch.utils._device as m

        self._produce_guard_code(
            guard, [f"utils_device.CURRENT_DEVICE == {m.CURRENT_DEVICE!r}"]
        )

    def SHAPE_ENV(self, guard: Guard):
        # Let's handle ShapeEnv guards.  To do this, we will resolve
        # shape variables to sources from tracked_fakes.  This must happen after
        # tensor checks.
        assert guard.name == ""
        output_graph = self.check_fn_manager.output_graph
        # NB: self.output_graph can be None in the debug_nops tests
        fs = output_graph.tracked_fakes
        constraint_inputs = [a.constraint_dims for a in fs]

        def get_sources(t_id, dim):
            # Looks up base sources mapped to a tensor id and uses them to create
            # sources for the corresponding tensor dimension.
            return [
                TensorPropertySource(source, TensorProperty.SIZE, dim)
                for source in output_graph.tracked_fakes_id_to_source[t_id]
            ]

        if output_graph.export_constraints:
            source_pairs: List[Tuple[Source, Source]] = []
            for constraint in output_graph.export_constraints:
                source, *other_sources = get_sources(constraint.t_id, constraint.dim)
                # When t.size()[dim] maps to src0, src1, ..., srcN, we add
                # constraints that make src0 "equal" to src1, ..., srcN.
                source_pairs.extend(
                    (source, other_source) for other_source in other_sources
                )
                if constraint.shared is not None:
                    # Moreover, when t.size()[dim] is specified equal to t'.size()[dim']
                    # and t'.size()[dim'] maps to src1', ..., srcN', we add
                    # constraints that also make src0 "equal" to src1', ..., srcN'.
                    other_sources = get_sources(
                        constraint.shared.t_id, constraint.shared.dim
                    )
                    source_pairs.extend(
                        (source, other_source) for other_source in other_sources
                    )
            equalities_inputs = EqualityConstraint(
                source_pairs=source_pairs,
                warn_only=False,
            )
        else:
            equalities_inputs = None
        guards = output_graph.shape_env.produce_guards(
            [a.fake for a in fs],
            [a.source for a in fs],
            constraint_inputs=constraint_inputs,
            equalities_inputs=equalities_inputs,
            source_ref=self.source_ref,
            # Export keeps static.
            ignore_static=(not self.check_fn_manager.output_graph.export),
        )
        output_graph.shape_env.freeze()
        for shape_guard in guards:
            self._produce_guard_code(guard, [shape_guard], shape_env=True)

    def TENSOR_MATCH(self, guard: Guard, value=None):
        if guard.is_nn_module():
            self.ID_MATCH(guard)
        else:
            if isinstance(value, TensorWeakRef):
                value = value()

            value = value if value is not None else self.get(guard.name)
            assert isinstance(value, torch.Tensor)

            tensor_name = self.arg_ref(guard)
            # [Note - On Export Tensor Guards]
            #
            # In eager mode, tensor guards are evaluated through C++, in guards.cpp
            # see [Note - On Eager Tensor Guards] for more info.
            #
            # In export mode, we instead maintain parallel logic between C++ and python
            # here, with an exception of checking the dispatch key - with the idea that a dispatch key
            # is an entirely runtime notion that would make no sense to keep in an exported graph.
            #
            # Now, this idea is okay, but to paraphrase @ezyang, this mental model is sufficient for now, although
            # not entirely true.
            # For example, suppose one of the input tensors had the negative dispatch key.
            # You should end up with a graph that is specialized for tensors that have a negative dispatch key.
            # If you allow a Tensor that does NOT have this bit set, you will accidentally run it "as if" it were negated.
            # Now, negative key only shows up for complex numbers, and most likely, the exported to target doesn't
            # support this feature at all, but the point stands that :some: tensor state only shows up on dispatch key.
            # TODO(voz): Either populate a dispatch_key check into the guards, or error on users passing in an unsupported
            # subset of keys during export.
            #
            # The list of tensor fields and calls we care about can be found in `terms` below.
            # TODO(voz): We are missing storage offset in all our tensor guards?
            code: List[str] = list()
            if self.check_fn_manager.output_graph.export:
                self.TYPE_MATCH(guard)
                terms = [
                    "dtype",
                    "device",
                    "requires_grad",
                    "ndimension()",
                ]

                for term in terms:
                    real_value = self.get(tensor_name + "." + term)
                    if istype(real_value, (torch.device, torch.dtype)):
                        # copy pasted from EQUALS_MATCH
                        code.append(f"str({tensor_name}.{term}) == {str(real_value)!r}")
                    else:
                        code.append(f"{tensor_name}.{term} == {real_value}")
            else:
                self.tensor_check_names.append(tensor_name)
                self.tensor_check_examples.append(value)
                self.tensor_check_guards.append(guard)

            # A frame is valid for reuse with dynamic dimensions if the new dynamic dimensions are a
            # strict subset of the old.
            #
            # The logic here is as follows:
            #
            # Every mark_dynamic directive is a user-knows-best command, which can incur a raise at tracing
            # time if we find guards that run counter to the user directive.
            # If compiling a frame with explicit dynamic dims X could cause an exception, we MUST NOT skip compiling.
            #
            # If the frame is compiled with any marked dynamic indices, let's call that set of indices X.
            # When we evaluated inputs against the guards, given the same tensor with potentially new dynamic indices,
            # let's call that set Y.
            #
            # When X is a strict subset of Y, the potential new raises introduced during compilation are a strict subset
            # of the raises we
            # could have encountered. The frame compiled under Y is safe to reuse with X.
            # When X is not a strict subset of Y, the non-overlapping new elements of X may cause new raises, and the
            # frame is no longer fit for reuse.
            #
            # This is the case because any newly introduced mark_dynamic directives have a chance of
            # raising, failing compilation. Any existing mark_dynamic indices that we lost are safe to lose
            # as all it means is that we have gotten rid of a user directive which could incur a raise at compile time.
            # In the case of when there is no Y, that is, there are no dynamic indices marked at all, the frame is safe
            # to reuse
            # as an empty set is a safe degeneration - that is, a strictly static tensor is always valid for a frame
            # compiled with that same
            # tensor + more onerous user directives.
            assert guard.source is not None
            static, reason = tensor_always_has_static_shape(
                value, is_tensor=True, guard_source=guard.source
            )
            if not static:
                if hasattr(value, "_dynamo_dynamic_indices"):
                    code.append(
                        f"(({tensor_name}._dynamo_dynamic_indices.issubset({value._dynamo_dynamic_indices})) if hasattr({tensor_name}, '_dynamo_dynamic_indices') else True)"  # noqa: B950
                    )
                # In the case of us not having any dynamic dimension indices, we compiled the frame with no chance of
                # raising for this specific tensor - and any inputs with more dynamic user directives specified must be recompiled.
                else:
                    code.append(
                        f"hasattr({tensor_name}, '_dynamo_dynamic_indices') == False"
                    )
            if len(code) > 0:
                self._produce_guard_code(guard, code)

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
            self.shape_env_code.append(GuardCodeList(code_list, guard))
        else:
            self.code.append(GuardCodeList(code_list, guard))

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
        # Not necessary to have weakref for Enum type, but there is a bug that
        # makes hasattr(guarded_object.__class__, "__weakref__") return True.
        if hasattr(guarded_object.__class__, "__weakref__") and not isinstance(
            guarded_object, enum.Enum
        ):
            obj_ref = weakref.ref(guarded_object)

        guard.set_export_info(
            func_name,
            guarded_object_type,
            code_list,
            obj_ref,
        )


# Common Sub-Expression Elimination for Python expressions.
#
# There are 2 steps to this pass:
#     1. Count the frequency of each sub-expression (i.e. inner
#        node in the AST tree)
#
#     2. Replace those that occur more than once by a fresh variable 'v'.
#        'v' will be defined in the 'preface' list (output argument to
#        'NodeTransformer')
#
# NB: the use of 'ast.unparse' while visiting the nodes makes this pass
# quadratic on the depth of the tree.
#
# NB: this pass creates a new variable for each AST node that is repeated
# more than 'USE_THRESHOLD'. e.g. if 'a.b.c.d' is used 10 times, 'a.b.c'
# and 'a.b' are also used 10 times. So, there will be a new variable for
# each of them.
class PyExprCSEPass:
    # Maximum number of times a given expression can be used without being
    # replaced by a fresh variable.
    USE_THRESHOLD = 1

    # Ad-Hoc: AST nodes this pass focuses on.
    ALLOWED_NODE_TYPES = (ast.Attribute, ast.Call, ast.Subscript)

    @dataclasses.dataclass
    class Config:
        expr_count: Dict[str, int]
        expr_to_name: Dict[str, str]

    class ExprCounter(ast.NodeVisitor):
        def __init__(self, config: "PyExprCSEPass.Config") -> None:
            self._config = config

        def visit(self, node: ast.AST) -> Any:
            if isinstance(node, PyExprCSEPass.ALLOWED_NODE_TYPES):
                self._config.expr_count[_ast_unparse(node)] += 1
            super().visit(node)

    class Replacer(ast.NodeTransformer):
        def __init__(
            self,
            config: "PyExprCSEPass.Config",
            gen_name: Callable[[], str],
        ) -> None:
            super().__init__()
            self._config = config
            self._gen_name = gen_name
            self.preface: List[str] = []

        def visit(self, node: ast.AST) -> Any:
            if isinstance(node, PyExprCSEPass.ALLOWED_NODE_TYPES):
                expr = _ast_unparse(node)

                # Replacement only occurs if a given expression is used more
                # than once.
                if self._config.expr_count[expr] > PyExprCSEPass.USE_THRESHOLD:
                    if expr not in self._config.expr_to_name:
                        # Parent 'visit' is called so that we CSE the inner expressions first.
                        #
                        # The resulting expression is used as right-hand-side of the variable
                        # assignment. i.e. we are CSE-ing the children before the parents.
                        #
                        # Indexing still uses the old 'node', since that's what was counted
                        # by the 'NodeVisitor'.
                        node_ = super().visit(node)
                        expr_ = _ast_unparse(node_)
                        var_name = self._gen_name()
                        self.preface.append(f"{var_name} = {expr_}")
                        self._config.expr_to_name[expr] = var_name
                    else:
                        var_name = self._config.expr_to_name[expr]
                    return ast.Name(var_name, ast.Load())

            return super().visit(node)

    def __init__(self) -> None:
        self._counter = 0
        self._config = self.Config(
            expr_count=collections.defaultdict(lambda: 0), expr_to_name={}
        )

    def _new_var(self, prefix: str = "_var") -> str:
        name = f"{prefix}{self._counter}"
        self._counter += 1
        return name

    def count(self, exprs: List[str]) -> None:
        counter = self.ExprCounter(self._config)
        for e in exprs:
            counter.visit(ast.parse(e))

    def replace(self, expr: str) -> Tuple[List[str], str]:
        replacer = self.Replacer(self._config, self._new_var)
        new_node = replacer.visit(ast.parse(expr))
        return replacer.preface, _ast_unparse(new_node)


# NB: Naively, you'd expect this to only be a function that produces
# the callable that constitutes the guard.  However, there is some
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
        guard_fail_fn: Optional[Callable[[Tuple[str, str]], None]] = None,
    ):
        guards = output_graph.guards if output_graph else None
        self.valid = True
        self._weakrefs: Dict[int, ReferenceType[object]] = {}
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
            self.lookup_weakrefs,
            combine_scopes(output_graph.global_scope, output_graph.local_scope),
            self,
            local=True,
        )
        global_builder = GuardBuilder(
            self.id_ref,
            source_ref,
            self.lookup_weakrefs,
            output_graph.global_scope,
            self,
            local=False,
        )

        # We need to transplant a copy here, because some guards
        # might get a cross ref between local and global, like L['mod_name'][G['some_key']]
        # the inverse is illegal.
        if "G" in global_builder.scope:
            local_builder.scope["G"] = global_builder.scope["G"]
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
                and (config.skip_nnmodule_hook_guards or "hooks" not in guard.name)
            ):
                continue
            guard.create(local_builder, global_builder)
        self.check_fn = self.compile_check_fn(
            local_builder, global_builder, guards, guard_fail_fn
        )
        self.check_fn.id_matched_objs = local_builder.id_matched_objs

    def compile_check_fn(
        self, local_builder, global_builder, guards_out, guard_fail_fn
    ):
        assert not (set(local_builder.argnames) & set(global_builder.argnames))
        # see parallel handling of ".0" / "___implicit0" in _eval_frame.c
        largs = local_builder.argnames
        largs += ["**___kwargs_ignored"]
        args = ",".join(largs)

        guards_log.debug("GUARDS:")

        # Don't report this guard, it's always the same, useless!
        code_parts = ["___guarded_code.valid"]
        base = os.path.dirname(__file__)

        def add_code_part(code, guard, log_only=False):
            if guards_log.isEnabledFor(logging.DEBUG):
                extra = ""
                if guard is not None:
                    if guard.user_stack:
                        for fs in reversed(guard.user_stack):
                            if fs.filename not in uninteresting_files():
                                break
                        else:
                            fs = guard.user_stack[-1]
                        extra = f"  # {format_frame(fs, line=True)}"
                    elif guard.stack:
                        extra = f"  # {format_frame(guard.stack.summary()[-1])}"

                guards_log.debug("%s", f"{code:<60}{extra}")

            if verbose_guards_log.isEnabledFor(logging.DEBUG):
                maybe_stack = ""
                maybe_user_stack = ""
                if guard is not None:
                    maybe_stack = f"\nStack:\n{''.join(guard.stack.format())}"
                    if guard.user_stack:
                        maybe_user_stack = (
                            f"\nUser stack:\n{''.join(guard.user_stack.format())}"
                        )
                verbose_guards_log.debug(
                    "Guard: %s%s%s",
                    code,
                    maybe_stack,
                    maybe_user_stack,
                )

            if not log_only:
                code_parts.append(code)

        # TODO: Maybe better not to repeatedly spam the same guard information
        # for each individual piece?  Not sure.
        for gcl in local_builder.code:
            for code in gcl.code_list:
                add_code_part(code, gcl.guard)

        for gcl in global_builder.code:
            for code in gcl.code_list:
                add_code_part(code, gcl.guard)

        tensor_check_names = (
            local_builder.tensor_check_names + global_builder.tensor_check_names
        )

        check_tensors_fn = None
        check_tensors_verbose_fn = None
        if tensor_check_names:
            assert (
                not self.output_graph.export
            ), "Illegal to set tensor_check_names in export."
            tensor_check_examples = (
                local_builder.tensor_check_examples
                + global_builder.tensor_check_examples
            )
            dynamic_dims_sizes = None
            dynamic_dims_strides = None

            def convert(size_or_stride):
                converted: List[Optional[int]] = []
                for dim in size_or_stride:
                    if is_concrete_int(dim):
                        converted.append(int(dim))
                    else:
                        converted.append(None)
                return converted

            dynamic_dims_sizes = [
                convert(
                    self.output_graph.tensor_weakref_to_sizes_strides[WeakIdRef(t)][
                        "size"
                    ]
                )
                for t in tensor_check_examples
            ]

            dynamic_dims_strides = [
                convert(
                    self.output_graph.tensor_weakref_to_sizes_strides[WeakIdRef(t)][
                        "stride"
                    ]
                )
                for t in tensor_check_examples
            ]

            tensor_guards = TensorGuards(
                *tensor_check_examples,
                dynamic_dims_sizes=dynamic_dims_sizes,
                dynamic_dims_strides=dynamic_dims_strides,
            )
            check_tensors_fn = tensor_guards.check
            check_tensors_verbose_fn = tensor_guards.check_verbose
            tensor_check_args = ", ".join(
                tensor_check_names + ["tensor_check_names=tensor_check_names"]
            )
            # Do this manually, to un-stagger the guards in log message
            code_parts.append(f"___check_tensors({tensor_check_args})")
            tensor_check_guards = (
                local_builder.tensor_check_guards + global_builder.tensor_check_guards
            )
            for i, name in enumerate(tensor_check_names):
                # This is a copy of what guards.cpp checks against
                # Keep this in sync with TensorCheck constructor
                t = tensor_check_examples[i]
                pytype = type(t)
                dispatch_key = (
                    torch._C._dispatch_keys(t)
                    | torch._C._dispatch_tls_local_include_set()
                ) - torch._C._dispatch_tls_local_exclude_set()
                dtype = t.dtype
                device_index = t.device.index
                requires_grad = t.requires_grad
                sizes = dynamic_dims_sizes[i]
                strides = dynamic_dims_strides[i]
                add_code_part(
                    f"check_tensor({name}, {pytype.__qualname__}, {dispatch_key}, {dtype}, "
                    f"device={device_index}, requires_grad={requires_grad}, size={sizes}, stride={strides})",
                    tensor_check_guards[i],
                    log_only=True,
                )

        aotautograd_guards: List[GuardEnvExpr] = (
            self.output_graph.tracing_context.guards_context.aotautograd_guards
            if self.output_graph
            else []
        )
        for guard in aotautograd_guards:
            if isinstance(guard, DuplicateInputs):
                source_a = guard.input_source_a
                source_b = guard.input_source_b
                add_code_part(f"{source_a.name()} is {source_b.name()}", None)
            else:
                raise RuntimeError(f"Unknown GuardEnvExpr: {guard}")

        # TODO: the "guard" here is actually just the top level SHAPE_ENV
        # which is useless.  Get ShapeEnv to pass in more provenance.
        for gcl in local_builder.shape_env_code:
            for code in gcl.code_list:
                add_code_part(code, gcl.guard)

        assert not global_builder.shape_env_code

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

        unique_code_parts = list(unique(code_parts))
        make_guard_fn_args = ", ".join(closure_vars.keys())
        guard_body, pycode = build_guard_function(unique_code_parts, make_guard_fn_args)

        if os.environ.get("TORCHDYNAMO_PRINT_GUARDS", None) == "1":
            print("GUARDS\n", guard_body)

        if is_guard_failure_reporting_enabled() or guard_fail_fn is not None:
            # Guard fail hook is called everytime guard eval fails. For a cache
            # lookup where there are multiple entries in the same cache line,
            # this can lead to very high performance overhead. So, we have
            # decided to hide this behing a config flag.
            set_guard_fail_hook(guard_fail_hook)

        out: Dict[str, Any] = dict()
        exec(pycode, global_builder.scope, out)
        guard_fn = out["___make_guard_fn"](*closure_vars.values())
        guard_fn.closure_vars = closure_vars
        # TODO(whc) maybe '.code_parts' was only kept around for the guard callback? so we don't need both
        guard_fn.args = largs
        guard_fn.code_parts = code_parts
        # Grab only G, but preserve "G" because guards access it as "G"
        guard_fn.global_scope = {
            "G": global_builder.scope["G"],
        }
        guard_fn.guard_fail_fn = guard_fail_fn
        return guard_fn

    def invalidate(self, ref):
        # A weakref is no longer valid, self.check_fn should return false
        # TODO(janimesh) - Free up cache entry after the cache entry formation
        # is in python, and the underlying data structure is a doubly linked
        # list.
        self.valid = False

    def id_ref(self, obj):
        """add a weakref, return the id"""
        try:
            if id(obj) not in self._weakrefs:
                weak_id = weakref.ref(obj)
                self._weakrefs[id(obj)] = weak_id
                weakref.finalize(weak_id, self.invalidate)
        except TypeError:
            pass  # cannot weakref bool object
        return id(obj)

    def lookup_weakrefs(self, obj):
        if id(obj) in self._weakrefs:
            return self._weakrefs[id(obj)]
        return None


def build_guard_function(code_parts, closure_args) -> Tuple[str, str]:
    from torch._inductor.utils import IndentedBuffer

    if HAS_UNPARSE_FUNCTIONS:
        csepass = PyExprCSEPass()
        csepass.count(code_parts)

        def replace(expr: str) -> Tuple[List[str], str]:
            return csepass.replace(expr)

    else:

        def replace(expr: str) -> Tuple[List[str], str]:
            return [], expr

    # Generate the inner body of the guard function.
    # i.e. if-chain of the guard expressions.
    guard_body = IndentedBuffer()
    for expr in code_parts:
        preface, expr = replace(expr)
        guard_body.writelines(preface)
        guard_body.writeline(f"if not ({expr}):")
        with guard_body.indent():
            guard_body.writeline("return False")

    # Wrap the inner body into the actual guard function.
    guard = IndentedBuffer()
    guard.writeline("def guard(L):")
    with guard.indent():
        guard.splice(guard_body)
        guard.writeline("return True")

    # Wrap the whole guard function into another function
    # with the closure variables.
    make_guard_fn = IndentedBuffer()
    make_guard_fn.writeline(f"def ___make_guard_fn({closure_args}):")
    with make_guard_fn.indent():
        make_guard_fn.splice(guard)
        make_guard_fn.writeline("return guard")

    return guard_body.getvalue(), make_guard_fn.getvalue()


stashed_first_fail_reason = None


def guard_fail_hook(
    guard_fn: GuardFn,
    code: types.CodeType,
    f_locals: Dict[str, object],
    index: int,
    last: bool,
) -> None:
    """
    called whenever a guard fails.
    """
    first = index == 0
    global stashed_first_fail_reason
    # Don't waste time computing the fail reason for guards we aren't going to report out.
    if not guard_fn.guard_fail_fn and not (first or last):
        return
    scope = {"L": f_locals, "G": guard_fn.global_scope["G"]}
    scope.update(guard_fn.closure_vars)
    scope["___check_tensors"] = scope["___check_tensors_verbose"]
    reason = None
    for part in guard_fn.code_parts:
        global_scope = dict(guard_fn.global_scope)
        global_scope["__compile_source__"] = part
        with report_compile_source_on_error():
            fail_reason = eval(part, global_scope, scope)
        # Only ___check_tensors knows how to return a fancy fail reason;
        # for everything else we just report the code that failed
        if isinstance(fail_reason, str):
            reason = fail_reason
            break
        elif isinstance(fail_reason, bool) and not fail_reason:
            reason = part
            break

    if first:
        stashed_first_fail_reason = reason

    if not last:
        return

    # Technically, we're failing our last guard, which is our oldest guard due to the
    # eval_frame.c logic that moves newest frames to head, but for logging purposes
    # it's more useful to see the 'first' failure (if we never got a hit) since it's
    # likely not yet been logged as a failure reason in a case of repeating failures.
    assert stashed_first_fail_reason
    guard_failures[orig_code_map[code]].append(stashed_first_fail_reason)
    stashed_first_fail_reason = None

    # TODO should we GuardFail our stashed_first_fail_reason too?
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


def guard_error_hook(
    guard_fn: GuardFn,
    code: types.CodeType,
    f_locals: Dict[str, object],
    index: int,
    last: bool,
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


def make_dupe_guard(obj_source, dupe_source):
    # Note - we may end up in a situation where we invoke something like
    # def fn(x, y)
    # with fn(x, x)
    # Prior to the addition of tracking to all relevant objects, we would handle this just fine by
    # eagerly re-entering VB and rewrapping inputs, correctly creating graphargs and placeholders. However,
    # with tracking on inputs, duplicate inputs or aliased relationships may end up getting erased here -
    # In the the fn(x, x) example call above look like a graph with a single input.
    # In order to ensure that we do not reuse fn(x, x) for fn(x, y), we create a duplicate input guard.

    # Note - we may not have a source, that is fine, it just means we had an object that is safe to have
    # leave unsourced - like a local list created and discharged entirely within a local scope.
    if dupe_source and dupe_source != obj_source:
        ser_source_is_local = is_from_local_source(dupe_source)
        source_is_local = is_from_local_source(obj_source)
        # Note - both must be local, or global, or we will run afoul of a lack of merging in how we currently
        # reconcile guards builder scopes in compile_check_fn. This technically means we miss a guard here,
        # so maybe we should do this refactor before we land this...
        # TODO(voz): Combine local and global guard builders.
        if ser_source_is_local == source_is_local:
            # Note - this is a little agressive - these being duplicate input does not always matter.
            # However, this should always be a sound guard to add here.
            return functools.partial(GuardBuilder.DUPLICATE_INPUT, source_b=dupe_source)
    return None
