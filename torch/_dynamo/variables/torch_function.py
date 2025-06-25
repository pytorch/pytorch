# mypy: ignore-errors

"""TorchDynamo support for __torch_function__ tensor subclasses.

This module implements support for tensor subclasses with __torch_function__ overrides.
A tensor subclass instance is represented as a TensorWithTFOverrideVariable, which handles
dispatching __torch_function__ on attribute accesses, method calls, and torch API calls.

Unsupported features:
- Triggering __torch_function__ on tensor subclass non-tensor custom attributes
- Graph breaking on mutating guardable tensor properties within a __torch_function__ context
  (can cause excessive recompiles in certain cases)
- Matching exact eager behavior of ignoring __torch_function__ objects in non-tensor
  argument positions of Torch API calls

Supported features:
- Static method implementations of __torch_function__ on custom objects (triggers on torch
  API calls with the object as any argument)
- Triggering __torch_function__ on torch API calls with tensor subclass arguments
- __torch_function__ calls on base tensor attribute access and method calls for tensor
  subclass instances
- Matches dispatch ordering behavior of eager __torch_function__ with subclass/object
  arguments in any position

See https://docs.google.com/document/d/1WBxBSvW3NXhRp9ncmtokJloMLCtF4AYNhJaffvHe8Kw/edit#heading=h.vacn73lozd9w
for more information on the design.
"""

import collections
import contextlib
import functools
import inspect
import operator
from typing import TYPE_CHECKING

import torch._C
import torch.utils._pytree as pytree
from torch._guards import Source
from torch.overrides import (
    _get_overloaded_args,
    BaseTorchFunctionMode,
    get_default_nowrap_functions,
    TorchFunctionMode,
)
from torch.utils._device import DeviceContext

from .. import graph_break_hints
from ..exc import unimplemented_v2
from ..guards import GuardBuilder, install_guard
from ..polyfills import NoEnterTorchFunctionMode
from ..source import AttrSource, GlobalSource, TorchFunctionModeStackSource, TypeSource
from ..utils import (
    class_has_getattribute,
    clear_torch_function_mode_stack,
    get_safe_global_name,
    has_torch_function,
    is_tensor_base_attr_getter,
    set_torch_function_mode_stack,
)
from .base import VariableTracker
from .constant import ConstantVariable
from .ctx_manager import GenericContextWrappingVariable
from .functions import UserMethodVariable
from .lazy import LazyVariableTracker
from .lists import TupleVariable
from .tensor import TensorSubclassVariable, TensorVariable
from .user_defined import UserDefinedObjectVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslator


bin_ops = [
    operator.pow,
    operator.mul,
    operator.matmul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.add,
    operator.lt,
    operator.gt,
    operator.ge,
    operator.le,
    operator.ne,
    operator.eq,
    operator.sub,
    operator.ipow,
    operator.imul,
    operator.imatmul,
    operator.ifloordiv,
    operator.itruediv,
    operator.imod,
    operator.iadd,
    operator.isub,
]

bin_int_ops = [
    operator.and_,
    operator.or_,
    operator.xor,
    operator.iand,
    operator.ixor,
    operator.ior,
]

un_int_ops = [operator.invert]

tensor_and_int_ops = [
    operator.lshift,
    operator.rshift,
    operator.ilshift,
    operator.irshift,
    operator.getitem,
]

un_ops = [
    operator.abs,
    operator.pos,
    operator.neg,
    operator.not_,  # Note: this has a local scalar dense call
    operator.length_hint,
]

BUILTIN_TO_TENSOR_FN_MAP = {}

# These functions represent the r* versions of the above ops
# Basically, if __add__(1, Tensor) is called, it is translated
# to __radd__(Tensor, 1).
# In the builtin var, we check if there is a tensor in the first args position,
# if not, we swap the args and use the r* version of the op.
BUILTIN_TO_TENSOR_RFN_MAP = {}


def populate_builtin_to_tensor_fn_map():
    global BUILTIN_TO_TENSOR_FN_MAP

    most_recent_func = None

    class GetMethodMode(BaseTorchFunctionMode):
        """
        Mode to extract the correct methods from torch function invocations
        (Used to get the correct torch.Tensor methods from builtins)
        """

        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            nonlocal most_recent_func
            most_recent_func = func
            return func(*args, **kwargs)

    inp0 = torch.ones(1)
    inp1 = torch.ones(1)
    inp0_int = torch.ones(1, dtype=torch.int32)
    inp1_int = torch.ones(1, dtype=torch.int32)
    with GetMethodMode():
        setups_and_oplists = [
            (lambda o: o(inp0), un_ops),
            (lambda o: o(inp0_int), un_int_ops),
            (lambda o: o(inp0, inp1), bin_ops),
            (lambda o: o(inp0_int, inp1_int), bin_int_ops),
            (lambda o: o(inp0_int, 0), tensor_and_int_ops),
        ]
        for setup_fn, op_list in setups_and_oplists:
            for op in op_list:
                setup_fn(op)
                assert most_recent_func is not None
                BUILTIN_TO_TENSOR_FN_MAP[op] = most_recent_func

        # gather the reverse functions
        rsetups_and_oplists = [
            (
                lambda o: o(1, inp1),
                bin_ops,
            ),  # Get r* ops, (ex. __sub__(int, Tensor) -> __rsub__(Tensor, int))
            (lambda o: o(1, inp1_int), bin_int_ops),
            (lambda o: o(0, inp0_int), tensor_and_int_ops),
        ]

        rskips = {operator.matmul, operator.imatmul, operator.getitem}
        for setup_fn, op_list in rsetups_and_oplists:
            for op in op_list:
                if op in rskips:
                    continue
                setup_fn(op)
                assert most_recent_func is not None
                if most_recent_func != BUILTIN_TO_TENSOR_FN_MAP[op]:
                    BUILTIN_TO_TENSOR_RFN_MAP[op] = most_recent_func


populate_builtin_to_tensor_fn_map()

banned_attrs = [
    fn.__self__.__name__
    for fn in get_default_nowrap_functions()
    if is_tensor_base_attr_getter(fn)
]


@functools.cache
def get_prev_stack_var_name():
    from ..bytecode_transformation import unique_id

    return unique_id("___prev_torch_function_mode_stack")


# Used to clear/restore the python torch function mode stack and temporarily restore it as needed
class TorchFunctionModeStackStateManager:
    def __init__(self):
        self.stack = []

    def __enter__(self):
        self.stack = torch.overrides._get_current_function_mode_stack()
        clear_torch_function_mode_stack()

    def __exit__(self, exc_type, exc_value, traceback):
        set_torch_function_mode_stack(self.stack)
        self.stack = []

    @contextlib.contextmanager
    def temp_restore_stack(self):
        prev = torch.overrides._get_current_function_mode_stack()
        set_torch_function_mode_stack(self.stack)
        try:
            yield
        finally:
            set_torch_function_mode_stack(prev)


torch_function_mode_stack_state_mgr = TorchFunctionModeStackStateManager()


class SymbolicTorchFunctionState:
    def __init__(self, py_stack):
        # This is annoyingly complicated because of how the torch function subclass + mode C API was designed
        # There are two exposed C knobs here as contexts: torch._C.DisableTorchFunction and torch._C.DisableTorchFunctionSubclass
        # These are their definitions:
        # 1) torch._C._is_torch_function_enabled indicates that neither of the above knobs have been entered
        # (if either are entered, this will be False)
        # 2) torch._C._is_torch_function_mode_enabled indicates that either the torch mode stack is empty OR
        # torch._C.DisableTorchFunction has been entered
        # To disambiguate these and keep myself sane I added a C API to check whether all torch function
        # concepts (modes and subclasses) are enabled.
        # This only returns true iff we have not entered torch._C.DisableTorchFunction and allows us to separate
        # the stack length from the enablement state of torch function modes.
        # This is important because now if a mode is pushed while dynamo is tracing, we know whether
        # or not torch function modes are enabled and whether we should trace it.
        self.torch_function_subclass_enabled = torch._C._is_torch_function_enabled()

        # This differs from the C API of the same name
        # this will only be false iff we have entered torch._C.DisableTorchFunction
        # and does not take into account the mode stack length, while the C API bundles these
        # two concepts
        self.torch_function_mode_enabled = (
            not torch._C._is_torch_function_all_disabled()
        )

        self.cur_mode = None

        TorchFunctionModeStackVariable.reset()

        self.mode_stack: collections.deque[TorchFunctionModeVariable] = (
            collections.deque()
        )

        for i, val in enumerate(py_stack):
            self.mode_stack.append(
                LazyVariableTracker.create(val, source=TorchFunctionModeStackSource(i))
            )

    def in_torch_function_mode(self):
        return len(self.mode_stack) > 0

    def pop_torch_function_mode(self):
        return self.mode_stack.pop()

    def push_torch_function_mode(self, mode_var):
        self.mode_stack.append(mode_var)

    def call_torch_function_mode(self, tx, fn, types, args, kwargs):
        with self._pop_mode_for_inlining() as cur_mode:
            return cur_mode.call_torch_function(tx, fn, types, args, kwargs)

    @contextlib.contextmanager
    def _pop_mode_for_inlining(self):
        old_mode = self.cur_mode
        self.cur_mode = self.pop_torch_function_mode()
        try:
            yield self.cur_mode
        finally:
            mode = self.cur_mode
            self.cur_mode = old_mode
            self.push_torch_function_mode(mode)


class TorchFunctionModeStackVariable(VariableTracker):
    """Fake VT to use as a dummy object, indicating the presence of torch function mode stack mutation"""

    # singleton value representing the global torch function mode stack
    # singleton (it exists in C++)
    stack_value_singleton = object()

    # offset is used to track if we have inserted/removed a
    # device context which is always placed at the bottom of the stack
    # if a device context is inserted, the graph will run this mutation
    # so when we want to reconstruct any other modes on the stack
    # their indices should be shifted right by 1 (+1)
    # Conversely, if there was a device context on the stack, and the graph
    # mutates the stack to remove that context (set default device to None)
    # each of the indices of other modes should be shifted left by 1 (-1)
    offset = 0

    def __init__(self, source, symbolic_stack):
        self.source = source
        self.symbolic_stack = symbolic_stack

    @classmethod
    def reset(cls):
        cls.offset = 0

    @classmethod
    def register_mutation(cls, tx: "InstructionTranslator"):
        if cls.stack_value_singleton not in tx.output.side_effects:
            var = cls(
                source=Source(),
                symbolic_stack=tx.symbolic_torch_function_state.mode_stack,
            )
            tx.output.side_effects.track_mutable(cls.stack_value_singleton, var)
            tx.output.side_effects.mutation(var)

    @classmethod
    def register_device_context_insertion(cls, tx: "InstructionTranslator"):
        stack = tx.symbolic_torch_function_state.mode_stack
        if stack and cls.is_device_context(stack[0]):
            return
        else:
            cls.offset += 1
            stack.insert(
                0,
                TorchFunctionModeVariable(
                    None, source=TorchFunctionModeStackSource(-cls.offset)
                ),
            )

    @classmethod
    def clear_default_device(cls, tx: "InstructionTranslator"):
        stack = tx.symbolic_torch_function_state.mode_stack
        if stack and cls.is_device_context(stack[0]):
            stack.popleft()
            cls.offset -= 1

    @staticmethod
    def is_device_context(var):
        return isinstance(var.value, DeviceContext) or var.value is None

    @classmethod
    def get_mode_index(cls, ind):
        return ind + cls.offset


class TorchFunctionModeVariable(GenericContextWrappingVariable):
    @staticmethod
    def is_supported_torch_function_mode(ty):
        # Supported in this sense means we can support graph breaks under the
        # context.
        # We are able to trace custom modes but if there are graph breaks under them
        # and they have a custom __enter__/__exit__ we don't handle this for the
        # same reason we don't handle generic context managers: there may be side effects
        # that are now affected by executing the function across two frames instead of one
        # Today we support the enter/exit of the default TorchFunctionMode as well as
        # DeviceContext (which is used for set_default_device)
        return issubclass(ty, (NoEnterTorchFunctionMode, DeviceContext)) or (
            not class_has_getattribute(ty)
            and inspect.getattr_static(ty, "__enter__") == TorchFunctionMode.__enter__
            and inspect.getattr_static(ty, "__exit__") == TorchFunctionMode.__exit__
        )

    def __init__(self, value, source=None, **kwargs):
        if value is not None:
            super().__init__(value, **kwargs)
        self.value = value
        self.cm_obj = value  # needed for BC with calling enter from CM code
        self.source = source

    def reconstruct(self, codegen: "PyCodegen"):
        # This shouldn't be called unless we have a source
        assert self.source
        self.source.reconstruct(codegen)

    def module_name(self):
        return self.value.__module__

    def fn_name(self):
        return type(self.value).__name__

    def python_type(self):
        return type(self.value)

    def call_torch_function(self, tx: "InstructionTranslator", fn, types, args, kwargs):
        return call_torch_function(
            tx,
            get_torch_function_fn(tx, self),
            fn,
            types,
            args,
            kwargs,
        )

    def enter(self, tx):
        from .torch import TorchInGraphFunctionVariable

        if isinstance(self.value, NoEnterTorchFunctionMode):
            return ConstantVariable.create(None)

        TorchInGraphFunctionVariable(
            torch._C._push_on_torch_function_stack
        ).call_function(tx, [self], {})
        return ConstantVariable.create(None)

    def exit(self, tx: "InstructionTranslator", *args):
        from .torch import TorchInGraphFunctionVariable

        TorchInGraphFunctionVariable(torch._C._pop_torch_function_stack).call_function(
            tx, [], {}
        )
        return ConstantVariable.create(None)

    def reconstruct_type(self, codegen: "PyCodegen"):
        ty = NoEnterTorchFunctionMode
        codegen(
            AttrSource(
                codegen.tx.import_source(ty.__module__),
                ty.__name__,
            )
        )

    def supports_graph_breaks(self):
        return True

    def exit_on_graph_break(self):
        return False


def _get_all_args(args, kwargs):
    return _flatten_vts(pytree.arg_tree_leaves(*args, **kwargs))


def _flatten_vts(vts):
    from collections import deque

    from .dicts import ConstDictVariable
    from .lists import ListVariable

    vts = deque(vts)
    output = []

    while vts:
        vt = vts.pop()

        if not vt.is_realized() and vt.peek_type() in (dict, list, tuple):
            vt.realize()

        if vt.is_realized():
            if isinstance(vt, ListVariable):
                vts.extend(vt.items)
            elif isinstance(vt, ConstDictVariable):
                vts.extend(vt.items.values())

        output.append(vt)

    return output


def _get_subclass_type(var):
    assert isinstance(var, (TensorWithTFOverrideVariable, UserDefinedObjectVariable))
    return var.python_type()


def _get_subclass_type_var(tx: "InstructionTranslator", var):
    assert isinstance(var, (TensorWithTFOverrideVariable, UserDefinedObjectVariable))
    if isinstance(var, TensorWithTFOverrideVariable):
        return var.class_type_var(tx)
    elif isinstance(var, UserDefinedObjectVariable):
        source = var.source and TypeSource(var.source)
        return VariableTracker.build(tx, var.python_type(), source)


def _is_attr_overridden(tx: "InstructionTranslator", var, name):
    import torch

    overridden = False
    try:
        attr_val = inspect.getattr_static(var.python_type(), name)
        overridden |= attr_val != getattr(torch.Tensor, name)
    except AttributeError:
        pass

    return overridden


def call_torch_function(tx, torch_function_var, fn, types, args, kwargs):
    # This emulates calling __torch_function__, which has a signature
    #   def __torch_function__(cls, func, types, args=(), kwargs=None):
    #
    # Also notice the `cls` is not explicitly passed in the reference
    # implementations:
    # 1. https://github.com/pytorch/pytorch/blob/8d81806211bc3c0ee6c2ef235017bacf1d775a85/torch/csrc/utils/python_arg_parser.cpp#L368-L374  # noqa: B950
    # 2. https://github.com/pytorch/pytorch/blob/8d81806211bc3c0ee6c2ef235017bacf1d775a85/torch/overrides.py#L1741-L1743
    tf_args = [
        fn,
        types,
        VariableTracker.build(tx, tuple(args)),
        VariableTracker.build(tx, kwargs),
    ]
    return torch_function_var.call_function(tx, tf_args, {})


def get_torch_function_fn(tx: "InstructionTranslator", vt):
    # The underlying function could be a classmethod, staticmethod, regular
    # function or a function with C-implementation. It doesn't matter as long as
    # they satisfy the calling convention in `call_torch_function`.
    from .builtin import BuiltinVariable

    args = [vt, ConstantVariable("__torch_function__")]
    func_vt = BuiltinVariable(getattr).call_function(tx, args, {})
    return func_vt


def can_dispatch_torch_function(tx: "InstructionTranslator", args, kwargs):
    has_overridden_args = any(
        has_torch_function(arg) for arg in _get_all_args(args, kwargs)
    )
    tf_state = tx.symbolic_torch_function_state
    return (has_overridden_args and tf_state.torch_function_subclass_enabled) or (
        tf_state.torch_function_mode_enabled and tf_state.in_torch_function_mode()
    )


def dispatch_torch_function(tx: "InstructionTranslator", fn, args, kwargs):
    """Gathers all args that are TensorWithTFOverrideVariable and dispatches based on the ordering in _get_overloaded_args"""

    all_args = _get_all_args(args, kwargs)
    overloaded_args = _get_overloaded_args(
        [arg for arg in all_args if has_torch_function(arg)],
        _get_subclass_type,
    )

    types = TupleVariable([_get_subclass_type_var(tx, arg) for arg in overloaded_args])

    if tx.symbolic_torch_function_state.in_torch_function_mode():
        res = tx.symbolic_torch_function_state.call_torch_function_mode(
            tx, fn, types, args, kwargs
        )
        if not (isinstance(res, ConstantVariable) and res.value is NotImplemented):
            return res

    for arg in overloaded_args:
        res = arg.call_torch_function(
            tx,
            fn,
            types,
            args,
            kwargs,
        )

        if not (isinstance(res, ConstantVariable) and res.value is NotImplemented):
            return res

    unimplemented_v2(
        gb_type="All __torch_function__ overrides returned NotImplemented due to TypeError from user code",
        context=f"{fn=}, {args=}, {kwargs=}",
        explanation=f"All __torch_function__ overrides for for function {fn} returned NotImplemented",
        hints=[
            *graph_break_hints.USER_ERROR,
        ],
    )


class TensorWithTFOverrideVariable(TensorVariable):
    """
    Represents a tensor subclass instance with a __torch_function__ override.
    """

    @classmethod
    def from_tensor_var(cls, tx, tensor_var, class_type, cls_source):
        # [Note: __torch_function__] coerce `tensor_var` into a
        # TensorWithTFOverrideVariable. In eager, this is just a type change.
        import torch

        # This simulates shallow-copying the tensor object.
        kwargs = dict(tensor_var.__dict__)
        input_tensor_type = kwargs.pop("class_type")
        assert input_tensor_type in (torch.Tensor, torch.nn.Parameter), (
            f"invalid class type {input_tensor_type} in TensorWithTFOverrideVariable.from_tensor_var"
        )
        var = cls(class_type=class_type, **kwargs)
        var.install_global(tx)
        return var

    def install_global(self, tx):
        # stash the subclass type to rewrap an output tensor if needed
        # this is needed because the actual type needs to be available
        # each time the compiled artifact is run and outputs a wrapped tensor.
        if self.global_mangled_class_name(tx) not in tx.output.global_scope:
            # Safe because global_mangled_class_name figures it out
            tx.output.install_global_unsafe(
                self.global_mangled_class_name(tx), self.class_type
            )

    def python_type(self):
        return self.class_type

    def class_type_var(self, tx):
        return TensorSubclassVariable(
            self.class_type, source=GlobalSource(self.global_mangled_class_name(tx))
        )

    def global_mangled_class_name(self, tx):
        return get_safe_global_name(
            tx, f"__subclass_{self.class_type.__name__}", self.class_type
        )

    def var_getattr(self, tx: "InstructionTranslator", name):
        # [Note: __torch_function__] We currently only support attributes that are defined on
        # base tensors, custom attribute accesses will graph break.
        import torch

        # I think only `_base` is breaking because we aren't modelling view
        # relationship perfectly in some scenarios.
        if name in banned_attrs:
            unimplemented_v2(
                gb_type="Unsupported tensor subclass attribute access",
                context=f"{name}",
                explanation="`torch.compile` currently can't trace this",
                hints=[
                    f"Avoid accessing {name} of tensor subclass in torch.compile region",
                    *graph_break_hints.SUPPORTABLE,
                ],
            )

        # Handle non-overridden attributes inherited from `torch.Tensor`.
        attr_is_overridden = _is_attr_overridden(tx, self, name)
        if (
            hasattr(torch.Tensor, name)
            and not attr_is_overridden
            and not inspect.ismethoddescriptor(getattr(torch.Tensor, name))
        ):
            args, kwargs = [self], {}
            if can_dispatch_torch_function(tx, args, kwargs):
                if self.source:
                    install_guard(
                        AttrSource(
                            AttrSource(self.source, "__class__"), name
                        ).make_guard(GuardBuilder.FUNCTION_MATCH)
                    )
                get_fn = VariableTracker.build(tx, getattr(torch.Tensor, name).__get__)

                return self.call_torch_function(
                    tx,
                    get_fn,
                    TupleVariable([self.class_type_var(tx)]),
                    args,
                    kwargs,
                )
        else:
            # `TensorVariable.var_getattr` doesn't handle user-defined
            # function/attribute well, so we explicitly handle them here.
            #
            # TODO move this logic into `TensorVariable`, or try to merge it
            # with similar logic in `UserDefinedObjectVariable`.
            try:
                attr = inspect.getattr_static(self.class_type, name)
            except AttributeError:
                pass
            else:
                import types

                cls_source = GlobalSource(self.global_mangled_class_name(tx))
                attr_source = AttrSource(cls_source, name)
                if isinstance(attr, types.FunctionType):
                    install_guard(attr_source.make_guard(GuardBuilder.FUNCTION_MATCH))
                    return UserMethodVariable(attr, self)

                elif isinstance(attr, property):
                    getter_source = AttrSource(attr_source, "fget")
                    getter = attr.fget
                    getter_var = UserMethodVariable(getter, self, source=getter_source)
                    return getter_var.call_function(tx, [], {})

                elif isinstance(attr, classmethod):
                    return UserMethodVariable(
                        attr.__func__, self.class_type_var(tx), source=attr_source
                    )

                elif attr_is_overridden:
                    unimplemented_v2(
                        gb_type="Unsupported tensor subclass overridden attribute access",
                        context=f"{name}",
                        explanation="`torch.compile` only support tracing certain types of overridden tensor subclass attributes",
                        hints=[
                            f"Avoid accessing {name} of tensor subclass in torch.compile region",
                            f"Renaming attribute `{name}` of type {self.class_type}",
                            *graph_break_hints.SUPPORTABLE,
                        ],
                    )

        return super().var_getattr(tx, name)

    def call_torch_function(self, tx: "InstructionTranslator", fn, types, args, kwargs):
        # NOTE this assumes `__torch_function__` isn't modified during tracing.
        if not hasattr(self, "torch_function_fn"):
            self.torch_function_fn = get_torch_function_fn(tx, self)

        return call_torch_function(
            tx,
            self.torch_function_fn,
            fn,
            types,
            args,
            kwargs,
        )

    def call_method(
        self,
        tx,
        name,
        args: "list[VariableTracker]",
        kwargs: "dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # This code block implements inlining the __torch_function__ override
        # of `call_method`.
        tf_args = [self] + args
        if can_dispatch_torch_function(tx, tf_args, kwargs):
            import torch

            if _is_attr_overridden(tx, self, name):
                unimplemented_v2(
                    gb_type="Tensor subclass overridden method call",
                    context=f"{name}",
                    explanation="`torch.compile` currently can't trace this",
                    hints=[
                        f"Avoid calling {name} of tensor subclass in torch.compile region",
                        f"Renaming method `{name}` of type {self.class_type}",
                        *graph_break_hints.SUPPORTABLE,
                    ],
                )

            # [Note: __torch_function__] Currently we only support methods that are defined on tensor
            # we will graph break in other cases this will need a bigger overhaul of extracting methods/comparing them for equality
            # We've established with the above check that the method is not overridden, so we guard that the method is the same
            # as the impl defined on tensor and retrieve it
            if self.source:
                source = AttrSource(AttrSource(self.source, "__class__"), name)
                value = inspect.getattr_static(self.python_type(), name)
            else:
                source = None
                value = getattr(torch.Tensor, name)
            func_var = VariableTracker.build(tx, value, source)
            return dispatch_torch_function(tx, func_var, tf_args, kwargs)
        else:
            return super().call_method(tx, name, args, kwargs)
