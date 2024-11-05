# mypy: ignore-errors

import builtins
import collections
import functools
import inspect
import itertools
import types
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import Never

import torch

from .. import polyfills, variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import (
    check_constant_args,
    check_unspec_or_constant_args,
    identity,
    is_function,
    is_wrapper_or_member_descriptor,
    istype,
    make_cell,
)
from .base import typestr, ValueMutationNew, VariableTracker
from .constant import ConstantVariable


try:
    from torch.distributed._composable.fsdp import _fsdp_param_group
except ModuleNotFoundError:
    _fsdp_param_group = None


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator
    from torch._guards import Source
    from torch._higher_order_ops.triton_kernel_wrap import (
        TritonGridType,
        TritonKernelType,
    )


_F = TypeVar("_F", bound=Callable)


def wrap_bound_arg(tx: "InstructionTranslator", val, source=None):
    # Source propagation is best effort since not every object we encounter has a source to begin with.
    if isinstance(val, VariableTracker):
        return val
    elif not source:
        return VariableTracker.build(tx, val)
    else:
        # Create a lazy variable to avoid guarding on __defaults__ unless really
        # needed.
        return variables.LazyVariableTracker.create(val, source)


def wrap_args_kwargs(tx: "InstructionTranslator", result):
    for k, v in list(result.items()):
        if isinstance(v, (tuple, dict)):
            # args/kwargs
            result[k] = wrap_bound_arg(tx, v)


def init_cellvars(parent, result, code):
    closure_cells = {}
    side_effects = parent.output.side_effects

    # for name in itertools.chain(code.co_cellvars, code.co_freevars):
    for name in code.co_cellvars:
        closure_cells[name] = side_effects.track_cell_new()
        if name in result:
            side_effects.store_cell(closure_cells[name], result.pop(name))

    return closure_cells


def _create_nested_fn(
    code, f_globals, name, defaults, closure, kwdefaults, annotations
):
    from types import FunctionType

    func = FunctionType(code, f_globals, name, defaults, closure)
    func.__kwdefaults__ = kwdefaults

    if isinstance(annotations, tuple):
        from itertools import pairwise

        annotations = dict(pairwise(annotations))

    # TypeError: __annotations__ must be set to a dict object
    assert annotations is None or isinstance(annotations, dict)
    func.__annotations__ = annotations

    return func


class BaseUserFunctionVariable(VariableTracker):
    def get_filename(self):
        return self.get_code().co_filename

    def get_name(self):
        return self.get_code().co_name

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return tx.inline_user_function_return(self, [*self.self_args(), *args], kwargs)

    def call_hasattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        result = False

        try:
            result = hasattr(self.get_function(), name)
        except NotImplementedError:
            if name == "__name__" and isinstance(self, NestedUserFunctionVariable):
                result = True
        return variables.ConstantVariable.create(result)

    def inspect_parameter_names(self):
        return list(inspect.signature(self.get_function()).parameters)

    def closure_vars(self, tx):
        return {}


class UserFunctionVariable(BaseUserFunctionVariable):
    """Some unsupported user-defined global function"""

    _nonvar_fields = {
        "fn",
        "is_constant",
        *BaseUserFunctionVariable._nonvar_fields,
    }

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))
        return cls(value, source=source)

    def __init__(self, fn, is_constant=False, **kwargs) -> None:
        super().__init__(**kwargs)
        if getattr(fn, "_dynamo_marked_constant", False):
            # This method should be treated as a constant for the purposes of compilation
            self.is_constant = True
        else:
            self.is_constant = False

        assert isinstance(
            fn, (types.FunctionType, torch.jit.ScriptFunction)
        ), f"expected FunctionType found {typestr(fn)} {fn}"
        # TODO(anijain2305) - Replace directly calling UserFunctionVariable with
        # VariableBuilder, which handles the wrapping of _torchdynamo_inline.
        # unpack @torch._dynamo.optimize()(fn) wrapped function
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        self.fn: types.FunctionType = fn

    def as_python_constant(self):
        if istype(self, UserFunctionVariable):
            return self.fn
        # subclasses (such as methods) usually aren't a constant
        return super().as_python_constant()

    def self_args(self):
        return []

    def get_function(self):
        return self.fn

    def get_code(self):
        return self.fn.__code__

    def python_type(self):
        return types.FunctionType

    def has_self(self):
        return getattr(self.fn, "__self__", None) is not None

    def get_globals(self):
        return self.fn.__globals__

    def bind_args(self, parent, args, kwargs):
        assert not self.is_constant
        tx = parent.output.root_tx
        wrap = functools.partial(wrap_bound_arg, tx=tx)

        fn: types.FunctionType = self.fn
        defaults = fn.__defaults__ or []
        defaults_sources = [
            None if self.source is None else DefaultsSource(self.source, idx)
            for idx, _ in enumerate(defaults)
        ]
        fake_func = types.FunctionType(
            fn.__code__,
            fn.__globals__,
            fn.__name__,
            tuple(
                [
                    wrap(val=arg, source=source)
                    for arg, source in zip(defaults, defaults_sources)
                ]
            ),
            fn.__closure__,
        )
        if fn.__kwdefaults__:
            kwdefaults_sources = {
                k: (
                    None
                    if self.source is None
                    else DefaultsSource(self.source, k, is_kw=True)
                )
                for k in fn.__kwdefaults__
            }
            fake_func.__kwdefaults__ = {
                k: wrap(val=v, source=kwdefaults_sources[k])
                for k, v in fn.__kwdefaults__.items()
            }

        bound = inspect.signature(fake_func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())

        wrap_args_kwargs(tx, result)
        closure_cells = init_cellvars(parent, result, fn.__code__)
        closure = self.fn.__closure__ or ()
        assert len(closure) == len(self.fn.__code__.co_freevars)
        for idx, name, cell in zip(
            itertools.count(), self.fn.__code__.co_freevars, closure
        ):
            if name == "__class__":
                source = AttrSource(self.source, "__class__") if self.source else None
                result[name] = variables.UserDefinedClassVariable(
                    cell.cell_contents,
                    source=source,
                )
            else:
                var = tx.match_nested_cell(name, cell)
                if var is not None:
                    # optimization for cleaner codegen
                    result[name] = var
                elif self.source:
                    side_effects = parent.output.side_effects
                    if cell in side_effects:
                        out = side_effects[cell]
                    else:
                        closure_cell = GetItemSource(
                            AttrSource(self.source, "__closure__"), idx
                        )
                        closure_cell_contents = AttrSource(
                            closure_cell, "cell_contents"
                        )
                        try:
                            contents_var = VariableTracker.build(
                                parent, cell.cell_contents, closure_cell_contents
                            )
                        except ValueError:
                            # Cell has not yet been assigned
                            contents_var = variables.DeletedVariable()

                        if id(cell) not in tx.mutated_closure_cell_ids:
                            # Optimistically don't allocate the cell, to
                            # reduce the number of side effects.  This is
                            # important for cond, as without it, any accesses
                            # to closures create side effects and cond doesn't
                            # support side effects.  If we're wrong and this
                            # closure cell gets written to, we will restart
                            # the analysis with this cell's name in the
                            # mutated list here
                            result[name] = contents_var
                            # Map the variable to the original cell so we can
                            # look it up later, see
                            # `InliningInstructionTranslator.STORE_DEREF`.
                            tx.contents_var_to_mutated_cell[contents_var] = cell
                            continue

                        # cells are written to with "cell_contents",
                        # so the source should just be the closure_cell, not its contents
                        out = side_effects.track_cell_existing(closure_cell, cell)
                        side_effects.store_cell(
                            out,
                            contents_var,
                        )

                    result[name] = out

                else:
                    result[name] = VariableTracker.build(tx, cell.cell_contents)

        return result, closure_cells

    def export_freevars(self, parent, child):
        pass

    def var_getattr(self, tx: "InstructionTranslator", name: str):
        source = self.source and AttrSource(self.source, name)
        try:
            subobj = inspect.getattr_static(self.fn, name)
        except AttributeError:
            return variables.GetAttrVariable(self, name, source=source)
        if source:
            return variables.LazyVariableTracker.create(subobj, source)
        return VariableTracker.build(tx, subobj)

    def call_hasattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        result = hasattr(self.fn, name)
        return variables.ConstantVariable.create(result)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if self.is_constant:
            return invoke_and_store_as_constant(
                tx, self.fn, self.get_name(), args, kwargs
            )
        if (
            tx.output.current_tracer.under_activation_checkpoint
            and not tx.output.current_tracer.allow_side_effects_under_checkpoint
        ):
            try:
                from torch.distributed._composable.fsdp._fsdp_state import FSDPState
            except Exception:
                FSDPState = None
            if FSDPState is not None and self.fn in [
                FSDPState._pre_forward,
                FSDPState._post_forward,
            ]:
                with torch._dynamo.side_effects.allow_side_effects_under_checkpoint(tx):
                    return super().call_function(tx, args, kwargs)
        return super().call_function(tx, args, kwargs)


class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(self, fn, obj, **kwargs) -> None:
        super().__init__(fn=fn, **kwargs)
        self.obj = obj

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.fn}, {self.obj})"

    def self_args(self):
        return [self.obj]

    def python_type(self):
        return types.MethodType

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # For nn.Module methods, redirecting to NNModuleVariable.call_method for optimized solution
        # rather than simple inlining. E.g, putting `call_method` op in FX graph for `forward` method
        # since we ensure `forward` of allowed modules can be traced by AOT safely.
        # Note this is not only for allowed modules, as user customized modules can extend from
        # allowed modules but using parent's `forward` method, which is also covered by this branch.

        # If we are tracing the higher order op, we want Dynamo to step inside
        # the module call so that Dynamo can see the underlying parameters and
        # buffers and raise them as inputs to the graph. The is_root_tracer
        # check bypasses the if condition for non-root tracers and directly
        # calls the super().call_function at the end, which is basically
        # equivalent of inlining the method.
        if tx.output.is_root_tracer() and isinstance(
            self.obj, variables.NNModuleVariable
        ):
            module_attr = getattr(self.fn, "__module__", "")
            # inline torch.nn.utils.parametrize
            if (
                module_attr is not None
                and module_attr.startswith("torch.nn.")
                and module_attr != "torch.nn.utils.parametrize"
                or self.is_constant
            ):
                return self.obj.call_method(
                    tx, self.fn.__name__, args, kwargs, constant=self.is_constant
                )
        elif (
            _fsdp_param_group is not None
            and self.fn is _fsdp_param_group.FSDPParamGroup.use_training_state
        ):
            return variables.TorchCtxManagerClassVariable(self.fn).call_function(
                tx, (self.obj, *args), kwargs
            )
        if self.is_constant:
            fn = getattr(self.obj.value, self.fn.__name__)
            return invoke_and_store_as_constant(tx, fn, self.get_name(), args, kwargs)
        return super().call_function(tx, args, kwargs)

    def inspect_parameter_names(self):
        return super().inspect_parameter_names()[1:]


class WrappedUserMethodVariable(UserMethodVariable):
    def __init__(self, wrapped, context, **kwargs) -> None:
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        super().__init__(wrapped.fn, wrapped.obj, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result


class WrappedUserFunctionVariable(UserFunctionVariable):
    def __init__(self, wrapped, context, **kwargs) -> None:
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        super().__init__(wrapped.fn, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result


def invoke_and_store_as_constant(tx: "InstructionTranslator", fn, name, args, kwargs):
    def convert(x):
        if isinstance(x, variables.TensorVariable):
            return x.get_real_value()
        return x.as_python_constant()

    args = [convert(x) for x in args]
    kwargs = {k: convert(v) for k, v in kwargs.items()}
    res = fn(*args, **kwargs)
    return tx.output.register_attr_or_module(
        res,
        name,
        source=ConstantSource(name),
    )


class NestedUserFunctionVariable(BaseUserFunctionVariable):
    _nonvar_fields = {
        "closure_scope",
        "f_globals",
        *BaseUserFunctionVariable._nonvar_fields,
    }

    def __init__(
        self,
        fn_name,
        code,
        f_globals,
        defaults,
        kwdefaults,
        annotations,
        closure,
        closure_scope,
        wrapped_reconstructible=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert isinstance(fn_name.as_python_constant(), str)
        assert isinstance(code.as_python_constant(), types.CodeType)
        assert isinstance(f_globals, dict)
        self.fn_name = fn_name
        self.code = code
        self.f_globals = f_globals
        self.defaults = defaults
        self.kwdefaults = kwdefaults
        self.annotations = annotations
        self.closure = closure
        if closure is None:
            closure_scope = None
        self.closure_scope = closure_scope
        # Either a source or a VT with .can_reconstruct() == True
        self.wrapped_reconstructible: Optional[
            Union[Source, VariableTracker]
        ] = wrapped_reconstructible

    def self_args(self):
        return []

    def get_code(self):
        return self.code.as_python_constant()

    def get_function(self):
        if self.closure:
            raise NotImplementedError
        func = types.FunctionType(
            self.code.as_python_constant(),
            self.f_globals,
            self.fn_name.as_python_constant(),
        )
        if self.defaults:
            func.__defaults__ = self.defaults.as_python_constant()
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.as_python_constant()
        if self.annotations:
            annotations = self.annotations.as_python_constant()
            if isinstance(annotations, tuple):
                from itertools import pairwise

                annotations = dict(pairwise(annotations))

            # TypeError: __annotations__ must be set to a dict object
            assert isinstance(annotations, dict)
            func.__annotations__ = annotations
        return func

    def has_closure(self):
        return self.closure is not None

    def has_self(self):
        return False

    def get_globals(self):
        return self.f_globals

    def bind_args(self, parent, args, kwargs):
        # Avoid circular import
        from .misc import ClosureVariable, NewCellVariable

        code = self.get_code()
        func = types.FunctionType(
            code,
            self.f_globals,
            self.fn_name.as_python_constant(),
            tuple(self.defaults.items) if self.defaults else None,
            tuple(make_cell(None) for _ in range(len(self.get_code().co_freevars))),
        )
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.keys_as_python_constant()
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())
        wrap_args_kwargs(parent.output.root_tx, result)
        closure_cells = init_cellvars(parent, result, code)

        for idx, name in enumerate(code.co_freevars):
            cell = self.closure.items[idx]
            assert name not in result
            # In the regular case, a cell is either a `ClosureVariable` or
            # `NewCellVariable`.
            if isinstance(cell, (ClosureVariable, NewCellVariable)):
                closure_cells[name] = cell
            else:
                # We model unmodified cells captured by `UserFunctionVariable` as
                # their contents, in tracer's `symbolic_locals`. See
                # `UserFunctionVariable::bind_args`.
                result[name] = cell

        return result, closure_cells

    def export_freevars(self, parent, child):
        code = self.get_code()
        for var in code.co_freevars:
            if var in child.symbolic_locals:
                parent.symbolic_locals[var] = child.symbolic_locals[var]

    def reconstruct(self, codegen):
        codegen.add_push_null(
            lambda: codegen.load_import_from(__name__, "_create_nested_fn")
        )
        codegen(self.code)
        codegen.extend_output([codegen._create_load_const(self.f_globals)])
        codegen(ConstantVariable.create(self.code.value.co_name))

        if self.defaults:
            codegen(self.defaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.closure:
            codegen(self.closure)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.kwdefaults:
            codegen(self.kwdefaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        if self.annotations:
            try:
                annotations = self.annotations.as_python_constant()
                codegen.extend_output([codegen._create_load_const(annotations)])
            except NotImplementedError:
                codegen(self.annotations)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        codegen.extend_output(create_call_function(7, False))

        if self.wrapped_reconstructible:
            codegen.add_push_null(
                lambda: codegen.load_import_from("functools", "wraps")
            )
            codegen(self.wrapped_reconstructible)
            codegen.extend_output(create_call_function(1, False))
            codegen.extend_output(create_rot_n(2))
            codegen.extend_output(create_call_function(1, True))


class SkipFunctionVariable(VariableTracker):
    _nonvar_fields = {
        "value",
        "reason",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, value, reason=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value
        self.reason = reason

    def as_python_constant(self):
        return self.value

    @classmethod
    def create_with_source(cls, value, source):
        if not is_wrapper_or_member_descriptor(value):
            # These descriptors are not guaranteed to return the same object on
            # attribute lookup. They are unlikely to be changed, so we can skip
            # guarding them.
            install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        return cls(value, source=source)

    @staticmethod
    @functools.lru_cache(None)
    def fold_through_function_to_wrapper():
        return {
            collections.namedtuple: variables.UserDefinedClassVariable,
        }

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if inspect.getattr_static(self.value, "_torchdynamo_disable", False):
            unimplemented(f"call torch._dynamo.disable() wrapped function {self.value}")
        # Fold through the functions(e.g, collections.namedtuple)
        # that inputs & outputs are all python constants
        elif (
            self.value in self.fold_through_function_to_wrapper().keys()
            and check_constant_args(args, kwargs)
        ):
            value = self.value(
                *[x.as_python_constant() for x in args],
                **{k: v.as_python_constant() for k, v in kwargs.items()},
            )
            return self.fold_through_function_to_wrapper().get(self.value)(
                value, mutation_type=ValueMutationNew()
            )
        elif (
            self.value is functools.wraps
            and not kwargs
            and len(args) == 1
            and (
                args[0].source is not None or args[0].can_reconstruct(tx.output.root_tx)
            )
        ):

            def wraps(fn):
                if isinstance(fn, variables.NestedUserFunctionVariable):
                    if args[0].source:
                        reconstructible = args[0].source
                    else:
                        reconstructible = args[0]
                    return fn.clone(wrapped_reconstructible=reconstructible)
                unimplemented(f"functools.wraps({fn})")

            return variables.LambdaVariable(wraps)
        else:
            try:
                path = inspect.getfile(self.value)
                msg = f"'skip function {self.value.__qualname__} in file {path}'"
            except TypeError:
                known_python_builtin_modules = {"_abc", "_warnings"}
                if self.value.__module__ in known_python_builtin_modules:
                    msg = (
                        f"Graph break due to unsupported Python builtin {self.value.__module__}.{self.value.__qualname__}. "
                        f"Please file an issue on GitHub "
                        f"so the PyTorch team can add support for it. "
                    )
                elif (
                    self.value.__module__ is not None
                    and self.value.__module__.startswith("optree")
                ):
                    msg = (
                        f"Graph break for an optree C/C++ function {self.value.__module__}.{self.value.__qualname__}."
                        f" Consider using torch.utils._pytree - "
                        f"https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py"
                    )
                    # also warn on it because most users won't see the graph break message
                    torch._dynamo.utils.warn_once(msg)
                else:
                    msg = (
                        f"Graph break due to unsupported builtin {self.value.__module__}.{self.value.__qualname__}. "
                        f"This function is either a Python builtin (e.g. _warnings.warn) "
                        f"or a third-party C/C++ Python extension (perhaps created with pybind). "
                        f"If it is a Python builtin, please file an issue on GitHub "
                        f"so the PyTorch team can add support for it and see the next case for a workaround. "
                        f"If it is a third-party C/C++ Python extension, please "
                        f"either wrap it into a PyTorch-understood custom operator "
                        f"(see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html "
                        f"for more details) or, if it is traceable, use "
                        f"torch.compiler.allow_in_graph."
                    )
                    # also warn on it because most users won't see the graph break message
                    torch._dynamo.utils.warn_once(msg)
            if self.value.__qualname__ == "allow_in_graph":
                msg = (
                    "Found an allow_in_graph decorator to a function which "
                    "is created inside the parent function that is getting "
                    "compiled. This is not supported for now."
                )
            msg += f"', {self.reason}'" if self.reason else ""
            unimplemented(msg)


class WrapperUserFunctionVariable(VariableTracker):
    """
    Used to represent a wrapper object that contains the actual callable as an
    attribute. For example, torch.jit.script/trace have the original function at
    their _torchdynamo_inline attribute. Similarly, functions with
    __script_if_tracing_wrapper have the original attr at "__original_fn".
    """

    def __init__(self, wrapper_obj, attr_to_trace, **kwargs) -> None:
        super().__init__(**kwargs)
        self.wrapper_obj = wrapper_obj
        self.attr_to_trace = attr_to_trace

    def var_getattr(self, tx: "InstructionTranslator", name):
        if name == self.attr_to_trace:
            val = getattr(self.wrapper_obj, self.attr_to_trace)
            source = self.source and AttrSource(self.source, name)
            return VariableTracker.build(tx, val, source)

        return super().var_getattr(tx, name)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return variables.UserFunctionVariable(
            polyfills.getattr_and_trace
        ).call_function(
            tx, [self, variables.ConstantVariable(self.attr_to_trace), *args], kwargs
        )


def _traceable_collective_remaps():
    # We can't rely on importing from distributed, since it's not always built
    if torch.distributed.is_available():
        from torch.distributed._functional_collectives import (
            traceable_collective_remaps,
        )

        return traceable_collective_remaps
    return {}


def _traceable_collectives_source(tx: "InstructionTranslator", fn):
    assert torch.distributed.is_available(), "Illegal invocation."
    assert fn in _traceable_collective_remaps().values()

    inner_name = fn.__name__
    path_source = tx.import_source("torch.distributed._functional_collectives")
    return AttrSource(path_source, inner_name)


class CollectiveFunctionRewriteVariable(UserFunctionVariable):
    """
    Some of the torch.distributed.* collective APIs are possible to rewrite to 'traceable' collectives.

    This class provides both a way to check if a function is remappable, and perform the remapping.

    In the case that a function is 'remappable' but only for some combinations of call-time arguments,
    we check the args at `call_function` time and fall back to graph-breaking if needed.  This is no worse
    than status-quo as we currently graph-break on all distributed.* collectives.
    """

    def __init__(self, fn, *, replacement_var, **kwargs) -> None:
        super().__init__(fn, **kwargs)
        assert isinstance(replacement_var, UserFunctionVariable)
        self.replacement_var = replacement_var

    @staticmethod
    def create(tx: "InstructionTranslator", old_fn, source, **options):
        new_fn, new_source = CollectiveFunctionRewriteVariable.rewrite(tx, old_fn)
        return CollectiveFunctionRewriteVariable(
            old_fn,
            replacement_var=UserFunctionVariable(new_fn, source=new_source, **options),
            source=source,
            **options,
        )

    @staticmethod
    def can_rewrite(variable):
        return (
            inspect.isfunction(variable) and variable in _traceable_collective_remaps()
        )

    @staticmethod
    def rewrite(tx: "InstructionTranslator", fn):
        new_fn = _traceable_collective_remaps()[fn]
        return new_fn, _traceable_collectives_source(tx, new_fn)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        # call_function must check any unsupported arguments and graph-break.
        # It's safe to assume args/kwargs from orig_fn map 1:1 to args/kwargs of remapped_fn,
        # since that's the contract for putting a mapping in `traceable_collective_remaps`
        import torch.distributed as dist
        from torch.distributed._functional_collectives import REDUCE_OP_TO_STR

        # Merge args into kwargs so positional and keyword args
        # can be processed the same way.
        signature = inspect.signature(self.fn)
        kwargs = dict(signature.bind(*args, **kwargs).arguments)
        args = ()

        if "async_op" in kwargs and kwargs["async_op"].as_python_constant():
            unimplemented(
                f"CollectiveFunctionRewriteVariable can't support async_op=True for {self.fn}"
            )

        if self.fn in (
            dist.all_reduce,
            dist.reduce_scatter_tensor,
            dist._reduce_scatter_base,
        ):
            reduce_op_var = kwargs.get("op")
            reduce_op = (
                reduce_op_var.value
                if reduce_op_var is not None
                else signature.parameters["op"].default
            )
            if reduce_op not in REDUCE_OP_TO_STR:
                raise ValueError(f"Unsupported all_reduce op: {reduce_op}")
            kwargs["op"] = variables.ConstantVariable.create(
                REDUCE_OP_TO_STR[reduce_op]
            )
        return self.replacement_var.call_function(tx, args, kwargs)


class FunctoolsPartialVariable(VariableTracker):
    def __init__(self, func: VariableTracker, args, keywords, **kwargs) -> None:
        super().__init__(**kwargs)
        self.func = func
        assert isinstance(args, list)
        self.args = args
        assert isinstance(keywords, dict)
        self.keywords = keywords

    def reconstruct(self, codegen):
        codegen.add_push_null(lambda: codegen.load_import_from("functools", "partial"))
        codegen(self.func)
        if self.args:
            codegen.foreach(self.args)
        if not self.keywords:
            codegen.extend_output(create_call_function(len(self.args) + 1, False))
            return

        codegen.foreach(self.keywords.values())
        keys = tuple(self.keywords.keys())
        codegen.extend_output(
            codegen.create_call_function_kw(len(keys) + len(self.args) + 1, keys, False)
        )

    def get_function(self):
        return self.as_python_constant()

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        merged_args = self.args + args
        merged_kwargs = {**self.keywords, **kwargs}
        return self.func.call_function(tx, merged_args, merged_kwargs)

    def call_hasattr(self, tx: "InstructionTranslator", name: str) -> VariableTracker:
        # functools.partial uses slots, so attributes are constant
        return variables.ConstantVariable.create(
            hasattr(functools.partial(identity), name)
        )

    def as_python_constant(self):
        return functools.partial(
            self.func.as_python_constant(),
            *[arg.as_python_constant() for arg in self.args],
            **{k: v.as_python_constant() for k, v in self.keywords.items()},
        )

    def guard_as_python_constant(self):
        """Similar to as_python_constant(), but add ID_MATCH guards to try to force things to become constants"""
        return functools.partial(
            self.func.guard_as_python_constant(),
            *[v.guard_as_python_constant() for v in self.args],
            **{k: v.guard_as_python_constant() for k, v in self.keywords.items()},
        )


class PolyfilledFunctionVariable(VariableTracker):
    _nonvar_fields = {
        "fn",
        "wrapped_fn",
        "traceable_fn",
        *VariableTracker._nonvar_fields,
    }

    @classmethod
    @functools.lru_cache(None)
    def _get_polyfill_handlers(cls) -> Dict[Callable[..., Any], types.FunctionType]:
        return {}

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))

        return cls(value, source=source)

    def __init__(self, fn: _F, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fn: _F = fn

        handler = self._get_polyfill_handlers().get(fn, fn)
        assert callable(handler), f"Polyfill handler {handler} is not callable for {fn}"
        for candidate_attr in (
            "__torch_dynamo_polyfill__",  # registered polyfill
            "__python_implementation__",  # self handler from third-party libraries
        ):
            candidate = getattr(handler, candidate_attr, None)
            if candidate:
                assert callable(candidate)
                traceable_fn = candidate
                break
        else:
            raise RuntimeError(
                f"Polyfill handler {handler} does not have a traceable function"
            )

        self.wrapped_fn: _F = handler
        self.traceable_fn: _F = traceable_fn

    @property
    def polyfill_fn(self) -> _F:
        return self.traceable_fn

    def can_constant_fold_through(self):
        return getattr(
            self.wrapped_fn, "__torch_dynamo_can_constant_fold_through__", False
        )

    def get_function(self):
        return self.as_python_constant()

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if self.can_constant_fold_through() and check_unspec_or_constant_args(
            args, kwargs
        ):
            result = (
                self.fn(  # use the original function which is faster than the polyfill
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                )
            )
            return VariableTracker.build(tx, result)

        # Special case for sum on tuple/list of ints
        if (
            self.fn is builtins.sum
            and len(args) == 1
            and not kwargs
            and isinstance(args[0], (variables.ListVariable, variables.TupleVariable))
            and all(
                (isinstance(x, variables.ConstantVariable) and isinstance(x.value, int))
                or (isinstance(x, variables.SymNodeVariable) and x.python_type() is int)
                for x in args[0].items
            )
        ):
            return variables.SymNodeVariable.create(
                tx,
                tx.output.create_proxy(
                    "call_function",
                    torch.sym_sum,
                    (tuple(a.as_proxy() for a in args[0].items),),
                    {},
                ),
                sym_num=torch.sym_sum(
                    [
                        (
                            x.value
                            if isinstance(x, variables.ConstantVariable)
                            else x.sym_num
                        )
                        for x in args[0].items
                    ]
                ),
            )

        traceable_function_variable = VariableTracker.build(tx, self.traceable_fn)
        return traceable_function_variable.call_function(tx, args, kwargs)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__call__":
            return self.call_function(tx, args, kwargs)

        method = getattr(self.fn, name, None)
        assert method is not None, f"Member {name} not found in {self.fn}"
        assert is_function(method), f"Member {name} is not callable in {self.fn}"
        options = {}
        if self.source:
            options["source"] = AttrSource(self.source, name)
        polyfilled_method_variable = PolyfilledFunctionVariable(method, **options)
        return polyfilled_method_variable.call_function(tx, args, kwargs)

    def as_python_constant(self):
        return self.fn


from torch._higher_order_ops.triton_kernel_wrap import (
    TMADescriptorMetadata,
    TritonHOPifier,
)


class DynamoTritonHOPifier(TritonHOPifier):
    def raise_unsupported(self, msg: str) -> Never:
        raise Unsupported(msg)

    def is_callable(self, maybe_callable: Any) -> bool:
        return isinstance(
            maybe_callable, (NestedUserFunctionVariable, UserFunctionVariable)
        )

    def get_value(self, val: Any) -> Any:
        return val.value

    def check_grid(self, grid) -> Tuple[torch.fx.proxy.Proxy, ...]:
        from .lists import BaseListVariable

        if isinstance(grid, BaseListVariable):
            return grid.as_proxy()
        else:
            unimplemented(f"grid for the triton kernel is {type(grid)}")

    def call_grid(self, grid, meta, tx):
        meta = {variables.ConstantVariable.create(k): v for k, v in meta.items()}
        grid = grid.call_function(tx, [meta], {})
        return grid

    def call_HOP(self, variable, grids, combined_args_raw, tx) -> ConstantVariable:
        from .constant import ConstantVariable
        from .dicts import ConstDictVariable

        # as we can only pass tensors as non-const args in fx graph,
        # here we replace TMA descriptors (TMADescriptorVariable
        # instances) with the underlying tensors, while moving the
        # TMA descriptor-related metadata to a separate argument,
        # so that we can reconstruct the TMA descriptors downstream
        tma_descriptor_metadata: TMADescriptorMetadata = {}
        for k in list(combined_args_raw.keys()):
            v = combined_args_raw[k]
            if isinstance(v, TMADescriptorVariable):
                tma_descriptor_metadata[k] = v.to_metadata()
                combined_args_raw[k] = v.data_ptr.from_tensor

        combined_args = {
            variables.ConstantVariable.create(k): v
            for k, v in combined_args_raw.items()
        }

        from torch._higher_order_ops.triton_kernel_wrap import (
            kernel_side_table,
            triton_kernel_wrapper_mutation,
        )

        # Combine args and kwargs and pass as a dict so that if user defined triton
        # kernel uses variables as 'grid' or 'kernel', it does not conflict with
        # parameters of the wrapper function
        constant_args = {
            k: v.as_python_constant()
            for k, v in combined_args_raw.items()
            if isinstance(v, ConstantVariable)
        }
        non_constant_args = {
            k: v
            for k, v in combined_args.items()
            if not isinstance(v, ConstantVariable)
        }

        for v in non_constant_args.values():
            v = v.realize()
            if not isinstance(v, (variables.TensorVariable, variables.SymNodeVariable)):
                self.raise_unsupported(
                    f"Unexpected argument type for a Triton kernel: {repr(v)}."
                )

        constant_args_idx = kernel_side_table.add_constant_args(constant_args)
        meta = ConstDictVariable(non_constant_args, dict)
        tx.output.create_proxy(
            "call_function",
            triton_kernel_wrapper_mutation,
            (),
            {
                "kernel_idx": variable.kernel_idx,
                "constant_args_idx": constant_args_idx,
                "grid": grids,
                "tma_descriptor_metadata": tma_descriptor_metadata,
                "kwargs": meta.as_proxy(),
            },
        )

        return variables.ConstantVariable(
            None,
        )


dynamo_triton_hopifier_singleton = DynamoTritonHOPifier()


class TritonKernelVariable(VariableTracker):
    grid: "TritonGridType"
    kernel: "TritonKernelType"
    kernel_idx: Optional[int]

    def __init__(self, kernel, kernel_idx, grid, **kwargs) -> None:
        super().__init__(**kwargs)
        dynamo_triton_hopifier_singleton.init_variable(self, kernel, kernel_idx, grid)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        return dynamo_triton_hopifier_singleton.call_triton_kernel(
            self, args, kwargs, tx
        )

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__getitem__":
            return dynamo_triton_hopifier_singleton.call_getitem(self, args)
        elif name == "run":
            return dynamo_triton_hopifier_singleton.call_run(self, args, kwargs, tx)

        # Bail out to parent's implementation
        return super().call_method(tx, name, args, kwargs)

    def specialize_symbolic(self, arg: Any) -> Any:
        from .constant import ConstantVariable
        from .tensor import SymNodeVariable

        # See [Note: Specialize tl.constexpr args in user-defined triton kernels]
        if isinstance(arg, SymNodeVariable):
            return ConstantVariable.create(arg.evaluate_expr())
        return arg


class TMADescriptorVariable(VariableTracker):
    def __init__(
        self,
        data_ptr: "variables.DataPtrVariable",
        dims: "List[ConstantVariable]",
        block_dims: "List[ConstantVariable]",
        element_size: "ConstantVariable",
        **kwargs,
    ):
        assert isinstance(data_ptr, variables.DataPtrVariable)
        super().__init__(**kwargs)
        self.data_ptr = data_ptr
        self.dims = dims
        self.block_dims = block_dims
        self.element_size = element_size

    def to_metadata(self):
        return (
            [dim.as_proxy() for dim in self.dims],
            [dim.as_proxy() for dim in self.block_dims],
            self.element_size.as_proxy(),
        )

    def reconstruct(self, codegen):
        codegen.add_push_null(
            lambda: codegen.load_import_from(
                "triton.tools.experimental_descriptor",
                f"create_{len(self.dims)}d_tma_descriptor",
            )
        )
        self.data_ptr.reconstruct(codegen)
        args = [*self.dims, *self.block_dims, self.element_size]
        codegen.foreach(args)
        codegen.call_function(len(args) + 1, False)


class CreateTMADescriptorVariable(VariableTracker):
    def __init__(
        self,
        rank: int,
        **kwargs,
    ) -> None:
        assert rank in (1, 2)
        super().__init__(**kwargs)
        self.rank = rank

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        ptr = kwargs["ptr"] if "ptr" in kwargs else args[0]

        if not isinstance(ptr, variables.DataPtrVariable):
            raise Unsupported(
                "Please ensure there were no graph breaks between "
                f"create_{self.rank}d_tma_descriptor and the upstream "
                ".data_ptr() call."
            )

        if self.rank == 1:
            assert len(args) + len(kwargs) == 4
            dims = [
                kwargs["dim"] if "dim" in kwargs else args[1],
            ]
            block_dims = [
                kwargs["block_dim"] if "block_dim" in kwargs else args[2],
            ]
        else:
            assert len(args) + len(kwargs) == 6
            dims = [
                kwargs["dim1"] if "dim1" in kwargs else args[1],
                kwargs["dim0"] if "dim0" in kwargs else args[2],
            ]
            block_dims = [
                kwargs["block_dim1"] if "block_dim1" in kwargs else args[3],
                kwargs["block_dim0"] if "block_dim0" in kwargs else args[4],
            ]
        element_size = kwargs["element_size"] if "element_size" in kwargs else args[-1]

        return TMADescriptorVariable(
            data_ptr=ptr,
            dims=dims,
            block_dims=block_dims,
            element_size=element_size,
        )
