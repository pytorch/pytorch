# mypy: ignore-errors

import collections
import functools
import inspect
import itertools
import types
from typing import Dict, List, Optional, TYPE_CHECKING, Union

import torch

from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import check_constant_args, get_first_attr, identity, istype, make_cell
from .base import MutableLocal, typestr, VariableTracker
from .constant import ConstantVariable
from .distributed import ProcessGroupVariable

if TYPE_CHECKING:
    from torch._guards import Source


def wrap_bound_arg(tx, val, source=None):
    # Source propagation is best effort since not every object we encounter has a source to begin with.
    if isinstance(val, VariableTracker):
        return val
    elif not source:
        from torch._dynamo.variables.builder import SourcelessBuilder

        return SourcelessBuilder()(tx, val)
    else:
        # Create a lazy variable to avoid guarding on __defaults__ unless really
        # needed.
        return variables.LazyVariableTracker.create(val, source)


def wrap_args_kwargs(tx, result):
    for k, v in list(result.items()):
        if isinstance(v, (tuple, dict)):
            # args/kwargs
            result[k] = wrap_bound_arg(tx, v)


def init_cellvars(parent, result, code):
    closure_cells = dict()
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
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        return tx.inline_user_function_return(
            self, list(self.self_args()) + list(args), kwargs
        )

    def call_hasattr(self, tx, name: str) -> VariableTracker:
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

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.CLOSURE_MATCH))
        return cls(
            value,
            source=source,
        )

    def __init__(self, fn, is_constant=False, **kwargs):
        super().__init__(**kwargs)
        if getattr(fn, "_dynamo_marked_constant", False):
            # This method should be treated as a constant for the purposes of compilation
            self.is_constant = True
        else:
            self.is_constant = False

        assert isinstance(
            fn, (types.FunctionType, torch.jit.ScriptFunction)
        ), f"expected FunctionType found {typestr(fn)} {fn}"
        # unpack @torch._dynamo.optimize()(fn) wrapped function
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)
        # unpack torch.jit.script_if_tracing
        if inspect.getattr_static(fn, "__script_if_tracing_wrapper", False):
            fn = inspect.getattr_static(fn, "__original_fn", fn)
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
                k: None
                if self.source is None
                else DefaultsSource(self.source, k, is_kw=True)
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
                    from .builder import VariableBuilder

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
                            contents_var = VariableBuilder(
                                parent, closure_cell_contents
                            )(cell.cell_contents)
                        except ValueError:
                            # Cell has not yet been assigned
                            contents_var = variables.DeletedVariable()

                        if (
                            closure_cell_contents.name()
                            not in tx.mutated_closure_cell_contents
                        ):
                            # Optimistically don't allocate the cell, to
                            # reduce the number of side effects.  This is
                            # important for cond, as without it, any accesses
                            # to closures create side effects and cond doesn't
                            # support side effects.  If we're wrong and this
                            # closure cell gets written to, we will restart
                            # the analysis with this cell's name in the
                            # mutated list here
                            result[name] = contents_var
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
                    from .builder import SourcelessBuilder

                    result[name] = SourcelessBuilder()(tx, cell.cell_contents)

        return result, closure_cells

    def export_freevars(self, parent, child):
        pass

    def call_hasattr(self, tx, name: str) -> VariableTracker:
        result = hasattr(self.fn, name)
        return variables.ConstantVariable.create(result)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        if self.is_constant:
            return invoke_and_store_as_constant(
                tx, self.fn, self.get_name(), args, kwargs
            )

        return super().call_function(tx, args, kwargs)


class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(self, fn, obj, **kwargs):
        super().__init__(fn=fn, **kwargs)
        self.obj = obj

    def __str__(self):
        return f"{self.__class__.__name__}({self.fn}, {self.obj})"

    def self_args(self):
        return [self.obj]

    def python_type(self):
        return types.MethodType

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
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
            if (
                module_attr is not None
                and module_attr.startswith("torch.nn.")
                or self.is_constant
            ):
                return self.obj.call_method(
                    tx, self.fn.__name__, args, kwargs, constant=self.is_constant
                )
        return super().call_function(tx, args, kwargs)

    def inspect_parameter_names(self):
        return super().inspect_parameter_names()[1:]


class WrappedUserMethodVariable(UserMethodVariable):
    def __init__(self, wrapped, context, **kwargs):
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        super().__init__(wrapped.fn, wrapped.obj, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result


class WrappedUserFunctionVariable(UserFunctionVariable):
    def __init__(self, wrapped, context, **kwargs):
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        super().__init__(wrapped.fn, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result


def invoke_and_store_as_constant(tx, fn, name, args, kwargs):
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
    ):
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
            raise NotImplementedError()
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
        from .misc import InlinedClosureVariable

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
            assert getattr(cell, name, name) == name
            assert name not in result
            if isinstance(cell, InlinedClosureVariable):
                # InlinedClosureVariable's are created from LOAD_CLOSURE's from
                # InliningInstructionTranslators when the variable name is not found in closure_cells.
                # They should remain outside of closure_cells, so that our callee (the
                # InliningInstructionTranslator that traces `func`) handles
                # the cell correctly - that is, the cell's contents are treated as if they
                # are local variables, like in UserFunctionVariable's bind_args for freevars.
                cand = parent
                while cand and name not in cand.symbolic_locals:
                    cand = cand.parent
                if cand is None:
                    raise RuntimeError(
                        f"Couldn't find {name} in the symbolic_locals of the inline interpreter stack"
                    )
                result[name] = cand.symbolic_locals[name]
            else:
                closure_cells[name] = self.closure.items[idx]

        return result, closure_cells

    def export_freevars(self, parent, child):
        code = self.get_code()
        for var in code.co_freevars:
            if var in child.symbolic_locals:
                parent.symbolic_locals[var] = child.symbolic_locals[var]

    def reconstruct(self, codegen):
        codegen.load_import_from(__name__, "_create_nested_fn")
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

        codegen.extend_output(create_call_function(7, push_null=True))

        if self.wrapped_reconstructible:
            codegen.load_import_from("functools", "wraps")
            codegen(self.wrapped_reconstructible)
            codegen.extend_output(create_call_function(1, True))
            codegen.extend_output(create_rot_n(2))
            codegen.extend_output(create_call_function(1, True))


class SkipFunctionVariable(VariableTracker):
    def __init__(self, value, reason=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.reason = reason

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        return cls(
            value,
            source=source,
        )

    @staticmethod
    @functools.lru_cache(None)
    def fold_through_function_to_wrapper():
        return {
            collections.namedtuple: variables.UserDefinedClassVariable,
        }

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
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
                value, mutable_local=MutableLocal()
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
            except TypeError:
                path = f"Builtin {self.value.__name__}"
            msg = f"'skip function {self.value.__qualname__} in file {path}'"
            msg += f"', {self.reason}'" if self.reason else ""
            unimplemented(msg)


def _traceable_collective_remaps():
    # We can't rely on importing from distributed, since it's not always built
    if torch.distributed.is_available():
        from torch.distributed._functional_collectives import (
            traceable_collective_remaps,
        )

        return traceable_collective_remaps
    return {}


def _traceable_collectives_source(tx, fn):
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

    def __init__(self, fn, *, replacement_var, **kwargs):
        super().__init__(fn, **kwargs)
        assert isinstance(replacement_var, UserFunctionVariable)
        self.replacement_var = replacement_var

    @staticmethod
    def create(tx, old_fn, source, **options):
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
    def rewrite(tx, fn):
        new_fn = _traceable_collective_remaps()[fn]
        return new_fn, _traceable_collectives_source(tx, new_fn)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
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

        if kwargs.get("group") is None or kwargs["group"].value is None:
            kwargs["group"] = ProcessGroupVariable.get_global_pg_variable()

        if self.fn == dist.all_reduce:
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
    def __init__(self, func: VariableTracker, args, keywords, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        assert isinstance(args, list)
        self.args = args
        assert isinstance(keywords, dict)
        self.keywords = keywords

    def reconstruct(self, codegen):
        codegen.load_import_from("functools", "partial")
        codegen(self.func)
        if self.args:
            codegen.foreach(self.args)
        if not self.keywords:
            codegen.extend_output(create_call_function(len(self.args) + 1, True))
            return

        codegen.foreach(self.keywords.values())
        keys = tuple(self.keywords.keys())
        codegen.extend_output(
            codegen.create_call_function_kw(len(keys) + len(self.args) + 1, keys, True)
        )

    def get_function(self):
        return self.as_python_constant()

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        merged_args = self.args + args
        merged_kwargs = {**self.keywords, **kwargs}
        return self.func.call_function(tx, merged_args, merged_kwargs)

    def call_hasattr(self, tx, name: str) -> VariableTracker:
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


class TritonKernelVariable(VariableTracker):
    def __init__(self, kernel, kernel_idx, grid, **kwargs):
        from triton.runtime.autotuner import Autotuner

        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        super().__init__(**kwargs)

        assert kernel is not None

        self.kernel = kernel
        self.kernel_idx = kernel_side_table.add_kernel(kernel)

        assert kernel_idx is None or self.kernel_idx == kernel_idx

        self.grid = grid

        if isinstance(kernel, Autotuner):
            # We only support configs and keys arguments of triton.autotune
            # Make sure other arguments are defaulted
            defaults = inspect.signature(Autotuner.__init__).parameters

            # Newer version of triton change attribute name from warmup to num_warmup and rep to num_rep.
            # The call to get_first_attr is to maintain backward-compatibility.
            if (
                (
                    "warmup" in defaults
                    and defaults["warmup"].default
                    != get_first_attr(kernel, "num_warmups", "warmup")
                )
                or (
                    "rep" in defaults
                    and defaults["rep"].default
                    != get_first_attr(kernel, "num_reps", "rep")
                )
                or (
                    "prune_configs_by" in defaults
                    and defaults["prune_configs_by"].default
                    != kernel.early_config_prune
                )
                # Set via reset_to_zero argument
                or len(kernel.reset_idx) != 0
                or len(kernel.restore_idx) != 0
            ):
                raise Unsupported(
                    "Only configs and keys are supported for triton.autotune"
                )

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        from triton.runtime.autotuner import Autotuner

        from .constant import ConstantVariable
        from .dicts import ConstDictVariable
        from .lists import BaseListVariable

        if self.grid is None:
            raise Unsupported("Triton kernels should always be called with a grid")

        # Both for grid's meta as well as for the kernel, we need combined
        # args and kwargs normalized
        names = (
            variables.ConstantVariable.create(name) for name in self.kernel.arg_names
        )
        kwargs = {variables.ConstantVariable.create(k): v for k, v in kwargs.items()}
        normalized_args = {**dict(zip(names, args)), **kwargs}

        configs = (
            [config.kwargs for config in self.kernel.configs]
            if isinstance(self.kernel, Autotuner)
            else [{}]
        )
        grids = []
        for config_args in configs:
            # If the grid is a function, then lets execute it and convert it to
            # a list
            grid = self.grid
            if isinstance(grid, (NestedUserFunctionVariable, UserFunctionVariable)):
                # Populate the special "meta" argument to call the grid function
                config_args = {
                    ConstantVariable.create(k): ConstantVariable.create(v)
                    for k, v in config_args.items()
                }
                meta = ConstDictVariable({**normalized_args, **config_args}, dict)
                grid = grid.call_function(tx, [meta], {})

            # Now, the grid must be a list either originally or through above
            # modification
            if isinstance(grid, BaseListVariable):
                grids.append(grid.as_proxy())
            else:
                unimplemented(f"grid for the triton kernel is {type(grid)}")

        for i in range(len(grids)):
            if not isinstance(grids[i], tuple):
                raise Unsupported("Only tuple grids are supported")
            # inductor expects all grids to be 3-tuple so lets make it
            if len(grids[i]) == 1:
                grids[i] = (grids[i][0], 1, 1)
            elif len(grids[i]) == 2:
                grids[i] = (grids[i][0], grids[i][1], 1)
            elif len(grids[i]) > 3:
                raise Unsupported("Grid can have at most rank 3")

        assert len(grids) != 0
        if len(set(grids)) == 1:
            # If there's only one unique grid, lets simplify
            grids = [grids[0]]

        from torch._higher_order_ops.triton_kernel_wrap import (
            triton_kernel_wrapper_mutation,
        )

        # Combine args and kwargs and pass as a dict so that if user defined triton
        # kernel uses variables as 'grid' or 'kernel', it does not conflict with
        # parameters of the wrapper function
        meta = ConstDictVariable(normalized_args, dict)
        tx.output.create_proxy(
            "call_function",
            triton_kernel_wrapper_mutation,
            (),
            {
                "kernel_idx": self.kernel_idx,
                "grid": grids,
                "kwargs": meta.as_proxy(),
            },
        )

        return variables.ConstantVariable(
            None,
        )

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__getitem__":
            # __getitem__ should only be called if we don't already have a grid
            # Only grid needs to be passed
            if self.grid is not None or len(args) != 1:
                raise Unsupported(
                    "Triton kernels should be called with only a single grid"
                )

            return TritonKernelVariable(
                kernel=self.kernel,
                kernel_idx=self.kernel_idx,
                grid=args[0],
            )
        elif name == "run":
            if "grid" not in kwargs:
                raise Unsupported("Triton kernel requires to be called with a grid")
            grid = kwargs.pop("grid")
            kwargs.pop("warmup", None)
            # rewrite kernel.run(*args, grid=grid) to kernel[grid](*args)
            return TritonKernelVariable(
                kernel=self.kernel, kernel_idx=self.kernel_idx, grid=grid
            ).call_function(tx, args, kwargs)

        # Bail out to parent's implementation
        return super().call_method(tx, name, args, kwargs)
