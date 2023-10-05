import functools
import inspect
import itertools
import types
from typing import Dict, List

import torch

from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented
from ..source import (
    AttrSource,
    ConstantSource,
    DefaultsSource,
    GetItemSource,
    GlobalSource,
)
from ..utils import make_cell
from .base import typestr, VariableTracker


def wrap_bound_arg(tx, val, options, source=None):
    # Source propagation is best effort since not every object we encounter has a source to begin with.
    assert (
        "source" not in options
    ), "Source needs to be separate from options due to recursive calls for lists/dicts"
    if isinstance(val, VariableTracker):
        return val
    elif not source:
        from torch._dynamo.variables.builder import SourcelessBuilder

        return SourcelessBuilder()(tx, val).add_options(options)
    else:
        from torch._dynamo.variables.builder import VariableBuilder

        return VariableBuilder(tx, source=source)(val).add_options(options)


def wrap_args_kwargs(tx, result, options):
    for k, v in list(result.items()):
        if isinstance(v, (tuple, dict)):
            # args/kwargs
            result[k] = wrap_bound_arg(tx, v, options)


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

    def num_parameters(self):
        return len(inspect.signature(self.get_function()).parameters)

    def closure_vars(self, tx):
        return {}


class UserFunctionVariable(BaseUserFunctionVariable):
    """Some unsupported user-defined global function"""

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
        options = VariableTracker.propagate([self])
        tx = parent.output.root_tx
        wrap = functools.partial(wrap_bound_arg, tx=tx, options=options)

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

        wrap_args_kwargs(tx, result, options)
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
                        contents_var = VariableBuilder(parent, closure_cell_contents)(
                            cell.cell_contents
                        )

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

                    result[name] = SourcelessBuilder()(
                        tx, cell.cell_contents
                    ).add_options(options)

        return result, closure_cells

    def export_freevars(self, parent, child):
        pass

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        if self.is_constant:
            options = VariableTracker.propagate(self, args, kwargs.values())
            return invoke_and_store_as_constant(
                tx, self.fn, self.get_name(), options, args, kwargs
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
                ).add_options(self)
        return super().call_function(tx, args, kwargs)

    def num_parameters(self):
        return super().num_parameters() - 1


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


def invoke_and_store_as_constant(tx, fn, name, options, args, kwargs):
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
        **options,
    )


class NestedUserFunctionVariable(BaseUserFunctionVariable):
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
        wraps_source=None,
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
        self.wraps_source = wraps_source

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
            func.__kwdefaults__ = self.kwdefaults.items
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())
        wrap_args_kwargs(parent.output.root_tx, result, VariableTracker.propagate(self))
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
        codegen(self.fn_name)

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
                if isinstance(self.annotations, variables.ConstDictVariable):
                    annotations = {
                        k: v.as_python_constant()
                        for k, v in self.annotations.items.items()
                    }
                else:
                    annotations = tuple(
                        [v.as_python_constant() for v in self.annotations.items]
                    )
                codegen.extend_output([codegen._create_load_const(annotations)])
            except NotImplementedError:
                codegen(self.annotations)
        else:
            codegen.extend_output([codegen.create_load_const(None)])

        codegen.extend_output(create_call_function(7, push_null=True))

        if self.wraps_source:
            codegen.load_import_from("functools", "wraps")
            codegen(self.wraps_source)
            codegen.extend_output(create_call_function(1, True))
            codegen.extend_output(create_rot_n(2))
            codegen.extend_output(create_call_function(1, True))

        return []


def _traceable_collective_remaps():
    # We can't rely on importing from distributed, since its not always built
    if torch.distributed.is_available():
        from torch.distributed._functional_collectives import (
            traceable_collective_remaps,
        )

        return traceable_collective_remaps
    return {}


def _traceable_collectives_source(fn):
    assert torch.distributed.is_available(), "Illegal invocation."
    from torch.distributed._functional_collectives import (
        all_gather_tensor_inplace,
        reduce_scatter_tensor_inplace,
    )

    valid_values = {all_gather_tensor_inplace, reduce_scatter_tensor_inplace}
    assert fn in valid_values
    inner_name = fn.__name__
    path_source = AttrSource(
        base=AttrSource(base=GlobalSource(global_name="torch"), member="distributed"),
        member="_functional_collectives",
    )
    return AttrSource(path_source, inner_name)


class CollectiveFunctionRewriteVariable(UserFunctionVariable):
    """
    Some of the torch.distributed.* collective APIs are possible to rewrite to 'traceable' collectives.

    This class provides both a way to check if a function is remappable, and perform the remapping.

    In the case that a function is 'remappable' but only for some combinations of call-time arguments,
    we check the args at `call_function` time and fall back to graph-breaking if needed.  This is no worse
    than status-quo as we currently graph-break on all distributed.* collectives.
    """

    def __init__(self, fn, *, orig_fn, orig_source, **kwargs):
        # orig_fn lets us implement any fn-specific args/kwargs restrictions inside call_function
        self.orig_fn = orig_fn
        self.orig_source = orig_source

        # remapped_fn gets stuffed in self.fn and used in super().call_function
        super().__init__(fn, **kwargs)

    @staticmethod
    def can_rewrite(variable):
        return (
            inspect.isfunction(variable) and variable in _traceable_collective_remaps()
        )

    @staticmethod
    def rewrite(fn):
        new_fn = _traceable_collective_remaps()[fn]
        return new_fn, _traceable_collectives_source(new_fn)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        # call_function must check any unsupported arguments and graph-break.
        # It's safe to assume args/kwargs from orig_fn map 1:1 to args/kwargs of remapped_fn,
        # since that's the contract for putting a mapping in `traceable_collective_remaps`
        if kwargs.get("async_op", False):
            # Put the old source back, this function will always graph break, but this ensures
            # we produce the correct guards.
            self.source = self.orig_source
            unimplemented(
                f"CollectiveFunctionRewriteVariable can't support async_op=True for {self.orig_fn}"
            )
        return super().call_function(tx, args, kwargs)


class FunctoolsPartialVariable(VariableTracker):
    def __init__(self, func, args, keywords, original=None, **kwargs):
        super().__init__(**kwargs)
        self.func = func
        assert isinstance(args, list)
        self.args = args
        assert isinstance(keywords, dict)
        self.keywords = keywords
        self.original = original

        self.guards.update(VariableTracker.propagate(func)["guards"])
        for arg in args:
            self.guards.update(VariableTracker.propagate(arg)["guards"])
        for val in keywords.values():
            self.guards.update(VariableTracker.propagate(val)["guards"])

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        options = VariableTracker.propagate([self])
        merged_args = self.args + args
        merged_kwargs = {**self.keywords, **kwargs}

        return self.func.call_function(tx, merged_args, merged_kwargs).add_options(
            options
        )

    def as_python_constant(self):
        if self.original:
            return self.original
        else:
            return functools.partial(
                self.func.fn,
                *[arg.as_python_constant for arg in self.args],
                **{k: v.as_python_constant() for k, v in self.keywords.items()},
            )
