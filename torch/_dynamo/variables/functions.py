import abc
import enum
import functools
import inspect
import itertools
import types
from typing import Dict, List

from .. import variables
from ..bytecode_transformation import create_instruction
from ..exc import unimplemented
from ..source import AttrSource, GetItemSource
from ..utils import make_cell
from .base import typestr, VariableTracker


def wrap_bound_arg(val, options):
    if isinstance(val, dict):
        return variables.ConstDictVariable(
            {k: wrap_bound_arg(v, options) for k, v in val.items()}, dict, **options
        )
    elif isinstance(val, (tuple, list)):
        cls = variables.BaseListVariable.cls_for(type(val))
        return cls([wrap_bound_arg(x, options) for x in val], **options)
    elif variables.ConstantVariable.is_literal(val):
        return variables.ConstantVariable(val, **options)
    elif isinstance(val, enum.Enum):
        return variables.EnumVariable(val, **options)
    elif isinstance(val, (type, abc.ABCMeta)):
        return variables.UserDefinedClassVariable(val, **options)
    else:
        assert isinstance(val, VariableTracker), typestr(val)
        return val


def wrap_args_kwargs(result, options):
    for k, v in list(result.items()):
        if isinstance(v, (tuple, dict)):
            # args/kwargs
            result[k] = wrap_bound_arg(v, options)


def init_cellvars(parent, result, code):
    closure_cells = dict()
    side_effects = parent.output.side_effects

    for name in code.co_cellvars:
        closure_cells[name] = side_effects.track_cell_new()
        if name in result:
            side_effects.store_cell(closure_cells[name], result.pop(name))

    return closure_cells


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
        super(UserFunctionVariable, self).__init__(**kwargs)
        if getattr(fn, "_dynamo_marked_constant", False):
            # This method should be treated as a constant for the purposes of compilation
            self.is_constant = True
        else:
            self.is_constant = False

        assert isinstance(
            fn, types.FunctionType
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
        wrap = functools.partial(wrap_bound_arg, options=options)

        fn: types.FunctionType = self.fn
        fake_func = types.FunctionType(
            fn.__code__,
            fn.__globals__,
            fn.__name__,
            tuple(map(wrap, fn.__defaults__ or [])),
            fn.__closure__,
        )
        if fn.__kwdefaults__:
            fake_func.__kwdefaults__ = {
                k: wrap(v) for k, v in fn.__kwdefaults__.items()
            }

        bound = inspect.signature(fake_func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())

        wrap_args_kwargs(result, options)
        closure_cells = init_cellvars(parent, result, fn.__code__)
        closure = self.fn.__closure__ or ()
        assert len(closure) == len(self.fn.__code__.co_freevars)
        for idx, name, cell in zip(
            itertools.count(), self.fn.__code__.co_freevars, closure
        ):
            if name == "__class__":
                result[name] = variables.UserDefinedClassVariable(cell.cell_contents)
            else:
                var = parent.output.root_tx.match_nested_cell(name, cell)
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

                        # cells are written to with "cell_contents",
                        # so the source should just be the closure_cell, not its contents
                        out = side_effects.track_cell_existing(closure_cell, cell)
                        side_effects.store_cell(
                            out,
                            VariableBuilder(parent, closure_cell_contents)(
                                cell.cell_contents
                            ),
                        )

                    result[name] = out

                else:
                    unimplemented("inline with __closure__")

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

        return super(UserFunctionVariable, self).call_function(tx, args, kwargs)


class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(self, fn, obj, **kwargs):
        super(UserMethodVariable, self).__init__(fn=fn, **kwargs)
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
        if (
            isinstance(self.obj, variables.NNModuleVariable)
            and getattr(self.fn, "__module__", "").startswith("torch.nn.")
            or self.is_constant
        ):
            return self.obj.call_method(
                tx, self.fn.__name__, args, kwargs, constant=self.is_constant
            ).add_options(self)
        return super().call_function(tx, args, kwargs)

    def num_parameters(self):
        return super(UserMethodVariable, self).num_parameters() - 1


class WrappedUserMethodVariable(UserMethodVariable):
    def __init__(self, wrapped, context, **kwargs):
        kwargs.pop("fn", None)
        kwargs.pop("obj", None)
        super(WrappedUserMethodVariable, self).__init__(
            wrapped.fn, wrapped.obj, **kwargs
        )
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
        super(WrappedUserFunctionVariable, self).__init__(wrapped.fn, **kwargs)
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
        **kwargs,
    ):
        super(NestedUserFunctionVariable, self).__init__(**kwargs)
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

        wrap_args_kwargs(result, VariableTracker.propagate(self))
        closure_cells = init_cellvars(parent, result, code)

        for idx, name in enumerate(code.co_freevars):
            assert getattr(self.closure.items[idx], name, name) == name
            assert name not in result
            closure_cells[name] = self.closure.items[idx]

        return result, closure_cells

    def export_freevars(self, parent, child):
        code = self.get_code()
        for var in code.co_freevars:
            if var in child.symbolic_locals:
                parent.symbolic_locals[var] = child.symbolic_locals[var]

    def reconstruct(self, codegen):
        flags = 0x00
        if self.defaults:
            flags |= 0x01
            codegen(self.defaults)
        if self.kwdefaults:
            flags |= 0x02
            codegen(self.kwdefaults)
        if isinstance(self.annotations, variables.ConstDictVariable) or isinstance(
            self.annotations, variables.TupleVariable
        ):
            flags |= 0x04
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
        if self.closure:
            flags |= 0x08
            codegen(self.closure)
        codegen(self.code)
        codegen(self.fn_name)
        return [create_instruction("MAKE_FUNCTION", flags)]
