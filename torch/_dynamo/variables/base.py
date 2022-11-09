import collections
import functools
import numbers
import operator
from typing import Any, Callable, Dict, List, Optional, Set

import torch
from torch.fx.immutable_collections import immutable_list

from .. import variables
from ..exc import unimplemented
from ..source import AttrSource, Source
from ..utils import (
    clone_input,
    dict_values,
    fake_tensors_available,
    get_fake_value,
    get_real_value,
    identity,
    istype,
    odict_values,
    preserve_rng_state,
    wrap_to_fake_tensor_and_record,
)


class MutableLocal:
    """
    Marker used to indicate this (list, iter, etc) was constructed in
    local scope and can be mutated safely in analysis without leaking
    state.
    """

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def wrap_fx_proxy(tx, proxy, example_value=None, **options):
    from .tensor import TensorVariable

    return wrap_fx_proxy_cls(
        target_cls=TensorVariable,
        tx=tx,
        proxy=proxy,
        example_value=example_value,
        **options,
    )


# Note: Unfortunate split due to some gross classes existing that subclass TensorVariable
# Should be compositional instead
def wrap_fx_proxy_cls(target_cls, tx, proxy, example_value=None, **options):
    from .. import config, variables
    from .constant import ConstantVariable
    from .lists import SizeVariable
    from .tensor import DynamicShapeVariable  # TODO(voz): Give it its own file

    if "guards" in options and options["guards"] is not None:
        tx.output.guards.update(options["guards"])

    assert "example_value" not in proxy.node.meta
    if not config.dynamic_propagation:
        if isinstance(example_value, torch.Tensor):
            options.update(target_cls.specialize(example_value))
        return target_cls(proxy, **options)

    use_fake_tensors = fake_tensors_available and config.fake_tensor_propagation

    initial_example_value = example_value

    def _clone_input(value):
        if isinstance(value, torch.Tensor):
            use_fake_tensors = fake_tensors_available and config.fake_tensor_propagation
            # tensor subclasses will not be converted to FakeTensors and need to be cloned
            if not use_fake_tensors or not isinstance(
                value, torch._subclasses.fake_tensor.FakeTensor
            ):
                # NB: ensure strides are preserved
                value = clone_input(value)

        return value

    with preserve_rng_state():
        if example_value is None:
            if use_fake_tensors:
                example_value = get_fake_value(proxy.node, tx)
            else:
                example_value = get_real_value(proxy.node, tx.output)

        else:
            proxy.tracer.real_value_cache[proxy.node] = _clone_input(example_value)
            if use_fake_tensors:
                fake_wrapper = functools.partial(wrap_to_fake_tensor_and_record, tx=tx)
                example_value = fake_wrapper(example_value)

    if isinstance(example_value, torch.Tensor):
        is_parameter = isinstance(example_value, torch.nn.Parameter)
        should_specialize = options.pop("should_specialize", False)
        if is_parameter or should_specialize:
            specialized_value = initial_example_value
        else:
            specialized_value = None

        example_value = _clone_input(example_value)
        proxy.node.meta["example_value"] = example_value
        specialized_props = target_cls.specialize(example_value)
        if use_fake_tensors and isinstance(
            example_value, torch._subclasses.fake_tensor.FakeTensor
        ):
            specialized_props["class_type"] = (
                torch.nn.Parameter if is_parameter else torch.Tensor
            )

        specialized_props["specialized_value"] = specialized_value

        options.update(specialized_props)
        return target_cls(proxy, **options)
    elif (
        hasattr(proxy.node.target, "__name__")
        and proxy.node.target.__name__ == "set_state"
        and isinstance(proxy.node.target.__self__, torch._C.Generator)
        or proxy.node.target == torch.random.set_rng_state
    ):
        from . import TorchVariable

        return TorchVariable(proxy.node.target)
    elif istype(example_value, (int, bool, float)) and not config.dynamic_shapes:
        return ConstantVariable(example_value)
    elif istype(example_value, (int, bool, float)) and config.dynamic_shapes:
        proxy.node.meta["example_value"] = example_value
        return DynamicShapeVariable.create(tx, proxy, example_value, **options)
    elif istype(example_value, torch.Size) and config.dynamic_shapes:
        proxy.node.meta["example_value"] = example_value
        sizes = []
        for i, v in enumerate(example_value):
            proxy_i = proxy[i]
            sizes.append(DynamicShapeVariable.create(tx, proxy_i, v, **options))
        return SizeVariable(sizes, proxy, **options)
    elif istype(example_value, int) and proxy.node.target in (
        torch.seed,
        operator.mod,
        # some mac builds are missing torch.distributed.get_rank()
        getattr(torch.distributed, "get_rank", _missing),
        getattr(torch.distributed, "get_world_size", _missing),
    ):
        proxy.node.meta["example_value"] = example_value
        return DynamicShapeVariable.create(tx, proxy, example_value, **options)
    elif istype(example_value, torch.Size) and all(
        [isinstance(x, int) for x in example_value]
    ):
        sizes = [variables.ConstantVariable(x) for x in example_value]
        return SizeVariable(sizes, **options)
    elif isinstance(example_value, (tuple, list)):
        unpacked = []
        for i, val in enumerate(example_value):
            if val is None:
                # nn.MultiheadAttention() can return None, see issue #175
                unpacked.append(
                    variables.ConstantVariable(None, **options),
                )
            else:
                unpacked.append(
                    wrap_fx_proxy(
                        tx,
                        proxy.tracer.create_proxy(
                            "call_function", operator.getitem, (proxy, i), {}
                        ),
                        example_value=val,
                        **options,
                    )
                )
        if istype(example_value, tuple):
            return variables.TupleVariable(unpacked, **options)
        elif istype(example_value, (list, immutable_list)):
            return variables.ListVariable(
                unpacked, mutable_local=MutableLocal(), **options
            )
        else:
            assert (
                example_value.__class__.__module__ == "torch.return_types"
                or hasattr(example_value, "_fields")
            ), ("namedtuple?")
            return variables.NamedTupleVariable(
                unpacked, example_value.__class__, **options
            )
    elif example_value is None or proxy.node.target is torch.manual_seed:
        return variables.ConstantVariable(None, **options)
    elif (
        isinstance(example_value, int)
        and proxy.node.target is torch._utils._element_size
    ):
        proxy.node.meta["example_value"] = example_value
        return variables.ConstantVariable(example_value, **options)
    elif (
        isinstance(example_value, numbers.Number)
        and (proxy.node.target == "item" or proxy.node.target in {math.sqrt, math.pow})
        and config.capture_scalar_outputs
    ):
        if use_fake_tensors:
            # item raw value should not be accessed
            return wrap_fx_proxy_cls(
                FakeItemVariable,
                tx=tx,
                proxy=proxy,
                example_value=torch.tensor(example_value),
                **options,
            )
        else:
            return wrap_fx_proxy_cls(
                UnspecializedPythonVariable,
                tx=tx,
                proxy=proxy,
                example_value=torch.tensor(example_value),
                raw_value=None if use_fake_tensors else example_value,
                need_unwrap=False,
                **options,
            )
    elif (
        proxy.node.target == torch._C._DisableFuncTorch
        or proxy.node.target == torch.cuda._is_in_bad_fork
    ):
        from . import UserDefinedObjectVariable

        return UserDefinedObjectVariable(example_value)
    elif isinstance(example_value, (torch.SymInt, torch.SymFloat)):
        proxy.node.meta["example_value"] = example_value
        return DynamicShapeVariable(proxy, example_value, **options)
    else:
        raise AssertionError(
            "torch.* op returned non-Tensor "
            + f"{typestr(example_value)} {proxy.node.op} {proxy.node.target}"
        )


class VariableTracker:
    """
    Base class for tracked locals and stack values

    VariableTracker instances are immutable and should be copied in
    order to change them.
    """

    # fields to leave unmodified in apply()
    _nonvar_fields = ["value"]

    @staticmethod
    def propagate(*vars: List[List["VariableTracker"]]):
        """Combine the guards from many VariableTracker into **kwargs for a new instance"""
        guards = set()

        def visit(var):
            if type(var) in (list, tuple, dict_values, odict_values):
                for i in var:
                    visit(i)
            elif isinstance(var, variables.BaseListVariable):
                guards.update(var.guards)
                for i in var.items:
                    visit(i)
            elif isinstance(var, variables.ConstDictVariable):
                guards.update(var.guards)
                visit(var.items.values())
            else:
                assert isinstance(var, VariableTracker), typestr(var)
                guards.update(var.guards)

        visit(vars)
        return {
            "guards": guards,
        }

    def clone(self, **kwargs):
        """Shallow copy with some (optional) changes"""
        args = dict(self.__dict__)
        args.update(kwargs)
        return self.__class__(**args)

    @classmethod
    def copy(cls, value):
        """Deeper (but not full) copy, leaving FX and user objects alone"""
        return cls.apply(identity, value)

    @classmethod
    def apply(
        cls, fn: Callable[["VariableTracker"], "VariableTracker"], value, cache=None
    ):
        """
        Walk this object and call fn on all the VariableTracker
        instances to produce a new VariableTracker with the results.
        """
        if cache is None:
            cache = dict()

        idx = id(value)
        if idx in cache:
            return cache[idx][0]

        if isinstance(value, VariableTracker):
            updated_dict = dict(value.__dict__)
            for key in updated_dict.keys():
                if key not in value._nonvar_fields:
                    updated_dict[key] = cls.apply(fn, updated_dict[key], cache)
            result = fn(value.clone(**updated_dict))
        elif istype(value, list):
            result = [cls.apply(fn, v, cache) for v in value]
        elif istype(value, tuple):
            result = tuple(cls.apply(fn, v, cache) for v in value)
        elif istype(value, collections.OrderedDict):
            result = collections.OrderedDict(
                cls.apply(fn, v, cache) for v in value.items()
            )
        elif istype(value, dict):
            result = {k: cls.apply(fn, v, cache) for k, v in list(value.items())}
        else:
            result = value

        # save `value` to keep it alive and ensure id() isn't reused
        cache[idx] = (result, value)
        return result

    def add_guard(self, guard):
        return self.clone(guards=set.union(self.guards, {guard}))

    def add_guards(self, guards):
        if guards is None:
            return self
        assert isinstance(guards, set)
        return self.clone(guards=set.union(self.guards, guards))

    def add_options(self, options, *more):
        if more:
            return self.add_options(options).add_options(*more)
        if isinstance(options, VariableTracker):
            return self.add_guards(options.guards)
        assert isinstance(options, dict)
        return self.add_guards(options.get("guards", set()))

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def __repr__(self):
        return str(self)

    def python_type(self):
        raise NotImplementedError(f"{self} has no type")

    def as_python_constant(self):
        """For constants"""
        raise NotImplementedError(f"{self} is not a constant")

    def is_python_constant(self):
        try:
            self.as_python_constant()
            return True
        except NotImplementedError:
            return False

    def as_specialized(self, tx):
        """
        For specialized variables, return itself,
        For unspecialized variables, convert to constant variable and return.
        """
        return self

    def can_make_guard(self):
        try:
            self.make_guard(None)
            return True
        except NotImplementedError:
            return False

    def make_guard(self, fn):
        if self.source:
            return self.source.make_guard(fn)
        raise NotImplementedError()

    def replace_guards(self, guards, *fns):
        name = self.source.name()
        new_guards = {g for g in (guards or []) if g.name != name}
        new_guards.update(self.source.make_guard(fn) for fn in fns)
        return new_guards

    def const_getattr(self, tx, name: str) -> Any:
        """getattr(self, name) returning a python constant"""
        raise NotImplementedError()

    def var_getattr(self, tx, name: str) -> "VariableTracker":
        """getattr(self, name) returning a new variable"""
        options = VariableTracker.propagate(self)
        value = self.const_getattr(tx, name)
        if not variables.ConstantVariable.is_literal(value):
            raise NotImplementedError()
        if self.source:
            options["source"] = AttrSource(self.source, name)
        return variables.ConstantVariable(value, **options)

    def is_proxy(self):
        try:
            self.as_proxy()
            return True
        except NotImplementedError:
            return False

    def as_proxy(self):
        raise NotImplementedError(str(self))

    def reconstruct(self, codegen):
        raise NotImplementedError()

    def unpack_var_sequence(self, tx):
        raise NotImplementedError()

    def has_unpack_var_sequence(self, tx):
        try:
            self.unpack_var_sequence(tx)
            return True
        except NotImplementedError:
            return False

    def num_parameters(self):
        unimplemented(f"num_parameters: {self}")

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        unimplemented(f"hasattr: {self}")

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        unimplemented(f"call_function {self} {args} {kwargs}")

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if name == "__len__" and self.has_unpack_var_sequence(tx):
            assert not (args or kwargs)
            return variables.ConstantVariable(
                len(self.unpack_var_sequence(tx)), **VariableTracker.propagate(self)
            )
        elif (
            name == "__getattr__"
            and len(args) == 1
            and args[0].is_python_constant()
            and not kwargs
        ):
            return self.var_getattr(tx, args[0].as_python_constant()).add_options(
                self, args[0]
            )
        raise unimplemented(f"call_method {self} {name} {args} {kwargs}")

    def __init__(
        self,
        guards: Optional[Set] = None,
        source: Source = None,
        mutable_local: MutableLocal = None,
    ):
        super(VariableTracker, self).__init__()
        self.guards = guards or set()
        self.source = source
        self.mutable_local = mutable_local


def typestr(*objs):
    if len(objs) == 1:
        (obj,) = objs
        if isinstance(obj, VariableTracker):
            return str(obj)
        else:
            return type(obj).__name__
    else:
        return " ".join(map(typestr, objs))
