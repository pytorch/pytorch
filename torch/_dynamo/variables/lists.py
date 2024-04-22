# mypy: ignore-errors

import collections
import functools
import inspect
import operator
import types
from typing import Dict, List, Optional

import torch
import torch.fx
from ..._guards import Source

from .. import polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..source import AttrSource, GetItemSource
from ..utils import (
    get_fake_value,
    guard_if_dyn,
    is_namedtuple,
    istype,
    iter_contains,
    namedtuple_fields,
    odict_values,
    set_example_value,
)
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable, UserMethodVariable


class BaseListVariable(VariableTracker):
    @staticmethod
    def cls_for_instance(obj):
        if is_namedtuple(obj):
            return functools.partial(NamedTupleVariable, tuple_cls=type(obj))
        return BaseListVariable.cls_for(type(obj))

    @staticmethod
    def cls_for(obj):
        return {
            iter: ListIteratorVariable,
            list: ListVariable,
            slice: SliceVariable,
            torch.Size: SizeVariable,
            tuple: TupleVariable,
            odict_values: ListVariable,
            torch.nn.ParameterList: ListVariable,
            torch.nn.ModuleList: ListVariable,
            collections.deque: DequeVariable,
        }[obj]

    def __init__(
        self,
        items: List[VariableTracker],
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        self.items: List[VariableTracker] = items

    def _as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def modified(self, items, **kwargs):
        return type(self)(items, **kwargs)

    @property
    def value(self):
        return self.as_python_constant()

    def as_python_constant(self):
        return self.python_type()([x.as_python_constant() for x in self.items])

    def as_proxy(self):
        assert self.python_type() is not SizeVariable
        return self.python_type()(self._as_proxy())

    def getitem_const(self, arg: VariableTracker):
        from .tensor import SymNodeVariable

        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        else:
            index = arg.as_python_constant()

        if isinstance(index, slice):
            if self.source is not None:
                return self.clone(
                    items=self.items[index],
                    source=GetItemSource(self.source, index),
                    mutable_local=MutableLocal() if self.mutable_local else None,
                )
            else:
                return self.clone(
                    items=self.items[index],
                    mutable_local=MutableLocal() if self.mutable_local else None,
                )
        else:
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index]

    def unpack_var_sequence(self, tx):
        return list(self.items)

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "__getitem__":
            from .tensor import TensorVariable

            assert not kwargs and len(args) == 1
            if isinstance(args[0], TensorVariable):
                value = get_fake_value(args[0].as_proxy().node, tx)
                if value.constant is not None and value.constant.numel() == 1:
                    value = variables.ConstantVariable.create(value.constant.item())
                else:
                    unimplemented("__getitem__ with non-constant tensor")
            else:
                value = args[0]
            return self.getitem_const(value)
        elif name == "__contains__":
            assert len(args) == 1
            assert not kwargs
            return iter_contains(self.unpack_var_sequence(tx), args[0], tx)
        elif name == "index":
            from .builder import SourcelessBuilder

            return tx.inline_user_function_return(
                SourcelessBuilder.create(tx, polyfill.index),
                [self] + list(args),
                kwargs,
            )

        return super().call_method(tx, name, args, kwargs)

    @staticmethod
    def list_compare(tx, op, left, right):
        return variables.UserFunctionVariable(polyfill.list_cmp).call_function(
            tx, [variables.BuiltinVariable(op), left, right], {}
        )


class RangeVariable(BaseListVariable):
    def __init__(self, items, **kwargs):
        items_to_map = items
        start = variables.ConstantVariable.create(0)
        stop = None
        step = variables.ConstantVariable.create(1)

        if len(items_to_map) == 1:
            (stop,) = items_to_map
        elif len(items_to_map) == 2:
            start, stop = items_to_map
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
        else:
            raise AssertionError

        assert stop is not None
        super().__init__([start, stop, step], **kwargs)

    def python_type(self):
        return range

    def as_python_constant(self):
        return range(*[x.as_python_constant() for x in self.items])

    def as_proxy(self):
        return self.python_type()(*self._as_proxy())

    def unpack_var_sequence(self, tx):
        return [variables.ConstantVariable.create(x) for x in self.as_python_constant()]

    def reconstruct(self, codegen):
        assert "range" not in codegen.tx.f_globals
        codegen.append_output(codegen.create_load_python_module(range, True))
        codegen.foreach(self.items)
        codegen.extend_output(create_call_function(3, False))

    def var_getattr(self, tx, name):
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented(f"range.{name}")
        return self.items[fields.index(name)]


class CommonListMethodsVariable(BaseListVariable):
    """
    Implement methods common to List and other List-like things
    """

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "append" and self.mutable_local:
            assert not kwargs
            (arg,) = args
            tx.output.side_effects.mutation(self)
            self.items.append(arg)
            return ConstantVariable.create(None)
        elif (
            name == "extend"
            and self.mutable_local
            and args
            and args[0].has_unpack_var_sequence(tx)
        ):
            assert not kwargs
            (arg,) = args
            seq = arg.unpack_var_sequence(tx)
            tx.output.side_effects.mutation(self)
            self.items.extend(seq)
            return ConstantVariable.create(None)
        elif name == "insert" and self.mutable_local:
            assert not kwargs
            idx, value = args
            const_idx = idx.as_python_constant()
            tx.output.side_effects.mutation(self)
            self.items.insert(const_idx, value)
            return ConstantVariable.create(None)
        elif name == "pop" and self.mutable_local:
            assert not kwargs
            tx.output.side_effects.mutation(self)
            return self.items.pop(*[a.as_python_constant() for a in args])
        elif name == "clear" and self.mutable_local:
            assert not kwargs and not args
            tx.output.side_effects.mutation(self)
            self.items.clear()
            return ConstantVariable.create(None)
        elif (
            name == "__setitem__"
            and self.mutable_local
            and args
            and args[0].is_python_constant()
        ):
            assert not kwargs
            key, value = args
            tx.output.side_effects.mutation(self)
            if isinstance(key, SliceVariable):
                self.items[key.as_python_constant()] = list(value.items)
            else:
                self.items[key.as_python_constant()] = value
            return ConstantVariable.create(None)
        elif name == "copy":
            # List copy() doesn't have args and kwargs
            assert not kwargs
            assert not args
            items = list(self.items)
            return self.modified(items, mutable_local=MutableLocal())
        elif name == "reverse" and self.mutable_local:
            assert not kwargs
            assert not args
            self.items.reverse()
            tx.output.side_effects.mutation(self)
            return ConstantVariable.create(None)
        else:
            return super().call_method(tx, name, args, kwargs)


class ListVariable(CommonListMethodsVariable):
    def python_type(self):
        return list

    def __repr__(self):
        return f"{self.__class__.__name__}(length={len(self.items)})"

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_LIST", arg=len(self.items)))

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if (
            name == "__setitem__"
            and self.mutable_local
            and args
            and args[0].is_python_constant()
        ):
            assert not kwargs
            key, value = args
            tx.output.side_effects.mutation(self)
            if isinstance(key, SliceVariable):
                if not value.has_unpack_var_sequence(tx):
                    unimplemented(
                        f"Missing dynamo support for expanding {value} into a list for slice assignment."
                    )
                self.items[key.as_python_constant()] = value.unpack_var_sequence(tx)
            else:
                self.items[key.as_python_constant()] = value
            return ConstantVariable.create(None)
        else:
            return super().call_method(tx, name, args, kwargs)

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        if self.python_type() is not list:
            return super().call_hasattr(tx, name)
        return variables.ConstantVariable.create(hasattr([], name))


class DequeVariable(CommonListMethodsVariable):
    def python_type(self):
        return collections.deque

    def reconstruct(self, codegen):
        assert "deque" not in codegen.tx.f_globals
        codegen.append_output(
            codegen.create_load_python_module(collections.deque, True)
        )
        codegen.foreach(self.items)
        codegen.extend_output(create_call_function(len(self.items), False))

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if (
            name == "__setitem__"
            and self.mutable_local
            and args
            and args[0].is_python_constant()
        ):
            assert not kwargs
            key, value = args
            assert key.is_python_constant() and isinstance(
                key.as_python_constant(), int
            )
            tx.output.side_effects.mutation(self)
            self.items[key.as_python_constant()] = value
            return ConstantVariable.create(None)
        elif name == "extendleft" and self.mutable_local:
            assert not kwargs

            (arg,) = args
            prefix = arg.unpack_var_sequence(tx)
            prefix.reverse()
            tx.output.side_effects.mutation(self)
            self.items = prefix + list(self.items)
            return ConstantVariable.create(None)
        elif name == "popleft" and self.mutable_local:
            assert not args
            assert not kwargs
            item = self.items[0]
            tx.output.side_effects.mutation(self)
            self.items = self.items[1:]
            return item
        elif name == "appendleft" and self.mutable_local:
            assert not kwargs
            tx.output.side_effects.mutation(self)
            self.items = [args[0]] + list(self.items)
            return ConstantVariable.create(None)
        else:
            return super().call_method(tx, name, args, kwargs)


class TupleVariable(BaseListVariable):
    def python_type(self):
        return tuple

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_TUPLE", arg=len(self.items)))

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        return super().call_method(tx, name, args, kwargs)

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        if self.python_type() is not tuple:
            return super().call_hasattr(tx, name)
        return variables.ConstantVariable.create(hasattr((), name))


class SizeVariable(TupleVariable):
    """torch.Size(...)"""

    _nonvar_fields = {
        "proxy",
        *TupleVariable._nonvar_fields,
    }

    def __init__(
        self,
        items: List[VariableTracker],
        proxy: Optional[torch.fx.Proxy] = None,
        **kwargs,
    ):
        self.proxy = proxy
        super().__init__(items, **kwargs)

    def python_type(self):
        return torch.Size

    def as_proxy(self):
        if self.proxy is not None:
            return self.proxy

        # torch.Size needs special handling.  Normally, we pun a list-like
        # container to directly contain Proxy/Node objects from FX, and FX
        # knows to look inside containers (via map_aggregate).  But torch.Size
        # is weird; although it subclasses from tuple, it doesn't allow
        # members which aren't int-like (rejecting Proxy and Node).  This
        # means we can't use the normal representation trick
        # torch.Size([proxy0, proxy1]).  I looked into seeing if I could
        # relax torch.Size in PyTorch proper, but if torch.Size constructor
        # sees a type that it doesn't recognize, it will try to call
        # __index__() on it, so there is no BC way to actually change this
        # behavior (though it occurs to me that I could have just added a
        # YOLO no checking alternate constructor.)
        #
        # To work around this problem, I represent a torch.Size proxy as
        # a straight up proxy, that would have been constructed by taking
        # the constituent proxies as arguments.  This trick can be generally
        # used for any construct that we need a proxy for but we can't
        # directly represent as an aggregate; I don't see very many examples
        # of this in torchdynamo though!

        # Look for a proxy.  If there are none, do the legacy behavior
        tracer = None
        proxies = self._as_proxy()
        for proxy in proxies:
            if isinstance(proxy, torch.fx.Proxy):
                tracer = proxy.tracer
                break

        if tracer is None:
            return torch.Size(proxies)

        proxy = tracer.create_proxy("call_function", torch.Size, (proxies,), {})
        set_example_value(
            proxy.node,
            torch.Size(
                [
                    p.node.meta["example_value"] if not isinstance(p, int) else p
                    for p in proxies
                ]
            ),
        )
        return proxy

    def reconstruct(self, codegen):
        codegen.load_import_from("torch", "Size")
        codegen.foreach(self.items)
        build_torch_size = [
            create_instruction("BUILD_TUPLE", arg=len(self.items)),
        ] + create_call_function(1, True)
        codegen.extend_output(build_torch_size)

    def unpack_var_sequence(self, tx):
        return list(self.items)

    def numel(self, tx):
        from .builtin import BuiltinVariable
        from .tensor import SymNodeVariable

        const_result = 1
        sym_sizes = []

        for v in self.items:
            if isinstance(v, ConstantVariable):
                const_result *= v.value
            else:
                assert isinstance(v, SymNodeVariable), type(v)
                # Delay proxy calls  until we know it will be necessary
                sym_sizes.append(v)

        result = ConstantVariable.create(const_result)
        if sym_sizes and const_result == 1:
            # Skip multiplying by 1
            result, *sym_sizes = sym_sizes

        if not sym_sizes or const_result == 0:
            return result

        mul = BuiltinVariable(operator.mul)
        for v in sym_sizes:
            result = mul.call_function(tx, [result, v], {})
        return result

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            out = self.get_item_dyn(tx, args[0])
            return out
        elif name == "numel":
            assert not args and not kwargs
            return self.numel(tx)

        return super().call_method(tx, name, args, kwargs)

    def get_item_dyn(self, tx, arg: VariableTracker):
        from .tensor import SymNodeVariable

        if isinstance(arg, SymNodeVariable):
            index = arg.sym_num
        else:
            index = arg.as_python_constant()
        if isinstance(index, slice):
            return SizeVariable(self.items[index])
        else:
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index]


class NamedTupleVariable(TupleVariable):
    _nonvar_fields = {
        "tuple_cls",
        *TupleVariable._nonvar_fields,
    }

    def __init__(self, items, tuple_cls, **kwargs):
        super().__init__(items, **kwargs)
        self.tuple_cls = tuple_cls

    def python_type(self):
        return self.tuple_cls

    def as_python_constant(self):
        return self.python_type()(*[x.as_python_constant() for x in self.items])

    def as_proxy(self):
        assert self.python_type() is not SizeVariable
        return self.python_type()(*self._as_proxy())

    def reconstruct(self, codegen):
        create_fn = getattr(self.tuple_cls, "_make", self.tuple_cls)
        codegen.append_output(codegen._create_load_const(create_fn))
        codegen.foreach(self.items)
        codegen.extend_output(
            [
                create_instruction("BUILD_TUPLE", arg=len(self.items)),
            ]
            + create_call_function(1, True)
        )

    def var_getattr(self, tx, name):
        def check_and_create_method():
            method = inspect.getattr_static(self.tuple_cls, name, None)
            if isinstance(method, classmethod):
                # We need the unbounded cls method to avoid the inline __self__
                return UserMethodVariable(
                    method.__func__,
                    variables.UserDefinedClassVariable(self.tuple_cls),
                )
            elif isinstance(method, staticmethod):
                return UserFunctionVariable(method.__func__)
            elif inspect.isfunction(method):
                return UserMethodVariable(method, self)
            else:
                return None

        fields = namedtuple_fields(self.tuple_cls)
        if name not in fields:
            method = check_and_create_method()
            if not method:
                super().var_getattr(tx, name)
            return method
        return self.items[fields.index(name)]

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        return variables.ConstantVariable.create(hasattr(self.tuple_cls, name))


class SliceVariable(BaseListVariable):
    def __init__(self, items, **kwargs):
        items_to_map = items
        start, stop, step = [variables.ConstantVariable.create(None)] * 3

        if len(items_to_map) == 1:
            (stop,) = items_to_map
        elif len(items_to_map) == 2:
            start, stop = items_to_map
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
        else:
            raise AssertionError

        if isinstance(start, variables.TensorVariable) or isinstance(
            stop, variables.TensorVariable
        ):
            unimplemented("Dynamic slicing on data-dependent value is not supported")

        super().__init__([start, stop, step], **kwargs)

    def as_proxy(self):
        return slice(*self._as_proxy())

    def python_type(self):
        return slice

    def as_python_constant(self):
        return slice(*[guard_if_dyn(x) for x in self.items])

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        codegen.append_output(create_instruction("BUILD_SLICE", arg=len(self.items)))

    def var_getattr(self, tx, name):
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented(f"slice.{name}")
        return self.items[fields.index(name)]


class ListIteratorVariable(VariableTracker):
    _nonvar_fields = {
        "index",
        *VariableTracker._nonvar_fields,
    }

    def __init__(self, items, index: int = 0, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(items, list)
        # Removing this check as it slows things down too much
        # https://github.com/pytorch/pytorch/pull/87533#issuecomment-1287574492

        # assert all(isinstance(x, VariableTracker) for x in items)
        self.items = items
        self.index = index

    def __repr__(self):
        return f"{self.__class__.__name__}(length={len(self.items)}, index={repr(self.index)})"

    def next_variable(self, tx):
        assert self.mutable_local
        old_index = self.index
        if old_index >= len(self.items):
            raise StopIteration
        tx.output.side_effects.mutation(self)
        self.index += 1
        return self.items[old_index]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ):
        if name == "__contains__":
            assert len(args) == 1
            assert not kwargs
            return iter_contains(self.items[self.index :], args[0], tx)

        return super().call_method(tx, name, args, kwargs)

    def as_python_constant(self):
        if self.index > 0:
            raise NotImplementedError
        return iter([x.as_python_constant() for x in self.items])

    def unpack_var_sequence(self, tx):
        return list(self.items[self.index :])

    def reconstruct(self, codegen):
        remaining_items = self.items[self.index :]
        codegen.foreach(remaining_items)
        codegen.extend_output(
            [
                create_instruction("BUILD_TUPLE", arg=len(remaining_items)),
                create_instruction("GET_ITER"),
            ]
        )


class TupleIteratorVariable(ListIteratorVariable):
    pass


class RestrictedListSubclassVariable(ListVariable):
    """
    This is a special case of UserDefinedObjectVariable where:
        1) The user subclasses list
        2) None of the list methods are overriden, merely some new methods are added

    In these cases, we can prevent graph breaks by not using the general
    UserDefinedObjectVariable machinery and instead treating it like
    a ListVariable.
    """

    _nonvar_fields = {"user_cls", "user_cls_source", *ListVariable._nonvar_fields}
    _allowed_names = {
        "__call__",
        "__module__",
        "__dict__",
        "__doc__",
        "__name__",
        "__qualname__",
    }
    _disallowed_names = {
        "__getattribute__",
        "__getattr__",
        "__setattr__",
    }

    @classmethod
    def _is_non_conflicting_subclass(
        cls,
        user_cls: type,
        python_cls: type,
    ):
        """Ensures user_cls inherits from python_cls (e.g. list) and does not override any methods on python_cls"""
        if (
            not istype(user_cls, type)
            or user_cls.__bases__ != (python_cls,)
            or user_cls.__mro__ != (user_cls, python_cls, object)
        ):
            return False  # not subclass
        return not any(
            hasattr(python_cls, name) or name in cls._disallowed_names
            for name in set(user_cls.__dict__.keys()) - cls._allowed_names
        )

    @classmethod
    def is_matching_cls(cls, user_cls: type):
        return cls._is_non_conflicting_subclass(user_cls, list)

    def __init__(self, items, *, user_cls: type, user_cls_source: Source, **kwargs):
        super().__init__(items=items, **kwargs)
        self.user_cls = user_cls
        self.user_cls_source = user_cls_source
        assert istype(user_cls, type)
        assert isinstance(user_cls_source, Source)

    def python_type(self):
        return self.user_cls

    def as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def as_python_constant(self):
        raise NotImplementedError

    def is_python_constant(self):
        return False

    @property
    def value(self):
        raise AttributeError("value")

    def modified(self, items, **kwargs):
        return type(self)(
            items,
            user_cls=self.user_cls,
            user_cls_source=self.user_cls_source,
            **kwargs,
        )

    def reconstruct(self, codegen):
        codegen(self.user_cls_source)
        super().reconstruct(codegen)
        codegen.extend_output(create_call_function(1, True))

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        if name in self.user_cls.__dict__:
            method = self.user_cls.__dict__[name]
            if isinstance(method, types.FunctionType):
                # inline the method
                source = AttrSource(self.user_cls_source, name)
                return UserMethodVariable(method, self, source=source).call_function(
                    tx, args, kwargs
                )
            unimplemented(
                f"RestrictedListSubclassVariable method {self.user_cls.__name__}.{name}"
            )
        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self, tx, args: "List[VariableTracker]", kwargs: "Dict[str, VariableTracker]"
    ) -> "VariableTracker":
        return self.call_method(tx, "__call__", args, kwargs)
