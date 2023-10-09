import collections
import dataclasses
import functools
import inspect
import operator
from typing import Any, Dict, List, Optional

import torch
import torch.fx

from .. import polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..guards import make_dupe_guard
from ..source import GetItemSource
from ..utils import (
    get_fake_value,
    guard_if_dyn,
    is_namedtuple,
    namedtuple_fields,
    odict_values,
)
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable, UserMethodVariable


def _listlike_contains_helper(items, search, tx, options):
    if search.is_python_constant():
        result = any(
            x.as_python_constant() == search.as_python_constant() for x in items
        )
        return variables.ConstantVariable.create(result, **options)

    from .builtin import BuiltinVariable

    result = None
    for x in items:
        check = BuiltinVariable(operator.eq).call_function(tx, [x, search], {})
        if result is None:
            result = check
        else:
            result = BuiltinVariable(operator.or_).call_function(
                tx, [check, result], {}
            )
    if result is None:
        result = ConstantVariable.create(None)
    return result


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
            set: SetVariable,
            odict_values: ListVariable,
            torch.nn.ParameterList: ListVariable,
            torch.nn.ModuleList: ListVariable,
            collections.deque: DequeVariable,
        }[obj]

    def __init__(
        self,
        items: List[VariableTracker],
        recursively_contains=None,
        regen_guards=True,
        **kwargs,
    ):
        super().__init__(recursively_contains=recursively_contains, **kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        # Sometimes, we know that we have passed in the guards from the items in the list
        if regen_guards:
            self.guards.update(VariableTracker.propagate(items)["guards"])

        self.items: List[VariableTracker] = items

    def _as_proxy(self):
        return [x.as_proxy() for x in self.items]

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
                ).add_options(arg, self)
            else:
                return self.clone(
                    items=self.items[index],
                    mutable_local=MutableLocal() if self.mutable_local else None,
                ).add_options(arg, self)
        else:
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index].add_options(arg, self)

    def unpack_var_sequence(self, tx):
        return [x.add_options(self) for x in self.items]

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
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
            return _listlike_contains_helper(self.items, args[0], tx, options)
        elif name == "index":
            from .builder import SourcelessBuilder

            assert len(kwargs) == 0
            assert len(args) > 0 and len(args) <= 3
            return tx.inline_user_function_return(
                SourcelessBuilder()(tx, polyfill.index), [self] + list(args), kwargs
            )

        return super().call_method(tx, name, args, kwargs)

    @staticmethod
    def list_compare(tx, op, left, right):
        from .builtin import BuiltinVariable

        eq_result = BaseListVariable.list_eq(tx, left, right)
        if op is operator.eq:
            return eq_result
        elif op is operator.ne:
            return BuiltinVariable(operator.not_).call_function(tx, [eq_result], {})
        else:
            unimplemented(f"list_compare {left} {op} {right}")

    @staticmethod
    def list_eq(tx, left, right):
        from .builtin import BuiltinVariable

        options = VariableTracker.propagate(left, right)

        # Most list-like variables implement comparison ops the same way,
        # so they can re-use this helper.
        # There are quirks though, like how `tuple([2]) == torch.Size([2])`,
        # but `tuple([2]) != list([2])`
        if len(left.items) != len(right.items):
            return ConstantVariable.create(False, **options)
        if len(left.items) == 0:
            return ConstantVariable.create(True, **options)

        # Generic list comparison works by iterating over left aka self and right the compared-to list.
        # If we hit here, their lengths are the same and they cannot be expressed as python constants.
        # So, we iterate over the zipped list items.
        comps = []
        for l, r in zip(left.items, right.items):
            comp = BuiltinVariable(operator.eq).call_function(tx, [l, r], {})
            if comp.is_python_constant() and not comp.as_python_constant():
                # early exit in false case
                return comp.add_options(options)
            comps.append(comp)

        return functools.reduce(
            lambda a, b: BuiltinVariable(operator.and_).call_function(tx, [a, b], {}),
            comps,
        ).add_options(options)


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
            raise AssertionError()

        assert stop is not None
        super().__init__([start, stop, step], **kwargs)

    def python_type(self):
        return range

    def as_python_constant(self):
        return range(*[x.as_python_constant() for x in self.items])

    def as_proxy(self):
        return self.python_type()(*self._as_proxy())

    def unpack_var_sequence(self, tx):
        return [
            variables.ConstantVariable.create(x).add_options(self)
            for x in self.as_python_constant()
        ]

    def reconstruct(self, codegen):
        assert "range" not in codegen.tx.f_globals
        codegen.append_output(codegen.create_load_python_module(range, True))
        codegen.foreach(self.items)
        return create_call_function(3, False)

    def var_getattr(self, tx, name):
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented(f"range.{name}")
        return self.items[fields.index(name)].add_options(self)


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
        options = VariableTracker.propagate(self, args, kwargs.values())
        if name == "append" and self.mutable_local:
            assert not kwargs
            (arg,) = args
            new_rec_contains = self.recursively_contains.union(arg.recursively_contains)
            if arg.mutable_local is not None:
                new_rec_contains.add(arg.mutable_local)
            tx.replace_all(
                self,
                type(self)(
                    self.items + [arg],
                    recursively_contains=new_rec_contains,
                    regen_guards=False,
                    **options,
                ),
            )
            return ConstantVariable.create(None)
        elif (
            name == "extend"
            and self.mutable_local
            and args
            and args[0].has_unpack_var_sequence(tx)
        ):
            assert not kwargs
            (arg,) = args
            return tx.replace_all(
                self,
                type(self)(
                    list(self.items) + list(arg.unpack_var_sequence(tx)),
                    regen_guards=False,
                    **options,
                ),
            )
        elif name == "insert" and self.mutable_local:
            assert not kwargs
            idx, value = args
            items = list(self.items)
            items.insert(idx.as_python_constant(), value)
            return tx.replace_all(
                self,
                type(self)(items, regen_guards=False, **options),
            )
        elif name == "pop" and self.mutable_local:
            assert not kwargs
            items = list(self.items)
            result = items.pop(*[a.as_python_constant() for a in args])
            tx.replace_all(
                self,
                type(self)(items, regen_guards=False, **options),
            )
            return result
        elif name == "clear" and self.mutable_local:
            assert not kwargs and not args
            return tx.replace_all(
                self,
                type(self)([], regen_guards=False, **options),
            )
        elif (
            name == "__setitem__"
            and self.mutable_local
            and args
            and args[0].is_python_constant()
        ):
            assert not kwargs
            key, value = args
            items = list(self.items)
            if isinstance(key, SliceVariable):
                items[key.as_python_constant()] = list(value.items)
            else:
                items[key.as_python_constant()] = value
            result = ListVariable(items, regen_guards=False, **options)
            return tx.replace_all(self, result)
        elif name == "copy":
            # List copy() doesn't have args and kwargs
            assert not kwargs
            assert not args
            items = list(self.items)
            return type(self)(
                items, regen_guards=False, mutable_local=MutableLocal(), **options
            )
        else:
            return super().call_method(tx, name, args, kwargs)


class ListVariable(CommonListMethodsVariable):
    def python_type(self):
        return list

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_LIST", arg=len(self.items))]

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if (
            name == "__setitem__"
            and self.mutable_local
            and args
            and args[0].is_python_constant()
        ):
            assert not kwargs
            key, value = args
            items = list(self.items)
            if isinstance(key, SliceVariable):
                if not value.has_unpack_var_sequence(tx):
                    unimplemented(
                        f"Missing dynamo support for expanding {value} into a list for slice assignment."
                    )
                items[key.as_python_constant()] = value.unpack_var_sequence(tx)
            else:
                items[key.as_python_constant()] = value
            result = ListVariable(items, regen_guards=False, **options)
            return tx.replace_all(self, result)
        else:
            return super().call_method(tx, name, args, kwargs)

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        if self.python_type() is not list:
            return super().call_hasattr(tx, name)
        options = VariableTracker.propagate(self)
        return variables.ConstantVariable.create(hasattr([], name), **options)


class DequeVariable(CommonListMethodsVariable):
    def python_type(self):
        return collections.deque

    def reconstruct(self, codegen):
        assert "deque" not in codegen.tx.f_globals
        codegen.append_output(
            codegen.create_load_python_module(collections.deque, True)
        )
        codegen.foreach(self.items)
        return create_call_function(len(self.items), False)

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
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
            items = list(self.items)
            items[key.as_python_constant()] = value
            result = DequeVariable(items, regen_guards=False, **options)
            return tx.replace_all(self, result)
        elif name == "extendleft" and self.mutable_local:
            assert not kwargs
            (arg,) = args
            return tx.replace_all(
                self,
                DequeVariable(
                    list(arg.unpack_var_sequence(tx)) + list(self.items),
                    regen_guards=False,
                    **options,
                ),
            )
        elif name == "popleft" and self.mutable_local:
            assert not args
            assert not kwargs
            items = collections.deque(self.items)
            result = items.popleft()
            tx.replace_all(
                self,
                DequeVariable(list(items), regen_guards=False, **options),
            )
            return result
        elif name == "appendleft" and self.mutable_local:
            assert not kwargs
            return tx.replace_all(
                self,
                DequeVariable(
                    [args[0]] + list(self.items),
                    regen_guards=False,
                    **options,
                ),
            )
        else:
            return super().call_method(tx, name, args, kwargs)


class TupleVariable(BaseListVariable):
    def python_type(self):
        return tuple

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_TUPLE", arg=len(self.items))]

    def call_method(
        self,
        tx,
        name,
        args: List["VariableTracker"],
        kwargs: Dict[str, "VariableTracker"],
    ) -> "VariableTracker":
        return super().call_method(tx, name, args, kwargs)


class SizeVariable(TupleVariable):
    """torch.Size(...)"""

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
        proxy.node.meta["example_value"] = torch.Size(
            [
                p.node.meta["example_value"] if not isinstance(p, int) else p
                for p in proxies
            ]
        )
        return proxy

    def reconstruct(self, codegen):
        codegen.load_import_from("torch", "Size")
        codegen.foreach(self.items)
        build_torch_size = [
            create_instruction("BUILD_TUPLE", arg=len(self.items)),
        ] + create_call_function(1, True)
        return build_torch_size

    def unpack_var_sequence(self, tx):
        return [x.add_options(self) for x in self.items]

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

        result = ConstantVariable.create(const_result).add_options(self)
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
        options = VariableTracker.propagate(self, args, kwargs.values())
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
            return SizeVariable(self.items[index]).add_options(arg, self)
        else:
            assert isinstance(index, (int, torch.SymInt))
            return self.items[index].add_options(arg, self)


class NamedTupleVariable(TupleVariable):
    def __init__(self, items, tuple_cls, **kwargs):
        super().__init__(items, **kwargs)
        self.tuple_cls = tuple_cls

    def python_type(self):
        return self.tuple_cls

    def as_python_constant(self):
        return self.python_type()(*[x.as_python_constant() for x in self.items])

    def reconstruct(self, codegen):
        create_fn = getattr(self.tuple_cls, "_make", self.tuple_cls)
        codegen.append_output(codegen._create_load_const(create_fn))
        codegen.foreach(self.items)
        return [
            create_instruction("BUILD_TUPLE", arg=len(self.items)),
        ] + create_call_function(1, True)

    def var_getattr(self, tx, name):
        def check_and_create_method():
            options = VariableTracker.propagate(self)
            method = inspect.getattr_static(self.tuple_cls, name, None)
            if isinstance(method, classmethod):
                # We need the unbounded cls method to avoid the inline __self__
                return UserMethodVariable(
                    method.__func__,
                    variables.UserDefinedClassVariable(self.tuple_cls, **options),
                )
            elif isinstance(method, staticmethod):
                return UserFunctionVariable(method.__func__, **options)
            elif inspect.isfunction(method):
                return UserMethodVariable(method, self, **options)
            else:
                return None

        fields = namedtuple_fields(self.tuple_cls)
        if name not in fields:
            method = check_and_create_method()
            if not method:
                unimplemented(f"NamedTupleVariable.{name}")
            return method
        return self.items[fields.index(name)].add_options(self)

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        options = VariableTracker.propagate(self)
        fields = namedtuple_fields(self.tuple_cls)
        return variables.ConstantVariable.create(name in fields, **options)


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
            raise AssertionError()

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
        return [create_instruction("BUILD_SLICE", arg=len(self.items))]

    def var_getattr(self, tx, name):
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented(f"slice.{name}")
        return self.items[fields.index(name)].add_options(self)


class ListIteratorVariable(VariableTracker):
    def __init__(self, items, index: int = 0, recursively_contains=None, **kwargs):
        super().__init__(recursively_contains=recursively_contains, **kwargs)
        assert isinstance(items, list)
        # Removing this check as it slows things down too much
        # https://github.com/pytorch/pytorch/pull/87533#issuecomment-1287574492

        # assert all(isinstance(x, VariableTracker) for x in items)
        self.items = items
        self.index = index

    def next_variables(self):
        assert self.mutable_local
        if self.index >= len(self.items):
            raise StopIteration()
        return self.items[self.index].add_options(self), ListIteratorVariable(
            self.items,
            self.index + 1,
            mutable_local=MutableLocal(),
            recursively_contains=self.recursively_contains,
            **VariableTracker.propagate([self]),
        )

    def as_python_constant(self):
        if self.index > 0:
            raise NotImplementedError()
        return iter([x.as_python_constant() for x in self.items])

    def unpack_var_sequence(self, tx):
        return [x.add_options(self) for x in self.items[self.index :]]

    def reconstruct(self, codegen):
        remaining_items = self.items[self.index :]
        codegen.foreach(remaining_items)
        return [
            create_instruction("BUILD_TUPLE", arg=len(remaining_items)),
            create_instruction("GET_ITER"),
        ]


class TupleIteratorVariable(ListIteratorVariable):
    pass


class SetVariable(VariableTracker):
    @dataclasses.dataclass
    class SetElement:
        vt: VariableTracker
        underlying_value: Any

        def __hash__(self) -> int:
            return hash(self.underlying_value)

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, SetVariable.SetElement):
                return False
            if isinstance(self.vt, variables.TensorVariable):
                return self.underlying_value is other.underlying_value
            else:
                return self.underlying_value == other.underlying_value

    def __init__(
        self,
        items: List[VariableTracker],
        recursively_contains=None,
        regen_guards=True,
        **kwargs,
    ):
        super().__init__(recursively_contains=recursively_contains, **kwargs)
        # Note - Set is still backed by a list, because we want set behavior over the contents,
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)

        self.items = []
        self._add(items)

        # Sometimes, we know that we have passed in the guards from the items in the set
        if regen_guards:
            self.guards.update(VariableTracker.propagate(items)["guards"])

    def as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def python_type(self):
        return set

    def reconstruct(self, codegen):
        codegen.load_import_from("builtins", "set")
        codegen.foreach(self.items)
        return [
            create_instruction("BUILD_SET", arg=len(self.items))
        ] + create_call_function(1, True)

    # Note - this is only used for producing a set
    def _as_set_element(self, vt):
        from .base import VariableTracker
        from .tensor import TensorVariable

        assert isinstance(vt, VariableTracker)

        if isinstance(vt, TensorVariable):
            tensor_node = vt.as_proxy().node
            return SetVariable.SetElement(vt, tensor_node)
        if isinstance(vt, ConstantVariable):
            return SetVariable.SetElement(vt, vt.value)

        unimplemented(f"Sets with {type(vt)} NYI")

    @property
    def _underlying_items(self):
        underlying_items = set()
        for current_item in self.items:
            assert (
                current_item not in underlying_items
            ), "Items modeling set invariant violated"
            underlying_items.add(self._as_set_element(current_item))
        return underlying_items

    def _add(self, item):
        underlying_items = self._underlying_items

        if isinstance(item, (list, set)):
            items_to_add = item
        else:
            items_to_add = [item]

        for item_to_add in items_to_add:
            set_element = self._as_set_element(item_to_add)
            if set_element not in underlying_items:
                underlying_items.add(set_element)
                self.items.append(set_element.vt)
            else:
                for e in underlying_items:
                    if hash(set_element) == hash(e):
                        alias_guard = make_dupe_guard(
                            e.vt.source, set_element.vt.source
                        )
                        if alias_guard:
                            e.vt = e.vt.add_guards(
                                {e.vt.source.make_guard(alias_guard)}
                            )

        return self.items

    def call_method(
        self,
        tx,
        name,
        args: List[VariableTracker],
        kwargs: Dict[str, VariableTracker],
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        # Somewhat duplicative of CommonListMethodsVariable - but better than to violate substitution
        # principles and end up with things like direct item access attempts on a set, or
        # getitem sources.
        if name == "add" and args and self.mutable_local:
            assert not kwargs
            item = args[0]
            result = SetVariable(
                self._add(item),
                mutable_local=self.mutable_local,
                regen_guards=False,
                **options,
            )
            tx.replace_all(self, result)
            return ConstantVariable.create(None)
        elif name == "pop" and self.mutable_local:
            assert not kwargs
            assert not args
            items = list(self.items)
            result = items.pop()
            tx.replace_all(
                self,
                SetVariable(items, regen_guards=False, **options),
            )
            return result
        elif name == "__len__":
            return ConstantVariable.create(len(self.items)).add_options(options)
        elif name == "__contains__":
            assert len(args) == 1
            assert not kwargs
            return _listlike_contains_helper(self.items, args[0], tx, options)
        else:
            return super().call_method(tx, name, args, kwargs)

    def getitem_const(self, arg: VariableTracker):
        raise RuntimeError("Illegal to getitem on a set")

    def as_python_constant(self):
        return self.python_type()([x.as_python_constant() for x in self.items])

    def unpack_var_sequence(self, tx):
        return [x.add_options(self) for x in self.items]
