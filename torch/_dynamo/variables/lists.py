from typing import Dict, List, Optional

import torch
import torch.fx

from .. import config, variables
from ..bytecode_transformation import create_instruction
from ..exc import unimplemented
from ..source import GetItemSource
from ..utils import namedtuple_fields, proxy_args_kwargs
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable


class BaseListVariable(VariableTracker):
    @staticmethod
    def cls_for(obj):
        return {
            iter: ListIteratorVariable,
            list: ListVariable,
            slice: SliceVariable,
            torch.Size: SizeVariable,
            tuple: TupleVariable,
        }[obj]

    def __init__(self, items: List[VariableTracker], **kwargs):
        super(BaseListVariable, self).__init__(**kwargs)
        assert isinstance(items, list)
        assert all(isinstance(x, VariableTracker) for x in items)
        self.items: List[VariableTracker] = items

    def _as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def as_python_constant(self):
        return self.python_type()([x.as_python_constant() for x in self.items])

    def as_proxy(self):
        assert self.python_type() is not SizeVariable
        return self.python_type()(self._as_proxy())

    def getitem_const(self, arg: VariableTracker):
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
            assert isinstance(index, int)
            return self.items[index].add_options(arg, self)

    def unpack_var_sequence(self, tx):
        return [x.add_options(self) for x in self.items]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            return self.getitem_const(args[0])
        elif name == "__add__":
            assert not kwargs and len(args) == 1
            return type(self)(self.items + args[0].items, **options)
        elif (
            name == "__contains__"
            and len(args) == 1
            and args[0].is_python_constant()
            and all(x.is_python_constant() for x in self.items)
        ):
            assert not kwargs
            search = args[0].as_python_constant()
            result = any(x.as_python_constant() == search for x in self.items)
            return variables.ConstantVariable(result, **options)

        return super(BaseListVariable, self).call_method(tx, name, args, kwargs)


class RangeVariable(BaseListVariable):
    def __init__(self, value, items=None, guards=None, **kwargs):
        if items is None:
            items = [variables.ConstantVariable(x, guards=guards) for x in value]
        super().__init__(items, guards=guards, **kwargs)
        self.value = value

    def python_type(self):
        return range

    def as_python_constant(self):
        return self.value

    def reconstruct(self, codegen):
        assert "range" not in codegen.tx.f_globals
        range_fn = codegen.create_load_global("range", add=True)
        if self.value.step == 1:
            if self.value.start == 0:
                return [
                    range_fn,
                    codegen.create_load_const(self.value.stop),
                    create_instruction("CALL_FUNCTION", 1),
                ]
            return [
                range_fn,
                codegen.create_load_const(self.value.start),
                codegen.create_load_const(self.value.stop),
                create_instruction("CALL_FUNCTION", 2),
            ]
        return [
            range_fn,
            codegen.create_load_const(self.value.start),
            codegen.create_load_const(self.value.stop),
            codegen.create_load_const(self.value.step),
            create_instruction("CALL_FUNCTION", 3),
        ]


class ListVariable(BaseListVariable):
    def python_type(self):
        return list

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_LIST", len(self.items))]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if name == "append" and self.mutable_local:
            assert not kwargs
            (arg,) = args
            tx.replace_all(
                self,
                ListVariable(self.items + [arg], **options),
            )
            return ConstantVariable(None)
        elif (
            name in ("extend", "__iadd__")
            and self.mutable_local
            and args
            and args[0].has_unpack_var_sequence(tx)
        ):
            assert not kwargs
            (arg,) = args
            return tx.replace_all(
                self,
                ListVariable(
                    list(self.items) + list(arg.unpack_var_sequence(tx)),
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
                ListVariable(items, **options),
            )
        elif name == "pop" and self.mutable_local:
            assert not kwargs
            items = list(self.items)
            result = items.pop(*[a.as_python_constant() for a in args])
            tx.replace_all(
                self,
                ListVariable(items, **options),
            )
            return result
        elif name == "clear" and self.mutable_local:
            assert not kwargs and not args
            return tx.replace_all(
                self,
                ListVariable([], **options),
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
            result = ListVariable(items, **options)
            return tx.replace_all(self, result)
        else:
            return super().call_method(tx, name, args, kwargs)


class TupleVariable(BaseListVariable):
    def python_type(self):
        return tuple

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_TUPLE", len(self.items))]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if (
            name in ("__add__", "__iadd__")
            and len(args) == 1
            and isinstance(args[0], TupleVariable)
        ):
            assert not kwargs
            return TupleVariable(self.items + args[0].items, **options)
        elif (
            name in ("__add__", "__iadd__")
            and len(args) == 1
            and isinstance(args[0], variables.ConstantVariable)
        ):
            assert not kwargs
            return TupleVariable(
                self.items + list(args[0].unpack_var_sequence(self)), **options
            )
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
            [p.node.meta["example_value"] for p in proxies]
        )
        return proxy

    def reconstruct(self, codegen):
        codegen.load_import_from("torch", "Size")
        codegen.foreach(self.items)
        build_torch_size = [
            create_instruction("BUILD_TUPLE", len(self.items)),
            create_instruction("CALL_FUNCTION", 1),
        ]
        return build_torch_size

    def unpack_var_sequence(self, tx):
        return [x.add_options(self) for x in self.items]

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        options = VariableTracker.propagate(self, args, kwargs.values())
        if name == "__getitem__":
            assert not kwargs and len(args) == 1
            if config.dynamic_shapes:
                out = self.get_item_dyn(tx, args[0])
            else:
                out = self.getitem_const(args[0])
            return out
        return super(SizeVariable, self).call_method(tx, name, args, kwargs)

    def get_item_dyn(self, tx, arg: VariableTracker):
        from .tensor import DynamicShapeVariable
        unimplemented("TODO: this logic cause jx_nest_base to fail")

        index = arg.as_python_constant()
        if isinstance(index, slice):
            def _dynamo_get_item_lambda(target, index):
                return torch.Size.__getitem__(target, index)

            parent_proxy = self.as_proxy()
            proxy = tx.output.create_proxy(
                "call_function",
                _dynamo_get_item_lambda,
                *proxy_args_kwargs([self, arg], {}),
                current_tx=tx,
            )
            items = self.items[index]

            def _unpack_into_example(item):
                if isinstance(item, DynamicShapeVariable):
                    return item.dyn_shape
                return item.as_python_constant()

            # Mirror the indexing into example_value for downstream correctness
            proxy.node.meta["example_value"] = parent_proxy.node.meta["example_value"][
                index
            ]
            return SizeVariable(items, proxy=proxy).add_options(arg, self)
        else:
            assert isinstance(index, int)
            return self.items[index].add_options(arg, self)


class ShapeVariable(TupleVariable):
    """
    Represents tensor.shape(...) and helps differentiate between a constant
    TupleVariable and ShapeVariable.
    """

    pass


class NamedTupleVariable(TupleVariable):
    def __init__(self, items, tuple_cls, **kwargs):
        super().__init__(items, **kwargs)
        self.tuple_cls = tuple_cls

    def python_type(self):
        return self.tuple_cls

    def reconstruct(self, codegen):
        create_fn = getattr(self.tuple_cls, "_make", self.tuple_cls)
        codegen.append_output(codegen._create_load_const(create_fn))
        codegen.foreach(self.items)
        return [
            create_instruction("BUILD_TUPLE", len(self.items)),
            create_instruction("CALL_FUNCTION", 1),
        ]

    def var_getattr(self, tx, name):
        fields = namedtuple_fields(self.tuple_cls)
        if name not in fields:
            unimplemented(f"NamedTupleVariable.{name}")
        return self.items[fields.index(name)].add_options(self)

    def call_hasattr(self, tx, name: str) -> "VariableTracker":
        options = VariableTracker.propagate(self)
        fields = namedtuple_fields(self.tuple_cls)
        return variables.ConstantVariable(name in fields, **options)


class SliceVariable(BaseListVariable):
    def __init__(self, items, **kwargs):
        from .tensor import DynamicShapeVariable

        if any([isinstance(x, DynamicShapeVariable) for x in items]):
            unimplemented("Dynamic slicing not supported")

        items_to_map = items
        start, stop, step = [variables.ConstantVariable(None)] * 3

        if len(items_to_map) == 1:
            (stop,) = items_to_map
        elif len(items_to_map) == 2:
            start, stop = items_to_map
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
        else:
            raise AssertionError()

        # Avoids a .item() call in the tensor slice that would attempt to get a
        # value out fake tensors, and which would determine the output shape of
        # the slice.  It is a workaround until
        # https://github.com/pytorch/pytorch/pull/83567 is landed and there is
        # more complete support for breaking on data dependent operators.
        if not config.capture_scalar_outputs:
            for limit in (start, stop, step):
                if isinstance(limit, (variables.TensorVariable, DynamicShapeVariable)):
                    unimplemented("Dynamic slicing not supported")

        super().__init__([start, stop, step], **kwargs)

    def as_proxy(self):
        return slice(*self._as_proxy())

    def python_type(self):
        return slice

    def as_python_constant(self):
        return slice(*[x.as_python_constant() for x in self.items])

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction("BUILD_SLICE", len(self.items))]

    def var_getattr(self, tx, name):
        fields = ["start", "stop", "step"]
        if name not in fields:
            unimplemented(f"slice.{name}")
        return self.items[fields.index(name)].add_options(self)


class ListIteratorVariable(VariableTracker):
    def __init__(self, items, index: int = 0, **kwargs):
        super(ListIteratorVariable, self).__init__(**kwargs)
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
            create_instruction("BUILD_TUPLE", len(remaining_items)),
            create_instruction("GET_ITER"),
        ]
