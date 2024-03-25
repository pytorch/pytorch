import logging
from typing import Any, Dict, Optional

import torch

from torch._library.utils import parse_namespace

log = logging.getLogger(__name__)


class AbstractClassRegistry:
    def __init__(self):
        self._registered_class: Dict[str, Any] = {}

    def has_impl(self, full_qualname: str):
        return full_qualname in self._registered_class

    def get_impl(self, full_qualname: str):
        self._check_registered(full_qualname)
        return self._registered_class[full_qualname]

    def register(self, full_qualname: str, abstract_class=None) -> None:
        if self.has_impl(full_qualname):
            raise RuntimeError(
                f"{full_qualname} is already registered. Please use deregister to deregister it first."
            )
        self._registered_class[full_qualname] = abstract_class

    def deregister(self, full_qualname: str) -> Any:
        if not self.has_impl(full_qualname):
            raise RuntimeError(
                f"Cannot deregister {full_qualname}. Please use impl_abstract_class to register it first."
                f" Or do you dereigster it twice?"
            )
        self._check_registered(full_qualname)
        return self._registered_class.pop(full_qualname)

    def clear(self):
        self._registered_class.clear()

    def _check_registered(self, full_qualname: str):
        if full_qualname not in self._registered_class:
            raise RuntimeError(
                f"{full_qualname} is not registered. Please use impl_abstract_class to register it first."
            )


global_abstract_class_registry = AbstractClassRegistry()


def create_abstract_obj(x: torch.ScriptObject):
    abstract_x = _abstract_obj_from_concrete(x)

    def _call_torchbind(method_name):
        from torch._higher_order_ops.torchbind import call_torchbind

        def wrapped(self_, *args, **kwargs):
            return call_torchbind(self_, method_name, *args, **kwargs)

        return wrapped

    abstract_x_wrapped = AbstractScriptObject(abstract_x)
    for name in x._method_names():  # type: ignore[attr-defined]
        attr = getattr(abstract_x, name, None)
        if attr:
            if not callable(attr):
                raise RuntimeError(f"Expect {name} to be a callable but got {attr}.")
            setattr(
                abstract_x_wrapped,
                name,
                _call_torchbind(name).__get__(abstract_x_wrapped),
            )
        else:
            log.warning("Abstract object of %s doesn't implement method %s.", x, name)
    return abstract_x_wrapped


def impl_abstract_class(qualname, abstract_class=None):
    r"""Register an abstract implementation for this class.

    It's in the same spirit of registering an abstract implementation for
    an operator with impl_abstract but with the difference that it
    associates a abstract class with the original torch bind class (registered
    with torch::class_).  In this way, , object of the class can be properly
    guarded and tracked by components in PT2 stack such as Dynamo and AOTAutograd.

    This API may be used as a decorator (see examples).

    Examples:
        # For a torch Bind class Foo defined in test_custom_class_registration.cpp:
        TORCH_LIBRARY(_TorchScriptTesting, m) {
            m.class_<Foo>("_Foo")
                .def(torch::init<int64_t, int64_t>())
                // .def(torch::init<>())
                .def("info", &Foo::info)
                .def("increment", &Foo::increment)
                .def("add", &Foo::add)
                .def("add_tensor", &Foo::add_tensor)
                .def("__eq__", &Foo::eq)
                .def("combine", &Foo::combine)
                .def_pickle(
                    [](c10::intrusive_ptr<Foo> self) { // __getstate__
                      return std::vector<int64_t>{self->x, self->y};
                    },
                    [](std::vector<int64_t> state) { // __setstate__
                      return c10::make_intrusive<Foo>(state[0], state[1]);
                });
        # We could register a abstract class abstractFoo in Python as follows:
        import torch

        @torch._library.impl_abstract_class("_TorchScriptTesting::_Foo")
        class abstractFoo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            @classmethod
            def from_concrete(cls, foo_obj):
                x, y = foo_obj.__getstate__()
                return cls(x, y)

            def add_tensor(self, z):
                return (self.x + self.y) * z1

    Temporal Limitations:
        - We don't support method call on the script object yet. Please use custom op to manipulate them. Please
          implement a custom op that takes the script object as input and call the method in the custom op.

    Examples:
        # CPU impl in test_custom_class_registration.cpp:

        at::Tensor takes_foo(c10::intrusive_ptr<Foo> foo, at::Tensor x) {
          return foo->add_tensor(x);
        }
        TORCH_LIBRARY_IMPL(_TorchScriptTesting, CPU, m) {
          m.impl("takes_foo", takes_foo);
        }

        # abstract impl in torchbind_impls.py:
        @torch.library.impl_abstract("_TorchScriptTesting::takes_foo")
        def foo_add_tensor(foo, z):
            return foo.add_tensor(z)
    """

    def inner(abstract_class):
        ns, name = parse_namespace(qualname)

        # This also checks whether the refered torch::class_ exists.
        torchbind_class = torch._C._get_custom_class_python_wrapper(ns, name)

        from_method = getattr(abstract_class, _CONVERT_FROM_REAL_NAME, None)
        if not from_method:
            raise RuntimeError(
                f"{abstract_class} doesn't define a classmethod from_concrete."
            )

        if not isinstance(
            abstract_class.__dict__[_CONVERT_FROM_REAL_NAME], classmethod
        ):
            raise RuntimeError(
                f"{_CONVERT_FROM_REAL_NAME} method is not a classmethod."
            )

        global_abstract_class_registry.register(
            _full_qual_class_name(qualname), abstract_class
        )
        return abstract_class

    if abstract_class is None:
        return inner
    return inner(abstract_class)


def deregister_abstract_impl(qualname):
    return global_abstract_class_registry.deregister(_full_qual_class_name(qualname))


def has_abstract_impl(full_qualname):
    return global_abstract_class_registry.has_impl(full_qualname)


def find_abstract_impl(full_qualname) -> Optional[Any]:
    if not has_abstract_impl(full_qualname):
        return None
    return global_abstract_class_registry.get_impl(full_qualname)


def _full_qual_class_name(qualname: str):
    ns, name = parse_namespace(qualname)
    return "__torch__.torch.classes." + ns + "." + name


# Return the namespace and class name of a script object.
def _ns_and_class_name(full_qualname: str):
    splits = full_qualname.split(".")
    assert len(splits) == 5
    _torch, torch_ns, classes, ns, class_name = splits
    return ns, class_name


def _find_abstract_class_for_script_object(x: torch.ScriptObject):
    full_qualname = x._type().qualified_name()  # type: ignore[attr-defined]
    ns, class_name = _ns_and_class_name(full_qualname)
    abstract_class = find_abstract_impl(full_qualname)
    if abstract_class is None:
        raise RuntimeError(
            f" ScriptObject's {full_qualname} haven't registered a abstract class."
            f" Please use impl_abstract_class({ns}::{class_name}) to annotate a abstract class for the script obj."
            f" Specifically, create a python class that implements a abstract version for all the methods"
            f" that're used in the program and put annotated class in the program e.g. after loading the library."
            f" The abstract methods can be written in the same way as a meta kernel for an operator but need to also"
            f" simulate the object's states when necessary. Be sure to add a {_CONVERT_FROM_REAL_NAME} classmethod"
            f" to enable creating a abstract obj from a real one."
        )
    return abstract_class


_CONVERT_FROM_REAL_NAME = "from_concrete"


class AbstractScriptObject:
    def __init__(self, wrapped_obj):
        self.wrapped_obj = wrapped_obj


def _abstract_obj_from_concrete(x):
    abstract_class = _find_abstract_class_for_script_object(x)

    from_concrete_method = getattr(abstract_class, _CONVERT_FROM_REAL_NAME, None)
    if not from_concrete_method:
        raise RuntimeError(
            f"{abstract_class} must define a classmethod {_CONVERT_FROM_REAL_NAME}"
            f" that converts the real object to the abstract object."
        )

    return abstract_class.from_concrete(x)
