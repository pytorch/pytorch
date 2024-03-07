from typing import Any, Dict, Optional

import torch

from torch._library.utils import parse_namespace


class AbstractClassRegistry:
    def __init__(self):
        self._registered_class: Dict[str, Any] = {}

    def has_impl(self, full_qualname: str):
        return full_qualname in self._registered_class

    def get_impl(self, full_qualname: str):
        self._check_registered(full_qualname)
        return self._registered_class[full_qualname]

    def register(self, full_qualname: str, fake_class=None) -> None:
        if self.has_impl(full_qualname):
            raise RuntimeError(
                f"{full_qualname} is already registered. Please use deregister to deregister it first."
            )
        self._registered_class[full_qualname] = fake_class

    def deregister(self, full_qualname: str) -> Any:
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


def _full_qual_class_name(qualname: str):
    ns, name = parse_namespace(qualname)
    return "__torch__.torch.classes." + ns + "." + name


def impl_abstract_class(qualname, fake_class=None):
    r"""Register an abstract implementation for this class.

    It's in the same spirit of registering an abstract implementation for
    an operator with impl_abstract but with the difference that it
    associates a fake class with the original torch bind class (registered
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
        # We could register a fake class FakeFoo in Python as follows:
        import torch

        @torch._library.impl_abstract_class("_TorchScriptTesting::_Foo")
        class FakeFoo:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            @classmethod
            def from_real(cls, foo_obj):
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

    def inner(fake_class):
        ns, name = parse_namespace(qualname)

        # Check whether the refered torch::class_ exists.
        torch._C._get_custom_class_python_wrapper(ns, name)

        if not isinstance(fake_class.__dict__.get("from_real", None), classmethod):
            raise RuntimeError(
                f"{fake_class} must define a classmethod from_real that converts the real object to the fake object."
            )

        global_abstract_class_registry.register(
            _full_qual_class_name(qualname), fake_class
        )
        return fake_class

    if fake_class is None:
        return inner
    return inner(fake_class)


def deregister_abstract_impl(qualname):
    return global_abstract_class_registry.deregister(_full_qual_class_name(qualname))


def has_fake_impl(full_qualname):
    return global_abstract_class_registry.has_impl(full_qualname)


def find_fake_impl(full_qualname) -> Optional[Any]:
    if not has_fake_impl(full_qualname):
        return None
    return global_abstract_class_registry.get_impl(full_qualname)
