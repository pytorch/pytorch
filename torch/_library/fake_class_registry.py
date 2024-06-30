# mypy: allow-untyped-defs
import logging
from typing import Any, Dict, Optional, Protocol, Tuple

import torch

from torch._library.utils import parse_namespace

log = logging.getLogger(__name__)


class FakeScriptObject:
    def __init__(self, wrapped_obj: Any, script_class_name: str, x: torch.ScriptObject):
        self.wrapped_obj = wrapped_obj

        # The fully qualified name of the class of original script object
        self.script_class_name = script_class_name
        self.real_obj = x


class FakeScriptMethod:
    def __init__(
        self,
        self_fake_obj: FakeScriptObject,
        method_name: str,
        schema: Optional[torch.FunctionSchema],
    ):
        self.self_fake_obj = self_fake_obj
        self.method_name = method_name
        self.schema = schema

    def __call__(self, *args, **kwargs):
        from torch._higher_order_ops.torchbind import call_torchbind

        return call_torchbind(self.self_fake_obj, self.method_name, *args, **kwargs)


class HasStaticMethodFromReal(Protocol):
    @classmethod
    def from_real(cls, real_obj: torch.ScriptObject):
        pass


class FakeClassRegistry:
    def __init__(self):
        self._registered_class: Dict[str, Any] = {}

    def has_impl(self, full_qualname: str) -> bool:
        return full_qualname in self._registered_class

    def get_impl(self, full_qualname: str) -> Any:
        self._check_registered(full_qualname)
        return self._registered_class[full_qualname]

    def register(self, full_qualname: str, fake_class=None) -> None:
        if self.has_impl(full_qualname):
            log.warning(
                "%s is already registered. Previous fake class is overrided with  %s.",
                full_qualname,
                fake_class,
            )
        self._registered_class[full_qualname] = fake_class

    def deregister(self, full_qualname: str) -> Any:
        if not self.has_impl(full_qualname):
            log.warning(
                "Cannot deregister %s. Please use register_fake_class to register it first."
                " Or do you dereigster it twice?",
                full_qualname,
            )
        else:
            return self._registered_class.pop(full_qualname)

    def clear(self) -> None:
        self._registered_class.clear()

    def _check_registered(self, full_qualname: str) -> None:
        if full_qualname not in self._registered_class:
            raise RuntimeError(
                f"{full_qualname} is not registered. Please use register_fake_class to register it first."
            )


global_fake_class_registry = FakeClassRegistry()


# TODO: add this check at compile time for __obj_flatten__.
def _check_valid_flat_script_obj(flat_x):
    if not isinstance(flat_x, tuple):
        raise RuntimeError("Expect flat x to be a tuple.")

    for tp in flat_x:
        if not isinstance(tp, tuple):
            raise RuntimeError("Expect flat x to be a tuple of tuples.")

        if not len(tp) == 2 or not isinstance(tp[0], str):
            raise RuntimeError(
                "Expect element of flat x to be a tuple of two elements with first element being a string"
            )


def to_fake_obj(fake_mode, x: torch.ScriptObject) -> FakeScriptObject:
    import torch.utils._pytree as pytree
    from torch.utils._python_dispatch import _disable_current_modes

    # x.__obj_flatten__() could be calling some tensor operations inside but we don't
    # want to call these ops in surrounding dispatch modes when executing it.
    # Otherwise, for example, the fake tensor modes will error out when the tensors inside
    # script obeject execute some operations like clone if allow_non_fake_input flag is set.
    with _disable_current_modes():
        flat_x = x.__obj_flatten__()  # type: ignore[attr-defined]

    _check_valid_flat_script_obj(flat_x)

    fake_flattened = pytree.tree_map_only(
        torch.Tensor,
        lambda t: fake_mode.from_tensor(t),
        flat_x,
    )

    fake_x = _find_fake_class_for_script_object(x).__obj_unflatten__(fake_flattened)

    fake_x_wrapped = FakeScriptObject(fake_x, x._type().qualified_name(), x)  # type: ignore[attr-defined]

    for name in x._method_names():  # type: ignore[attr-defined]
        attr = getattr(fake_x, name, None)
        if attr:
            if not callable(attr):
                raise RuntimeError(f"Expect {name} to be a callable but got {attr}.")

            real_attr = getattr(x, name)  # type: ignore[attr-defined]

            # real attr sometimes is not torch.ScriptMethod thus doesn't have schema e.g. __init___ or __eq__
            method_schema: Optional[torch.FunctionSchema] = None
            if isinstance(real_attr, torch.ScriptMethod):
                method_schema = real_attr.schema  # type: ignore[attr-defined]

            setattr(
                fake_x_wrapped,
                name,
                FakeScriptMethod(fake_x_wrapped, name, method_schema),
            )
        else:
            log.warning("fake object of %s doesn't implement method %s.", x, name)
    return fake_x_wrapped


def register_fake_class(qualname, fake_class: Optional[HasStaticMethodFromReal] = None):
    r"""Register a fake implementation for this class.

    It's in the same spirit of registering a fake implementation for
    an operator but with the difference that it
    associates a fake class with the original torch bind class (registered
    with torch::class_). In this way, torch.compile can handle them properly
    in components such as Dynamo and AOTAutograd.

    This API may be used as a decorator (see example). For the fake class, users
    are required to provide a from_real classmethod that takes a real object and
    returns an instance of the fake class. All tensors in the fake object should also
    be properly fakified with to_fake_tensor() in from_real.


    Examples:
        # For a custom class Foo defined in test_custom_class_registration.cpp:

        TORCH_LIBRARY(_TorchScriptTesting, m) {
          m.class_<TensorQueue>("_TensorQueue")
            .def(torch::init<at::Tensor>())
            .def("push", &TensorQueue::push)
            .def("pop", &TensorQueue::pop)
            .def("top", &TensorQueue::top)
            .def("size", &TensorQueue::size)
            .def("clone_queue", &TensorQueue::clone_queue)
            .def("__obj_flatten__", &TensorQueue::__obj_flatten__)
            .def_pickle(
                // __getstate__
                [](const c10::intrusive_ptr<TensorQueue>& self)
                    -> c10::Dict<std::string, at::Tensor> {
                  return self->serialize();
                },
                // __setstate__
                [](c10::Dict<std::string, at::Tensor> data)
                    -> c10::intrusive_ptr<TensorQueue> {
                  return c10::make_intrusive<TensorQueue>(std::move(data));
                });
            };
        # We could register a fake class FakeTensorQueue in Python as follows:
        import torch

        @torch._library.register_fake_class("_TorchScriptTesting::_TensorQueue")
        class FakeTensorQueue:
            def __init__(self, queue):
                self.queue = queue

            @classmethod
            def __obj_unflatten__(cls, flattened_ctx):
                return cls(**dict(ctx))

            def push(self, x):
                self.queue.append(x)

            def pop(self):
                return self.queue.pop(0)

            def size(self):
                return len(self.queue)

    In this example, the original TensorQeue need to addd a __obj_flatten__ method
    to the class TensorQueue and the flattend result is passed into FakeTensorQueue's
    __obj_unflatten__ as inputs to create a fake class. This protocol allows pytorch to look
    at the contents of the script object and properly handle them in the subsystems
    like dynamo, aot_aotugrad or more.
    """

    def inner(fake_class: HasStaticMethodFromReal):
        ns, name = parse_namespace(qualname)

        # This also checks whether the refered torch::class_ exists.
        torchbind_class = torch._C._get_custom_class_python_wrapper(ns, name)

        from_method = getattr(fake_class, _CONVERT_FROM_REAL_NAME, None)
        if not from_method:
            raise RuntimeError(
                f"{fake_class} doesn't define a classmethod {_CONVERT_FROM_REAL_NAME}."
            )

        if not isinstance(fake_class.__dict__[_CONVERT_FROM_REAL_NAME], classmethod):
            raise RuntimeError(
                f"{_CONVERT_FROM_REAL_NAME} method is not a classmethod."
            )

        global_fake_class_registry.register(_full_qual_class_name(qualname), fake_class)
        return fake_class

    if fake_class is None:
        return inner
    return inner(fake_class)


def deregister_fake_class(qualname):
    return global_fake_class_registry.deregister(_full_qual_class_name(qualname))


def has_fake_class(full_qualname) -> bool:
    return global_fake_class_registry.has_impl(full_qualname)


def find_fake_class(full_qualname) -> Optional[Any]:
    if not has_fake_class(full_qualname):
        return None
    return global_fake_class_registry.get_impl(full_qualname)


def _full_qual_class_name(qualname: str) -> str:
    ns, name = parse_namespace(qualname)
    return "__torch__.torch.classes." + ns + "." + name


# Return the namespace and class name from fully qualified name.
def _ns_and_class_name(full_qualname: str) -> Tuple[str, str]:
    splits = full_qualname.split(".")
    assert len(splits) == 5
    _torch, torch_ns, classes, ns, class_name = splits
    return ns, class_name


def _find_fake_class_for_script_object(x: torch.ScriptObject) -> Any:
    full_qualname = x._type().qualified_name()  # type: ignore[attr-defined]
    ns, class_name = _ns_and_class_name(full_qualname)
    fake_class = find_fake_class(full_qualname)
    if fake_class is None:
        raise RuntimeError(
            f" ScriptObject's {full_qualname} haven't registered a fake class."
            f" Please use register_fake_class({ns}::{class_name}) to annotate a fake class for the script obj."
            f" Specifically, create a python class that implements a fake version for all the methods"
            f" that're used in the program and put annotated class in the program e.g. after loading the library."
            f" The fake methods can be written in the same way as a meta kernel for an operator but need to additionally"
            f" simulate the object's states. Be sure to add a {_CONVERT_FROM_REAL_NAME} classmethod"
            f" to enable creating a fake obj from a real one."
        )
    return fake_class


_CONVERT_FROM_REAL_NAME = "__obj_unflatten__"


def _fake_obj_from_real(fake_mode, x) -> Any:
    fake_class = _find_fake_class_for_script_object(x)

    from_real_method = getattr(fake_class, _CONVERT_FROM_REAL_NAME, None)
    if not from_real_method:
        raise RuntimeError(
            f"{fake_class} must define a classmethod {_CONVERT_FROM_REAL_NAME}"
            f" that converts the real object to the fake object."
        )

    # from_real defined by user need the ctx to fakify the tensor states.
    ctx = torch._library.fake_impl.FakeImplCtx(fake_mode, None)
    with torch._library.fake_impl.set_ctx_getter(lambda: ctx):
        return fake_class.from_real(x)
