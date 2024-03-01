from typing import Any, Dict

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
        self._registered_class[full_qualname] = fake_class

    def deregister(self, full_qualname: str) -> Any:
        self._check_registered(full_qualname)
        return self._registered_class.pop(full_qualname)

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
    def inner(fake_class):
        global global_abstract_class_registry
        global_abstract_class_registry.register(
            _full_qual_class_name(qualname), fake_class
        )
        return fake_class

    if fake_class is None:
        return inner
    return inner(fake_class)


def deregister_abstract_impl(qualname):
    global global_abstract_class_registry
    return global_abstract_class_registry.deregister(_full_qual_class_name(qualname))
