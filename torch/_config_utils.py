import contextlib
import copy
import dataclasses
import functools
import inspect
import pickle
import types
import unittest
from types import ModuleType
from typing import Any, Dict, Set

import torch


class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    def __call__(self, func):
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):  # type: ignore[misc, valid-type]
                @classmethod
                def setUpClass(cls):
                    self.__enter__()  # type: ignore[attr-defined]
                    try:
                        super().setUpClass()
                    except Exception:
                        self.__exit__(None, None, None)  # type: ignore[attr-defined]
                        raise

                @classmethod
                def tearDownClass(cls):
                    try:
                        super().tearDownClass()
                    finally:
                        self.__exit__(None, None, None)  # type: ignore[attr-defined]

            _TestCase.__name__ = func.__name__  # type:ignore[attr-defined]
            _TestCase.__qualname__ = func.__qualname__
            _TestCase.__module__ = func.__module__
            return _TestCase

        return super().__call__(func)


def _dataclass_obj_to_flat_dict(dc):
    fields = getattr(type(dc), "__dataclass_fields__", {})
    result = {}
    for name, field in fields.items():
        if not field.metadata.get("skip_pickle", False):
            value = getattr(dc, name)
            if dataclasses.is_dataclass(value):
                for k2, v2 in _dataclass_obj_to_flat_dict(value).items():
                    result[f"{name}.{k2}"] = v2
            else:
                result[name] = value
    return result


def _codegen_changes_of_dataclass_obj(dc, name):
    values = _dataclass_obj_to_flat_dict(dc)
    defaults = _dataclass_obj_to_flat_dict(type(dc)())

    result = []
    for k, v in values.items():
        if defaults[k] != v:
            result.append(f"{name}.{k} = {v!r}")
    return "\n".join(result)


class ConfigMixin:
    """Mixin class shared between dataclasses that meant to represent a config.

    Usage:
        @dataclass
        class SomeConfig(ConfigMixin):
            a: int
            b: int
            ...
            c: SomeOtherNestedConfig

        Note: c the nested config should also inherit ConfigMixin.
        ie.

        @dataclass
        class SomeOtherNestedConfig(ConfigMixin):
            d: ...

    This mixin will:
        1. Make the subclass pickable by allowing one to mark non-picklable
           field with {'skip_pickle': True} metadata.
        2. `save_config` which returns the config as bytes, and
           `load_config` what replaces fields of an instance with the content
           of serialized string. Note: these are legacy methods, it's better
           to use pickle directly.
        3. .to_dict will create a flat dict:
            in the SomeConfig above, it will return a dictionary with keys
            'a', 'b', 'c.d'
        4. .codegen_config will create a string of python code with
           modifications of this config compared to the default values.
    """

    def __getstate__(self):
        start = {}
        for name, field in self._fields().items():
            if not field.metadata.get("skip_pickle", False):
                start[name] = getattr(self, name)
        return start

    def __setstate__(self, state):
        self.__init__()  # type: ignore[misc]
        self.__dict__.update(state)

    def save_config(self):
        return pickle.dumps(self, protocol=2)

    def load_config(self, content):
        state = pickle.loads(content)
        self.__dict__.update(state.__dict__)
        return self

    def _update_single(self, key, val):
        pieces = key.split(".")
        current = self
        for token in pieces[:-1]:
            current = getattr(current, token)
        setattr(current, pieces[-1], val)

    def _get_single(self, key):
        pieces = key.split(".")
        current = self
        for token in pieces:
            current = getattr(current, token)
        return current

    def update(self, content_dict):
        for k, v in content_dict.items():
            self._update_single(k, v)

    @classmethod
    def _fields(cls):
        return getattr(cls, "__dataclass_fields__", {})

    def __setattr__(self, key, val):
        if (
            not inspect.isclass(val)
            and key not in type(self).__dict__
            and key not in self._fields()
        ):
            raise AttributeError(
                f"Trying to set attribute {key} that is not part of this config {type(self).__name__}"
            )
        super().__setattr__(key, val)

    def to_dict(self):
        flatdict = _dataclass_obj_to_flat_dict(self)
        return BoundDict(flatdict, self)

    @classmethod
    def is_fbcode(cls):
        return not hasattr(torch.version, "git_version")

    def patch(self, arg1=None, arg2=None, **kwargs):
        """
        Decorator and/or context manager to make temporary changes to a config.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2):
            @config.patch({"name1": val1, "name2", val2})
            def foo(...):
                ...

        As a context manager:

            with config.patch("name", val):
                ...
        """
        if arg1 is not None:
            if arg2 is not None:
                # patch("key", True) syntax
                changes = {arg1: arg2}
            else:
                # patch({"key": True}) syntax
                changes = arg1
            assert not kwargs
        else:
            # patch(key=True) syntax
            changes = kwargs
            assert arg2 is None
        assert isinstance(changes, dict), f"expected `dict` got {type(changes)}"
        prior: Dict[str, Any] = {}
        config = self

        class ConfigPatch(ContextDecorator):
            def __enter__(self):
                assert not prior
                for key in changes.keys():
                    # KeyError on invalid entry
                    prior[key] = config._get_single(key)
                config.update(changes)

            def __exit__(self, exc_type, exc_val, exc_tb):
                config.update(prior)
                prior.clear()

        return ConfigPatch()

    def codegen_config(self, name=None):
        """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """
        lines = []
        if name is None:
            name = self.__name__  # type: ignore[attr-defined]
        return _codegen_changes_of_dataclass_obj(self, name)


class BoundDict(dict):
    def __init__(self, orig, config):
        super().__init__(orig)
        self._config = config

    def __setitem__(self, key, val):
        self._config._update_single(key, val)
        super().__setitem__(key, val)


def make_config_dataclass(name, config_module):
    fields = []
    module_name = ".".join(config_module.__name__.split(".")[:-1])

    ignored_fields: Set[str] = getattr(config_module, "_save_config_ignore", set())
    for fname, default_value in config_module.__dict__.items():
        if callable(default_value) or isinstance(default_value, ModuleType):
            # Module level functions and imported modules are
            # usually not part of config.
            continue
        if fname.startswith("__"):
            continue
        annotation = config_module.__annotations__.get(fname)
        assert (
            annotation is not None
        ), f"Please specify type annotation for {fname} in {config_module.__name__}"
        should_skip = fname in ignored_fields
        field = dataclasses.field(
            default_factory=functools.partial(copy.copy, default_value),
            metadata={"skip_pickle": should_skip},
        )
        fields.append((fname, annotation, field))
    fields.append(("__name__", str, dataclasses.field(default=config_module.__name__)))
    cls = dataclasses.make_dataclass(
        name, fields, bases=(ConfigMixin, types.ModuleType)
    )
    cls.__dataclass_fields__["__name__"].default = config_module.__name__  # type: ignore[attr-defined]

    # NOTE: this is to make pickle work. In Python 3.12 make_dataclass
    # will take a module argument that it would set __module__ field inside.
    cls.__module__ = module_name
    return cls


def install_config_module(classname, module):
    orig_name = module.__name__
    module.__class__ = make_config_dataclass(classname, module)
    module.__init__()  # call constructor by hand
    module.__name__ = orig_name
