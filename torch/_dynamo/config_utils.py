import contextlib

import copy
import pickle
import unittest
from types import FunctionType, ModuleType
from typing import Any, Dict, Set
from unittest import mock

# Types saved/loaded in configs
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)


def install_config_module(module):
    """
    Converts a module-level config into a `ConfigModule()`
    """

    class ConfigModuleInstance(ConfigModule):
        _bypass_keys = set()

    def visit(source, dest, prefix):
        """Walk the module structure and move everything to module._config"""
        for key, value in list(source.__dict__.items()):
            if key.startswith("__") or isinstance(value, (ModuleType, FunctionType)):
                continue

            name = f"{prefix}{key}"
            if isinstance(value, CONFIG_TYPES):
                config[name] = value
                default[name] = value
                if dest is module:
                    delattr(module, key)
            elif isinstance(value, type):
                assert value.__module__ == module.__name__
                # a subconfig with `class Blah:` syntax
                proxy = SubConfigProxy(module, f"{name}.")
                visit(value, proxy, f"{name}.")
                setattr(dest, key, proxy)
            else:
                raise AssertionError(f"Unhandled config {key}={value} ({type(value)})")

    config = dict()
    default = dict()
    visit(module, module, "")
    module._config = config
    module._default = default
    module._allowed_keys = set(config.keys())
    module.__class__ = ConfigModuleInstance


class ConfigModule(ModuleType):
    # The default values of the configuration settings.  This can be used to
    # determine if the config has been changed or not.
    _default: Dict[str, Any]
    # The actual configuration settings.  E.g., torch._dynamo.config.debug
    # would live as "debug" in the key, and torch._inductor.config.triton.cudagraphs
    # maps as "triton.cudagraphs"
    _config: Dict[str, Any]
    _allowed_keys: Set[str]
    _bypass_keys: Set[str]

    def __init__(self):
        raise NotImplementedError(
            f"use {__name__}.install_config_module(sys.modules[__name__])"
        )

    def __setattr__(self, name, value):
        if name in self._bypass_keys:
            super().__setattr__(name, value)
        elif name not in self._allowed_keys:
            raise AttributeError(f"{self.__name__}.{name} does not exist")
        else:
            self._config[name] = value

    def __getattr__(self, name):
        try:
            return self._config[name]
        except KeyError:
            # make hasattr() work properly
            raise AttributeError(f"{self.__name__}.{name} does not exist")

    def __delattr__(self, name):
        # must support delete because unittest.mock.patch deletes
        # then recreate things
        del self._config[name]

    def save_config(self):
        """Convert config to a pickled blob"""
        config = dict(self._config)
        for key in config.get("_save_config_ignore", ()):
            config.pop(key)
        return pickle.dumps(config, protocol=2)

    def codegen_config(self):
        """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """
        lines = []
        mod = self.__name__
        for k, v in self._config.items():
            if k in self._config.get("_save_config_ignore", ()):
                continue
            if v == self._default[k]:
                continue
            lines.append(f"{mod}.{k} = {v!r}")
        return "\n".join(lines)

    def load_config(self, data):
        """Restore from a prior call to save_config()"""
        self.to_dict().update(pickle.loads(data))

    def to_dict(self):
        return self._config

    def get_config_copy(self):
        return copy.deepcopy(self._config)

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
        prior = {}
        config = self

        class ConfigPatch(ContextDecorator):
            def __enter__(self):
                assert not prior
                for key in changes.keys():
                    # KeyError on invalid entry
                    prior[key] = config._config[key]
                config._config.update(changes)

            def __exit__(self, exc_type, exc_val, exc_tb):
                config._config.update(prior)
                prior.clear()

        return ConfigPatch()


class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    def __call__(self, func):
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):
                @classmethod
                def setUpClass(cls):
                    self.__enter__()
                    try:
                        super().setUpClass()
                    except Exception:
                        self.__exit__(None, None, None)
                        raise

                @classmethod
                def tearDownClass(cls):
                    try:
                        super().tearDownClass()
                    finally:
                        self.__exit__(None, None, None)

            _TestCase.__name__ = func.__name__
            _TestCase.__qualname__ = func.__qualname__
            _TestCase.__module__ = func.__module__

            return _TestCase

        return super().__call__(func)


class SubConfigProxy:
    """
    Shim to redirect to main config.
    `config.triton.cudagraphs` maps to _config["triton.cudagraphs"]
    """

    def __init__(self, config, prefix):
        # `super().__setattr__` to bypass custom `__setattr__`
        super().__setattr__("_config", config)
        super().__setattr__("_prefix", prefix)

    def __setattr__(self, name, value):
        return self._config.__setattr__(self._prefix + name, value)

    def __getattr__(self, name):
        return self._config.__getattr__(self._prefix + name)

    def __delattr__(self, name):
        return self._config.__delattr__(self._prefix + name)


def patch_object(obj, name, value):
    """
    Workaround `mock.patch.object` issue with ConfigModule
    """
    if isinstance(obj, ConfigModule):
        return obj.patch(name, value)
    return mock.patch.object(obj, name, value)
