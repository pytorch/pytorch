import contextlib

import copy
import inspect
import io
import itertools
import pickle
import tokenize
import unittest
import warnings
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

    def visit(source, dest, prefix, ignore_all=False):
        """
        Walk the module structure and move everything to
        module._config and module._compile_ignored
        """
        ignored = get_assignments_with_compile_ignored_comments(source)
        for key, value in list(source.__dict__.items()):
            if key.startswith("__") or isinstance(value, (ModuleType, FunctionType)):
                continue

            name = f"{prefix}{key}"
            if isinstance(value, CONFIG_TYPES):
                if ignore_all or key in ignored:
                    compile_ignored[name] = value
                else:
                    config[name] = value
                default[name] = value
                if dest is module:
                    delattr(module, key)
            elif isinstance(value, type):
                assert value.__module__ == module.__name__
                # a subconfig with `class Blah:` syntax
                proxy = SubConfigProxy(module, f"{name}.")
                visit(value, proxy, f"{name}.", ignore_all=name in ignored)
                setattr(dest, key, proxy)
            else:
                raise AssertionError(f"Unhandled config {key}={value} ({type(value)})")

    config = dict()
    compile_ignored = dict()
    default = dict()

    visit(module, module, "")
    module._config = config
    module._compile_ignored = compile_ignored
    module._default = default

    config_keys, compile_ignored_keys = set(config.keys()), set(compile_ignored.keys())
    assert config_keys.isdisjoint(compile_ignored_keys)
    module._allowed_keys = config_keys | compile_ignored_keys
    module._compile_ignored_keys = set(compile_ignored.keys())

    module.__class__ = ConfigModuleInstance


# Gets all the keys (i.e. assignments) with a @compile_ignored comment
def get_assignments_with_compile_ignored_comments(module):
    source_code = inspect.getsource(module)
    assignments = set()

    # Tokenize the source code to retrieve comments
    tokens = tokenize.tokenize(io.BytesIO(source_code.encode("utf-8")).readline)
    current_comment = "", -1
    prev_name = ""
    prev_assigned = "", -1

    for token in tokens:
        if token.type == tokenize.COMMENT:
            maybe_current = token.string.strip()
            if "@compile_ignored" in maybe_current:
                current_comment = maybe_current, token.start[0]
                if token.start[0] == prev_assigned[1]:
                    # Check if the current assignment is followed with
                    # a same-line comment with '@compile_ignored'
                    assignments.add(prev_assigned[0])
        elif token.type == tokenize.NAME:
            prev_name = token.string
        elif token.type == tokenize.OP and token.string == "=":
            prev_assigned = prev_name, token.start[0]
            # Check if the current assignment follows a comment with '@compile_ignored'
            if (
                "@compile_ignored" in current_comment[0]
                and current_comment[1] == token.start[0] - 1
            ):
                assignments.add(prev_name)
    return assignments


class ConfigModule(ModuleType):
    # The default values of the configuration settings.  This can be used to
    # determine if the config has been changed or not.
    _default: Dict[str, Any]
    # The actual configuration settings.  E.g., torch._dynamo.config.debug
    # would live as "debug" in the key, and torch._inductor.config.triton.cudagraphs
    # maps as "triton.cudagraphs"
    _config: Dict[str, Any]
    # The same as _config, but for keys that are annotated wtih @compile_ignored
    # in a comment on the line preceding or the same line
    _compile_ignored: Dict[str, Any]
    _allowed_keys: Set[str]
    _bypass_keys: Set[str]
    _compile_ignored_keys: Set[str]

    def __init__(self):
        raise NotImplementedError(
            f"use {__name__}.install_config_module(sys.modules[__name__])"
        )

    def __hasattr__(self, name, value):
        return name in self._config or name in self._compile_ignored

    def __setattr__(self, name, value):
        if name in self._bypass_keys:
            super().__setattr__(name, value)
        elif name in self._compile_ignored_keys:
            self._compile_ignored[name] = value
        elif name in self._allowed_keys:
            self._config[name] = value
        else:
            raise AttributeError(f"{self.__name__}.{name} does not exist")

    def __getattr__(self, name):
        try:
            if name in self._compile_ignored_keys:
                return self._compile_ignored[name]
            else:
                return self._config[name]
        except KeyError:
            # make hasattr() work properly
            raise AttributeError(f"{self.__name__}.{name} does not exist")

    def __delattr__(self, name):
        # must support delete because unittest.mock.patch deletes
        # then recreate things
        if name in self._compile_ignored_keys:
            del self._compile_ignored[name]
        else:
            del self._config[name]

    def save_config(self):
        """Convert config to a pickled blob"""
        config = {**self._config, **self._compile_ignored}
        for key in config.get("_save_config_ignore", ()):
            config.pop(key)
        return pickle.dumps(config, protocol=2)

    def codegen_config(self):
        """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """
        lines = []
        mod = self.__name__
        has_saved_config_ignore = hasattr(self, "_save_config_ignore")
        for k, v in itertools.chain(
            self._config.items(), self._compile_ignored.items()
        ):
            if has_saved_config_ignore and k in self._save_config_ignore:
                continue
            if v == self._default[k]:
                continue
            lines.append(f"{mod}.{k} = {v!r}")
        return "\n".join(lines)

    def to_dict(self):
        warnings.warn(
            (
                "config.to_dict() has been deprecated. It may no longer change the underlying config.",
                "use config.shallow_copy_dict() or config.get_config_copy() instead",
            ),
            DeprecationWarning,
        )
        return self.shallow_copy_dict()

    def shallow_copy_dict(self):
        return {**self._config, **self._compile_ignored}

    def load_dict(self, d):
        assert set(d.keys()) == self._allowed_keys
        for k, v in d.items():
            if k in self._compile_ignored_keys:
                self._compile_ignored[k] = v
            else:
                self._config[k] = v

    def load_config(self, data):
        """Restore from a prior call to save_config()"""
        for k, v in pickle.loads(data).items():
            if k in self._compile_ignored_keys:
                self._compile_ignored[k] = v
            else:
                self._config[k] = v

    def get_config_copy(self):
        return {**copy.deepcopy(self._config), **copy.deepcopy(self._compile_ignored)}

    def patch(self, arg1=None, arg2=None, **kwargs):
        """
        Decorator and/or context manager to make temporary changes to a config.

        As a decorator:

            @config.patch("name", val)
            @config.patch(name1=val1, name2=val2)
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
        prior_ignored = {}
        config = self

        class ConfigPatch(ContextDecorator):
            def __enter__(self):
                assert not prior
                for key, val in changes.items():
                    # KeyError on invalid entry
                    if key in config._compile_ignored_keys:
                        prior_ignored[key] = config._compile_ignored[key]
                        config._compile_ignored[key] = val
                    else:
                        prior[key] = config._config[key]
                        config._config[key] = val

            def __exit__(self, exc_type, exc_val, exc_tb):
                config._config.update(prior)
                config._compile_ignored.update(prior_ignored)
                prior.clear()
                prior_ignored.clear()

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
