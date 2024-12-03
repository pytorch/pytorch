import contextlib
import copy
import hashlib
import inspect
import io
import os
import pickle
import sys
import tokenize
import unittest
import warnings
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, List, NoReturn, Optional, Set, Union
from typing_extensions import deprecated
from unittest import mock

from torch._utils_internal import justknobs_check


@dataclass
class Config:
    """Represents a config with richer behaviour than just a default value.
    ::
        i.e.
        foo = Config(justknob="//foo:bar", default=False)
        install_config_module(...)

    This configs must be installed with install_config_module to be used

    Precedence Order:
        env_name_force: If set, this environment variable overrides everything
        user_override: If a user sets a value (i.e. foo.bar=True), that
            has precedence over everything after this.
        env_name_default: If set, this environment variable will override everything
            after this.
        justknob: If this pytorch installation supports justknobs, that will
            override defaults, but will not override the user_override precendence.
        default: This value is the lowest precendance, and will be used if nothing is
            set.

    Environment Variables:
        These are interpreted to be either "0" or "1" to represent true and false.

    Arguments:
        justknob: the name of the feature / JK. In OSS this is unused.
        default: is the value to default this knob to in OSS.
        env_name_force: The environment variable to read that is a FORCE
            environment variable. I.e. it overrides everything
        env_name_default: The environment variable to read that changes the
            default behaviour. I.e. user overrides take preference.
    """

    default: Any = True
    justknob: Optional[str] = None
    env_name_default: Optional[str] = None
    env_name_force: Optional[str] = None
    value_type: Optional[type] = None

    def __init__(
        self,
        default: Any = True,
        justknob: Optional[str] = None,
        env_name_default: Optional[str] = None,
        env_name_force: Optional[str] = None,
        value_type: Optional[type] = None,
    ):
        # python 3.9 does not support kw_only on the dataclass :(.
        self.default = default
        self.justknob = justknob
        self.env_name_default = env_name_default
        self.env_name_force = env_name_force
        self.value_type = value_type
        if self.justknob is not None:
            assert isinstance(
                self.default, bool
            ), f"justknobs only support booleans, {self.default} is not a boolean"


# Types saved/loaded in configs
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)


def _read_env_variable(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value == "1":
        return True
    if value == "0":
        return False
    return None


def install_config_module(module: ModuleType) -> None:
    """
    Converts a module-level config into a `ConfigModule()`.

    See _config_typing.pyi for instructions on how to get the converted module to typecheck.
    """

    class ConfigModuleInstance(ConfigModule):
        # __annotations__ is written to by Sphinx autodoc
        _bypass_keys = set({"_is_dirty", "_hash_digest", "__annotations__"})

    def visit(
        source: Union[ModuleType, type],
        dest: Union[ModuleType, SubConfigProxy],
        prefix: str,
    ) -> None:
        """Walk the module structure and move everything to module._config"""
        if sys.version_info[:2] < (3, 10):
            type_hints = getattr(source, "__annotations__", {})
        else:
            type_hints = inspect.get_annotations(source)
        for key, value in list(source.__dict__.items()):
            if (
                key.startswith("__")
                or isinstance(value, (ModuleType, FunctionType))
                or (hasattr(value, "__module__") and value.__module__ == "typing")
                # Handle from torch.utils._config_module import Config
                or (isinstance(value, type) and issubclass(value, Config))
            ):
                continue

            name = f"{prefix}{key}"
            if isinstance(value, CONFIG_TYPES):
                annotated_type = type_hints.get(key, None)
                config[name] = _ConfigEntry(
                    Config(default=value, value_type=annotated_type)
                )
                if dest is module:
                    delattr(module, key)
            elif isinstance(value, Config):
                config[name] = _ConfigEntry(value)

                if dest is module:
                    delattr(module, key)
            elif isinstance(value, type):
                assert value.__module__ == module.__name__
                # a subconfig with `class Blah:` syntax
                proxy = SubConfigProxy(module, f"{name}.")
                visit(value, proxy, f"{name}.")
                if dest is module:
                    setattr(dest, key, proxy)
                else:
                    dest.__dict__[key] = proxy
            else:
                raise AssertionError(f"Unhandled config {key}={value} ({type(value)})")

    config: Dict[str, _ConfigEntry] = {}

    compile_ignored_keys = get_assignments_with_compile_ignored_comments(module)

    visit(module, module, "")
    module._config = config  # type: ignore[attr-defined]
    module._compile_ignored_keys = compile_ignored_keys  # type: ignore[attr-defined]
    module.__class__ = ConfigModuleInstance
    module._is_dirty = True  # type: ignore[attr-defined]
    module._hash_digest = None  # type: ignore[attr-defined]


COMPILE_IGNORED_MARKER = "@compile_ignored"


# Gets all the keys (i.e. assignments) with a @compile_ignored comment
def get_assignments_with_compile_ignored_comments(module: ModuleType) -> Set[str]:
    source_code = inspect.getsource(module)
    assignments = set()

    # Tokenize the source code to retrieve comments
    tokens = tokenize.tokenize(io.BytesIO(source_code.encode("utf-8")).readline)
    current_comment = "", -1
    prev_name = ""

    for token in tokens:
        if token.type == tokenize.COMMENT:
            prev_name = ""
            maybe_current = token.string.strip()
            if COMPILE_IGNORED_MARKER in maybe_current:
                assert current_comment == (
                    "",
                    -1,
                ), f"unconsumed {COMPILE_IGNORED_MARKER}"
                current_comment = maybe_current, token.start[0]
        elif token.type == tokenize.NAME:
            # Only accept the first name token, to handle if you have
            # something like foo: Bar = ...
            if not prev_name:
                prev_name = token.string
        elif token.type == tokenize.OP and token.string == "=":
            # Check if the current assignment follows a comment
            # with COMPILE_IGNORED_MARKER
            if (
                COMPILE_IGNORED_MARKER in current_comment[0]
                and current_comment[1] == token.start[0] - 1
            ):
                assignments.add(prev_name)
                current_comment = "", -1  # reset
            prev_name = ""
    assert current_comment == ("", -1), f"unconsumed {COMPILE_IGNORED_MARKER}"
    return assignments


_UNSET_SENTINEL = object()


@dataclass
class _ConfigEntry:
    # The default value specified in the configuration
    default: Any
    # The type of the configuration value
    value_type: type
    # The value specified by the user when they overrode the configuration
    # _UNSET_SENTINEL indicates the value is not set.
    user_override: Any = _UNSET_SENTINEL
    # The justknob to check for this config
    justknob: Optional[str] = None
    # environment variables are read at install time
    env_value_force: Any = _UNSET_SENTINEL
    env_value_default: Any = _UNSET_SENTINEL
    # Used to work arounds bad assumptions in unittest.mock.patch
    # The code to blame is
    # https://github.com/python/cpython/blob/94a7a4e22fb8f567090514785c69e65298acca42/Lib/unittest/mock.py#L1637
    # Essentially, mock.patch requires, that if __dict__ isn't accessible
    # (which it isn't), that after delattr is called on the object, the
    # object must throw when hasattr is called. Otherwise, it doesn't call
    # setattr again.
    # Technically we'll have an intermediate state of hiding the config while
    # mock.patch is unpatching itself, but it calls setattr after the delete
    # call so the final state is correct. It's just very unintuitive.
    # upstream bug - python/cpython#126886
    hide: bool = False

    def __init__(self, config: Config):
        self.default = config.default
        self.value_type = (
            config.value_type if config.value_type is not None else type(self.default)
        )
        self.justknob = config.justknob
        if config.env_name_default is not None:
            if (env_value := _read_env_variable(config.env_name_default)) is not None:
                self.env_value_default = env_value
        if config.env_name_force is not None:
            if (env_value := _read_env_variable(config.env_name_force)) is not None:
                self.env_value_force = env_value


class ConfigModule(ModuleType):
    # NOTE: This should be kept in sync with _config_typing.pyi.

    # The actual configuration settings.  E.g., torch._dynamo.config.debug
    # would live as "debug" in the key, and torch._inductor.config.triton.cudagraphs
    # maps as "triton.cudagraphs". See discussion on the class for meaning of various sub items
    _config: Dict[str, _ConfigEntry]
    _bypass_keys: Set[str]
    _compile_ignored_keys: Set[str]
    _is_dirty: bool
    _hash_digest: Optional[bytes]

    def __init__(self) -> None:
        raise NotImplementedError(
            f"use {__name__}.install_config_module(sys.modules[__name__])"
        )

    def __setattr__(self, name: str, value: object) -> None:
        if name in self._bypass_keys:
            super().__setattr__(name, value)
        elif name not in self._config:
            raise AttributeError(f"{self.__name__}.{name} does not exist")
        else:
            self._config[name].user_override = value
            self._is_dirty = True
            self._config[name].hide = False

    def __getattr__(self, name: str) -> Any:
        try:
            config = self._config[name]

            if config.hide:
                raise AttributeError(f"{self.__name__}.{name} does not exist")

            if config.env_value_force is not _UNSET_SENTINEL:
                return config.env_value_force

            if config.user_override is not _UNSET_SENTINEL:
                return config.user_override

            if config.env_value_default is not _UNSET_SENTINEL:
                return config.env_value_default

            if config.justknob is not None:
                # JK only supports bools and ints
                return justknobs_check(name=config.justknob, default=config.default)

            # Note that reference types can still be modified, so we
            # copy them to user_overrides in case the user overrides
            # them
            if isinstance(config.default, (list, set, dict)):
                config.user_override = copy.deepcopy(config.default)
                return config.user_override
            return config.default

        except KeyError as e:
            # make hasattr() work properly
            raise AttributeError(f"{self.__name__}.{name} does not exist") from e

    def __delattr__(self, name: str) -> None:
        self._is_dirty = True
        # must support delete because unittest.mock.patch deletes
        # then recreate things
        self._config[name].user_override = _UNSET_SENTINEL
        self._config[name].hide = True

    def _is_default(self, name: str) -> bool:
        return self._config[name].user_override is _UNSET_SENTINEL

    def _get_dict(
        self,
        ignored_keys: Optional[List[str]] = None,
        ignored_prefixes: Optional[List[str]] = None,
        skip_default: bool = False,
    ) -> Dict[str, Any]:
        """Export a dictionary of current configuration keys and values.

        This function is design to provide a single point which handles
        accessing config options and exporting them into a dictionary.
        This is used by a number of different user facing export methods
        which all have slightly different semantics re: how and what to
        skip.

        Arguments:
            ignored_keys are keys that should not be exported.
            ignored_prefixes are prefixes that if a key matches should
                not be exported
            skip_default does two things. One if a key has not been modified
                it skips it. The other is it modified the logging behaviour
                to match what codegen already did for modified skipped keys
        """
        config: Dict[str, Any] = {}
        for key in self._config:
            if ignored_keys and key in ignored_keys:
                if skip_default and not self._is_default(key):
                    warnings.warn(
                        f"Skipping serialization of {key} value {getattr(self, key)}"
                    )
                continue
            if ignored_prefixes:
                if any(key.startswith(prefix) for prefix in ignored_prefixes):
                    continue
            if skip_default and self._is_default(key):
                continue
            config[key] = copy.deepcopy(getattr(self, key))
        return config

    def get_type(self, config_name: str) -> type:
        return self._config[config_name].value_type

    def save_config(self) -> bytes:
        """Convert config to a pickled blob"""
        ignored_keys = getattr(self, "_save_config_ignore", [])
        return pickle.dumps(
            self._get_dict(ignored_keys=ignored_keys),
            protocol=2,
        )

    def save_config_portable(self) -> Dict[str, Any]:
        """Convert config to portable format"""
        prefixes = ["_"]
        prefixes.extend(getattr(self, "_cache_config_ignore_prefix", []))
        return self._get_dict(ignored_prefixes=prefixes)

    def codegen_config(self) -> str:
        """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """
        lines = []
        mod = self.__name__
        for k, v in self._get_dict(
            ignored_keys=getattr(self, "_save_config_ignore", []), skip_default=True
        ).items():
            lines.append(f"{mod}.{k} = {v!r}")
        return "\n".join(lines)

    def get_hash(self) -> bytes:
        """Hashes the configs that are not compile_ignored"""
        if self._is_dirty or self._hash_digest is None:
            dict_to_hash = self._get_dict(ignored_keys=list(self._compile_ignored_keys))
            string_to_hash = repr(sorted(dict_to_hash.items()))
            self._hash_digest = hashlib.md5(string_to_hash.encode("utf-8")).digest()
            self._is_dirty = False
        return self._hash_digest

    @deprecated(
        "`config.to_dict()` has been deprecated. It no longer changes the underlying config."
        " use `config.get_config_copy()` instead if you just want a copy of the config, or "
        "config.load_config if you need mutable access",
        category=FutureWarning,
    )
    def to_dict(self) -> Dict[str, Any]:
        return self.get_config_copy()

    @deprecated(
        "`config.shallow_copy_dict()` has been deprecated. It no longer changes the underlying config."
        " use `config.get_config_copy()` instead if you just want a copy of the config, or "
        "config.load_config if you need mutable access",
        category=FutureWarning,
    )
    def shallow_copy_dict(self) -> Dict[str, Any]:
        return self.get_config_copy()

    def load_config(self, maybe_pickled_config: Union[bytes, Dict[str, Any]]) -> None:
        """Restore from a prior call to save_config() or shallow_copy_dict()"""
        if not isinstance(maybe_pickled_config, dict):
            config = pickle.loads(maybe_pickled_config)
        else:
            config = maybe_pickled_config
        for k, v in config.items():
            if k in self._config:
                setattr(self, k, v)
            else:
                warnings.warn(
                    f"key {k} with value {v} is not understood by this config"
                )

    def get_config_copy(self) -> Dict[str, Any]:
        return self._get_dict()

    def patch(
        self,
        arg1: Optional[Union[str, Dict[str, Any]]] = None,
        arg2: Any = None,
        **kwargs: Dict[str, Any],
    ) -> "ContextDecorator":
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
        changes: Dict[str, Any]
        if arg1 is not None:
            if arg2 is not None:
                assert isinstance(arg1, str)
                # patch("key", True) syntax
                changes = {arg1: arg2}
            else:
                assert isinstance(arg1, dict)
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
            def __enter__(self) -> None:
                assert not prior
                for key in changes.keys():
                    # KeyError on invalid entry
                    prior[key] = config.__getattr__(key)
                for k, v in changes.items():
                    config.__setattr__(k, v)

            def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
                for k, v in prior.items():
                    config.__setattr__(k, v)
                prior.clear()

        return ConfigPatch()

    def _make_closure_patcher(self, **changes: Dict[str, Any]) -> Any:
        """
        A lower-overhead version of patch() for things on the critical path.

        Usage:

            # do this off the critical path
            change_fn = config.make_closure_patcher(foo=True)

            ...

            revert = change_fn()
            try:
              ...
            finally:
                revert()

        """
        config = self._config

        def change() -> Callable[[], None]:
            prior = {k: config[k].user_override for k in changes}
            for k, v in changes.items():
                self._config[k].user_override = v

            def revert() -> None:
                for k, v in prior.items():
                    self._config[k].user_override = v

            return revert

        return change


class ContextDecorator(contextlib.ContextDecorator):
    """
    Same as contextlib.ContextDecorator, but with support for
    `unittest.TestCase`
    """

    def __enter__(self) -> None:
        raise NotImplementedError("NYI")

    def __exit__(self, exc_type, exc_val, exc_tb) -> NoReturn:  # type: ignore[no-untyped-def]
        raise NotImplementedError("NYI")

    def __call__(self, func: Callable[[Any], Any]) -> Any:
        if isinstance(func, type) and issubclass(func, unittest.TestCase):

            class _TestCase(func):  # type: ignore[valid-type, misc]
                @classmethod
                def setUpClass(cls) -> None:
                    self.__enter__()
                    try:
                        super().setUpClass()
                    except Exception:
                        self.__exit__(None, None, None)
                        raise

                @classmethod
                def tearDownClass(cls) -> None:
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

    def __init__(self, config: object, prefix: str):
        # `super().__setattr__` to bypass custom `__setattr__`
        super().__setattr__("_config", config)
        super().__setattr__("_prefix", prefix)

    def __setattr__(self, name: str, value: object) -> None:
        return self._config.__setattr__(self._prefix + name, value)

    def __getattr__(self, name: str) -> Any:
        return self._config.__getattr__(self._prefix + name)

    def __delattr__(self, name: str) -> None:
        return self._config.__delattr__(self._prefix + name)


def patch_object(obj: object, name: str, value: object) -> object:
    """
    Workaround `mock.patch.object` issue with ConfigModule
    """
    if isinstance(obj, ConfigModule):
        return obj.patch(name, value)
    return mock.patch.object(obj, name, value)


def get_tristate_env(name: str, default: Any = None) -> Optional[bool]:
    value = os.environ.get(name)
    if value == "1":
        return True
    if value == "0":
        return False
    return default
