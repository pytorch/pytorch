import contextlib
import copy
import hashlib
import importlib
import inspect
import io
import os
import pickle
import sys
import tokenize
import unittest
from dataclasses import dataclass
from types import FunctionType, ModuleType
from typing import (
    Any,
    Callable,
    Generic,
    NoReturn,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import deprecated
from unittest import mock

from torch._utils_internal import justknobs_check


# Types saved/loaded in configs
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)


# Duplicated, because mypy needs these types statically
T = TypeVar("T", bound=Union[int, float, bool, None, str, list, set, tuple, dict])


_UNSET_SENTINEL = object()


@dataclass
class _Config(Generic[T]):
    """Represents a config with richer behaviour than just a default value.
    ::
        i.e.
        foo = Config(justknob="//foo:bar", default=False)
        install_config_module(...)

    This configs must be installed with install_config_module to be used

    Precedence Order:
        alias: If set, the directly use the value of the alias.
        env_name_force: If set, this environment variable has precedence over
            everything after this.
            If multiple env variables are given, the precendence order is from
            left to right.
        user_override: If a user sets a value (i.e. foo.bar=True), that
            has precedence over everything after this.
        env_name_default: If set, this environment variable will override everything
            after this.
            If multiple env variables are given, the precendence order is from
            left to right.
        justknob: If this pytorch installation supports justknobs, that will
            override defaults, but will not override the user_override precendence.
        default: This value is the lowest precendance, and will be used if nothing is
            set.

    Environment Variables:
        These are interpreted to be either "0" or "1" to represent true and false.

    Arguments:
        justknob: the name of the feature / JK. In OSS this is unused.
        default: is the value to default this knob to in OSS.
        alias: The alias config to read instead.
        env_name_force: The environment variable, or list of, to read that is a FORCE
            environment variable. I.e. it overrides everything except for alias.
        env_name_default: The environment variable, or list of, to read that changes the
            default behaviour. I.e. user overrides take preference.
    """

    default: Union[T, object]
    justknob: Optional[str] = None
    env_name_default: Optional[list[str]] = None
    env_name_force: Optional[list[str]] = None
    alias: Optional[str] = None

    def __init__(
        self,
        default: Union[T, object] = _UNSET_SENTINEL,
        justknob: Optional[str] = None,
        env_name_default: Optional[Union[str, list[str]]] = None,
        env_name_force: Optional[Union[str, list[str]]] = None,
        value_type: Optional[type] = None,
        alias: Optional[str] = None,
    ):
        # python 3.9 does not support kw_only on the dataclass :(.
        self.default = default
        self.justknob = justknob
        self.env_name_default = _Config.string_or_list_of_string_to_list(
            env_name_default
        )
        self.env_name_force = _Config.string_or_list_of_string_to_list(env_name_force)
        self.value_type = value_type
        self.alias = alias
        if self.alias is not None:
            assert (
                default is _UNSET_SENTINEL
                and justknob is None
                and env_name_default is None
                and env_name_force is None
            ), "if alias is set, none of {default, justknob and env var} can be set"

    @staticmethod
    def string_or_list_of_string_to_list(
        val: Optional[Union[str, list[str]]]
    ) -> Optional[list[str]]:
        if val is None:
            return None
        if isinstance(val, str):
            return [val]
        assert isinstance(val, list)
        return val


# In runtime, we unbox the Config[T] to a T, but typechecker cannot see this,
# so in order to allow for this dynamic behavior to work correctly with
# typechecking we are going to lie to the typechecker that Config[T] returns
# a T.
if TYPE_CHECKING:

    def Config(
        default: Union[T, object] = _UNSET_SENTINEL,
        justknob: Optional[str] = None,
        env_name_default: Optional[Union[str, list[str]]] = None,
        env_name_force: Optional[Union[str, list[str]]] = None,
        value_type: Optional[type] = None,
        alias: Optional[str] = None,
    ) -> T:
        ...

else:

    def Config(
        default: Union[T, object] = _UNSET_SENTINEL,
        justknob: Optional[str] = None,
        env_name_default: Optional[Union[str, list[str]]] = None,
        env_name_force: Optional[Union[str, list[str]]] = None,
        value_type: Optional[type] = None,
        alias: Optional[str] = None,
    ) -> _Config[T]:
        return _Config(
            default, justknob, env_name_default, env_name_force, value_type, alias
        )


def _read_env_variable(name: str) -> Optional[Union[bool, str]]:
    value = os.environ.get(name)
    if value == "1":
        return True
    if value == "0":
        return False
    return value


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
                or (isinstance(value, type) and issubclass(value, _Config))
            ):
                continue

            name = f"{prefix}{key}"
            annotated_type = type_hints.get(key, None)
            if isinstance(value, CONFIG_TYPES):
                config[name] = _ConfigEntry(
                    _Config(default=value, value_type=annotated_type)
                )
                if dest is module:
                    delattr(module, key)
            elif isinstance(value, _Config):
                if annotated_type is not None and value.value_type is None:
                    value.value_type = annotated_type

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

    config: dict[str, _ConfigEntry] = {}

    compile_ignored_keys = get_assignments_with_compile_ignored_comments(module)

    visit(module, module, "")
    module._config = config  # type: ignore[attr-defined]
    module._compile_ignored_keys = compile_ignored_keys  # type: ignore[attr-defined]
    module.__class__ = ConfigModuleInstance
    module._is_dirty = True  # type: ignore[attr-defined]
    module._hash_digest = None  # type: ignore[attr-defined]


COMPILE_IGNORED_MARKER = "@compile_ignored"


# Gets all the keys (i.e. assignments) with a @compile_ignored comment
def get_assignments_with_compile_ignored_comments(module: ModuleType) -> set[str]:
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
    alias: Optional[str] = None

    def __init__(self, config: _Config):
        self.default = config.default
        self.value_type = (
            config.value_type if config.value_type is not None else type(self.default)
        )
        self.justknob = config.justknob
        self.alias = config.alias
        if config.env_name_default is not None:
            for val in config.env_name_default:
                if (env_value := _read_env_variable(val)) is not None:
                    self.env_value_default = env_value
                    break
        if config.env_name_force is not None:
            for val in config.env_name_force:
                if (env_value := _read_env_variable(val)) is not None:
                    self.env_value_force = env_value
                    break

        # Ensure justknobs and envvars are allowlisted types
        if self.justknob is not None and self.default is not None:
            assert isinstance(
                self.default, bool
            ), f"justknobs only support booleans, {self.default} is not a boolean"
        if self.value_type is not None and (
            config.env_name_default is not None or config.env_name_force is not None
        ):
            assert self.value_type in (
                bool,
                str,
                Optional[bool],
                Optional[str],
            ), f"envvar configs only support (optional) booleans or strings, {self.value_type} is neither"


class ConfigModule(ModuleType):
    # NOTE: This should be kept in sync with _config_typing.pyi.

    # The actual configuration settings.  E.g., torch._dynamo.config.debug
    # would live as "debug" in the key, and torch._inductor.config.triton.cudagraphs
    # maps as "triton.cudagraphs". See discussion on the class for meaning of various sub items
    _config: dict[str, _ConfigEntry]
    _bypass_keys: set[str]
    _compile_ignored_keys: set[str]
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
        elif self._config[name].alias is not None:
            self._set_alias_val(self._config[name], value)
        else:
            self._config[name].user_override = value
            self._is_dirty = True
            self._config[name].hide = False

    def __getattr__(self, name: str) -> Any:
        try:
            config = self._config[name]

            if config.hide:
                raise AttributeError(f"{self.__name__}.{name} does not exist")

            alias_val = self._get_alias_val(config)
            if alias_val is not _UNSET_SENTINEL:
                return alias_val

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

    def _get_alias_module_and_name(
        self, entry: _ConfigEntry
    ) -> Optional[tuple[ModuleType, str]]:
        alias = entry.alias
        if alias is None:
            return None
        module_name, constant_name = alias.rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise AttributeError("config alias {alias} does not exist") from e
        return module, constant_name

    def _get_alias_val(self, entry: _ConfigEntry) -> Any:
        data = self._get_alias_module_and_name(entry)
        if data is None:
            return _UNSET_SENTINEL
        module, constant_name = data
        constant_value = getattr(module, constant_name)
        return constant_value

    def _set_alias_val(self, entry: _ConfigEntry, val: Any) -> None:
        data = self._get_alias_module_and_name(entry)
        assert data is not None
        module, constant_name = data
        setattr(module, constant_name, val)

    def _is_default(self, name: str) -> bool:
        """
        Returns true if the config is at its default value.
        configs overriden by the env are not considered default.
        """
        config_val = self._config[name]
        # The config is not overridden by the user, and the env_value_default
        # is different from the default value (meaning user has set the env to
        # change the default value).
        not_set_env_default = (
            config_val.env_value_default is _UNSET_SENTINEL
            or config_val.env_value_default == config_val.default
        )
        not_set_env_force = (
            config_val.env_value_force is _UNSET_SENTINEL
            or config_val.env_value_force == config_val.default
        )

        unset = config_val.user_override is _UNSET_SENTINEL
        # Handle reference types specially to avoid spammy warnings
        if isinstance(config_val.default, (list, set, dict)):
            unset = unset or config_val.user_override == config_val.default
        return unset and not_set_env_default and not_set_env_force

    def _get_dict(
        self,
        ignored_keys: Optional[list[str]] = None,
        ignored_prefixes: Optional[list[str]] = None,
        skip_default: bool = False,
    ) -> dict[str, Any]:
        """Export a dictionary of current configuration keys and values.

        This function is design to provide a single point which handles
        accessing config options and exporting them into a dictionary.
        This is used by a number of different user facing export methods
        which all have slightly different semantics re: how and what to
        skip.
        If a config is aliased, it skips this config.

        Arguments:
            ignored_keys are keys that should not be exported.
            ignored_prefixes are prefixes that if a key matches should
                not be exported
            skip_default does two things. One if a key has not been modified
                it skips it.
        """
        config: dict[str, Any] = {}
        for key in self._config:
            if ignored_keys and key in ignored_keys:
                continue
            if ignored_prefixes:
                if any(key.startswith(prefix) for prefix in ignored_prefixes):
                    continue
            if skip_default and self._is_default(key):
                continue
            if self._config[key].alias is not None:
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

    def save_config_portable(self) -> dict[str, Any]:
        """Convert config to portable format"""
        prefixes = ["_"]
        prefixes.extend(getattr(self, "_cache_config_ignore_prefix", []))
        return self._get_dict(ignored_prefixes=prefixes)

    def codegen_config(self) -> str:
        """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """

        # additional imports required
        imports = set()

        def get_module_name(func: Callable, add_dot: bool) -> str:
            module_name = func.__module__
            if module_name == "builtins":
                module_name = ""
            if add_dot and module_name != "":
                module_name += "."
            return module_name

        def add_import(func: Callable) -> None:
            module_name = get_module_name(func, False)
            if module_name:
                imports.add(module_name)

        def list_of_callables_to_string(v: Union[list, set]) -> list[str]:
            return [f"{get_module_name(item, True)}{item.__name__}" for item in v]

        def importable_callable(v: Any) -> bool:
            # functools.partial has no attributes below but is a callable
            return callable(v) and hasattr(v, "__module__") and hasattr(v, "__name__")

        def get_config_line(mod, k, v) -> str:  # type: ignore[no-untyped-def]
            """
            Return a string version of the config line.
            Handle v when v is a callable, or a list/dict of callables. Add import statements for callables if necessary.
            We assume that the value of a single config won't be a mix of callables and non-callables.

            Example output:
                import logging
                import _warnings
                torch._dynamo.config.reorderable_logging_functions = { _warnings.warn, logging.warn, print }
            """
            if importable_callable(v):
                add_import(v)
                return f"{mod}.{k} = {get_module_name(v, True)}{v.__name__}"
            elif isinstance(v, (list, set)) and all(
                importable_callable(item) for item in v
            ):
                for item in v:
                    add_import(item)
                v_list = list_of_callables_to_string(v)
                if isinstance(v, list):
                    return f"{mod}.{k} = {v_list}"
                else:
                    return f"{mod}.{k} = {{ {', '.join(v_list)} }}"
            else:
                return f"{mod}.{k} = {v!r}"

        lines = []
        mod = self.__name__
        for k, v in self._get_dict(
            ignored_keys=getattr(self, "_save_config_ignore", []), skip_default=True
        ).items():
            lines.append(get_config_line(mod, k, v))
        for import_name in imports:
            lines.insert(0, f"import {import_name}")
        return "\n".join(lines)

    def get_hash(self) -> bytes:
        """Hashes the configs that are not compile_ignored"""
        if self._is_dirty or self._hash_digest is None:
            dict_to_hash = self._get_dict(ignored_keys=list(self._compile_ignored_keys))
            string_to_hash = repr(sorted(dict_to_hash.items()))
            self._hash_digest = hashlib.md5(
                string_to_hash.encode("utf-8"), usedforsecurity=False
            ).digest()
            self._is_dirty = False
        return self._hash_digest

    @deprecated(
        "`config.to_dict()` has been deprecated. It no longer changes the underlying config."
        " use `config.get_config_copy()` instead if you just want a copy of the config, or "
        "config.load_config if you need mutable access",
        category=FutureWarning,
    )
    def to_dict(self) -> dict[str, Any]:
        return self.get_config_copy()

    @deprecated(
        "`config.shallow_copy_dict()` has been deprecated. It no longer changes the underlying config."
        " use `config.get_config_copy()` instead if you just want a copy of the config, or "
        "config.load_config if you need mutable access",
        category=FutureWarning,
    )
    def shallow_copy_dict(self) -> dict[str, Any]:
        return self.get_config_copy()

    def load_config(self, maybe_pickled_config: Union[bytes, dict[str, Any]]) -> None:
        """Restore from a prior call to save_config() or shallow_copy_dict()"""
        if not isinstance(maybe_pickled_config, dict):
            config = pickle.loads(maybe_pickled_config)
        else:
            config = maybe_pickled_config
        for k, v in config.items():
            if k in self._config:
                setattr(self, k, v)
            else:
                from torch._dynamo.utils import warn_once

                warn_once(f"key {k} with value {v} is not understood by this config")

    def get_config_copy(self) -> dict[str, Any]:
        return self._get_dict()

    def patch(
        self,
        arg1: Optional[Union[str, dict[str, Any]]] = None,
        arg2: Any = None,
        **kwargs: dict[str, Any],
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
        changes: dict[str, Any]
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
        prior: dict[str, Any] = {}
        config = self

        class ConfigPatch(ContextDecorator):
            def __init__(self) -> None:
                self.changes = changes

            def __enter__(self) -> None:
                assert not prior
                for key in self.changes.keys():
                    # KeyError on invalid entry
                    prior[key] = config.__getattr__(key)
                for k, v in self.changes.items():
                    config.__setattr__(k, v)

            def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
                for k, v in prior.items():
                    config.__setattr__(k, v)
                prior.clear()

        return ConfigPatch()

    def _make_closure_patcher(self, **changes: dict[str, Any]) -> Any:
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
