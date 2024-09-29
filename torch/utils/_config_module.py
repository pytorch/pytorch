import contextlib
import copy
import hashlib
import inspect
import io
import pickle
import tokenize
import unittest
import warnings
from types import FunctionType, ModuleType
from typing import Any, Callable, Dict, NoReturn, Optional, Set, Union
from typing_extensions import deprecated
from unittest import mock


# Types saved/loaded in configs
CONFIG_TYPES = (int, float, bool, type(None), str, list, set, tuple, dict)


def install_config_module(module: ModuleType) -> None:
    """
    Converts a module-level config into a `ConfigModule()`.

    See _config_typing.pyi for instructions on how to get the converted module to typecheck.
    """

    class ConfigModuleInstance(ConfigModule):
        _bypass_keys = set({"_is_dirty", "_hash_digest"})

    def visit(
        source: Union[ModuleType, type],
        dest: Union[ModuleType, SubConfigProxy],
        prefix: str,
    ) -> None:
        """Walk the module structure and move everything to module._config"""
        for key, value in list(source.__dict__.items()):
            if (
                key.startswith("__")
                or isinstance(value, (ModuleType, FunctionType))
                or (hasattr(value, "__module__") and value.__module__ == "typing")
            ):
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
                if dest is module:
                    setattr(dest, key, proxy)
                else:
                    dest.__dict__[key] = proxy
            else:
                raise AssertionError(f"Unhandled config {key}={value} ({type(value)})")

    config: Dict[str, Any] = {}
    default: Dict[str, Any] = {}

    compile_ignored_keys = get_assignments_with_compile_ignored_comments(module)

    visit(module, module, "")
    module._config = config  # type: ignore[attr-defined]
    module._default = default  # type: ignore[attr-defined]
    module._allowed_keys = set(config.keys())  # type: ignore[attr-defined]
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


class ConfigModule(ModuleType):
    # NOTE: This should be kept in sync with _config_typing.pyi.

    # The default values of the configuration settings.  This can be used to
    # determine if the config has been changed or not.
    _default: Dict[str, Any]
    # The actual configuration settings.  E.g., torch._dynamo.config.debug
    # would live as "debug" in the key, and torch._inductor.config.triton.cudagraphs
    # maps as "triton.cudagraphs"
    _config: Dict[str, Any]
    _allowed_keys: Set[str]
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
        elif name not in self._allowed_keys:
            raise AttributeError(f"{self.__name__}.{name} does not exist")
        else:
            self._config[name] = value

    def __getattr__(self, name: str) -> Any:
        try:
            return self._config[name]
        except KeyError as e:
            # make hasattr() work properly
            raise AttributeError(f"{self.__name__}.{name} does not exist") from e

    def __delattr__(self, name: str) -> None:
        # must support delete because unittest.mock.patch deletes
        # then recreate things
        del self._config[name]

    def save_config(self) -> bytes:
        """Convert config to a pickled blob"""
        config = dict(self._config)
        for key in config.get("_save_config_ignore", ()):
            config.pop(key)
        return pickle.dumps(config, protocol=2)

    def save_config_portable(self) -> Dict[str, Any]:
        """Convert config to portable format"""
        config: Dict[str, Any] = {}
        for key in sorted(self._config):
            if key.startswith("_"):
                continue
            if any(
                key.startswith(e) for e in self._config["_cache_config_ignore_prefix"]
            ):
                continue
            config[key] = self._config[key]
        return config

    def codegen_config(self) -> str:
        """Convert config to Python statements that replicate current config.
        This does NOT include config settings that are at default values.
        """
        lines = []
        mod = self.__name__
        for k, v in self._config.items():
            if k in self._config.get("_save_config_ignore", ()):
                if v != self._default[k]:
                    warnings.warn(f"Skipping serialization of {k} value {v}")
                continue
            if v == self._default[k]:
                continue
            lines.append(f"{mod}.{k} = {v!r}")
        return "\n".join(lines)

    def get_hash(self) -> bytes:
        """Hashes the configs that are not compile_ignored"""
        if self._is_dirty or self._hash_digest is None:
            dict_to_hash = {
                k: v
                for k, v in self._config.items()
                if k not in self._compile_ignored_keys
            }
            string_to_hash = repr(sorted(dict_to_hash.items()))
            self._hash_digest = hashlib.md5(string_to_hash.encode("utf-8")).digest()
            self._is_dirty = False
        return self._hash_digest

    @deprecated(
        "`config.to_dict()` has been deprecated. It may no longer change the underlying config."
        " use `config.shallow_copy_dict()` or `config.get_config_copy()` instead",
        category=FutureWarning,
    )
    def to_dict(self) -> Dict[str, Any]:
        return self.shallow_copy_dict()

    def shallow_copy_dict(self) -> Dict[str, Any]:
        return {**self._config}

    def load_config(self, maybe_pickled_config: Union[bytes, Dict[str, Any]]) -> None:
        """Restore from a prior call to save_config() or shallow_copy_dict()"""
        if not isinstance(maybe_pickled_config, dict):
            config = pickle.loads(maybe_pickled_config)
        else:
            config = maybe_pickled_config
        self._config.update(config)

    def get_config_copy(self) -> Dict[str, Any]:
        return copy.deepcopy(self._config)

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
        dirty = False

        class ConfigPatch(ContextDecorator):
            def __enter__(self) -> None:
                assert not prior
                nonlocal dirty
                for key in changes.keys():
                    # KeyError on invalid entry
                    prior[key] = config._config[key]
                    dirty = key not in config._compile_ignored_keys
                config._config.update(changes)
                config._is_dirty = dirty

            def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[no-untyped-def]
                nonlocal dirty
                config._config.update(prior)
                config._is_dirty = dirty
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
            prior = {k: config[k] for k in changes}
            config.update(changes)

            def revert() -> None:
                config.update(prior)

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
