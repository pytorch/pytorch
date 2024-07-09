from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import re
import sys
import textwrap
from dataclasses import fields, is_dataclass
from enum import auto, Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    Literal,
    NoReturn,
    Sequence,
    TYPE_CHECKING,
    TypeVar,
)
from typing_extensions import Self

from torchgen.code_template import CodeTemplate


if TYPE_CHECKING:
    from argparse import Namespace


REPO_ROOT = Path(__file__).absolute().parent.parent


# Many of these functions share logic for defining both the definition
# and declaration (for example, the function signature is the same), so
# we organize them into one function that takes a Target to say which
# code we want.
#
# This is an OPEN enum (we may add more cases to it in the future), so be sure
# to explicitly specify with Literal[Target.XXX] or Literal[Target.XXX, Target.YYY]
# what targets are valid for your use.
class Target(Enum):
    # top level namespace (not including at)
    DEFINITION = auto()
    DECLARATION = auto()
    # TORCH_LIBRARY(...) { ... }
    REGISTRATION = auto()
    # namespace { ... }
    ANONYMOUS_DEFINITION = auto()
    # namespace cpu { ... }
    NAMESPACED_DEFINITION = auto()
    NAMESPACED_DECLARATION = auto()


# Matches "foo" in "foo, bar" but not "foobar". Used to search for the
# occurrence of a parameter in the derivative formula
IDENT_REGEX = r"(^|\W){}($|\W)"


# TODO: Use a real parser here; this will get bamboozled
def split_name_params(schema: str) -> tuple[str, list[str]]:
    m = re.match(r"(\w+)(\.\w+)?\((.*)\)", schema)
    if m is None:
        raise RuntimeError(f"Unsupported function schema: {schema}")
    name, _, params = m.groups()
    return name, params.split(", ")


T = TypeVar("T")
S = TypeVar("S")

# These two functions purposely return generators in analogy to map()
# so that you don't mix up when you need to list() them


# Map over function that may return None; omit Nones from output sequence
def mapMaybe(func: Callable[[T], S | None], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        r = func(x)
        if r is not None:
            yield r


# Map over function that returns sequences and cat them all together
def concatMap(func: Callable[[T], Sequence[S]], xs: Iterable[T]) -> Iterator[S]:
    for x in xs:
        yield from func(x)


# Conveniently add error context to exceptions raised.  Lets us
# easily say that an error occurred while processing a specific
# context.
@contextlib.contextmanager
def context(msg_fn: Callable[[], str]) -> Iterator[None]:
    try:
        yield
    except Exception as e:
        # TODO: this does the wrong thing with KeyError
        msg = msg_fn()
        msg = textwrap.indent(msg, "  ")
        msg = f"{e.args[0]}\n{msg}" if e.args else msg
        e.args = (msg,) + e.args[1:]
        raise


# A little trick from https://github.com/python/mypy/issues/6366
# for getting mypy to do exhaustiveness checking
# TODO: put this somewhere else, maybe
def assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError(f"Unhandled type: {type(x).__name__}")


@functools.lru_cache(maxsize=None)
def _read_template(template_fn: str) -> CodeTemplate:
    return CodeTemplate.from_file(template_fn)


# String hash that's stable across different executions, unlike builtin hash
def string_stable_hash(s: str) -> int:
    sha1 = hashlib.sha1(s.encode("latin1")).digest()
    return int.from_bytes(sha1, byteorder="little")


# A small abstraction for writing out generated files and keeping track
# of what files have been written (so you can write out a list of output
# files)
class FileManager:
    install_dir: str
    template_dir: str
    dry_run: bool
    filenames: set[str]

    def __init__(self, install_dir: str, template_dir: str, dry_run: bool) -> None:
        self.install_dir = install_dir
        self.template_dir = template_dir
        self.filenames = set()
        self.dry_run = dry_run

    def _write_if_changed(self, filename: str, contents: str) -> None:
        old_contents: str | None
        try:
            with open(filename) as f:
                old_contents = f.read()
        except OSError:
            old_contents = None
        if contents != old_contents:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "w") as f:
                f.write(contents)

    # Read from template file and replace pattern with callable (type could be dict or str).
    def substitute_with_template(
        self, template_fn: str, env_callable: Callable[[], str | dict[str, Any]]
    ) -> str:
        template_path = os.path.join(self.template_dir, template_fn)
        env = env_callable()
        if isinstance(env, dict):
            if "generated_comment" not in env:
                generator_default = REPO_ROOT / "torchgen" / "gen.py"
                try:
                    generator = Path(
                        sys.modules["__main__"].__file__ or generator_default
                    ).absolute()
                except (KeyError, AttributeError):
                    generator = generator_default.absolute()

                try:
                    generator_path = generator.relative_to(REPO_ROOT).as_posix()
                except ValueError:
                    generator_path = generator.name

                env = {
                    **env,  # copy the original dict instead of mutating it
                    "generated_comment": (
                        "@" + f"generated by {generator_path} from {template_fn}"
                    ),
                }
            template = _read_template(template_path)
            return template.substitute(env)
        elif isinstance(env, str):
            return env
        else:
            assert_never(env)

    def write_with_template(
        self,
        filename: str,
        template_fn: str,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> None:
        filename = f"{self.install_dir}/{filename}"
        assert filename not in self.filenames, "duplicate file write {filename}"
        self.filenames.add(filename)
        if not self.dry_run:
            substitute_out = self.substitute_with_template(
                template_fn=template_fn,
                env_callable=env_callable,
            )
            self._write_if_changed(filename=filename, contents=substitute_out)

    def write(
        self,
        filename: str,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> None:
        self.write_with_template(filename, filename, env_callable)

    def write_sharded(
        self,
        filename: str,
        items: Iterable[T],
        *,
        key_fn: Callable[[T], str],
        env_callable: Callable[[T], dict[str, list[str]]],
        num_shards: int,
        base_env: dict[str, Any] | None = None,
        sharded_keys: set[str],
    ) -> None:
        everything: dict[str, Any] = {"shard_id": "Everything"}
        shards: list[dict[str, Any]] = [
            {"shard_id": f"_{i}"} for i in range(num_shards)
        ]
        all_shards = [everything] + shards

        if base_env is not None:
            for shard in all_shards:
                shard.update(base_env)

        for key in sharded_keys:
            for shard in all_shards:
                if key in shard:
                    assert isinstance(
                        shard[key], list
                    ), "sharded keys in base_env must be a list"
                    shard[key] = shard[key].copy()
                else:
                    shard[key] = []

        def merge_env(into: dict[str, list[str]], from_: dict[str, list[str]]) -> None:
            for k, v in from_.items():
                assert k in sharded_keys, f"undeclared sharded key {k}"
                into[k] += v

        if self.dry_run:
            # Dry runs don't write any templates, so incomplete environments are fine
            items = ()

        for item in items:
            key = key_fn(item)
            sid = string_stable_hash(key) % num_shards
            env = env_callable(item)

            merge_env(shards[sid], env)
            merge_env(everything, env)

        dot_pos = filename.rfind(".")
        if dot_pos == -1:
            dot_pos = len(filename)
        base_filename = filename[:dot_pos]
        extension = filename[dot_pos:]

        for shard in all_shards:
            shard_id = shard["shard_id"]
            self.write_with_template(
                f"{base_filename}{shard_id}{extension}", filename, lambda: shard
            )

        # filenames is used to track compiled files, but FooEverything.cpp isn't meant to be compiled
        self.filenames.discard(
            f"{self.install_dir}/{base_filename}Everything{extension}"
        )

    def write_outputs(self, variable_name: str, filename: str) -> None:
        """Write a file containing the list of all outputs which are
        generated by this script."""
        content = "set({}\n    {})".format(
            variable_name,
            "\n    ".join('"' + name + '"' for name in sorted(self.filenames)),
        )
        self._write_if_changed(filename, content)

    def template_dir_for_comments(self) -> str:
        """
        This needs to be deterministic. The template dir is an absolute path
        that varies across builds. So, just use the path relative to this file,
        which will point to the codegen source but will be stable.
        """
        return os.path.relpath(self.template_dir, os.path.dirname(__file__))


# Helper function to generate file manager
def make_file_manager(
    options: Namespace, install_dir: str | None = None
) -> FileManager:
    template_dir = os.path.join(options.source_path, "templates")
    install_dir = install_dir if install_dir else options.install_dir
    return FileManager(
        install_dir=install_dir, template_dir=template_dir, dry_run=options.dry_run
    )


# Helper function to create a pretty representation for dataclasses
def dataclass_repr(
    obj: Any,
    indent: int = 0,
    width: int = 80,
) -> str:
    # built-in pprint module support dataclasses from python 3.10
    if sys.version_info >= (3, 10):
        from pprint import pformat

        return pformat(obj, indent, width)

    return _pformat(obj, indent=indent, width=width)


def _pformat(
    obj: Any,
    indent: int,
    width: int,
    curr_indent: int = 0,
) -> str:
    assert is_dataclass(obj), f"obj should be a dataclass, received: {type(obj)}"

    class_name = obj.__class__.__name__
    # update current indentation level with class name
    curr_indent += len(class_name) + 1

    fields_list = [(f.name, getattr(obj, f.name)) for f in fields(obj) if f.repr]

    fields_str = []
    for name, attr in fields_list:
        # update the current indent level with the field name
        # dict, list, set and tuple also add indent as done in pprint
        _curr_indent = curr_indent + len(name) + 1
        if is_dataclass(attr):
            str_repr = _pformat(attr, indent, width, _curr_indent)
        elif isinstance(attr, dict):
            str_repr = _format_dict(attr, indent, width, _curr_indent)
        elif isinstance(attr, (list, set, tuple)):
            str_repr = _format_list(attr, indent, width, _curr_indent)
        else:
            str_repr = repr(attr)

        fields_str.append(f"{name}={str_repr}")

    indent_str = curr_indent * " "
    body = f",\n{indent_str}".join(fields_str)
    return f"{class_name}({body})"


def _format_dict(
    attr: dict[Any, Any],
    indent: int,
    width: int,
    curr_indent: int,
) -> str:
    curr_indent += indent + 3
    dict_repr = []
    for k, v in attr.items():
        k_repr = repr(k)
        v_str = (
            _pformat(v, indent, width, curr_indent + len(k_repr))
            if is_dataclass(v)
            else repr(v)
        )
        dict_repr.append(f"{k_repr}: {v_str}")

    return _format(dict_repr, indent, width, curr_indent, "{", "}")


def _format_list(
    attr: list[Any] | set[Any] | tuple[Any, ...],
    indent: int,
    width: int,
    curr_indent: int,
) -> str:
    curr_indent += indent + 1
    list_repr = [
        _pformat(l, indent, width, curr_indent) if is_dataclass(l) else repr(l)
        for l in attr
    ]
    start, end = ("[", "]") if isinstance(attr, list) else ("(", ")")
    return _format(list_repr, indent, width, curr_indent, start, end)


def _format(
    fields_str: list[str],
    indent: int,
    width: int,
    curr_indent: int,
    start: str,
    end: str,
) -> str:
    delimiter, curr_indent_str = "", ""
    # if it exceed the max width then we place one element per line
    if len(repr(fields_str)) >= width:
        delimiter = "\n"
        curr_indent_str = " " * curr_indent

    indent_str = " " * indent
    body = f", {delimiter}{curr_indent_str}".join(fields_str)
    return f"{start}{indent_str}{body}{end}"


class NamespaceHelper:
    """A helper for constructing the namespace open and close strings for a nested set of namespaces.

    e.g. for namespace_str torch::lazy,

    prologue:
    namespace torch {
    namespace lazy {

    epilogue:
    } // namespace lazy
    } // namespace torch
    """

    def __init__(
        self, namespace_str: str, entity_name: str = "", max_level: int = 2
    ) -> None:
        # cpp_namespace can be a colon joined string such as torch::lazy
        cpp_namespaces = namespace_str.split("::")
        assert (
            len(cpp_namespaces) <= max_level
        ), f"Codegen doesn't support more than {max_level} level(s) of custom namespace. Got {namespace_str}."
        self.cpp_namespace_ = namespace_str
        self.prologue_ = "\n".join([f"namespace {n} {{" for n in cpp_namespaces])
        self.epilogue_ = "\n".join(
            [f"}} // namespace {n}" for n in reversed(cpp_namespaces)]
        )
        self.namespaces_ = cpp_namespaces
        self.entity_name_ = entity_name

    @staticmethod
    def from_namespaced_entity(
        namespaced_entity: str, max_level: int = 2
    ) -> NamespaceHelper:
        """
        Generate helper from nested namespaces as long as class/function name. E.g.: "torch::lazy::add"
        """
        names = namespaced_entity.split("::")
        entity_name = names[-1]
        namespace_str = "::".join(names[:-1])
        return NamespaceHelper(
            namespace_str=namespace_str, entity_name=entity_name, max_level=max_level
        )

    @property
    def prologue(self) -> str:
        return self.prologue_

    @property
    def epilogue(self) -> str:
        return self.epilogue_

    @property
    def entity_name(self) -> str:
        return self.entity_name_

    # Only allow certain level of namespaces
    def get_cpp_namespace(self, default: str = "") -> str:
        """
        Return the namespace string from joining all the namespaces by "::" (hence no leading "::").
        Return default if namespace string is empty.
        """
        return self.cpp_namespace_ if self.cpp_namespace_ else default


class OrderedSet(Generic[T]):
    storage: dict[T, Literal[None]]

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        if iterable is None:
            self.storage = {}
        else:
            self.storage = dict.fromkeys(iterable)

    def __contains__(self, item: T) -> bool:
        return item in self.storage

    def __iter__(self) -> Iterator[T]:
        return iter(self.storage.keys())

    def update(self, items: OrderedSet[T]) -> None:
        self.storage.update(items.storage)

    def add(self, item: T) -> None:
        self.storage[item] = None

    def copy(self) -> OrderedSet[T]:
        ret: OrderedSet[T] = OrderedSet()
        ret.storage = self.storage.copy()
        return ret

    @staticmethod
    def union(*args: OrderedSet[T]) -> OrderedSet[T]:
        ret = args[0].copy()
        for s in args[1:]:
            ret.update(s)
        return ret

    def __or__(self, other: OrderedSet[T]) -> OrderedSet[T]:
        return OrderedSet.union(self, other)

    def __ior__(self, other: OrderedSet[T]) -> Self:
        self.update(other)
        return self

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OrderedSet):
            return self.storage == other.storage
        else:
            return set(self.storage.keys()) == other
