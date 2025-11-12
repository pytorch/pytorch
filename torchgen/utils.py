from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import re
import sys
import textwrap
from dataclasses import is_dataclass
from enum import auto, Enum
from pathlib import Path
from pprint import pformat
from typing import Any, Generic, TYPE_CHECKING, TypeVar
from typing_extensions import assert_never, Self

from torchgen.code_template import CodeTemplate


if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Callable, Iterable, Iterator, Sequence


TORCHGEN_ROOT = Path(__file__).absolute().parent
REPO_ROOT = TORCHGEN_ROOT.parent


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


@functools.cache
def _read_template(template_fn: str) -> CodeTemplate:
    return CodeTemplate.from_file(template_fn)


# String hash that's stable across different executions, unlike builtin hash
def string_stable_hash(s: str) -> int:
    sha1 = hashlib.sha1(s.encode("latin1"), usedforsecurity=False).digest()
    return int.from_bytes(sha1, byteorder="little")


# A small abstraction for writing out generated files and keeping track
# of what files have been written (so you can write out a list of output
# files)
class FileManager:
    def __init__(
        self,
        install_dir: str | Path,
        template_dir: str | Path,
        dry_run: bool,
    ) -> None:
        self.install_dir = Path(install_dir)
        self.template_dir = Path(template_dir)
        self.files: set[Path] = set()
        self.dry_run = dry_run

    @property
    def filenames(self) -> frozenset[str]:
        return frozenset({file.as_posix() for file in self.files})

    def _write_if_changed(self, filename: str | Path, contents: str) -> None:
        file = Path(filename)
        old_contents: str | None = None
        try:
            old_contents = file.read_text(encoding="utf-8")
        except OSError:
            pass
        if contents != old_contents:
            # Create output directory if it doesn't exist
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(contents, encoding="utf-8")

    # Read from template file and replace pattern with callable (type could be dict or str).
    def substitute_with_template(
        self,
        template_fn: str | Path,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> str:
        assert not Path(template_fn).is_absolute(), (
            f"template_fn must be relative: {template_fn}"
        )
        template_path = self.template_dir / template_fn
        env = env_callable()
        if isinstance(env, dict):
            if "generated_comment" not in env:
                generator_default = TORCHGEN_ROOT / "gen.py"
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
            substitute_out = template.substitute(env)
            # Ensure an extra blank line between the class/function definition
            # and the docstring of the previous class/function definition.
            # NB: It is generally not recommended to have docstrings in pyi stub
            #     files. But if there are any, we need to ensure that the file
            #     is properly formatted.
            return re.sub(
                r'''
                (""")\n+             # match triple quotes
                (
                    (\s*@.+\n)*     # match decorators if any
                    \s*(class|def)  # match class/function definition
                )
                ''',
                r"\g<1>\n\n\g<2>",
                substitute_out,
                flags=re.VERBOSE,
            )
        if isinstance(env, str):
            return env
        assert_never(env)

    def write_with_template(
        self,
        filename: str | Path,
        template_fn: str | Path,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> None:
        filename = Path(filename)
        assert not filename.is_absolute(), f"filename must be relative: {filename}"
        file = self.install_dir / filename
        assert file not in self.files, f"duplicate file write {file}"
        self.files.add(file)
        if not self.dry_run:
            substitute_out = self.substitute_with_template(
                template_fn=template_fn,
                env_callable=env_callable,
            )
            self._write_if_changed(filename=file, contents=substitute_out)

    def write(
        self,
        filename: str | Path,
        env_callable: Callable[[], str | dict[str, Any]],
    ) -> None:
        self.write_with_template(filename, filename, env_callable)

    def write_sharded(
        self,
        filename: str | Path,
        items: Iterable[T],
        *,
        key_fn: Callable[[T], str],
        env_callable: Callable[[T], dict[str, list[str]]],
        num_shards: int,
        base_env: dict[str, Any] | None = None,
        sharded_keys: set[str],
    ) -> None:
        self.write_sharded_with_template(
            filename,
            filename,
            items,
            key_fn=key_fn,
            env_callable=env_callable,
            num_shards=num_shards,
            base_env=base_env,
            sharded_keys=sharded_keys,
        )

    def write_sharded_with_template(
        self,
        filename: str | Path,
        template_fn: str | Path,
        items: Iterable[T],
        *,
        key_fn: Callable[[T], str],
        env_callable: Callable[[T], dict[str, list[str]]],
        num_shards: int,
        base_env: dict[str, Any] | None = None,
        sharded_keys: set[str],
    ) -> None:
        file = Path(filename)
        assert not file.is_absolute(), f"filename must be relative: {filename}"
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
                    assert isinstance(shard[key], list), (
                        "sharded keys in base_env must be a list"
                    )
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

        for shard in all_shards:
            shard_id = shard["shard_id"]
            self.write_with_template(
                file.with_stem(f"{file.stem}{shard_id}"),
                template_fn,
                lambda: shard,
            )

        # filenames is used to track compiled files, but FooEverything.cpp isn't meant to be compiled
        self.files.discard(self.install_dir / file.with_stem(f"{file.stem}Everything"))

    def write_outputs(self, variable_name: str, filename: str | Path) -> None:
        """Write a file containing the list of all outputs which are generated by this script."""
        content = "\n".join(
            (
                "set(",
                variable_name,
                # Use POSIX paths to avoid invalid escape sequences on Windows
                *(f'    "{file.as_posix()}"' for file in sorted(self.files)),
                ")",
            )
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
    options: Namespace,
    install_dir: str | Path | None = None,
    template_dir: str | Path | None = None,
) -> FileManager:
    if template_dir is None:
        template_dir = os.path.join(options.source_path, "templates")
    install_dir = install_dir if install_dir else options.install_dir
    return FileManager(
        install_dir=install_dir,
        template_dir=template_dir,
        dry_run=options.dry_run,
    )


# Helper function to create a pretty representation for dataclasses
def dataclass_repr(
    obj: Any,
    indent: int = 0,
    width: int = 80,
) -> str:
    return pformat(obj, indent, width)


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
            pformat(v, indent, width, curr_indent + len(k_repr))
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
        pformat(l, indent, width, curr_indent) if is_dataclass(l) else repr(l)
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
        self,
        namespace_str: str,
        entity_name: str = "",
        max_level: int = 2,
    ) -> None:
        # cpp_namespace can be a colon joined string such as torch::lazy
        cpp_namespaces = namespace_str.split("::")
        assert len(cpp_namespaces) <= max_level, (
            f"Codegen doesn't support more than {max_level} level(s) of custom namespace. Got {namespace_str}."
        )
        self.cpp_namespace_ = namespace_str
        self.prologue_ = "\n".join([f"namespace {n} {{" for n in cpp_namespaces])
        self.epilogue_ = "\n".join(
            [f"}} // namespace {n}" for n in reversed(cpp_namespaces)]
        )
        self.namespaces_ = cpp_namespaces
        self.entity_name_ = entity_name

    @staticmethod
    def from_namespaced_entity(
        namespaced_entity: str,
        max_level: int = 2,
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
    storage: dict[T, None]

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
