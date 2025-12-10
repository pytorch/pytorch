import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from pathlib import PurePath
from typing import Any, ClassVar

from .registry import _import_class, get_filesystem_class
from .spec import AbstractFileSystem


class FilesystemJSONEncoder(json.JSONEncoder):
    include_password: ClassVar[bool] = True

    def default(self, o: Any) -> Any:
        if isinstance(o, AbstractFileSystem):
            return o.to_dict(include_password=self.include_password)
        if isinstance(o, PurePath):
            cls = type(o)
            return {"cls": f"{cls.__module__}.{cls.__name__}", "str": str(o)}

        return super().default(o)

    def make_serializable(self, obj: Any) -> Any:
        """
        Recursively converts an object so that it can be JSON serialized via
        :func:`json.dumps` and :func:`json.dump`, without actually calling
        said functions.
        """
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, Mapping):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, Sequence):
            return [self.make_serializable(v) for v in obj]

        return self.default(obj)


class FilesystemJSONDecoder(json.JSONDecoder):
    def __init__(
        self,
        *,
        object_hook: Callable[[dict[str, Any]], Any] | None = None,
        parse_float: Callable[[str], Any] | None = None,
        parse_int: Callable[[str], Any] | None = None,
        parse_constant: Callable[[str], Any] | None = None,
        strict: bool = True,
        object_pairs_hook: Callable[[list[tuple[str, Any]]], Any] | None = None,
    ) -> None:
        self.original_object_hook = object_hook

        super().__init__(
            object_hook=self.custom_object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            strict=strict,
            object_pairs_hook=object_pairs_hook,
        )

    @classmethod
    def try_resolve_path_cls(cls, dct: dict[str, Any]):
        with suppress(Exception):
            fqp = dct["cls"]

            path_cls = _import_class(fqp)

            if issubclass(path_cls, PurePath):
                return path_cls

        return None

    @classmethod
    def try_resolve_fs_cls(cls, dct: dict[str, Any]):
        with suppress(Exception):
            if "cls" in dct:
                try:
                    fs_cls = _import_class(dct["cls"])
                    if issubclass(fs_cls, AbstractFileSystem):
                        return fs_cls
                except Exception:
                    if "protocol" in dct:  # Fallback if cls cannot be imported
                        return get_filesystem_class(dct["protocol"])

                    raise

        return None

    def custom_object_hook(self, dct: dict[str, Any]):
        if "cls" in dct:
            if (obj_cls := self.try_resolve_fs_cls(dct)) is not None:
                return AbstractFileSystem.from_dict(dct)
            if (obj_cls := self.try_resolve_path_cls(dct)) is not None:
                return obj_cls(dct["str"])

        if self.original_object_hook is not None:
            return self.original_object_hook(dct)

        return dct

    def unmake_serializable(self, obj: Any) -> Any:
        """
        Inverse function of :meth:`FilesystemJSONEncoder.make_serializable`.
        """
        if isinstance(obj, dict):
            obj = self.custom_object_hook(obj)
        if isinstance(obj, dict):
            return {k: self.unmake_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.unmake_serializable(v) for v in obj]

        return obj
