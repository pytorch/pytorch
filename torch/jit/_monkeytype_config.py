# mypy: allow-untyped-defs
import inspect
import sys
import typing
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from types import CodeType
from typing import Optional

import torch


_IS_MONKEYTYPE_INSTALLED = True
try:
    import monkeytype  # type: ignore[import]

    # pyrefly: ignore [import-error, missing-import]
    from monkeytype import trace as monkeytype_trace
    from monkeytype.config import _startswith, LIB_PATHS  # type: ignore[import]
    from monkeytype.db.base import (  # type: ignore[import]
        CallTraceStore,
        CallTraceStoreLogger,
        CallTraceThunk,
    )
    from monkeytype.tracing import CallTrace, CodeFilter  # type: ignore[import]
except ImportError:
    _IS_MONKEYTYPE_INSTALLED = False


# Checks whether a class is defined in `torch.*` modules
def is_torch_native_class(cls):
    if not hasattr(cls, "__module__"):
        return False

    parent_modules = cls.__module__.split(".")
    if not parent_modules:
        return False

    root_module = sys.modules.get(parent_modules[0])
    return root_module is torch


def get_type(type):
    """Convert the given type to a torchScript acceptable format."""
    if isinstance(type, str):
        return type
    elif inspect.getmodule(type) == typing:
        # If the type is a type imported from typing
        # like Tuple, List, Dict then replace `typing.`
        # with a null string. This needs to be done since
        # typing.List is not accepted by TorchScript.
        type_to_string = str(type)
        return type_to_string.replace(type.__module__ + ".", "")
    elif is_torch_native_class(type):
        # If the type is a subtype of torch module, then TorchScript expects a fully qualified name
        # for the type which is obtained by combining the module name and type name.
        return type.__module__ + "." + type.__name__
    else:
        # For all other types use the name for the type.
        return type.__name__


def get_optional_of_element_type(types):
    """Extract element type, return as `Optional[element type]` from consolidated types.

    Helper function to extracts the type of the element to be annotated to Optional
    from the list of consolidated types and returns `Optional[element type]`.
    TODO: To remove this check once Union support lands.
    """
    elem_type = types[1] if type(None) is types[0] else types[0]
    elem_type = get_type(elem_type)

    # Optional type is internally converted to Union[type, NoneType], which
    # is not supported yet in TorchScript. Hence, representing the optional type as string.
    return "Optional[" + elem_type + "]"


def get_qualified_name(func):
    return func.__qualname__


if _IS_MONKEYTYPE_INSTALLED:

    class JitTypeTraceStoreLogger(CallTraceStoreLogger):
        """A JitTypeCallTraceLogger that stores logged traces in a CallTraceStore."""

        def log(self, trace: CallTrace) -> None:
            # pyrefly: ignore [missing-attribute]
            self.traces.append(trace)

    class JitTypeTraceStore(CallTraceStore):
        def __init__(self) -> None:
            super().__init__()
            # A dictionary keeping all collected CallTrace
            # key is fully qualified name of called function
            # value is list of all CallTrace
            self.trace_records: dict[str, list] = defaultdict(list)

        def add(self, traces: Iterable[CallTrace]) -> None:
            for t in traces:
                qualified_name = get_qualified_name(t.func)
                self.trace_records[qualified_name].append(t)

        def filter(
            self,
            qualified_name: str,
            qualname_prefix: Optional[str] = None,
            limit: int = 2000,
        ) -> list[CallTraceThunk]:
            return self.trace_records[qualified_name]

        def analyze(self, qualified_name: str) -> dict:
            # Analyze the types for the given module
            # and create a dictionary of all the types
            # for arguments.
            records = self.trace_records[qualified_name]
            all_args = defaultdict(set)
            for record in records:
                for arg, arg_type in record.arg_types.items():
                    all_args[arg].add(arg_type)
            return all_args

        def consolidate_types(self, qualified_name: str) -> dict:
            all_args = self.analyze(qualified_name)
            # If there are more types for an argument,
            # then consolidate the type to `Any` and replace the entry
            # by type `Any`.
            for arg, types in all_args.items():
                types = list(types)
                type_length = len(types)
                if type_length == 2 and type(None) in types:
                    # TODO: To remove this check once Union support in TorchScript lands.
                    all_args[arg] = get_optional_of_element_type(types)
                elif type_length > 1:
                    all_args[arg] = "Any"
                elif type_length == 1:
                    all_args[arg] = get_type(types[0])
            return all_args

        def get_args_types(self, qualified_name: str) -> dict:
            return self.consolidate_types(qualified_name)

    class JitTypeTraceConfig(monkeytype.config.Config):
        def __init__(self, s: JitTypeTraceStore) -> None:
            super().__init__()
            self.s = s

        def trace_logger(self) -> JitTypeTraceStoreLogger:
            """Return a JitCallTraceStoreLogger that logs to the configured trace store."""
            # pyrefly: ignore [bad-argument-count]
            return JitTypeTraceStoreLogger(self.trace_store())

        def trace_store(self) -> CallTraceStore:
            return self.s

        def code_filter(self) -> Optional[CodeFilter]:
            return jit_code_filter

else:
    # When MonkeyType is not installed, we provide dummy class definitions
    # for the below classes.
    class JitTypeTraceStoreLogger:  # type:  ignore[no-redef]
        def __init__(self) -> None:
            pass

    class JitTypeTraceStore:  # type:  ignore[no-redef]
        def __init__(self) -> None:
            self.trace_records = None

    class JitTypeTraceConfig:  # type:  ignore[no-redef]
        def __init__(self) -> None:
            pass

    monkeytype_trace = None  # type: ignore[assignment]  # noqa: F811


def jit_code_filter(code: CodeType) -> bool:
    """Codefilter for Torchscript to trace forward calls.

    The custom CodeFilter is required while scripting a FX Traced forward calls.
    FX Traced forward calls have `code.co_filename` start with '<' which is used
    to exclude tracing of stdlib and site-packages in the default code filter.
    Since we need all forward calls to be traced, this custom code filter
    checks for code.co_name to be 'forward' and enables tracing for all such calls.
    The code filter is similar to default code filter for monkeytype and
    excludes tracing of stdlib and site-packages.
    """
    # Filter code without a source file and exclude this check for 'forward' calls.
    if code.co_name != "forward" and (
        not code.co_filename or code.co_filename[0] == "<"
    ):
        return False

    filename = Path(code.co_filename).resolve()
    return not any(_startswith(filename, lib_path) for lib_path in LIB_PATHS)
