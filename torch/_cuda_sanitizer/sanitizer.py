import functools
import sys
from typing import Any, Dict, Iterator, List, Set, Tuple, TypeVar

import torch
import torch._cuda_sanitizer.trace_checker as trace_checker
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


TK = TypeVar("TK")
TVa = TypeVar("TVa")
TVb = TypeVar("TVb")


def zip_by_key(a: Dict[TK, TVa], b: Dict[TK, TVb]) -> Iterator[Tuple[TK, TVa, TVb]]:
    for arg, value in a.items():
        if arg in b:
            yield arg, value, b[arg]


def zip_arguments(
    schema: torch.FunctionSchema, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Iterator[Tuple[torch.Argument, Any]]:
    schema_args = schema.arguments[: len(args)]
    schema_kwargs = {arg.name: arg for arg in schema.arguments[len(args) :]}

    yield from zip(schema_args, args)

    for _, argument, value in zip_by_key(schema_kwargs, kwargs):
        yield (argument, value)


class ArgumentHandler:
    def __init__(self):
        self.dataptrs_read: Set[int] = set()
        self.dataptrs_written: Set[int] = set()
        self.tensor_names: Dict[int, List[str]] = dict()

    def _handle_argument(self, value: Any, is_write: bool, name: str) -> None:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            data_ptr = value.data_ptr()
            if is_write:
                self.dataptrs_written.add(data_ptr)
            else:
                self.dataptrs_read.add(data_ptr)
            self.tensor_names.setdefault(data_ptr, []).append(name)

    def parse_inputs(
        self,
        schema: torch.FunctionSchema,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        for argument, value in zip_arguments(schema, args, kwargs):
            is_write = False
            if hasattr(argument, "alias_info") and argument.alias_info is not None:
                is_write = getattr(argument.alias_info, "is_write", False)
            tree_map(
                functools.partial(
                    self._handle_argument, is_write=is_write, name=argument.name
                ),
                value,
            )

    def parse_outputs(self, outputs: Any) -> None:
        tree_map(
            functools.partial(self._handle_argument, is_write=True, name="output"),
            outputs,
        )


class CUDASanitizerDispatchMode(TorchDispatchMode):
    def __init__(self):
        self.checker = trace_checker.TraceChecker()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        argument_handler = ArgumentHandler()
        argument_handler.parse_inputs(func._schema, args, kwargs)

        outputs = func(*args, **kwargs)

        argument_handler.parse_outputs(outputs)

        errors = self.checker._handle_kernel_launch(
            torch.cuda.current_stream().cuda_stream,
            list(argument_handler.dataptrs_read - argument_handler.dataptrs_written),
            list(argument_handler.dataptrs_written),
            func._schema,
            argument_handler.tensor_names,
        )
        if errors:
            for error in errors:
                print(error)
            sys.exit()

        return outputs


class CUDASanitizer:
    """Manages the lifetime of a CUDASanitizer dispatch mode object.

    The CUDASanitizer class wraps the entering/exiting functions of the dispatch mode
    context manager in its constructor/destructor, respectively. This is to explicitly
    set the lifetime of the dispatch mode object to that of the application.
    This approach was deemed more elegant than using the atexit module.
    """

    def __init__(self):
        self.dispatch = CUDASanitizerDispatchMode()
        self.dispatch.__enter__()

    def __del__(self):
        self.dispatch.__exit__(None, None, None)
