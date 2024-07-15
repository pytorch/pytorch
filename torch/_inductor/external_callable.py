import inspect
from typing import Any, Callable, List

external_matmul: List[Callable[..., Any]] = []


def register_external_matmul_call(func: Callable[..., Any]) -> None:
    params = inspect.signature(func).parameters
    if len(params) != 3:
        raise Exception(
            f"required 3 params for matmul but got {len(params)}"
        )  # noqa: TRY002
    external_matmul.append(func)
