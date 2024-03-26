import copy
import functools
import time
from typing import Any, Callable, Dict, List, Tuple, TypeVar

from typing_extensions import ParamSpec

import torch.distributed.c10d_logger as c10d_logger
from torch.distributed.checkpoint.logging_handlers import DCP_LOGGER_NAME

__all__: List[str] = []

global _dcp_logger
_dcp_logger = c10d_logger._get_or_create_logger(DCP_LOGGER_NAME)

_T = TypeVar("_T")
_P = ParamSpec("_P")


def _parse_dcp_method_args(*args, **kwargs) -> Tuple[Tuple[Any], Dict[str, Any]]:
    """
    Parse method arguments into positional and keyword arguments.
    """
    args, kwargs = copy.copy(args), copy.copy(kwargs)

    # remove objects which are too large/complicated to log
    # we can do this because the save/load API's force kwargs for all cases except state_dict
    state_dict = kwargs.pop("state_dict", None)
    if not state_dict:
        args = args[:0]

    storage_writer = kwargs.pop("storage_writer", None)
    storage_reader = kwargs.pop("storage_reader", None)

    # handled in the c10d logger
    kwargs["group"] = kwargs.pop("process_group", None)

    # checkpoint ID can be passed in through the serializer or through the checkpoint id directly
    if kwargs.get("checkpoint_id") is None and (
        serializer := storage_writer or storage_reader
    ):
        kwargs["checkpoint_id"] = getattr(serializer, "checkpoint_id", None)

    return args, kwargs


def _get_msg_dict(func_name, *args, **kwargs) -> Dict[str, Any]:
    log_args, log_kwargs = _parse_dcp_method_args(*args, **kwargs)
    return c10d_logger._get_msg_dict(func_name, *log_args, **log_kwargs)


def _dcp_method_logger(
    log_exceptions: bool = False, **wrapper_kwargs: Any
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:  # pyre-ignore
    """This method decorator logs the start, end, and exception of wrapped events."""

    def decorator(func: Callable[_P, _T]):
        @functools.wraps(func)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            msg_dict = _get_msg_dict(
                func.__name__, *args, **{**wrapper_kwargs, **kwargs}
            )

            # log start event
            msg_dict["event"] = "start"
            t0 = time.time_ns()
            msg_dict["time"] = t0
            _dcp_logger.debug(msg_dict)

            # exceptions
            try:
                result = func(*args, **kwargs)
            except Exception as error:
                if log_exceptions:
                    msg_dict["event"] = "exception"
                    msg_dict["error"] = f"{error}"
                    msg_dict["time"] = time.time_ns()
                    _dcp_logger.error(msg_dict)
                raise

            # end event
            msg_dict["event"] = "end"
            t1 = time.time_ns()
            msg_dict["time"] = time.time_ns()
            msg_dict["times_spent"] = t1 - t0
            _dcp_logger.debug(msg_dict)

            return result

        return wrapper

    return decorator
