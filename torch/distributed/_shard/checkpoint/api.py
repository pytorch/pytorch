from typing import Dict, Tuple, Any
import traceback as tb

WRAPPED_EXCEPTION = Tuple[BaseException, tb.StackSummary]

def _wrap_exception(exc: BaseException) -> WRAPPED_EXCEPTION:
    return (exc, tb.extract_tb(exc.__traceback__))

def _is_wrapped_exception(obj: Any) -> bool:
    if not isinstance(obj, tuple):
        return False
    if len(obj) != 2:
        return False
    return isinstance(obj[0], BaseException) and isinstance(obj[1], tb.StackSummary)

class CheckpointException(BaseException):
    """
    Exception raised if failure was detected as part of a checkpoint load or save.
    """
    def __init__(self, msg: str, failures: Dict[int, WRAPPED_EXCEPTION]):
        super().__init__(msg, failures)
        self._failures = failures

    @property
    def failures(self) -> Dict[int, WRAPPED_EXCEPTION]:
        """
        Returns:
            Dict of failed nodes and their associated exception.
              Keys are node ranks and values are exceptions
        """
        return self._failures

    def __str__(self):
        str = f"CheckpointException ranks:{self._failures.keys()}\n"
        for rank, exc_pair in self._failures.items():
            exc, trace = exc_pair
            str += f"Traceback (most recent call last): (RANK {rank})\n"
            if trace is not None:
                str += "".join(tb.format_list(trace))
            str += "".join(tb.format_exception_only(type(exc), value=exc))
        return str
