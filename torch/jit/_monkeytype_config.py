import inspect
import typing
from typing import Optional, Iterable, List, Dict
from collections import defaultdict

_IS_MONKEYTYPE_INSTALLED = True
try:
    import monkeytype  # type: ignore[import]
    from monkeytype import trace as monkeytype_trace
    from monkeytype.db.base import CallTraceThunk, CallTraceStore, CallTraceStoreLogger  # type: ignore[import]
    from monkeytype.config import default_code_filter  # type: ignore[import]
    from monkeytype.tracing import CallTrace, CodeFilter  # type: ignore[import]
except ImportError:
    _IS_MONKEYTYPE_INSTALLED = False

def get_qualified_name(func):
    return func.__qualname__

if _IS_MONKEYTYPE_INSTALLED:

    class JitTypeTraceStoreLogger(CallTraceStoreLogger):
        """A JitTypeCallTraceLogger that stores logged traces in a CallTraceStore."""
        def __init__(self, store: CallTraceStore):
            super().__init__(store)

        def log(self, trace: CallTrace) -> None:
            self.traces.append(trace)

    class JitTypeTraceStore(CallTraceStore):
        def __init__(self):
            super().__init__()
            # A dictionary keeping all collected CallTrace
            # key is fully qualified name of called function
            # value is list of all CallTrace
            self.trace_records: Dict[str, list] = defaultdict(list)

        def add(self, traces: Iterable[CallTrace]):
            for t in traces:
                qualified_name = get_qualified_name(t.func)
                self.trace_records[qualified_name].append(t)

        def filter(
            self,
            qualified_name: str,
            qualname_prefix: Optional[str] = None,
            limit: int = 2000
        ) -> List[CallTraceThunk]:
            return self.trace_records[qualified_name]

        def analyze(self, qualified_name: str) -> Dict:
            # Analyze the types for the given module
            # and create a dictionary of all the types
            # for arguments.
            records = self.trace_records[qualified_name]
            all_args = defaultdict(set)  # type:  ignore[var-annotated]
            for record in records:
                for arg, arg_type in record.arg_types.items():
                    all_args[arg].add(arg_type)
            return all_args

        def consolidate_types(self, qualified_name: str) -> Dict:
            all_args = self.analyze(qualified_name)
            # If there are more types for an argument,
            # then consolidate the type to `Any` and replace the entry
            # by type `Any`.
            for arg, types in all_args.items():
                _all_type = " "
                for _type in types:
                    # If the type is a type imported from typing
                    # like Tuple, List, Dict then replace "typing."
                    # with a null string.
                    if inspect.getmodule(_type) == typing:
                        _type_to_string = str(_type)
                        _all_type += _type_to_string.replace('typing.', '') + ','
                    else:
                        _all_type += _type.__name__ + ','
                _all_type = _all_type.lstrip(" ")  # Remove any trailing spaces

                if len(types) > 1:
                    all_args[arg] = {'Any'}
                else:
                    all_args[arg] = {_all_type[:-1]}
            return all_args

        def get_args_types(self, qualified_name: str) -> Dict:
            return self.consolidate_types(qualified_name)

    class JitTypeTraceConfig(monkeytype.config.Config):
        def __init__(self, s: JitTypeTraceStore):
            super().__init__()
            self.s = s

        def trace_logger(self) -> JitTypeTraceStoreLogger:
            """
            Returns a JitCallTraceStoreLogger that logs to the configured
            trace store.
            """
            return JitTypeTraceStoreLogger(self.trace_store())

        def trace_store(self) -> CallTraceStore:
            return self.s

        def code_filter(self) -> Optional[CodeFilter]:
            return default_code_filter
else:
    # When MonkeyType is not installed, we provide dummy class definitions
    # for the below classes.
    class JitTypeTraceStoreLogger:  # type:  ignore[no-redef]
        def __init__(self):
            pass

    class JitTypeTraceStore:  # type:  ignore[no-redef]
        def __init__(self):
            self.trace_records = None

    class JitTypeTraceConfig:  # type:  ignore[no-redef]
        def __init__(self):
            pass

    monkeytype_trace = None  # type:  ignore[assignment]  # noqa: F811
