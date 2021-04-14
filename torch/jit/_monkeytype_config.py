# !/usr/bin/env python
# -*- coding: utf-8 -*-
# mypy: ignore-errors

from typing import Optional, Iterable, List, Dict
from collections import defaultdict

_IS_MONKEYTYPE_INSTALLED = True
try:
    import monkeytype
    from monkeytype import trace as monkeytype_trace
    from monkeytype.db.base import CallTraceStore, CallTraceStoreLogger
    from monkeytype.config import default_code_filter
    from monkeytype.tracing import CallTrace, CodeFilter
except ImportError:
    _IS_MONKEYTYPE_INSTALLED = False
    print("Warning: monkeytype is not installed. Please install monkeytype to enable Profile-Directed Typing in TorchScript")

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
            self.trace_records = defaultdict(list)

        def add(self, traces: Iterable[CallTrace]):
            for t in traces:
                qualified_name = get_qualified_name(t.func)
                self.trace_records[qualified_name].append(t)

        def filter(
            self,
            qualified_name: str,
            qualname_prefix: Optional[str] = None,
            limit: int = 2000
        ) -> List[CallTrace]:
            return self.trace_records[qualified_name]

        def analyze(self, qualified_name: str) -> Dict:
            # Analyze the types for the given module
            # and create a dictionary of all the types
            # for arguments and return values this module can take
            records = self.trace_records[qualified_name]
            all_args = defaultdict(set)
            for record in records:
                for arg, arg_type in record.arg_types.items():
                    all_args[arg].add(arg_type)
                all_args["return_type"].add(record.return_type)
            return all_args

        def consolidate_types(self, qualified_name: str) -> Dict:
            all_args = self.analyze(qualified_name)
            # If there are more types for an argument,
            # then consolidate the type to `Any` and replace the entry
            # by type `Any`.
            # This currently handles only cases with only one return value
            # TODO: Consolidate types for multiple return values
            for args, types in all_args.items():
                if len(types) > 1:
                    all_args[args] = {'Any'}
                else:
                    _element_string_type = types.pop()
                    all_args[args] = {_element_string_type.__name__}
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
    class JitTypeTraceStoreLogger:
        def __init__(self):
            self.traces = None

    class JitTypeTraceStore:
        def __init__(self):
            self.trace_records = None

    class JitTypeTraceConfig:
        def __init__(self):
            self.s = None

    monkeytype_trace = None  # noqa: F811
