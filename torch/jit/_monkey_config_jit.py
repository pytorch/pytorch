from typing import Optional, Iterable, List
from collections import defaultdict

_IS_MONKEYTYPE_INSTALLED = True
try:
    import monkeytype  # type: ignore
    from monkeytype.db.base import CallTraceStore, CallTraceStoreLogger  # type: ignore
    from monkeytype.config import default_code_filter  # type: ignore
    from monkeytype.tracing import CallTrace, CodeFilter, CallTraceLogger  # type: ignore
except ImportError:
    _IS_MONKEYTYPE_INSTALLED = False
    print("Warning: monkeytype is not installed. Please install monkeytype to enable static type annotation")

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
            self.trace_records = defaultdict(list)  # type: ignore

        def add(self, traces: Iterable[CallTrace]):
            for t in traces:
                qualified_name = get_qualified_name(t.func)
                self.trace_records[qualified_name].append(t)

        def filter(self, qualified_name: str, qualname_prefix: Optional[str] = None, limit: int = 2000) -> List[CallTrace]:  # type: ignore[override]
            return self.trace_records[qualified_name]

    class JitTypeTraceConfig(monkeytype.config.Config):
        def __init__(self, s: JitTypeTraceStore):
            super().__init__()
            self.s = s

        def trace_logger(self) -> JitTypeTraceStoreLogger:
            """
            Return the JitCallTraceStoreLogger for logging call traces.
            """
            return JitTypeTraceStoreLogger(self.trace_store())

        def trace_store(self) -> CallTraceStore:
            return self.s

        def code_filter(self) -> Optional[CodeFilter]:
            return default_code_filter

    type_trace_db = JitTypeTraceStore()
