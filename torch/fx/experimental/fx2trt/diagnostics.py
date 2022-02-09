from contextvars import ContextVar
import time
import traceback
from dataclasses import dataclass
import inspect
import contextlib
import logging
import os
import os.path
import typing as t
import shutil
import tempfile


TWrite = t.Union[str, bytes]
WriteObj = t.Union[TWrite, t.Callable[[], TWrite]]

_CURRENT_WRITER: ContextVar["DiagnosticsWriter"] = ContextVar("_CURRENT_WRITER")
_CURRENT_COLLECTOR: ContextVar["DiagnosticsCollector"] = ContextVar("_CURRENT_COLLECTOR")
# Allows a collector to indicate subsequent collections should be suppressed to
# avoid duplicate collections.
_SUBSEQUENT_COLLECT_SUPPRESSED_BY: ContextVar[object] = ContextVar("_SUBSEQUENT_COLLECT_SUPPRESSED_BY")
# Indicates current execution context is within a context manager by
# `collect_when`. Only when it's set do we actually write diagnostics.
_IS_IN_COLLECT_CONTEXT: ContextVar[bool] = ContextVar("_IS_IN_COLLECT_CONTEXT")
_LOGGER = logging.getLogger(__name__)


@dataclass
class CollectionConditionContext:
    exception: t.Optional[Exception]

CollectionCondition = t.Callable[[CollectionConditionContext], bool]


def collect_when(condition: "CollectionCondition", supress_subsequent_collect: bool = True):
    """See `DiagnosticsCollector.collect_when`"""
    return get_current_collector().collect_when(condition, supress_subsequent_collect)


def collect():
    return collect_when(CollectionConditions.always())


def collect_when_fail():
    return collect_when(CollectionConditions.when_fail())


def write(file_name: str, text: WriteObj):
    return get_current_writer().write(file_name, text)


def get_current_writer() -> "DiagnosticsWriter":
    """Get the writer for current execution context.

    Lazily instantiates and registers one if not already done.
    """
    current_writer = _CURRENT_WRITER.get(None)
    if not current_writer:
        current_writer = DiagnosticsWriter()
        _CURRENT_WRITER.set(current_writer)
    return current_writer


def get_current_collector() -> "DiagnosticsCollector":
    current_collector = _CURRENT_COLLECTOR.get(None)
    if not current_collector:
        current_collector = DiagnosticsCollector()
        _CURRENT_COLLECTOR.set(current_collector)
    return current_collector


def set_current_collector(collector: "DiagnosticsCollector"):
    _CURRENT_COLLECTOR.set(collector)


class DiagnosticsWriter:

    # the root dir in which the diagnostics will be written
    _root_dir: str

    def __init__(self):
        self._root_dir = tempfile.mkdtemp(prefix="fx2trt.")
        _LOGGER.info(f"Initializing DiagnosticsWriter with root_dir: {self._root_dir}")

    def write(self, file_name: str, data: WriteObj):
        """
        TODO: Can be disabled by regex on file_name
        """
        # Only write if we are inside a collect_when() context.
        if not _IS_IN_COLLECT_CONTEXT.get(False):
            return

        try:
            res, err = _res_or_err(data)
            if err:
                to_write = err.encode("utf-8")
            else:
                if isinstance(res, str):
                    to_write = res.encode("utf-8")
                elif isinstance(res, bytes):
                    to_write = res
                else:
                    raise TypeError(f"Unknown data type: {type(res)}")
            self._write(file_name, to_write)
        except Exception as e:
            # Log the error and swallow the exception, as this should not
            # propagated into business logic
            _LOGGER.warning(f"Error writing diagnostics: {e}")

    def root_dir(self) -> str:
        return self._root_dir

    def _write(self, file_name: str, to_write: bytes):
        # ms granularity - no naming collash, otherwise file will be
        # overwritten.
        ts = int(time.time() * 1000)
        file_name = f"{file_name}.{ts}"
        fn = os.path.join(self.root_dir(), file_name)
        with open(fn, "wb") as f:
            f.write(to_write)


class CollectionConditions:
    @classmethod
    def any(cls, *conditions: "CollectionCondition") -> "CollectionCondition":
        return lambda ctx: any(
            cond(ctx) for cond in conditions
        )

    @classmethod
    def all(cls, *conditions: "CollectionCondition") -> "CollectionCondition":
        return lambda ctx: all(
            cond(ctx) for cond in conditions
        )

    @classmethod
    def always(cls) -> "CollectionCondition":
        """Always collect"""
        return lambda ctx: True

    @classmethod
    def never(cls) -> "CollectionCondition":
        """Never collect"""
        return lambda ctx: False

    @classmethod
    def when_fail(cls) -> "CollectionCondition":
        """Collect when failed"""
        ctx: CollectionConditionContext
        return lambda ctx: ctx.exception is not None

    @classmethod
    def when_called_by_function(cls, func_name: str) -> "CollectionCondition":
        def _when_called_by_function(ctx: CollectionConditionContext) -> bool:
            frames = inspect.stack()
            for frame in frames:
                if frame[3] == func_name:
                    return True
            return False
        return _when_called_by_function


class DiagnosticsCollector:
    @contextlib.contextmanager
    def collect_when(self, condition: "CollectionCondition", supress_subsequent_collect: bool = True):
        """
        Context manager to collect diagnostics when the enclosed code completes
        and *any* of the given condition is met.

        Args:
            condition:
                the condition only when met should the collection be done
            supress_subsequent_collect:
                When true, suppress any collections registered by this function
                call. This is to ensure duplicate collections registered across
                the callstack by different components. In this case, only the
                outermost component will collect.

                When false, always collect (subject to given condition) regardless
                of earlier collection registration's suppression.

        Returns:
            a context manager that handles the collection when its enclosed
            code finished run.
        """
        this_collection_handle = object()
        suppressed_by = _SUBSEQUENT_COLLECT_SUPPRESSED_BY.get(None)
        reset_suppressed_by = False
        if supress_subsequent_collect:
            if suppressed_by and suppressed_by != this_collection_handle:
                # Disable this collection since it's suppressed by a previously
                # installed collection
                condition = CollectionConditions.never()
            else:
                suppressed_by = this_collection_handle
                _SUBSEQUENT_COLLECT_SUPPRESSED_BY.set(suppressed_by)
                # don't forget to reset it in `finanlly`
                reset_suppressed_by = True

        is_in_collect_context_tok = _IS_IN_COLLECT_CONTEXT.set(True)
        exception: t.Optional[Exception] = None
        try:
            yield
        except Exception as e:
            exception = e
            raise
        finally:
            if reset_suppressed_by:
                _SUBSEQUENT_COLLECT_SUPPRESSED_BY.set(None)
            if self._test_condition(condition, CollectionConditionContext(exception)):
                try:
                    self.collect()
                except Exception as e:
                    _LOGGER.warning(
                        f"Error while collecting diagnostics (THIS EXCEPTION IS HANDLED):\n"
                        f"{e}\n"
                        f"{traceback.format_exc()}"
                    )
            _IS_IN_COLLECT_CONTEXT.reset(is_in_collect_context_tok)

    def collect(self) -> str:
        """Collect the diagnostics. Overridable in sub-classes."""
        return ""

    @classmethod
    def _test_condition(
        cls,
        cond: CollectionCondition,
        ctx: CollectionConditionContext
    ) -> bool:
        try:
            return cond(ctx)
        except Exception as e:
            _LOGGER.warning(f"Error while testing condition: {e}")
            return False


class ZipDiagnosticsCollector(DiagnosticsCollector):
    _write: DiagnosticsWriter
    _last_zip_path_for_test: str = ""  # for test purpose only

    def __init__(self, writer: DiagnosticsWriter):
        self._write = writer

    def collect(self) -> str:
        _, fp = tempfile.mkstemp()
        try:
            zip_path = shutil.make_archive(fp, "zip", self._write.root_dir())
            self._last_zip_path_for_test = zip_path
            return zip_path
        finally:
            os.remove(fp)


def _res_or_err(data: WriteObj) -> t.Tuple[TWrite, str]:
    if isinstance(data, (str, bytes)):
        return data, ""
    if not callable(data):
        raise TypeError(
            f"data must be a callable that returns actual data to"
            f"write, but got {type(data)}"
        )
    try:
        return data(), ""
    except Exception as e:
        _LOGGER.warning(f"Error getting data to write: {e}")
        return "", str(e)
