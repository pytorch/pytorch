"""Utilities to handle communication between parent worker."""
import dataclasses
import datetime
import inspect
import io
import marshal
import os
import pathlib
import pickle
import sys
import textwrap
import traceback
import types
import typing


class ExceptionUnpickler(pickle.Unpickler):

    @classmethod
    def load_bytes(cls, data: bytes) -> typing.Type[Exception]:
        result = cls(io.BytesIO(data)).load()

        # Make sure we have an Exception class, but not an instantiated
        # Exception.
        if not issubclass(result, Exception):
            raise pickle.UnpicklingError(f"{result} is not an Exception")

        if isinstance(result, Exception):
            raise pickle.UnpicklingError(
                f"{result} is an Exception instance, not a class.")

        return result   # type: ignore[no-any-return]

    def find_class(self, module: str, name: str) -> typing.Any:
        if module != "builtins":
            raise pickle.UnpicklingError(f"Invalid object: {module}.{name}")
        return super().find_class(module, name)


class UnserializableException(Exception):

    def __init__(self, type_repr: str, args_repr: str) -> None:
        self.type_repr = type_repr
        self.args_repr = args_repr
        super().__init__(type_repr, args_repr)


class ChildTraceException(Exception):
    pass


@dataclasses.dataclass(init=True, frozen=True)
class SerializedException:
    _is_serializable: bool
    _type_bytes: bytes
    _args_bytes: bytes

    # Fallbacks for UnserializableException
    _type_repr: str
    _args_repr: str

    _traceback_print: str

    @staticmethod
    def from_exception(e: Exception, tb: types.TracebackType) -> "SerializedException":
        """Best effort attempt to serialize Exception.

        Because this will be used to communicate from a subprocess to its
        parent, we want to surface as much information as possible. It is
        not possible to serialize a traceback because it is too intertwined
        with the runtime; however what we really want is the traceback so we
        can print it. We can grab that string and send it without issue.

        ExceptionUnpickler explicitly refuses to load any non-builtin exception
        (for the same reason we prefer `marshal` to `pickle`), so we won't be
        able to serialize all cases. However we don't want to simply give up
        as this will make it difficult for a user to diagnose what's going on.
        So instead we extract what information we can, and raise an
        UnserializableException in the main process with whatever we were able
        to scrape up from the child process.
        """
        try:
            print_file = io.StringIO()
            traceback.print_exception(
                etype=type(e),
                value=e,
                tb=tb,
                file=print_file,
            )
            print_file.seek(0)
            traceback_print: str = print_file.read()

        except Exception:
            traceback_print = textwrap.dedent("""
                Traceback
                    Failed to extract traceback from worker. This is not expected.
            """).strip()

        try:
            args_bytes: bytes = marshal.dumps(e.args)
            type_bytes = pickle.dumps(e.__class__)

            # Make sure we'll be able to get something out on the other side.
            revived_type = ExceptionUnpickler.load_bytes(data=type_bytes)
            revived_e = revived_type(*marshal.loads(args_bytes))
            is_serializable: bool = True

        except Exception:
            is_serializable = False
            args_bytes = b""
            type_bytes = b""

        # __repr__ can contain arbitrary code, so we can't trust it to noexcept.
        def hardened_repr(o: typing.Any) -> str:
            try:
                return repr(o)

            except Exception:
                return "< Unknown >"

        return SerializedException(
            _is_serializable=is_serializable,
            _type_bytes=type_bytes,
            _args_bytes=args_bytes,
            _type_repr=hardened_repr(e.__class__),
            _args_repr=hardened_repr(getattr(e, "args", None)),
            _traceback_print=traceback_print,
        )

    @staticmethod
    def raise_from(
        serialized_e: "SerializedException",
        extra_context: typing.Optional[str] = None,
    ) -> None:
        """Revive `serialized_e`, and raise.

        We raise the revived exception type (if possible) so that any higher
        try catch logic will see the original exception type. In other words:
        ```
            try:
                worker.run("assert False")
            except AssertionError:
                ...
        ```

        will flow identically to:

        ```
            try:
                assert False
            except AssertionError:
                ...
        ```

        If for some reason we can't move the true exception type to the main
        process (e.g. a custom Exception) we raise UnserializableException as
        a fallback.
        """
        if serialized_e._is_serializable:
            revived_type = ExceptionUnpickler.load_bytes(data=serialized_e._type_bytes)
            e = revived_type(*marshal.loads(serialized_e._args_bytes))
        else:
            e = UnserializableException(serialized_e._type_repr, serialized_e._args_repr)

        traceback_str = serialized_e._traceback_print
        if extra_context:
            traceback_str = f"{traceback_str}\n{extra_context}"

        raise e from ChildTraceException(traceback_str)


class FileBasedCommunication:

    def __init__(self, communication_dir: str):
        self._communication_dir = communication_dir

        self._begin_sentinel = os.path.join(self._communication_dir, "begin")
        self._exception_payload = os.path.join(self._communication_dir, "exception_data")
        self._finished_sentinel = os.path.join(self._communication_dir, "finished")

    def reset(self) -> None:
        for fpath in (
            self._begin_sentinel,
            self._exception_payload,
            self._finished_sentinel
        ):
            if os.path.exists(fpath):
                os.remove(fpath)

    def save_exception(self, serialized_e: SerializedException) -> None:
        with open(self._exception_payload, "wb") as f:
            marshal.dump(dataclasses.asdict(serialized_e), f)

    def get_exception(self) -> typing.Optional[SerializedException]:
        if os.path.exists(self._exception_payload):
            with open(self._exception_payload, "rb") as f:
                kwargs = marshal.load(f)

            # We can't marshal SerializedException as it is a custom class,
            # however all of the fields are builtin types so we can revive the
            # field dict as kwargs. Because the worker and parent are running
            # the same code we don't need to worry about versioning.
            return SerializedException(**kwargs)
        return None

    def declare_begin(self) -> None:
        pathlib.Path(self._begin_sentinel).touch()

    def declare_finished(self) -> None:
        pathlib.Path(self._finished_sentinel).touch()

    def is_finished(self) -> bool:
        return os.path.exists(self._finished_sentinel)


def _log_progress(suffix: str) -> None:
    now = datetime.datetime.now().strftime("[%Y-%m-%d] %H:%M:%S.%f")
    print(f"\n{now}: TIMER_SUBPROCESS_{suffix}")


def subprocess_snippet_handler(
    snippet: str,
    communication_dir: str
) -> None:
    communication_handler = FileBasedCommunication(communication_dir)
    try:
        communication_handler.declare_begin()
        _log_progress("BEGIN")

        # In Python, `global` means global to a module, not global to the
        # program. So if we simply call `globals()`, we will get the globals
        # for this module (which contains lots of implementation details),
        # not the globals from the from the calling context. So instead we grab
        # the calling frame exec with those globals.
        calling_frame = inspect.stack()[1].frame

        exec(  # noqa: P204
            compile(snippet, "<subprocess-worker>", "exec"),
            calling_frame.f_globals,
        )

        _log_progress("SUCCESS")

    except Exception as e:
        tb = sys.exc_info()[2]
        assert tb is not None
        serialized_e = SerializedException.from_exception(e, tb)
        communication_handler.save_exception(serialized_e)
        _log_progress("FAILED")

    finally:
        _log_progress("FINISHED")
        sys.stdout.flush()
        sys.stderr.flush()
        communication_handler.declare_finished()
