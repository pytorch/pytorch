#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

import functools
import json
import os
import shutil
import signal
import tempfile
import unittest
from unittest import mock

from torch.distributed.elastic.multiprocessing.errors import (
    ChildFailedError,
    ProcessFailure,
    record,
)
from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler


class SentinelError(Exception):
    # exists so that we can validate that
    # the correct error is raised and propagated
    pass


@record
def raise_exception_fn():
    raise SentinelError("foobar")


@record
def raise_system_exit_exception_fn(exit_code: int = 1):
    exp = SystemExit()
    exp.code = exit_code
    raise exp


@record
def good_fn():
    print("hello world")


@record
def raise_child_failure_error_fn(name, child_error_file=""):
    if child_error_file:
        with mock.patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": child_error_file}):
            ErrorHandler().record_exception(SentinelError("foobar"))
    pf = ProcessFailure(local_rank=0, pid=997, exitcode=1, error_file=child_error_file)
    raise ChildFailedError(name, {0: pf})


def raise_sentinel_error_fn():
    raise SentinelError("foobar")


def return_ok_fn():
    return "ok"


@record
def function_for_testing():
    return None


# A functools.partial has no __qualname__. MultiprocessContext._wrap runs
# record(fn)(*args_) in every spawned worker, and torchrec's elastic_launch
# passes a partial as fn -- the exact entrypoint shape that regressed in
# D116228049. `@record` needs a `def`, but `@record` is just `fn = record(fn)`,
# so wrapping a partial with record() is the faithful decorator equivalent.
partial_entrypoint_fn = functools.partial(raise_sentinel_error_fn)
record_wrapped_partial_fn = record(partial_entrypoint_fn)


class FnNameCapturingErrorHandler(ErrorHandler):
    """Captures the entrypoint fn_name that ``@record`` threads via handler state.

    Deliberately overrides ``initialize``/``record_exception`` with the
    pre-fn_name signatures to prove that old-style subclasses still receive the
    entrypoint name (through ``self._fn_name``) and are not broken by ``@record``.
    """

    def __init__(self) -> None:
        self.initialize_fn_name: str | None = None
        self.record_exception_fn_name: str | None = None
        self.record_success_fn_name: str | None = None

    def initialize(self) -> None:
        self.initialize_fn_name = self._fn_name
        super().initialize()

    def record_exception(self, e: BaseException) -> None:
        self.record_exception_fn_name = self._fn_name
        super().record_exception(e)

    def record_success(self) -> None:
        self.record_success_fn_name = self._fn_name
        super().record_success()


def read_resource_file(resource_file: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), resource_file)) as fp:
        return "".join(fp.readlines())


class ApiTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)
        self.test_error_file = os.path.join(self.test_dir, "error.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_failure_incorrect_reply_file(self):
        content = {"unknown_key": "unknown_value"}
        with open(self.test_error_file, "w") as fp:
            json.dump(content, fp)
        with self.assertRaises(Exception):
            ProcessFailure(
                local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
            )

    def failure_with_error_file(self, exception):
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            ErrorHandler().record_exception(exception)
        return ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
        )

    def failure_without_error_file(self, exitcode):
        return ProcessFailure(
            local_rank=0, pid=997, exitcode=exitcode, error_file="ignored.json"
        )

    def test_process_failure_new_format(self):
        error_data = {"message": "test error message", "timestamp": 10}
        with open(self.test_error_file, "w") as fp:
            json.dump(error_data, fp)
        pf = ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
        )
        self.assertEqual("test error message", pf.message)
        self.assertEqual(10, pf.timestamp)

    def test_process_mast_error_format(self):
        error_data = {"message": "test error message", "timestamp": "10"}
        with open(self.test_error_file, "w") as fp:
            json.dump(error_data, fp)
        pf = ProcessFailure(
            local_rank=0, pid=997, exitcode=1, error_file=self.test_error_file
        )
        self.assertEqual("test error message", pf.message)
        self.assertEqual(10, pf.timestamp)

    def test_process_failure(self):
        pf = self.failure_with_error_file(exception=SentinelError("foobar"))
        self.assertEqual(0, pf.local_rank)
        self.assertEqual(997, pf.pid)
        self.assertEqual(1, pf.exitcode)
        self.assertEqual(self.test_error_file, pf.error_file)
        self.assertEqual(
            pf.error_file_data["message"]["extraInfo"]["timestamp"], str(pf.timestamp)
        )
        self.assertTrue(pf.message)  # check not None and not "" (empty string)
        self.assertEqual("<N/A>", pf.signal_name())

    def test_process_failure_signal(self):
        pf = self.failure_without_error_file(exitcode=-signal.SIGSEGV)
        self.assertEqual("SIGSEGV", pf.signal_name())
        self.assertEqual(
            f"Signal {signal.SIGSEGV} (SIGSEGV) received by PID {pf.pid}", pf.message
        )

    def test_format_msg_enriches_root_signal_failure(self):
        # The root signal failure's rendered message is passed through the
        # handler's enrichment seam at format time (no-op base; a build-swapped
        # handler may append device-fault context). The seam works on the local
        # rendered string, so the failure's stored message is not mutated.
        with open(self.test_error_file, "w") as fp:
            json.dump({"message": "Fatal signal received", "timestamp": 10}, fp)
        handler = mock.MagicMock()
        handler.maybe_enrich_signal_failure_message.side_effect = (
            lambda message, error_file: message + "\n[device fault context]"
        )
        with mock.patch(
            "torch.distributed.elastic.multiprocessing.errors.get_error_handler",
            return_value=handler,
        ):
            pf = ProcessFailure(
                local_rank=0,
                pid=997,
                exitcode=-signal.SIGABRT,
                error_file=self.test_error_file,
            )
            rendered = str(ChildFailedError("trainer", {0: pf}))
        self.assertIn("[device fault context]", rendered)
        self.assertNotIn("[device fault context]", pf.message)
        handler.maybe_enrich_signal_failure_message.assert_called_once_with(
            mock.ANY, self.test_error_file
        )

    def test_process_failure_no_error_file(self):
        pf = self.failure_without_error_file(exitcode=138)
        self.assertEqual("<N/A>", pf.signal_name())
        self.assertEqual("<N/A>", pf.error_file)
        self.assertEqual(
            "To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html",
            pf.message,
        )

    def test_child_failed_error(self):
        pf0 = self.failure_with_error_file(exception=SentinelError("rank 0"))
        pf1 = self.failure_with_error_file(exception=SentinelError("rank 1"))
        pf2 = self.failure_without_error_file(exitcode=138)
        ex = ChildFailedError("trainer.par", {0: pf0, 1: pf1, 2: pf2})
        self.assertEqual(pf0, ex.get_first_failure()[1])
        # print is intentional and should prints something like this:
        """
        *********************************************
              trainer.par FAILED
        =============================================
        Root Cause:
        [0]:
          time: 2020-11-25_21:22:31
          rank: 0 (local_rank: 0)
          exitcode: 1 (pid: 997)
          error_file: /tmp/ApiTesttbb37ier/error.json
          traceback: "SentinelError: rank 0"
        =============================================
        Other Failures:
        [1]:
          time: 2020-11-25_21:22:31
          rank: 1 (local_rank: 0)
          exitcode: 1 (pid: 997)
          error_file: /tmp/ApiTesttbb37ier/error.json
          msg: "SentinelError: rank 1"
        [2]:
          time: 2020-11-25_21:22:31
          rank: 2 (local_rank: 0)
          exitcode: 138 (pid: 997)
          error_file: <N/A>
          traceback: To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html
        *********************************************
        """
        print(ex)

    def test_record(self):
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(SentinelError):
                raise_exception_fn()

        with open(self.test_error_file) as fp:
            err = json.load(fp)
            self.assertIsNotNone(err["message"]["message"])
            self.assertIsNotNone(err["message"]["extraInfo"]["py_callstack"])
            self.assertIsNotNone(err["message"]["extraInfo"]["timestamp"])

    def test_record_system_exit(self):
        with mock.patch.dict(os.environ, {}):
            raise_system_exit_exception_fn(exit_code=0)

        # no error file should have been generated
        self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_system_exit_erronr(self):
        with mock.patch.dict(os.environ, {}):
            with self.assertRaises(SystemExit):
                raise_system_exit_exception_fn()

        # no error file should have been generated
        self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_no_error_file(self):
        with mock.patch.dict(os.environ, {}):
            with self.assertRaises(SentinelError):
                raise_exception_fn()

        # no error file should have been generated
        self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_good_fn(self):
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            good_fn()
            # function did not error; no error file should be produced
            self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_child_failure(self):
        trainer_log_dir = os.path.join(self.test_dir, "trainer", "0")
        os.makedirs(trainer_log_dir)
        trainer_error_file = os.path.join(trainer_log_dir, "error.json")

        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(ChildFailedError) as cm:
                raise_child_failure_error_fn("trainer", trainer_error_file)
            pf = cm.exception.get_first_failure()[1]
            # compare worker error file with reply file and overridden error code
            with open(pf.error_file) as f:
                expect = json.load(f)
            expect["message"]["errorCode"] = pf.exitcode
            with open(self.test_error_file) as f:
                actual = json.load(f)
            self.assertTrue(
                json.dumps(expect, sort_keys=True),
                json.dumps(actual, sort_keys=True),
            )

    def test_record_child_failure_no_child_error_file(self):
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(ChildFailedError):
                raise_child_failure_error_fn("trainer")

            # @record should only copy child error file when ChildFailedError
            # is raised - it should NOT record ChildFailedError itself
            # it SHOULD re-raise ChildFailedError for any upstream system
            # to handle it.
            self.assertFalse(os.path.isfile(self.test_error_file))

    def test_child_failed_error_signal_name_in_message(self):
        pf = self.failure_without_error_file(exitcode=-signal.SIGSEGV)
        ex = ChildFailedError("trainer.par", {0: pf})
        error_msg = str(ex)
        self.assertIn("(SIGSEGV)", error_msg)
        self.assertIn(f"exitcode  : {-signal.SIGSEGV}", error_msg)

    def test_record_passes_fn_name_to_error_handler(self):
        # a subclass using the pre-fn_name signatures must still receive the
        # entrypoint name via handler state and must not raise from @record
        error_handler = FnNameCapturingErrorHandler()
        wrapped = record(raise_sentinel_error_fn, error_handler=error_handler)

        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(SentinelError):
                wrapped()

        self.assertEqual("raise_sentinel_error_fn", error_handler.initialize_fn_name)
        self.assertEqual(
            "raise_sentinel_error_fn", error_handler.record_exception_fn_name
        )

    def test_record_partial_entrypoint_without_qualname(self):
        # Regression test for D116228049: a functools.partial entrypoint has no
        # __qualname__. Before the fix, @record's f.__qualname__ raised
        # AttributeError in every spawned worker; getattr(f, "__qualname__",
        # None) now threads None instead of crashing.
        self.assertFalse(hasattr(partial_entrypoint_fn, "__qualname__"))

        # real-world path (default handler): only the underlying error surfaces,
        # not AttributeError from @record.
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(SentinelError):
                record_wrapped_partial_fn()

        # the missing __qualname__ is threaded to the handler as None (no fn
        # attribution), rather than raising. Use a fresh error file: the write
        # above leaves error.json read-only to preserve the first failure.
        error_handler = FnNameCapturingErrorHandler()
        capture_error_file = os.path.join(self.test_dir, "capture_error.json")
        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": capture_error_file}
        ):
            with self.assertRaises(SentinelError):
                record(partial_entrypoint_fn, error_handler=error_handler)()

        self.assertIsNone(error_handler.initialize_fn_name)
        self.assertIsNone(error_handler.record_exception_fn_name)

    def test_record_decorated_fn_threads_qualname(self):
        # Counterpart to the partial test: a plain @record def HAS __qualname__,
        # so getattr(f, "__qualname__", None) takes the non-fallback branch and
        # threads the function's qualified name. Calling the decorated fn also
        # shows the shared line runs on the normal path without raising.
        self.assertIsNone(function_for_testing())

        error_handler = FnNameCapturingErrorHandler()
        result = record(function_for_testing.__wrapped__, error_handler=error_handler)()

        self.assertIsNone(result)
        self.assertEqual("function_for_testing", error_handler.initialize_fn_name)
        self.assertEqual("function_for_testing", error_handler.record_success_fn_name)

    def test_record_does_not_write_fn_name_to_error_file(self):
        # extraInfo is a map<string,string> for downstream consumers, so @record
        # threads fn_name to the handler but the base handler does not persist it
        wrapped = record(raise_sentinel_error_fn, error_handler=ErrorHandler())

        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(SentinelError):
                wrapped()

        with open(self.test_error_file) as fp:
            err = json.load(fp)
        self.assertNotIn("fn_name", err["message"]["extraInfo"])

    def test_record_calls_record_success_on_success(self):
        # @record must invoke record_success and return the fn's value on the
        # no-exception path, without recording an exception or error file
        error_handler = mock.MagicMock(spec=ErrorHandler)
        wrapped = record(return_ok_fn, error_handler=error_handler)

        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            result = wrapped()

        self.assertEqual("ok", result)
        error_handler.record_success.assert_called_once()
        error_handler.record_exception.assert_not_called()
        self.assertFalse(os.path.isfile(self.test_error_file))

    def test_record_does_not_call_record_success_on_failure(self):
        error_handler = mock.MagicMock(spec=ErrorHandler)
        wrapped = record(raise_sentinel_error_fn, error_handler=error_handler)

        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(SentinelError):
                wrapped()

        error_handler.record_success.assert_not_called()
        error_handler.record_exception.assert_called_once()

    def test_record_success_receives_fn_name(self):
        # record_success runs while the entrypoint fn_name is still set on the handler
        error_handler = FnNameCapturingErrorHandler()
        wrapped = record(return_ok_fn, error_handler=error_handler)

        result = wrapped()

        self.assertEqual("ok", result)
        self.assertEqual("return_ok_fn", error_handler.record_success_fn_name)

    def test_record_success_error_not_recorded_as_failure(self):
        # if record_success() raises, it must propagate and NOT be routed through
        # record_exception (which would misreport a successful run as a failure)
        error_handler = mock.MagicMock(spec=ErrorHandler)
        error_handler.record_success.side_effect = RuntimeError("telemetry boom")
        wrapped = record(return_ok_fn, error_handler=error_handler)

        with mock.patch.dict(
            os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}
        ):
            with self.assertRaises(RuntimeError):
                wrapped()

        error_handler.record_exception.assert_not_called()
