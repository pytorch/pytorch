#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

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


def read_resource_file(resource_file: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), resource_file)) as fp:
        return "".join(fp.readlines())


class ApiTest(unittest.TestCase):
    def setUp(self):
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
            expect = json.load(open(pf.error_file))
            expect["message"]["errorCode"] = pf.exitcode
            actual = json.load(open(self.test_error_file))
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
