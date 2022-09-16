#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

import filecmp
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from torch.distributed.elastic.multiprocessing.errors.error_handler import ErrorHandler
from torch.distributed.elastic.multiprocessing.errors.handlers import get_error_handler


def raise_exception_fn():
    raise RuntimeError("foobar")


class GetErrorHandlerTest(unittest.TestCase):
    def test_get_error_handler(self):
        self.assertTrue(isinstance(get_error_handler(), ErrorHandler))


class ErrorHandlerTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=self.__class__.__name__)
        self.test_error_file = os.path.join(self.test_dir, "error.json")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("faulthandler.enable")
    def test_initialize(self, fh_enable_mock):
        ErrorHandler().initialize()
        fh_enable_mock.assert_called_once()

    @patch("faulthandler.enable", side_effect=RuntimeError)
    def test_initialize_error(self, fh_enable_mock):
        # makes sure that initialize handles errors gracefully
        ErrorHandler().initialize()
        fh_enable_mock.assert_called_once()

    def test_record_exception(self):
        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}):
            eh = ErrorHandler()
            eh.initialize()

            try:
                raise_exception_fn()
            except Exception as e:
                eh.record_exception(e)

            with open(self.test_error_file, "r") as fp:
                err = json.load(fp)
                # error file content example:
                # {
                #   "message": {
                #     "message": "RuntimeError: foobar",
                #     "extraInfo": {
                #       "py_callstack": "Traceback (most recent call last):\n  <... OMITTED ...>",
                #       "timestamp": "1605774851"
                #     }
                #   }
            self.assertIsNotNone(err["message"]["message"])
            self.assertIsNotNone(err["message"]["extraInfo"]["py_callstack"])
            self.assertIsNotNone(err["message"]["extraInfo"]["timestamp"])

    def test_record_exception_no_error_file(self):
        # make sure record does not fail when no error file is specified in env vars
        with patch.dict(os.environ, {}):
            eh = ErrorHandler()
            eh.initialize()
            try:
                raise_exception_fn()
            except Exception as e:
                eh.record_exception(e)

    def test_dump_error_file(self):
        src_error_file = os.path.join(self.test_dir, "src_error.json")
        eh = ErrorHandler()
        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": src_error_file}):
            eh.record_exception(RuntimeError("foobar"))

        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": self.test_error_file}):
            eh.dump_error_file(src_error_file)
            self.assertTrue(filecmp.cmp(src_error_file, self.test_error_file))

        with patch.dict(os.environ, {}):
            eh.dump_error_file(src_error_file)
            # just validate that dump_error_file works when
            # my error file is not set
            # should just log an error with src_error_file pretty printed

    def test_dump_error_file_overwrite_existing(self):
        dst_error_file = os.path.join(self.test_dir, "dst_error.json")
        src_error_file = os.path.join(self.test_dir, "src_error.json")
        eh = ErrorHandler()
        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": dst_error_file}):
            eh.record_exception(RuntimeError("foo"))

        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": src_error_file}):
            eh.record_exception(RuntimeError("bar"))

        with patch.dict(os.environ, {"TORCHELASTIC_ERROR_FILE": dst_error_file}):
            eh.dump_error_file(src_error_file)
            self.assertTrue(filecmp.cmp(src_error_file, dst_error_file))
