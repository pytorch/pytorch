#!/usr/bin/env python3
# Owner(s): ["oncall: r2p"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import ctypes
import os
import shutil
import sys
import tempfile
import unittest

from torch.distributed.elastic.multiprocessing.redirects import (
    redirect,
    redirect_stderr,
    redirect_stdout,
)


libc = ctypes.CDLL("libc.so.6")
c_stderr = ctypes.c_void_p.in_dll(libc, "stderr")


class RedirectsTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix=f"{self.__class__.__name__}_")

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_redirect_invalid_std(self):
        with self.assertRaises(ValueError):
            with redirect("stdfoo", os.path.join(self.test_dir, "stdfoo.log")):
                pass

    def test_redirect_stdout(self):
        stdout_log = os.path.join(self.test_dir, "stdout.log")

        # printing to stdout before redirect should go to console not stdout.log
        print("foo first from python")
        libc.printf(b"foo first from c\n")
        os.system("echo foo first from cmd")

        with redirect_stdout(stdout_log):
            print("foo from python")
            libc.printf(b"foo from c\n")
            os.system("echo foo from cmd")

        # make sure stdout is restored
        print("foo again from python")
        libc.printf(b"foo again from c\n")
        os.system("echo foo again from cmd")

        with open(stdout_log) as f:
            # since we print from python, c, cmd -> the stream is not ordered
            # do a set comparison
            lines = set(f.readlines())
            self.assertEqual(
                {"foo from python\n", "foo from c\n", "foo from cmd\n"}, lines
            )

    def test_redirect_stderr(self):
        stderr_log = os.path.join(self.test_dir, "stderr.log")

        print("bar first from python")
        libc.fprintf(c_stderr, b"bar first from c\n")
        os.system("echo bar first from cmd 1>&2")

        with redirect_stderr(stderr_log):
            print("bar from python", file=sys.stderr)
            libc.fprintf(c_stderr, b"bar from c\n")
            os.system("echo bar from cmd 1>&2")

        print("bar again from python")
        libc.fprintf(c_stderr, b"bar again from c\n")
        os.system("echo bar again from cmd 1>&2")

        with open(stderr_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"bar from python\n", "bar from c\n", "bar from cmd\n"}, lines
            )

    def test_redirect_both(self):
        stdout_log = os.path.join(self.test_dir, "stdout.log")
        stderr_log = os.path.join(self.test_dir, "stderr.log")

        print("first stdout from python")
        libc.printf(b"first stdout from c\n")

        print("first stderr from python", file=sys.stderr)
        libc.fprintf(c_stderr, b"first stderr from c\n")

        with redirect_stdout(stdout_log), redirect_stderr(stderr_log):
            print("redir stdout from python")
            print("redir stderr from python", file=sys.stderr)
            libc.printf(b"redir stdout from c\n")
            libc.fprintf(c_stderr, b"redir stderr from c\n")

        print("again stdout from python")
        libc.fprintf(c_stderr, b"again stderr from c\n")

        with open(stdout_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"redir stdout from python\n", "redir stdout from c\n"}, lines
            )

        with open(stderr_log) as f:
            lines = set(f.readlines())
            self.assertEqual(
                {"redir stderr from python\n", "redir stderr from c\n"}, lines
            )

    def _redirect_large_buffer(self, print_fn, num_lines=500_000):
        stdout_log = os.path.join(self.test_dir, "stdout.log")

        with redirect_stdout(stdout_log):
            for i in range(num_lines):
                print_fn(i)

        with open(stdout_log) as fp:
            actual = {int(line.split(":")[1]) for line in fp}
            expected = set(range(num_lines))
            self.assertSetEqual(expected, actual)

    def test_redirect_large_buffer_py(self):
        def py_print(i):
            print(f"py:{i}")

        self._redirect_large_buffer(py_print)

    def test_redirect_large_buffer_c(self):
        def c_print(i):
            libc.printf(bytes(f"c:{i}\n", "utf-8"))

        self._redirect_large_buffer(c_print)
