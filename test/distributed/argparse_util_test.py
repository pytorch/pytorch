#!/usr/bin/env python3
# Owner(s): ["oncall: distributed"]

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import unittest
from argparse import ArgumentParser

from torch.distributed.argparse_util import check_env, env


class ArgParseUtilTest(unittest.TestCase):
    def setUp(self):
        # remove any lingering environment variables
        for e in os.environ.keys():
            if e.startswith("PET_"):
                del os.environ[e]

    def test_env_string_arg_no_env(self):
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default="bar")

        self.assertEqual("bar", parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_env_string_arg_env(self):
        os.environ["PET_FOO"] = "env_baz"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default="bar")

        self.assertEqual("env_baz", parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_env_int_arg_no_env(self):
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default=1, type=int)

        self.assertEqual(1, parser.parse_args([]).foo)
        self.assertEqual(2, parser.parse_args(["-f", "2"]).foo)
        self.assertEqual(2, parser.parse_args(["--foo", "2"]).foo)

    def test_env_int_arg_env(self):
        os.environ["PET_FOO"] = "3"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default=1, type=int)

        self.assertEqual(3, parser.parse_args([]).foo)
        self.assertEqual(2, parser.parse_args(["-f", "2"]).foo)
        self.assertEqual(2, parser.parse_args(["--foo", "2"]).foo)

    def test_env_no_default_no_env(self):
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env)

        self.assertIsNone(parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_env_no_default_env(self):
        os.environ["PET_FOO"] = "env_baz"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env)

        self.assertEqual("env_baz", parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_env_required_no_env(self):
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, required=True)

        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_env_required_env(self):
        os.environ["PET_FOO"] = "env_baz"
        parser = ArgumentParser()
        parser.add_argument("-f", "--foo", action=env, default="bar", required=True)

        self.assertEqual("env_baz", parser.parse_args([]).foo)
        self.assertEqual("baz", parser.parse_args(["-f", "baz"]).foo)
        self.assertEqual("baz", parser.parse_args(["--foo", "baz"]).foo)

    def test_check_env_no_env(self):
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env)

        self.assertFalse(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["-v"]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_check_env_default_no_env(self):
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env, default=True)

        self.assertTrue(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["-v"]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_check_env_env_zero(self):
        os.environ["PET_VERBOSE"] = "0"
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env)

        self.assertFalse(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_check_env_env_one(self):
        os.environ["PET_VERBOSE"] = "1"
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env)

        self.assertTrue(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_check_env_default_env_zero(self):
        os.environ["PET_VERBOSE"] = "0"
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env, default=True)

        self.assertFalse(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)

    def test_check_env_default_env_one(self):
        os.environ["PET_VERBOSE"] = "1"
        parser = ArgumentParser()
        parser.add_argument("-v", "--verbose", action=check_env, default=True)

        self.assertTrue(parser.parse_args([]).verbose)
        self.assertTrue(parser.parse_args(["--verbose"]).verbose)
