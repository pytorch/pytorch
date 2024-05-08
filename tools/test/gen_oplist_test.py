#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import unittest
from unittest.mock import MagicMock

from tools.code_analyzer.gen_oplist import throw_if_any_op_includes_overloads


class GenOplistTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_throw_if_any_op_includes_overloads(self):
        selective_builder = MagicMock()
        selective_builder.operators = MagicMock()
        selective_builder.operators.items.return_value = [
            ("op1", MagicMock(include_all_overloads=True)),
            ("op2", MagicMock(include_all_overloads=False)),
            ("op3", MagicMock(include_all_overloads=True)),
        ]

        self.assertRaises(
            Exception, throw_if_any_op_includes_overloads, selective_builder
        )

        selective_builder.operators.items.return_value = [
            ("op1", MagicMock(include_all_overloads=False)),
            ("op2", MagicMock(include_all_overloads=False)),
            ("op3", MagicMock(include_all_overloads=False)),
        ]

        # Here we do not expect it to throw an exception since none of the ops
        # include all overloads.
        throw_if_any_op_includes_overloads(selective_builder)
