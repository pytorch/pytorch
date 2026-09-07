"""Baseline behaviour of the TorchFuzz CLI helpers.

`fuzzer.py` is the entry point every fuzzing run goes through, but its argument
parsing and path resolution had no tests. These lock the current contract so a
later change to parsing, seed substitution, or the exit code and stream a
rejected invocation uses has to update an assertion instead of moving silently.
"""

import io
import logging
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from torchfuzz.fuzzer import (
    _parse_supported_ops_with_weights,
    _resolve_generate_only_path,
    main,
)


class ParseSupportedOpsWithWeightsTest(unittest.TestCase):
    def test_empty_spec_yields_nothing(self) -> None:
        self.assertEqual(([], {}), _parse_supported_ops_with_weights(""))

    def test_bare_operator_has_no_weight(self) -> None:
        ops, weights = _parse_supported_ops_with_weights("torch.add")

        self.assertEqual(["torch.add"], ops)
        self.assertEqual({}, weights)

    def test_weight_is_parsed_as_float(self) -> None:
        ops, weights = _parse_supported_ops_with_weights("torch.matmul=5")

        self.assertEqual(["torch.matmul"], ops)
        self.assertEqual({"torch.matmul": 5.0}, weights)

    def test_surrounding_whitespace_is_ignored(self) -> None:
        ops, weights = _parse_supported_ops_with_weights(" torch.add , torch.mul = 2 ")

        self.assertEqual(["torch.add", "torch.mul"], ops)
        self.assertEqual({"torch.mul": 2.0}, weights)

    def test_empty_entries_are_skipped(self) -> None:
        ops, _weights = _parse_supported_ops_with_weights("torch.add,,torch.mul")

        self.assertEqual(["torch.add", "torch.mul"], ops)

    def test_unparsable_weight_drops_the_operator_entirely(self) -> None:
        # Not "weight ignored": the whole entry is skipped, so the operator
        # never reaches the allowlist and a typo silently narrows the run.
        ops, weights = _parse_supported_ops_with_weights("torch.add=abc,torch.mul")

        self.assertEqual(["torch.mul"], ops)
        self.assertEqual({}, weights)


class ResolveGenerateOnlyPathTest(unittest.TestCase):
    def test_pattern_without_marker_is_returned_unchanged(self) -> None:
        self.assertEqual("out.py", _resolve_generate_only_path("out.py", 7))

    def test_marker_is_replaced_by_the_seed(self) -> None:
        self.assertEqual("out7.py", _resolve_generate_only_path("out?.py", 7))

    def test_marker_run_sets_the_zero_padding_width(self) -> None:
        self.assertEqual("out007.py", _resolve_generate_only_path("out???.py", 7))

    def test_seed_wider_than_the_marker_is_not_truncated(self) -> None:
        self.assertEqual("out1234.py", _resolve_generate_only_path("out??.py", 1234))

    def test_split_marker_runs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "together"):
            _resolve_generate_only_path("o?ut?.py", 7)

    def test_missing_marker_is_rejected_when_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "must contain a"):
            _resolve_generate_only_path("out.py", 7, require_marker=True)


class ArgparseContractTest(unittest.TestCase):
    """Exit code and output stream used for a rejected invocation.

    Bounce tooling and shell callers key off both, so changing either is a
    contract change rather than a refactor.
    """

    def _reject(self, argv: list[str], *, stream: str = "stdout") -> tuple[object, str]:
        """Run `main()` with `argv`, returning its exit code and that stream.

        `main()` initialises the device plugin and calls `logging.basicConfig`
        before it validates arguments. Both are neutralised here: otherwise a
        parsing test can fail for reasons unrelated to parsing, and every later
        test in the process inherits a reconfigured root logger.
        """
        root = logging.getLogger()
        level, handlers = root.level, list(root.handlers)

        def restore() -> None:
            root.setLevel(level)
            root.handlers[:] = handlers

        self.addCleanup(restore)

        captured = io.StringIO()
        redirect = redirect_stderr if stream == "stderr" else redirect_stdout
        with mock.patch("torchfuzz.fuzzer.initialize_codegen"), mock.patch(
            "torchfuzz.fuzzer.get_template_names", return_value=["default"]
        ), mock.patch.object(sys, "argv", argv), redirect(captured):
            with self.assertRaises(SystemExit) as raised:
                main()
        return raised.exception.code, captured.getvalue()

    def test_conflicting_generate_only_exits_one_and_writes_stdout(self) -> None:
        code, out = self._reject(
            ["fuzzer.py", "--generate-only", "out.py", "--stop-at-first-failure"]
        )

        self.assertEqual(1, code)
        self.assertIn("--generate-only cannot be used", out)

    def test_missing_start_exits_one_and_writes_stdout(self) -> None:
        code, out = self._reject(["fuzzer.py", "--count", "2"])

        self.assertEqual(1, code)
        self.assertIn("--start is required", out)

    def test_missing_count_exits_one_and_writes_stdout(self) -> None:
        code, out = self._reject(["fuzzer.py", "--start", "0"])

        self.assertEqual(1, code)
        self.assertIn("--count is required", out)

    def test_multi_seed_generate_only_requires_a_seed_marker(self) -> None:
        code, out = self._reject(
            ["fuzzer.py", "--generate-only", "out.py", "--start", "0", "--count", "2"]
        )

        self.assertEqual(1, code)
        self.assertIn("must contain a", out)

