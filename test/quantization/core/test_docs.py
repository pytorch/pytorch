# Owner(s): ["oncall: quantization"]

import re
import contextlib
from pathlib import Path

import torch

from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    SingleLayerLinearModel,
)
from torch.testing._internal.common_quantized import override_quantized_engine
from torch.testing._internal.common_utils import IS_ARM64, IS_FBCODE
import unittest


@unittest.skipIf(IS_FBCODE, "some path issues in fbcode")
class TestQuantizationDocs(QuantizationTestCase):
    r"""
    The tests in this section import code from the quantization docs and check that
    they actually run without errors. In cases where objects are undefined in the code snippet,
    they must be provided in the test. The imports seem to behave a bit inconsistently,
    they can be imported either in the test file or passed as a global input
    """

    def run(self, result=None):
        with override_quantized_engine("qnnpack") if IS_ARM64 else contextlib.nullcontext():
            super().run(result)

    def _get_code(
        self, path_from_pytorch, unique_identifier, offset=2, short_snippet=False
    ):
        r"""
        This function reads in the code from the docs given a unique identifier.
        Most code snippets have a 2 space indentation, for other indentation levels,
        change the offset `arg`. the `short_snippet` arg can be set to allow for testing
        of smaller snippets, the check that this arg controls is used to make sure that
        we are not accidentally only importing a blank line or something.
        """

        def get_correct_path(path_from_pytorch):
            r"""
            Current working directory when CI is running test seems to vary, this function
            looks for docs relative to this test file.
            """
            core_dir = Path(__file__).parent
            assert core_dir.match("test/quantization/core/"), (
                "test_docs.py is in an unexpected location. If you've been "
                "moving files around, ensure that the test and build files have "
                "been updated to have the correct relative path between "
                "test_docs.py and the docs."
            )
            pytorch_root = core_dir.parent.parent.parent
            return pytorch_root / path_from_pytorch

        path_to_file = get_correct_path(path_from_pytorch)
        if path_to_file:
            with open(path_to_file) as file:
                content = file.readlines()

            # it will register as having a newline at the end in python
            if "\n" not in unique_identifier:
                unique_identifier += "\n"

            assert unique_identifier in content, f"could not find {unique_identifier} in {path_to_file}"

            # get index of first line of code
            line_num_start = content.index(unique_identifier) + 1

            # next find where the code chunk ends.
            # this regex will match lines that don't start
            # with a \n or "  " with number of spaces=offset
            r = r = re.compile("^[^\n," + " " * offset + "]")
            # this will return the line of first line that matches regex
            line_after_code = next(filter(r.match, content[line_num_start:]))
            last_line_num = content.index(line_after_code)

            # remove the first `offset` chars of each line and gather it all together
            code = "".join(
                [x[offset:] for x in content[line_num_start + 1 : last_line_num]]
            )

            # want to make sure we are actually getting some code,
            assert last_line_num - line_num_start > 3 or short_snippet, (
                f"The code in {path_to_file} identified by {unique_identifier} seems suspiciously short:"
                f"\n\n###code-start####\n{code}###code-end####"
            )
            return code

        return None

    def _test_code(self, code, global_inputs=None):
        r"""
        This function runs `code` using any vars in `global_inputs`
        """
        # if couldn't find the
        if code is not None:
            expr = compile(code, "test", "exec")
            exec(expr, global_inputs)

    def test_quantization_doc_ptdq(self):
        path_from_pytorch = "docs/source/quantization.rst"
        unique_identifier = "PTDQ API Example::"
        code = self._get_code(path_from_pytorch, unique_identifier)
        self._test_code(code)

    def test_quantization_doc_ptsq(self):
        path_from_pytorch = "docs/source/quantization.rst"
        unique_identifier = "PTSQ API Example::"
        code = self._get_code(path_from_pytorch, unique_identifier)
        self._test_code(code)

    def test_quantization_doc_qat(self):
        path_from_pytorch = "docs/source/quantization.rst"
        unique_identifier = "QAT API Example::"

        def _dummy_func(*args, **kwargs):
            return None

        input_fp32 = torch.randn(1, 1, 1, 1)
        global_inputs = {"training_loop": _dummy_func, "input_fp32": input_fp32}
        code = self._get_code(path_from_pytorch, unique_identifier)
        self._test_code(code, global_inputs)

    def test_quantization_doc_fx(self):
        path_from_pytorch = "docs/source/quantization.rst"
        unique_identifier = "FXPTQ API Example::"

        input_fp32 = SingleLayerLinearModel().get_example_inputs()
        global_inputs = {"UserModel": SingleLayerLinearModel, "input_fp32": input_fp32}

        code = self._get_code(path_from_pytorch, unique_identifier)
        self._test_code(code, global_inputs)

    def test_quantization_doc_custom(self):
        path_from_pytorch = "docs/source/quantization.rst"
        unique_identifier = "Custom API Example::"

        global_inputs = {"nnq": torch.ao.nn.quantized}

        code = self._get_code(path_from_pytorch, unique_identifier)
        self._test_code(code, global_inputs)
