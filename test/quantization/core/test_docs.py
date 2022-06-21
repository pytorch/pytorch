# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    SingleLayerLinearModel,
)


class TestQuantizationDocs(QuantizationTestCase):
    r"""
    The tests in this section import code from the quantization docs and check that
    they actually run. In cases where objects are undefined in the code snippet, they
    must be provided in the test.
    """

    def _get_code(self, filename, first_line, last_line, offset=2, strict=True):
        r"""
        This function reads in the code from the docs, note that first and last
        line refer to the line number (first line of doc is 1), the offset due to
        python file reading is handled within this function. Most code snippets
        have a 2 space indentation, for other indentation levels, change offset,
        strict=True is to check that the line before the first line and the line after
        the last line are `newlines`, this is to ensure that the addition of a new line
        in the docs does not shift the code chunk out of the selection window.
        """
        file = open(filename)
        content = file.readlines()
        if strict:
            assert content[first_line - 2] == "\n" and content[last_line] == "\n", (
                "The line before and after the code chunk should be a newline."
                " If new material was added to {}, please update this test with"
                "the new code chunk line numbers, previously the lines were "
                "{} to {}".format(filename, first_line, last_line)
            )

        code_to_test = ""
        for i in range(first_line - 2, last_line):
            code_to_test += content[i][offset:]
        file.close()
        return code_to_test

    def _test_code(self, code, global_inputs=None):
        r"""
        This function runs `code` using any vars in `global_inputs`
        """
        expr = compile(code, "test", "exec")
        exec(expr, global_inputs)
        # is there a better way to check for no error than just running it?

    def test_quantization_doc_ptdq(self):
        filename = "./docs/source/quantization.rst"
        first_line = 74
        last_line = 96
        code = self._get_code(filename, first_line, last_line)
        self._test_code(code)

    def test_quantization_doc_ptsq(self):
        filename = "./docs/source/quantization.rst"
        first_line = 129
        last_line = 187
        code = self._get_code(filename, first_line, last_line)
        self._test_code(code)

    def test_quantization_doc_qat(self):
        filename = "./docs/source/quantization.rst"
        first_line = 227
        last_line = 283

        def _dummy_func(*args, **kwargs):
            return None

        input_fp32 = torch.randn(1, 1, 1, 1)
        global_inputs = {"training_loop": _dummy_func, "input_fp32": input_fp32}

        code = self._get_code(filename, first_line, last_line)
        self._test_code(code, global_inputs)

    def test_quantization_doc_fx(self):
        filename = "./docs/source/quantization.rst"
        first_line = 330
        last_line = 383

        def _dummy_func(*args, **kwargs):
            return None

        input_fp32 = SingleLayerLinearModel().get_example_inputs()
        global_inputs = {"UserModel": SingleLayerLinearModel, "input_fp32": input_fp32}

        code = self._get_code(filename, first_line, last_line)
        self._test_code(code, global_inputs)
