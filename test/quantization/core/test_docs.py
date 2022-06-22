# Owner(s): ["oncall: quantization"]

import torch
from torch.testing._internal.common_quantization import (
    QuantizationTestCase,
    SingleLayerLinearModel,
)
from os.path import exists
from os import getcwd


class TestQuantizationDocs(QuantizationTestCase):
    r"""
    The tests in this section import code from the quantization docs and check that
    they actually run. In cases where objects are undefined in the code snippet, they
    must be provided in the test.
    """

    def _get_code(self, path_from_pytorch, first_line, last_line, offset=2, strict=True):
        r"""
        This function reads in the code from the docs, note that first and last
        line refer to the line number (first line of doc is 1), the offset due to
        python file reading is handled within this function. Most code snippets
        have a 2 space indentation, for other indentation levels, change offset,
        strict=True is to check that the line before the first line and the line after
        the last line are `newlines`, this is to ensure that the addition of a new line
        in the docs does not shift the code chunk out of the selection window.
        """
        def get_correct_path(path_from_pytorch):
            r"""
            Current working directory when CI is running test seems to vary, this function
            looks for the pytorch directory and if it finds it looks for the path to the
            file and if the file exists returns that path, otherwise keeps looking. Will
            only work if cwd contains pytorch or is somewhere in the pytorch repo.
            """

            # check if cwd contains pytorch
            if exists('./pytorch' + path_from_pytorch):
                return './pytorch' + path_from_pytorch

            # check if pytorch is cwd or a parent of cwd
            cur_dir_path = getcwd()
            folders = cur_dir_path.split('/')[::-1]
            path_prefix = './'
            for folder in folders:
                if folder == 'pytorch' and exists(path_prefix + path_from_pytorch):
                    return(path_prefix + path_from_pytorch)
                path_prefix = '.' + path_prefix
            # if not found
            return None

        path_to_file = get_correct_path(path_from_pytorch)
        if path_to_file:
            file = open(path_to_file)
            content = file.readlines()
            if strict:
                assert content[first_line - 2] == "\n" and content[last_line] == "\n", (
                    "The line before and after the code chunk should be a newline."
                    "If new material was added to {}, please update this test with"
                    "the new code chunk line numbers, previously the lines were "
                    "{} to {}".format(path_to_file, first_line, last_line)
                )

            code_to_test = ""
            for i in range(first_line - 2, last_line):
                code_to_test += content[i][offset:]
            file.close()
        else:
            code_to_test = None
        return code_to_test


    def _test_code(self, code, global_inputs=None):
        r"""
        This function runs `code` using any vars in `global_inputs`
        """
        if code is not None:  # the path doesn't work for some CI runs
            expr = compile(code, "test", "exec")
            exec(expr, global_inputs)

    def test_quantization_doc_ptdq(self):
        path_from_pytorch = "docs/source/quantization.rst"
        first_line = 74
        last_line = 96
        code = self._get_code(path_from_pytorch, first_line, last_line)
        self._test_code(code)

    def test_quantization_doc_ptsq(self):
        path_from_pytorch = "docs/source/quantization.rst"
        first_line = 129
        last_line = 187
        code = self._get_code(path_from_pytorch, first_line, last_line)
        self._test_code(code)

    def test_quantization_doc_qat(self):
        path_from_pytorch = "docs/source/quantization.rst"
        first_line = 227
        last_line = 283

        def _dummy_func(*args, **kwargs):
            return None

        input_fp32 = torch.randn(1, 1, 1, 1)
        global_inputs = {"training_loop": _dummy_func, "input_fp32": input_fp32}

        code = self._get_code(path_from_pytorch, first_line, last_line)
        self._test_code(code, global_inputs)

    def test_quantization_doc_fx(self):
        path_from_pytorch = "docs/source/quantization.rst"
        first_line = 330
        last_line = 383

        def _dummy_func(*args, **kwargs):
            return None

        input_fp32 = SingleLayerLinearModel().get_example_inputs()
        global_inputs = {"UserModel": SingleLayerLinearModel, "input_fp32": input_fp32}

        code = self._get_code(path_from_pytorch, first_line, last_line)
        self._test_code(code, global_inputs)
