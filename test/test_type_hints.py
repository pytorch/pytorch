from __future__ import print_function

import inspect
import os
import re
import subprocess
import sys
import tempfile
import unittest

import torch
from common_utils import TestCase, run_tests

try:
    import mypy  # noqa: F401

    HAVE_MYPY = True
except ImportError:
    HAVE_MYPY = False


def get_examples_from_docstring(docstr):
    """
    Extracts all runnable python code from the examples
    in docstrings; returns a list of lines.
    """
    # TODO: Figure out if there's a way to use doctest directly to
    # implement this
    example_file_lines = []
    # the detection is a bit hacky because there isn't a nice way of detecting
    # where multiline commands end. Thus we keep track of how far we got in beginning
    # and continue to add lines until we have a compileable Python statement.
    exampleline_re = re.compile(r"^\s+(?:>>>|\.\.\.) (.*)$")
    beginning = ""
    for l in docstr.split("\n"):
        if beginning:
            m = exampleline_re.match(l)
            if m:
                beginning += m.group(1)
            else:
                beginning += l
        else:
            m = exampleline_re.match(l)
            if m:
                beginning += m.group(1)
        if beginning:
            complete = True
            try:
                compile(beginning, "", "exec")
            except SyntaxError:
                complete = False
            if complete:
                # found one
                example_file_lines += beginning.split("\n")
                beginning = ""
            else:
                beginning += "\n"
    return ["    " + l for l in example_file_lines]


def get_artificial_file_prelude():
    return [
        "import torch",
        "import torch.nn.functional as F",
        "import math  # type: ignore",  # mypy complains about floats where SupportFloat is expected
        "import numpy  # type: ignore",
        "import io  # type: ignore",
        "import itertools  # type: ignore",
        "",
        # for requires_grad_ example
        # NB: We are parsing this file as Python 2, so we must use
        # Python 2 type annotation syntax
        "def preprocess(inp):",
        "    # type: (torch.Tensor) -> torch.Tensor",
        "    return inp",
    ]


def get_all_examples():
    """get_all_examples() -> str

    This function grabs (hopefully all) examples from the torch documentation
    strings and puts them in one nonsensical module returned as a string.
    """
    blacklist = {
        "_np",
        "refine_names",
        "rename",
        "names",
        "align_as",
        "align_to",
        "unflatten",
    }

    example_file_lines = get_artificial_file_prelude()

    for fname in dir(torch):
        fn = getattr(torch, fname)
        docstr = inspect.getdoc(fn)
        if docstr and fname not in blacklist:
            e = get_examples_from_docstring(docstr)
            if e:
                example_file_lines.append("\n\ndef example_torch_{}():".format(fname))
                example_file_lines += e

    for fname in dir(torch.Tensor):
        fn = getattr(torch.Tensor, fname)
        docstr = inspect.getdoc(fn)
        if docstr and fname not in blacklist:
            e = get_examples_from_docstring(docstr)
            if e:
                example_file_lines.append(
                    "\n\ndef example_torch_tensor_{}():".format(fname)
                )
                example_file_lines += e

    return "\n".join(example_file_lines)


# These are tutorial files that don't currently typecheck.
# Cut this down to produce more coverage!
BLACKLISTED_TUTORIALS = {
    "beginner_source": [
        "audio_preprocessing_tutorial.py",
        "aws_distributed_training_tutorial.py",
        "blitz/autograd_tutorial.py",
        "blitz/cifar10_tutorial.py",
        "blitz/data_parallel_tutorial.py",
        "blitz/neural_networks_tutorial.py",
        "blitz/tensor_tutorial.py",
        "chatbot_tutorial.py",
        "data_loading_tutorial.py",
        "dcgan_faces_tutorial.py",
        "deploy_seq2seq_hybrid_frontend_tutorial.py",
        "examples_autograd/tf_two_layer_net.py",
        "examples_autograd/two_layer_net_autograd.py",
        "examples_autograd/two_layer_net_custom_function.py",
        "examples_nn/two_layer_net_nn.py",
        "fgsm_tutorial.py",
        "former_torchies/autograd_tutorial_old.py",
        "former_torchies/nnft_tutorial.py",
        "former_torchies/parallelism_tutorial.py",
        "former_torchies/tensor_tutorial_old.py",
        "hybrid_frontend/learning_hybrid_frontend_through_example_tutorial.py",
        "Intro_to_TorchScript_tutorial.py",
        "nlp/advanced_tutorial.py",
        "nlp/deep_learning_tutorial.py",
        "nlp/pytorch_tutorial.py",
        "nlp/sequence_models_tutorial.py",
        "nlp/word_embeddings_tutorial.py",
        "nn_tutorial.py",
        "text_sentiment_ngrams_tutorial.py",
        "torchtext_translation_tutorial.py",
        "transfer_learning_tutorial.py",
        "transformer_tutorial.py",
    ],
    "intermediate_source": [
        "char_rnn_classification_tutorial.py",
        "char_rnn_generation_tutorial.py",
        "flask_rest_api_tutorial.py",
        "model_parallel_tutorial.py",
        "named_tensor_tutorial.py",
        "reinforcement_q_learning.py",
        "seq2seq_translation_tutorial.py",
        "spatial_transformer_tutorial.py",
    ],
    "advanced_source": [
        "dynamic_quantization_tutorial.py",
        "neural_style_tutorial.py",
        "numpy_extensions_tutorial.py",
        "static_quantization_tutorial.py",
        "super_resolution_with_onnxruntime.py",
    ],
}


class TestTypeHints(TestCase):
    def typecheck_artificial_file(self, smoketest_filename, text):
        fn = os.path.join(os.path.dirname(__file__), smoketest_filename)
        with open(fn, "w") as f:
            print(text, file=f)

        # OK, so here's the deal.  mypy treats installed packages
        # and local modules differently: if a package is installed,
        # mypy will refuse to use modules from that package for type
        # checking unless the module explicitly says that it supports
        # type checking. (Reference:
        # https://mypy.readthedocs.io/en/latest/running_mypy.html#missing-imports
        # )
        #
        # Now, PyTorch doesn't support typechecking, and we shouldn't
        # claim that it supports typechecking (it doesn't.) However, not
        # claiming we support typechecking is bad for this test, which
        # wants to use the partial information we get from the bits of
        # PyTorch which are typed to check if it typechecks.  And
        # although mypy will work directly if you are working in source,
        # some of our tests involve installing PyTorch and then running
        # its tests.
        #
        # The guidance we got from Michael Sullivan and Joshua Oreman,
        # and also independently developed by Thomas Viehmann,
        # is that we should create a fake directory and add symlinks for
        # the packages that should typecheck.  So that is what we do
        # here.
        #
        # If you want to run mypy by hand, and you run from PyTorch
        # root directory, it should work fine to skip this step (since
        # mypy will preferentially pick up the local files first).  The
        # temporary directory here is purely needed for CI.  For this
        # reason, we also still drop the generated file in the test
        # source folder, for ease of inspection when there are failures.
        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                os.symlink(
                    os.path.dirname(torch.__file__),
                    os.path.join(tmp_dir, "torch"),
                    target_is_directory=True,
                )
            except OSError:
                raise unittest.SkipTest("cannot symlink")
            try:
                subprocess.run(
                    [
                        sys.executable,
                        "-mmypy",
                        "--follow-imports",
                        "silent",
                        "--check-untyped-defs",
                        os.path.abspath(fn),
                    ],
                    cwd=tmp_dir,
                    check=True,
                )
            except subprocess.CalledProcessError:
                raise AssertionError(
                    "mypy failed.  Look below this error for mypy's output."
                )

    @unittest.skipIf(sys.version_info[0] == 2, "no type hints for Python 2")
    @unittest.skipIf(not HAVE_MYPY, "need mypy")
    def test_doc_examples(self):
        """
        Run documentation examples through mypy.
        """
        self.typecheck_artificial_file(
            "generated_type_hints_smoketest.py", get_all_examples()
        )

    @unittest.skipIf(sys.version_info[0] == 2, "no type hints for Python 2")
    @unittest.skipIf(not HAVE_MYPY, "need mypy")
    def test_tutorials(self):
        """
        Run code from the tutorials through mypy.
        """
        prelude = "\n".join(get_artificial_file_prelude()) + "\n"
        tutorials_directory = "third_party/tutorials"

        for tutorial_subdirectory in [
            "beginner_source",
            "intermediate_source",
            "advanced_source",
        ]:
            blacklisted_tutorial_paths = set(
                os.path.join(tutorials_directory, tutorial_subdirectory, subpath)
                for subpath in BLACKLISTED_TUTORIALS.get(tutorial_subdirectory, [])
            )

            def should_include_file(path):
                return path.endswith(".py") and path not in blacklisted_tutorial_paths

            path = os.path.join(tutorials_directory, tutorial_subdirectory)
            for root, dirs, files in os.walk(path):
                for fname in files:
                    filepath = os.path.join(root, fname)
                    if should_include_file(filepath):
                        with open(filepath, "r") as contents:
                            source = prelude + contents.read()
                        self.typecheck_artificial_file(
                            "generated_tutorial_type_hints_smoketest.py", source
                        )


if __name__ == "__main__":
    run_tests()
