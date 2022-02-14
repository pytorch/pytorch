# Owner(s): ["module: tests"]

from torch.testing._internal.common_utils import TestCase, run_tests
from torch import _torch_docs
import doctest
import torch
import math
import numpy
import itertools
import ast
import sys
import re


# ast.Str deprecated in 3.8
if sys.version_info >= (3, 8):
    AstStrType = ast.Constant
else:
    AstStrType = ast.Str


def joinattr(a):
    if type(a) == ast.Name:
        return a.id
    elif type(a) == ast.Attribute:
        return joinattr(a.value) + '.' + a.attr


def find_docstring_node(node):
    docstring_node = None

    if getattr(getattr(node, 'func', None), 'id', None) == 'add_docstr':
        documented_function = joinattr(node.args[0])
        if type(node.args[1]) == AstStrType:
            # Handles hardcoded docstring case
            docstring_node = node.args[1]
        elif (type(node.args[1]) == ast.Call and
              type(node.args[1].func) == ast.Attribute and
              type(node.args[1].func.value) == AstStrType):
            # This handles the "docstring".format(**kwargs) case
            docstring_node = node.args[1].func.value
        elif (type(node.args[1]) == ast.BinOp and
              type(node.args[1].op) == ast.Add):
            # Handles compound docstrings with two parts
            # Here I assume that the second part contains examples
            if type(node.args[1].right) == AstStrType:
                # Both parts are constants
                docstring_node = node.args[1].right
            else:
                # Second part is a formatted string
                docstring_node = node.args[1].right.func.value
        else:
            raise Exception("Edge case " + documented_function + ":" + ast.dump(node))

    return docstring_node


class DoctTestVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.parser = doctest.DocTestParser()
        # Modify the default parser to allow blanklines in the expected output.
        # This is useful because Tensors with many dim's tend to print
        # with blanklines
        self.parser._EXAMPLE_RE = re.compile(r'''
        # Source consists of a PS1 line followed by zero or more PS2 lines.
        (?P<source>
            (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line
            (?:\n           [ ]*  \.\.\. .*)*)  # PS2 lines
        \n?
        # Want consists of any non-blank lines that do not start with PS1.
        (?P<want> (?:(?!\.\.[ ])        # Not the start of a new section
                     (?![ ]*>>>)        # Not a line starting with PS1
                     # But any other line that startswith some whitespace, including empty
                     # lines to allow for tensors with many axes as these tend to include
                     # empty lines when printed.
                     ([ ]+.+$\n?|\n)?
                  )*)
        ''', re.MULTILINE | re.VERBOSE)
        self.tests = {}

    def visit_Call(self, node):
        docstring_node = find_docstring_node(node)
        if docstring_node is not None:
            documented_function = joinattr(node.args[0])
            globs = {"torch": torch, "math": math, "F": torch.nn.functional,
                     "numpy": numpy, "itertools": itertools}
            test = self.parser.get_doctest(
                docstring_node.s,
                globs=globs,
                name=documented_function,
                filename=_torch_docs.__file__,
                lineno=docstring_node.lineno
            )
            if test.examples:
                test.examples = [
                    doctest.Example(
                        source="torch.manual_seed(0)\n",
                        want="<torch._C.Generator object at 0x...>"
                    )
                ] + test.examples

                self.tests[documented_function] = test


class TestDocs(TestCase):
    pass


def make_test(doctest_runner: doctest.DocTestRunner, doc_test: doctest.DocTest):
    def test_docstring_example(self):
        result: doctest.TestResults = doctest_runner.run(doc_test)
        self.assertEqual(result.failed, 0)

    return test_docstring_example


# Generates tests
# Note: test generation must be done at file scope, not within main, or
# pytest will fail.
txt = open(_torch_docs.__file__).read()
tree = ast.parse(txt)
doctest_visitor = DoctTestVisitor()
doctest_visitor.visit(tree)
runner = doctest.DocTestRunner(
    optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
)
cuda_available = torch.cuda.is_available()
for name, test in doctest_visitor.tests.items():
    want = [ex.want for ex in test.examples]
    needs_cuda = any("device='cuda" in w for w in want)
    if needs_cuda and not cuda_available:
        continue
    setattr(TestDocs, f"test_{name.replace(',', '_')}", make_test(runner, test))

if __name__ == '__main__':
    run_tests()
