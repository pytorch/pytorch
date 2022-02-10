from torch.testing._internal.common_utils import TestCase, run_tests
import doctest
import torch
import torch.nn.functional as F
import math
import numpy
import itertools
import ast
import re


ALWAYS_PASS = "__always_pass__"
_TORCH_DOCS = "../torch/_torch_docs.py"


def joinattr(a):
    if type(a) == ast.Name:
        return a.id
    elif type(a) == ast.Attribute:
        return joinattr(a.value) + '.' + a.attr


def find_docstring_node(node):
    docstring_node = None

    if getattr(getattr(node, 'func', None), 'id', None) == 'add_docstr':
        documented_function = joinattr(node.args[0])
        if type(node.args[1]) == ast.Constant:
            # Handles hardcoded docstring case
            docstring_node = node.args[1]
        elif (type(node.args[1]) == ast.Call and
              type(node.args[1].func) == ast.Attribute and
              type(node.args[1].func.value) == ast.Constant):
            # This handles the "docstring".format(**kwargs) case
            docstring_node = node.args[1].func.value
        elif (type(node.args[1]) == ast.BinOp and
              type(node.args[1].op) == ast.Add):
            # Handles compound docstrings with two parts
            # Here I assume that the second part contains examples
            if type(node.args[1].right) == ast.Constant:
                # Both parts are constants
                docstring_node = node.args[1].right
            else:
                # Second part is a formatted string
                docstring_node = node.args[1].right.func.value
        else:
            raise Exception("Edge case " + documented_function + ":" + ast.dump(node))

    return docstring_node


class AlwaysPassManualSeed(doctest.OutputChecker):
    def check_output(self, want, got, optionflags):
        if want == F"{ALWAYS_PASS}\n":
            return True
        else:
            got = re.sub(r"\n\n", "\n", got)
            return super().check_output(want, got, optionflags)


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
        (?P<want> (?:(?!\.\.[ ])  # Not the start of a new section
                     (?![ ]*>>>)  # Not a line starting with PS1 (including blanklines)
                     .+$\n?       # But any other line
                  )*)
        ''', re.MULTILINE | re.VERBOSE)
        self.runner = doctest.DocTestRunner(
            checker=AlwaysPassManualSeed(),
            optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS
        )
        self.num_attempted = 0
        self.num_failed = 0

    def visit_Call(self, node):
        docstring_node = find_docstring_node(node)
        if docstring_node is not None:
            documented_function = joinattr(node.args[0])
            globs = {'torch': torch, 'math': math, 'F': torch.nn.functional,
                'numpy': numpy, 'itertools': itertools}
            test = self.parser.get_doctest(
                docstring_node.s,
                globs=globs,
                name=documented_function,
                filename=_TORCH_DOCS,
                lineno=docstring_node.lineno
            )
            if test.examples:
                test.examples = [
                    doctest.Example(source="torch.manual_seed(0);\n", want=ALWAYS_PASS)
                ] + test.examples
            results = self.runner.run(test)
            self.num_attempted += results.attempted
            self.num_failed += results.failed


class TestDocs(TestCase):

    def test_docs(self):
        torch_docs_fn = _TORCH_DOCS
        txt = open(torch_docs_fn).read()
        tree = ast.parse(txt)
        doctest_visitor = DoctTestVisitor()
        doctest_visitor.visit(tree)
        print("Total attempted: {}, total failed: {}".format(
            doctest_visitor.num_attempted,
            doctest_visitor.num_failed
        ))
        self.assertEqual(doctest_visitor.num_failed, 0)


if __name__ == '__main__':
    run_tests()
