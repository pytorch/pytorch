import os
import tempfile

import sympy
from sympy.testing.pytest import raises
from sympy.parsing.latex.lark import LarkLaTeXParser, TransformToSymPyExpr, parse_latex_lark
from sympy.external import import_module

lark = import_module("lark")

# disable tests if lark is not present
disabled = lark is None

grammar_file = os.path.join(os.path.dirname(__file__), "../latex/lark/grammar/latex.lark")

modification1 = """
%override DIV_SYMBOL: DIV
%override MUL_SYMBOL: MUL | CMD_TIMES
"""

modification2 = r"""
%override number: /\d+(,\d*)?/
"""

def init_custom_parser(modification, transformer=None):
    with open(grammar_file, encoding="utf-8") as f:
        latex_grammar = f.read()

    latex_grammar += modification

    with tempfile.NamedTemporaryFile() as f:
        f.write(bytes(latex_grammar, encoding="utf8"))

        parser = LarkLaTeXParser(grammar_file=f.name, transformer=transformer)

    return parser

def test_custom1():
    # Removes the parser's ability to understand \cdot and \div.

    parser = init_custom_parser(modification1)

    with raises(lark.exceptions.UnexpectedCharacters):
        parser.doparse(r"a \cdot b")
        parser.doparse(r"x \div y")

class CustomTransformer(TransformToSymPyExpr):
    def number(self, tokens):
        if "," in tokens[0]:
            # The Float constructor expects a dot as the decimal separator
            return sympy.core.numbers.Float(tokens[0].replace(",", "."))
        else:
            return sympy.core.numbers.Integer(tokens[0])

def test_custom2():
    # Makes the parser parse commas as the decimal separator instead of dots

    parser = init_custom_parser(modification2, CustomTransformer)

    with raises(lark.exceptions.UnexpectedCharacters):
        # Asserting that the default parser cannot parse numbers which have commas as
        # the decimal separator
        parse_latex_lark("100,1")
        parse_latex_lark("0,009")

    parser.doparse("100,1")
    parser.doparse("0,009")
    parser.doparse("2,71828")
    parser.doparse("3,14159")
