import os
from textwrap import dedent
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.mathml import apply_xsl



lxml = import_module('lxml')

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_xxe.py"))


def test_xxe():
    assert os.path.isfile(path)
    if not lxml:
        skip("lxml not installed.")

    mml = dedent(
        rf"""
        <!--?xml version="1.0" ?-->
        <!DOCTYPE replace [<!ENTITY ent SYSTEM "file://{path}"> ]>
        <userInfo>
        <firstName>John</firstName>
        <lastName>&ent;</lastName>
        </userInfo>
        """
    )
    xsl = 'mathml/data/simple_mmlctop.xsl'

    res = apply_xsl(mml, xsl)
    assert res == \
        '<?xml version="1.0"?>\n<userInfo>\n<firstName>John</firstName>\n<lastName/>\n</userInfo>\n'
