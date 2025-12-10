from sympy.liealgebras.dynkin_diagram import DynkinDiagram

def test_DynkinDiagram():
    c = DynkinDiagram("A3")
    diag = "0---0---0\n1   2   3"
    assert c == diag
    ct = DynkinDiagram(["B", 3])
    diag2 = "0---0=>=0\n1   2   3"
    assert ct == diag2
