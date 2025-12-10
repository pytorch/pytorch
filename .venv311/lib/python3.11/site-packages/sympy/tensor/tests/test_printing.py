from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead
from sympy import I

def test_printing_TensMul():
    R3 = TensorIndexType('R3', dim=3)
    p, q = tensor_indices("p q", R3)
    K = TensorHead("K", [R3])

    assert repr(2*K(p)) == "2*K(p)"
    assert repr(-K(p)) == "-K(p)"
    assert repr(-2*K(p)*K(q)) == "-2*K(p)*K(q)"
    assert repr(-I*K(p)) == "-I*K(p)"
    assert repr(I*K(p)) == "I*K(p)"
