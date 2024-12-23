from sympy.physics.quantum.circuitplot import labeller, render_label, Mz, CreateOneQubitGate,\
     CreateCGate
from sympy.physics.quantum.gate import CNOT, H, SWAP, CGate, S, T
from sympy.external import import_module
from sympy.testing.pytest import skip

mpl = import_module('matplotlib')

def test_render_label():
    assert render_label('q0') == r'$\left|q0\right\rangle$'
    assert render_label('q0', {'q0': '0'}) == r'$\left|q0\right\rangle=\left|0\right\rangle$'

def test_Mz():
    assert str(Mz(0)) == 'Mz(0)'

def test_create1():
    Qgate = CreateOneQubitGate('Q')
    assert str(Qgate(0)) == 'Q(0)'

def test_createc():
    Qgate = CreateCGate('Q')
    assert str(Qgate([1],0)) == 'C((1),Q(0))'

def test_labeller():
    """Test the labeller utility"""
    assert labeller(2) == ['q_1', 'q_0']
    assert labeller(3,'j') == ['j_2', 'j_1', 'j_0']

def test_cnot():
    """Test a simple cnot circuit. Right now this only makes sure the code doesn't
    raise an exception, and some simple properties
    """
    if not mpl:
        skip("matplotlib not installed")
    else:
        from sympy.physics.quantum.circuitplot import CircuitPlot

    c = CircuitPlot(CNOT(1,0),2,labels=labeller(2))
    assert c.ngates == 2
    assert c.nqubits == 2
    assert c.labels == ['q_1', 'q_0']

    c = CircuitPlot(CNOT(1,0),2)
    assert c.ngates == 2
    assert c.nqubits == 2
    assert c.labels == []

def test_ex1():
    if not mpl:
        skip("matplotlib not installed")
    else:
        from sympy.physics.quantum.circuitplot import CircuitPlot

    c = CircuitPlot(CNOT(1,0)*H(1),2,labels=labeller(2))
    assert c.ngates == 2
    assert c.nqubits == 2
    assert c.labels == ['q_1', 'q_0']

def test_ex4():
    if not mpl:
        skip("matplotlib not installed")
    else:
        from sympy.physics.quantum.circuitplot import CircuitPlot

    c = CircuitPlot(SWAP(0,2)*H(0)* CGate((0,),S(1)) *H(1)*CGate((0,),T(2))\
                    *CGate((1,),S(2))*H(2),3,labels=labeller(3,'j'))
    assert c.ngates == 7
    assert c.nqubits == 3
    assert c.labels == ['j_2', 'j_1', 'j_0']
