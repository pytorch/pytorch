from sympy.physics.optics.polarization import (jones_vector, stokes_vector,
    jones_2_stokes, linear_polarizer, phase_retarder, half_wave_retarder,
    quarter_wave_retarder, transmissive_filter, reflective_filter,
    mueller_matrix, polarizing_beam_splitter)
from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.matrices.dense import Matrix


def test_polarization():
    assert jones_vector(0, 0) == Matrix([1, 0])
    assert jones_vector(pi/2, 0) == Matrix([0, 1])
    #################################################################
    assert stokes_vector(0, 0) == Matrix([1, 1, 0, 0])
    assert stokes_vector(pi/2, 0) == Matrix([1, -1, 0, 0])
    #################################################################
    H = jones_vector(0, 0)
    V = jones_vector(pi/2, 0)
    D = jones_vector(pi/4, 0)
    A = jones_vector(-pi/4, 0)
    R = jones_vector(0, pi/4)
    L = jones_vector(0, -pi/4)

    res = [Matrix([1, 1, 0, 0]),
           Matrix([1, -1, 0, 0]),
           Matrix([1, 0, 1, 0]),
           Matrix([1, 0, -1, 0]),
           Matrix([1, 0, 0, 1]),
           Matrix([1, 0, 0, -1])]

    assert [jones_2_stokes(e) for e in [H, V, D, A, R, L]] == res
    #################################################################
    assert linear_polarizer(0) == Matrix([[1, 0], [0, 0]])
    #################################################################
    delta = symbols("delta", real=True)
    res = Matrix([[exp(-I*delta/2), 0], [0, exp(I*delta/2)]])
    assert phase_retarder(0, delta) == res
    #################################################################
    assert half_wave_retarder(0) == Matrix([[-I, 0], [0, I]])
    #################################################################
    res = Matrix([[exp(-I*pi/4), 0], [0, I*exp(-I*pi/4)]])
    assert quarter_wave_retarder(0) == res
    #################################################################
    assert transmissive_filter(1) == Matrix([[1, 0], [0, 1]])
    #################################################################
    assert reflective_filter(1) == Matrix([[1, 0], [0, -1]])

    res = Matrix([[S(1)/2, S(1)/2, 0, 0],
                  [S(1)/2, S(1)/2, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 0]])
    assert mueller_matrix(linear_polarizer(0)) == res
    #################################################################
    res = Matrix([[1, 0, 0, 0], [0, 0, 0, -I], [0, 0, 1, 0], [0, -I, 0, 0]])
    assert polarizing_beam_splitter() == res
