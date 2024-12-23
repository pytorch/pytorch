import os
import tempfile
from sympy.core.symbol import (Symbol, symbols)
from sympy.codegen.ast import (
    Assignment, Print, Declaration, FunctionDefinition, Return, real,
    FunctionCall, Variable, Element, integer
)
from sympy.codegen.fnodes import (
    allocatable, ArrayConstructor, isign, dsign, cmplx, kind, literal_dp,
    Program, Module, use, Subroutine, dimension, assumed_extent, ImpliedDoLoop,
    intent_out, size, Do, SubroutineCall, sum_, array, bind_C
)
from sympy.codegen.futils import render_as_module
from sympy.core.expr import unchanged
from sympy.external import import_module
from sympy.printing.codeprinter import fcode
from sympy.utilities._compilation import has_fortran, compile_run_strings, compile_link_import_strings
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip, XFAIL

cython = import_module('cython')
np = import_module('numpy')


def test_size():
    x = Symbol('x', real=True)
    sx = size(x)
    assert fcode(sx, source_format='free') == 'size(x)'


@may_xfail
def test_size_assumed_shape():
    if not has_fortran():
        skip("No fortran compiler found.")
    a = Symbol('a', real=True)
    body = [Return((sum_(a**2)/size(a))**.5)]
    arr = array(a, dim=[':'], intent='in')
    fd = FunctionDefinition(real, 'rms', [arr], body)
    render_as_module([fd], 'mod_rms')

    (stdout, stderr), info = compile_run_strings([
        ('rms.f90', render_as_module([fd], 'mod_rms')),
        ('main.f90', (
            'program myprog\n'
            'use mod_rms, only: rms\n'
            'real*8, dimension(4), parameter :: x = [4, 2, 2, 2]\n'
            'print *, dsqrt(7d0) - rms(x)\n'
            'end program\n'
        ))
    ], clean=True)
    assert '0.00000' in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK


@XFAIL  # https://github.com/sympy/sympy/issues/20265
@may_xfail
def test_ImpliedDoLoop():
    if not has_fortran():
        skip("No fortran compiler found.")

    a, i = symbols('a i', integer=True)
    idl = ImpliedDoLoop(i**3, i, -3, 3, 2)
    ac = ArrayConstructor([-28, idl, 28])
    a = array(a, dim=[':'], attrs=[allocatable])
    prog = Program('idlprog', [
        a.as_Declaration(),
        Assignment(a, ac),
        Print([a])
    ])
    fsrc = fcode(prog, standard=2003, source_format='free')
    (stdout, stderr), info = compile_run_strings([('main.f90', fsrc)], clean=True)
    for numstr in '-28 -27 -1 1 27 28'.split():
        assert numstr in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK


@may_xfail
def test_Program():
    x = Symbol('x', real=True)
    vx = Variable.deduced(x, 42)
    decl = Declaration(vx)
    prnt = Print([x, x+1])
    prog = Program('foo', [decl, prnt])
    if not has_fortran():
        skip("No fortran compiler found.")

    (stdout, stderr), info = compile_run_strings([('main.f90', fcode(prog, standard=90))], clean=True)
    assert '42' in stdout
    assert '43' in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK


@may_xfail
def test_Module():
    x = Symbol('x', real=True)
    v_x = Variable.deduced(x)
    sq = FunctionDefinition(real, 'sqr', [v_x], [Return(x**2)])
    mod_sq = Module('mod_sq', [], [sq])
    sq_call = FunctionCall('sqr', [42.])
    prg_sq = Program('foobar', [
        use('mod_sq', only=['sqr']),
        Print(['"Square of 42 = "', sq_call])
    ])
    if not has_fortran():
        skip("No fortran compiler found.")
    (stdout, stderr), info = compile_run_strings([
        ('mod_sq.f90', fcode(mod_sq, standard=90)),
        ('main.f90', fcode(prg_sq, standard=90))
    ], clean=True)
    assert '42' in stdout
    assert str(42**2) in stdout
    assert stderr == ''


@XFAIL  # https://github.com/sympy/sympy/issues/20265
@may_xfail
def test_Subroutine():
    # Code to generate the subroutine in the example from
    # http://www.fortran90.org/src/best-practices.html#arrays
    r = Symbol('r', real=True)
    i = Symbol('i', integer=True)
    v_r = Variable.deduced(r, attrs=(dimension(assumed_extent), intent_out))
    v_i = Variable.deduced(i)
    v_n = Variable('n', integer)
    do_loop = Do([
        Assignment(Element(r, [i]), literal_dp(1)/i**2)
    ], i, 1, v_n)
    sub = Subroutine("f", [v_r], [
        Declaration(v_n),
        Declaration(v_i),
        Assignment(v_n, size(r)),
        do_loop
    ])
    x = Symbol('x', real=True)
    v_x3 = Variable.deduced(x, attrs=[dimension(3)])
    mod = Module('mymod', definitions=[sub])
    prog = Program('foo', [
        use(mod, only=[sub]),
        Declaration(v_x3),
        SubroutineCall(sub, [v_x3]),
        Print([sum_(v_x3), v_x3])
    ])

    if not has_fortran():
        skip("No fortran compiler found.")

    (stdout, stderr), info = compile_run_strings([
        ('a.f90', fcode(mod, standard=90)),
        ('b.f90', fcode(prog, standard=90))
    ], clean=True)
    ref = [1.0/i**2 for i in range(1, 4)]
    assert str(sum(ref))[:-3] in stdout
    for _ in ref:
        assert str(_)[:-3] in stdout
    assert stderr == ''


def test_isign():
    x = Symbol('x', integer=True)
    assert unchanged(isign, 1, x)
    assert fcode(isign(1, x), standard=95, source_format='free') == 'isign(1, x)'


def test_dsign():
    x = Symbol('x')
    assert unchanged(dsign, 1, x)
    assert fcode(dsign(literal_dp(1), x), standard=95, source_format='free') == 'dsign(1d0, x)'


def test_cmplx():
    x = Symbol('x')
    assert unchanged(cmplx, 1, x)


def test_kind():
    x = Symbol('x')
    assert unchanged(kind, x)


def test_literal_dp():
    assert fcode(literal_dp(0), source_format='free') == '0d0'


@may_xfail
def test_bind_C():
    if not has_fortran():
        skip("No fortran compiler found.")
    if not cython:
        skip("Cython not found.")
    if not np:
        skip("NumPy not found.")

    a = Symbol('a', real=True)
    s = Symbol('s', integer=True)
    body = [Return((sum_(a**2)/s)**.5)]
    arr = array(a, dim=[s], intent='in')
    fd = FunctionDefinition(real, 'rms', [arr, s], body, attrs=[bind_C('rms')])
    f_mod = render_as_module([fd], 'mod_rms')

    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([
            ('rms.f90', f_mod),
            ('_rms.pyx', (
                "#cython: language_level={}\n".format("3") +
                "cdef extern double rms(double*, int*)\n"
                "def py_rms(double[::1] x):\n"
                "    cdef int s = x.size\n"
                "    return rms(&x[0], &s)\n"))
        ], build_dir=folder)
        assert abs(mod.py_rms(np.array([2., 4., 2., 2.])) - 7**0.5) < 1e-14
