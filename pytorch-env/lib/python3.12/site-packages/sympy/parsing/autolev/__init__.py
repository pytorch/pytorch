from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on

@doctest_depends_on(modules=('antlr4',))
def parse_autolev(autolev_code, include_numeric=False):
    """Parses Autolev code (version 4.1) to SymPy code.

    Parameters
    =========
    autolev_code : Can be an str or any object with a readlines() method (such as a file handle or StringIO).
    include_numeric : boolean, optional
        If True NumPy, PyDy, or other numeric code is included for numeric evaluation lines in the Autolev code.

    Returns
    =======
    sympy_code : str
        Equivalent SymPy and/or numpy/pydy code as the input code.


    Example (Double Pendulum)
    =========================
    >>> my_al_text = ("MOTIONVARIABLES' Q{2}', U{2}'",
    ... "CONSTANTS L,M,G",
    ... "NEWTONIAN N",
    ... "FRAMES A,B",
    ... "SIMPROT(N, A, 3, Q1)",
    ... "SIMPROT(N, B, 3, Q2)",
    ... "W_A_N>=U1*N3>",
    ... "W_B_N>=U2*N3>",
    ... "POINT O",
    ... "PARTICLES P,R",
    ... "P_O_P> = L*A1>",
    ... "P_P_R> = L*B1>",
    ... "V_O_N> = 0>",
    ... "V2PTS(N, A, O, P)",
    ... "V2PTS(N, B, P, R)",
    ... "MASS P=M, R=M",
    ... "Q1' = U1",
    ... "Q2' = U2",
    ... "GRAVITY(G*N1>)",
    ... "ZERO = FR() + FRSTAR()",
    ... "KANE()",
    ... "INPUT M=1,G=9.81,L=1",
    ... "INPUT Q1=.1,Q2=.2,U1=0,U2=0",
    ... "INPUT TFINAL=10, INTEGSTP=.01",
    ... "CODE DYNAMICS() some_filename.c")
    >>> my_al_text = '\\n'.join(my_al_text)
    >>> from sympy.parsing.autolev import parse_autolev
    >>> print(parse_autolev(my_al_text, include_numeric=True))
    import sympy.physics.mechanics as _me
    import sympy as _sm
    import math as m
    import numpy as _np
    <BLANKLINE>
    q1, q2, u1, u2 = _me.dynamicsymbols('q1 q2 u1 u2')
    q1_d, q2_d, u1_d, u2_d = _me.dynamicsymbols('q1_ q2_ u1_ u2_', 1)
    l, m, g = _sm.symbols('l m g', real=True)
    frame_n = _me.ReferenceFrame('n')
    frame_a = _me.ReferenceFrame('a')
    frame_b = _me.ReferenceFrame('b')
    frame_a.orient(frame_n, 'Axis', [q1, frame_n.z])
    frame_b.orient(frame_n, 'Axis', [q2, frame_n.z])
    frame_a.set_ang_vel(frame_n, u1*frame_n.z)
    frame_b.set_ang_vel(frame_n, u2*frame_n.z)
    point_o = _me.Point('o')
    particle_p = _me.Particle('p', _me.Point('p_pt'), _sm.Symbol('m'))
    particle_r = _me.Particle('r', _me.Point('r_pt'), _sm.Symbol('m'))
    particle_p.point.set_pos(point_o, l*frame_a.x)
    particle_r.point.set_pos(particle_p.point, l*frame_b.x)
    point_o.set_vel(frame_n, 0)
    particle_p.point.v2pt_theory(point_o,frame_n,frame_a)
    particle_r.point.v2pt_theory(particle_p.point,frame_n,frame_b)
    particle_p.mass = m
    particle_r.mass = m
    force_p = particle_p.mass*(g*frame_n.x)
    force_r = particle_r.mass*(g*frame_n.x)
    kd_eqs = [q1_d - u1, q2_d - u2]
    forceList = [(particle_p.point,particle_p.mass*(g*frame_n.x)), (particle_r.point,particle_r.mass*(g*frame_n.x))]
    kane = _me.KanesMethod(frame_n, q_ind=[q1,q2], u_ind=[u1, u2], kd_eqs = kd_eqs)
    fr, frstar = kane.kanes_equations([particle_p, particle_r], forceList)
    zero = fr+frstar
    from pydy.system import System
    sys = System(kane, constants = {l:1, m:1, g:9.81},
    specifieds={},
    initial_conditions={q1:.1, q2:.2, u1:0, u2:0},
    times = _np.linspace(0.0, 10, 10/.01))
    <BLANKLINE>
    y=sys.integrate()
    <BLANKLINE>
    """

    _autolev = import_module(
        'sympy.parsing.autolev._parse_autolev_antlr',
        import_kwargs={'fromlist': ['X']})

    if _autolev is not None:
        return _autolev.parse_autolev(autolev_code, include_numeric)
