import os

from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.parsing.autolev import parse_autolev

antlr4 = import_module("antlr4")

if not antlr4:
    disabled = True

FILE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(os.path.realpath(__file__))))


def _test_examples(in_filename, out_filename, test_name=""):

    in_file_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                                in_filename)
    correct_file_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                                     out_filename)
    with open(in_file_path) as f:
        generated_code = parse_autolev(f, include_numeric=True)

    with open(correct_file_path) as f:
        for idx, line1 in enumerate(f):
            if line1.startswith("#"):
                break
            try:
                line2 = generated_code.split('\n')[idx]
                assert line1.rstrip() == line2.rstrip()
            except Exception:
                msg = 'mismatch in ' + test_name + ' in line no: {0}'
                raise AssertionError(msg.format(idx+1))


def test_rule_tests():

    l = ["ruletest1", "ruletest2", "ruletest3", "ruletest4", "ruletest5",
         "ruletest6", "ruletest7", "ruletest8", "ruletest9", "ruletest10",
         "ruletest11", "ruletest12"]

    for i in l:
        in_filepath = i + ".al"
        out_filepath = i + ".py"
        _test_examples(in_filepath, out_filepath, i)


def test_pydy_examples():

    l = ["mass_spring_damper", "chaos_pendulum", "double_pendulum",
         "non_min_pendulum"]

    for i in l:
        in_filepath = os.path.join("pydy-example-repo", i + ".al")
        out_filepath = os.path.join("pydy-example-repo", i + ".py")
        _test_examples(in_filepath, out_filepath, i)


def test_autolev_tutorial():

    dir_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                            'autolev-tutorial')

    if os.path.isdir(dir_path):
        l = ["tutor1", "tutor2", "tutor3", "tutor4", "tutor5", "tutor6",
             "tutor7"]
        for i in l:
            in_filepath = os.path.join("autolev-tutorial", i + ".al")
            out_filepath = os.path.join("autolev-tutorial", i + ".py")
            _test_examples(in_filepath, out_filepath, i)


def test_dynamics_online():

    dir_path = os.path.join(FILE_DIR, 'autolev', 'test-examples',
                            'dynamics-online')

    if os.path.isdir(dir_path):
        ch1 = ["1-4", "1-5", "1-6", "1-7", "1-8", "1-9_1", "1-9_2", "1-9_3"]
        ch2 = ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-7", "2-8", "2-9",
               "circular"]
        ch3 = ["3-1_1", "3-1_2", "3-2_1", "3-2_2", "3-2_3", "3-2_4", "3-2_5",
               "3-3"]
        ch4 = ["4-1_1", "4-2_1", "4-4_1", "4-4_2", "4-5_1", "4-5_2"]
        chapters = [(ch1, "ch1"), (ch2, "ch2"), (ch3, "ch3"), (ch4, "ch4")]
        for ch, name in chapters:
            for i in ch:
                in_filepath = os.path.join("dynamics-online", name, i + ".al")
                out_filepath = os.path.join("dynamics-online", name, i + ".py")
                _test_examples(in_filepath, out_filepath, i)


def test_output_01():
    """Autolev example calculates the position, velocity, and acceleration of a
    point and expresses in a single reference frame::

          (1) FRAMES C,D,F
          (2) VARIABLES FD'',DC''
          (3) CONSTANTS R,L
          (4) POINTS O,E
          (5) SIMPROT(F,D,1,FD)
       -> (6) F_D = [1, 0, 0; 0, COS(FD), -SIN(FD); 0, SIN(FD), COS(FD)]
          (7) SIMPROT(D,C,2,DC)
       -> (8) D_C = [COS(DC), 0, SIN(DC); 0, 1, 0; -SIN(DC), 0, COS(DC)]
          (9) W_C_F> = EXPRESS(W_C_F>, F)
       -> (10) W_C_F> = FD'*F1> + COS(FD)*DC'*F2> + SIN(FD)*DC'*F3>
          (11) P_O_E>=R*D2>-L*C1>
          (12) P_O_E>=EXPRESS(P_O_E>, D)
       -> (13) P_O_E> = -L*COS(DC)*D1> + R*D2> + L*SIN(DC)*D3>
          (14) V_E_F>=EXPRESS(DT(P_O_E>,F),D)
       -> (15) V_E_F> = L*SIN(DC)*DC'*D1> - L*SIN(DC)*FD'*D2> + (R*FD'+L*COS(DC)*DC')*D3>
          (16) A_E_F>=EXPRESS(DT(V_E_F>,F),D)
       -> (17) A_E_F> = L*(COS(DC)*DC'^2+SIN(DC)*DC'')*D1> + (-R*FD'^2-2*L*COS(DC)*DC'*FD'-L*SIN(DC)*FD'')*D2> + (R*FD''+L*COS(DC)*DC''-L*SIN(DC)*DC'^2-L*SIN(DC)*FD'^2)*D3>

    """

    if not antlr4:
        skip('Test skipped: antlr4 is not installed.')

    autolev_input = """\
FRAMES C,D,F
VARIABLES FD'',DC''
CONSTANTS R,L
POINTS O,E
SIMPROT(F,D,1,FD)
SIMPROT(D,C,2,DC)
W_C_F>=EXPRESS(W_C_F>,F)
P_O_E>=R*D2>-L*C1>
P_O_E>=EXPRESS(P_O_E>,D)
V_E_F>=EXPRESS(DT(P_O_E>,F),D)
A_E_F>=EXPRESS(DT(V_E_F>,F),D)\
"""

    sympy_input = parse_autolev(autolev_input)

    g = {}
    l = {}
    exec(sympy_input, g, l)

    w_c_f = l['frame_c'].ang_vel_in(l['frame_f'])
    # P_O_E> means "the position of point E wrt to point O"
    p_o_e = l['point_e'].pos_from(l['point_o'])
    v_e_f = l['point_e'].vel(l['frame_f'])
    a_e_f = l['point_e'].acc(l['frame_f'])

    # NOTE : The Autolev outputs above were manually transformed into
    # equivalent SymPy physics vector expressions. Would be nice to automate
    # this transformation.
    expected_w_c_f = (l['fd'].diff()*l['frame_f'].x +
                      cos(l['fd'])*l['dc'].diff()*l['frame_f'].y +
                      sin(l['fd'])*l['dc'].diff()*l['frame_f'].z)

    assert (w_c_f - expected_w_c_f).simplify() == 0

    expected_p_o_e = (-l['l']*cos(l['dc'])*l['frame_d'].x +
                      l['r']*l['frame_d'].y +
                      l['l']*sin(l['dc'])*l['frame_d'].z)

    assert (p_o_e - expected_p_o_e).simplify() == 0

    expected_v_e_f = (l['l']*sin(l['dc'])*l['dc'].diff()*l['frame_d'].x -
                      l['l']*sin(l['dc'])*l['fd'].diff()*l['frame_d'].y +
                      (l['r']*l['fd'].diff() +
                       l['l']*cos(l['dc'])*l['dc'].diff())*l['frame_d'].z)
    assert (v_e_f - expected_v_e_f).simplify() == 0

    expected_a_e_f = (l['l']*(cos(l['dc'])*l['dc'].diff()**2 +
                              sin(l['dc'])*l['dc'].diff().diff())*l['frame_d'].x +
                      (-l['r']*l['fd'].diff()**2 -
                       2*l['l']*cos(l['dc'])*l['dc'].diff()*l['fd'].diff() -
                       l['l']*sin(l['dc'])*l['fd'].diff().diff())*l['frame_d'].y +
                      (l['r']*l['fd'].diff().diff() +
                       l['l']*cos(l['dc'])*l['dc'].diff().diff() -
                       l['l']*sin(l['dc'])*l['dc'].diff()**2 -
                       l['l']*sin(l['dc'])*l['fd'].diff()**2)*l['frame_d'].z)
    assert (a_e_f - expected_a_e_f).simplify() == 0
