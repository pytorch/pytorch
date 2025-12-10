try:
    from ctypes import c_float, c_int, c_double
except ImportError:
    pass

import pyglet.gl as pgl
from sympy.core import S


def get_model_matrix(array_type=c_float, glGetMethod=pgl.glGetFloatv):
    """
    Returns the current modelview matrix.
    """
    m = (array_type*16)()
    glGetMethod(pgl.GL_MODELVIEW_MATRIX, m)
    return m


def get_projection_matrix(array_type=c_float, glGetMethod=pgl.glGetFloatv):
    """
    Returns the current modelview matrix.
    """
    m = (array_type*16)()
    glGetMethod(pgl.GL_PROJECTION_MATRIX, m)
    return m


def get_viewport():
    """
    Returns the current viewport.
    """
    m = (c_int*4)()
    pgl.glGetIntegerv(pgl.GL_VIEWPORT, m)
    return m


def get_direction_vectors():
    m = get_model_matrix()
    return ((m[0], m[4], m[8]),
            (m[1], m[5], m[9]),
            (m[2], m[6], m[10]))


def get_view_direction_vectors():
    m = get_model_matrix()
    return ((m[0], m[1], m[2]),
            (m[4], m[5], m[6]),
            (m[8], m[9], m[10]))


def get_basis_vectors():
    return ((1, 0, 0), (0, 1, 0), (0, 0, 1))


def screen_to_model(x, y, z):
    m = get_model_matrix(c_double, pgl.glGetDoublev)
    p = get_projection_matrix(c_double, pgl.glGetDoublev)
    w = get_viewport()
    mx, my, mz = c_double(), c_double(), c_double()
    pgl.gluUnProject(x, y, z, m, p, w, mx, my, mz)
    return float(mx.value), float(my.value), float(mz.value)


def model_to_screen(x, y, z):
    m = get_model_matrix(c_double, pgl.glGetDoublev)
    p = get_projection_matrix(c_double, pgl.glGetDoublev)
    w = get_viewport()
    mx, my, mz = c_double(), c_double(), c_double()
    pgl.gluProject(x, y, z, m, p, w, mx, my, mz)
    return float(mx.value), float(my.value), float(mz.value)


def vec_subs(a, b):
    return tuple(a[i] - b[i] for i in range(len(a)))


def billboard_matrix():
    """
    Removes rotational components of
    current matrix so that primitives
    are always drawn facing the viewer.

    |1|0|0|x|
    |0|1|0|x|
    |0|0|1|x| (x means left unchanged)
    |x|x|x|x|
    """
    m = get_model_matrix()
    # XXX: for i in range(11): m[i] = i ?
    m[0] = 1
    m[1] = 0
    m[2] = 0
    m[4] = 0
    m[5] = 1
    m[6] = 0
    m[8] = 0
    m[9] = 0
    m[10] = 1
    pgl.glLoadMatrixf(m)


def create_bounds():
    return [[S.Infinity, S.NegativeInfinity, 0],
            [S.Infinity, S.NegativeInfinity, 0],
            [S.Infinity, S.NegativeInfinity, 0]]


def update_bounds(b, v):
    if v is None:
        return
    for axis in range(3):
        b[axis][0] = min([b[axis][0], v[axis]])
        b[axis][1] = max([b[axis][1], v[axis]])


def interpolate(a_min, a_max, a_ratio):
    return a_min + a_ratio * (a_max - a_min)


def rinterpolate(a_min, a_max, a_value):
    a_range = a_max - a_min
    if a_max == a_min:
        a_range = 1.0
    return (a_value - a_min) / float(a_range)


def interpolate_color(color1, color2, ratio):
    return tuple(interpolate(color1[i], color2[i], ratio) for i in range(3))


def scale_value(v, v_min, v_len):
    return (v - v_min) / v_len


def scale_value_list(flist):
    v_min, v_max = min(flist), max(flist)
    v_len = v_max - v_min
    return [scale_value(f, v_min, v_len) for f in flist]


def strided_range(r_min, r_max, stride, max_steps=50):
    o_min, o_max = r_min, r_max
    if abs(r_min - r_max) < 0.001:
        return []
    try:
        range(int(r_min - r_max))
    except (TypeError, OverflowError):
        return []
    if r_min > r_max:
        raise ValueError("r_min cannot be greater than r_max")
    r_min_s = (r_min % stride)
    r_max_s = stride - (r_max % stride)
    if abs(r_max_s - stride) < 0.001:
        r_max_s = 0.0
    r_min -= r_min_s
    r_max += r_max_s
    r_steps = int((r_max - r_min)/stride)
    if max_steps and r_steps > max_steps:
        return strided_range(o_min, o_max, stride*2)
    return [r_min] + [r_min + e*stride for e in range(1, r_steps + 1)] + [r_max]


def parse_option_string(s):
    if not isinstance(s, str):
        return None
    options = {}
    for token in s.split(';'):
        pieces = token.split('=')
        if len(pieces) == 1:
            option, value = pieces[0], ""
        elif len(pieces) == 2:
            option, value = pieces
        else:
            raise ValueError("Plot option string '%s' is malformed." % (s))
        options[option.strip()] = value.strip()
    return options


def dot_product(v1, v2):
    return sum(v1[i]*v2[i] for i in range(3))


def vec_sub(v1, v2):
    return tuple(v1[i] - v2[i] for i in range(3))


def vec_mag(v):
    return sum(v[i]**2 for i in range(3))**(0.5)
