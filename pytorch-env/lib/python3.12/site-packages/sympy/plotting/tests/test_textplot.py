from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.plotting.textplot import textplot_str

from sympy.utilities.exceptions import ignore_warnings


def test_axes_alignment():
    x = Symbol('x')
    lines = [
        '      1 |                                                     ..',
        '        |                                                  ...  ',
        '        |                                                ..     ',
        '        |                                             ...       ',
        '        |                                          ...          ',
        '        |                                        ..             ',
        '        |                                     ...               ',
        '        |                                  ...                  ',
        '        |                                ..                     ',
        '        |                             ...                       ',
        '      0 |--------------------------...--------------------------',
        '        |                       ...                             ',
        '        |                     ..                                ',
        '        |                  ...                                  ',
        '        |               ...                                     ',
        '        |             ..                                        ',
        '        |          ...                                          ',
        '        |       ...                                             ',
        '        |     ..                                                ',
        '        |  ...                                                  ',
        '     -1 |_______________________________________________________',
        '         -1                         0                          1'
    ]
    assert lines == list(textplot_str(x, -1, 1))

    lines = [
        '      1 |                                                     ..',
        '        |                                                 ....  ',
        '        |                                              ...      ',
        '        |                                           ...         ',
        '        |                                       ....            ',
        '        |                                    ...                ',
        '        |                                 ...                   ',
        '        |                             ....                      ',
        '      0 |--------------------------...--------------------------',
        '        |                      ....                             ',
        '        |                   ...                                 ',
        '        |                ...                                    ',
        '        |            ....                                       ',
        '        |         ...                                           ',
        '        |      ...                                              ',
        '        |  ....                                                 ',
        '     -1 |_______________________________________________________',
        '         -1                         0                          1'
    ]
    assert lines == list(textplot_str(x, -1, 1, H=17))


def test_singularity():
    x = Symbol('x')
    lines = [
        '     54 | .                                                     ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ','        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '   27.5 |--.----------------------------------------------------',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |   .                                                   ',
        '        |    \\                                                  ',
        '        |     \\                                                 ',
        '        |      ..                                               ',
        '        |        ...                                            ',
        '        |           .............                               ',
        '      1 |_______________________________________________________',
        '         0                          0.5                        1'
    ]
    assert lines == list(textplot_str(1/x, 0, 1))

    lines = [
        '      0 |                                                 ......',
        '        |                                         ........      ',
        '        |                                 ........              ',
        '        |                           ......                      ',
        '        |                      .....                            ',
        '        |                  ....                                 ',
        '        |               ...                                     ',
        '        |             ..                                        ',
        '        |          ...                                          ',
        '        |         /                                             ',
        '     -2 |-------..----------------------------------------------',
        '        |      /                                                ',
        '        |     /                                                 ',
        '        |    /                                                  ',
        '        |   .                                                   ',
        '        |                                                       ',
        '        |  .                                                    ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '     -4 |_______________________________________________________',
        '         0                          0.5                        1'
    ]
    # RuntimeWarning: divide by zero encountered in log
    with ignore_warnings(RuntimeWarning):
        assert lines == list(textplot_str(log(x), 0, 1))


def test_sinc():
    x = Symbol('x')
    lines = [
        '      1 |                          . .                          ',
        '        |                         .   .                         ',
        '        |                                                       ',
        '        |                        .     .                        ',
        '        |                                                       ',
        '        |                       .       .                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                      .         .                      ',
        '        |                                                       ',
        '    0.4 |-------------------------------------------------------',
        '        |                     .           .                     ',
        '        |                                                       ',
        '        |                    .             .                    ',
        '        |                                                       ',
        '        |    .....                                     .....    ',
        '        |  ..     \\         .               .         /     ..  ',
        '        | /        \\                                 /        \\ ',
        '        |/          \\      .                 .      /          \\',
        '        |            \\    /                   \\    /            ',
        '   -0.2 |_______________________________________________________',
        '         -10                        0                          10'
    ]
    # RuntimeWarning: invalid value encountered in double_scalars
    with ignore_warnings(RuntimeWarning):
        assert lines == list(textplot_str(sin(x)/x, -10, 10))


def test_imaginary():
    x = Symbol('x')
    lines = [
        '      1 |                                                     ..',
        '        |                                                   ..  ',
        '        |                                                ...    ',
        '        |                                              ..       ',
        '        |                                            ..         ',
        '        |                                          ..           ',
        '        |                                        ..             ',
        '        |                                      ..               ',
        '        |                                    ..                 ',
        '        |                                   /                   ',
        '    0.5 |----------------------------------/--------------------',
        '        |                                ..                     ',
        '        |                               /                       ',
        '        |                              .                        ',
        '        |                                                       ',
        '        |                             .                         ',
        '        |                            .                          ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '      0 |_______________________________________________________',
        '         -1                         0                          1'
    ]
    # RuntimeWarning: invalid value encountered in sqrt
    with ignore_warnings(RuntimeWarning):
        assert list(textplot_str(sqrt(x), -1, 1)) == lines

    lines = [
        '      1 |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '      0 |-------------------------------------------------------',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '        |                                                       ',
        '     -1 |_______________________________________________________',
        '         -1                         0                          1'
    ]
    assert list(textplot_str(S.ImaginaryUnit, -1, 1)) == lines
