import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
    implicit_multiplication_application,
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
    split_symbols,
    split_symbols_custom,
    _token_splittable
)
from sympy.testing.pytest import raises


def test_implicit_multiplication():
    cases = {
        '5x': '5*x',
        'abc': 'a*b*c',
        '3sin(x)': '3*sin(x)',
        '(x+1)(x+2)': '(x+1)*(x+2)',
        '(5 x**2)sin(x)': '(5*x**2)*sin(x)',
        '2 sin(x) cos(x)': '2*sin(x)*cos(x)',
        'pi x': 'pi*x',
        'x pi': 'x*pi',
        'E x': 'E*x',
        'EulerGamma y': 'EulerGamma*y',
        'E pi': 'E*pi',
        'pi (x + 2)': 'pi*(x+2)',
        '(x + 2) pi': '(x+2)*pi',
        'pi sin(x)': 'pi*sin(x)',
    }
    transformations = standard_transformations + (convert_xor,)
    transformations2 = transformations + (split_symbols,
                                          implicit_multiplication)
    for case in cases:
        implicit = parse_expr(case, transformations=transformations2)
        normal = parse_expr(cases[case], transformations=transformations)
        assert(implicit == normal)

    application = ['sin x', 'cos 2*x', 'sin cos x']
    for case in application:
        raises(SyntaxError,
               lambda: parse_expr(case, transformations=transformations2))
    raises(TypeError,
           lambda: parse_expr('sin**2(x)', transformations=transformations2))


def test_implicit_application():
    cases = {
        'factorial': 'factorial',
        'sin x': 'sin(x)',
        'tan y**3': 'tan(y**3)',
        'cos 2*x': 'cos(2*x)',
        '(cot)': 'cot',
        'sin cos tan x': 'sin(cos(tan(x)))'
    }
    transformations = standard_transformations + (convert_xor,)
    transformations2 = transformations + (implicit_application,)
    for case in cases:
        implicit = parse_expr(case, transformations=transformations2)
        normal = parse_expr(cases[case], transformations=transformations)
        assert(implicit == normal), (implicit, normal)

    multiplication = ['x y', 'x sin x', '2x']
    for case in multiplication:
        raises(SyntaxError,
               lambda: parse_expr(case, transformations=transformations2))
    raises(TypeError,
           lambda: parse_expr('sin**2(x)', transformations=transformations2))


def test_function_exponentiation():
    cases = {
        'sin**2(x)': 'sin(x)**2',
        'exp^y(z)': 'exp(z)^y',
        'sin**2(E^(x))': 'sin(E^(x))**2'
    }
    transformations = standard_transformations + (convert_xor,)
    transformations2 = transformations + (function_exponentiation,)
    for case in cases:
        implicit = parse_expr(case, transformations=transformations2)
        normal = parse_expr(cases[case], transformations=transformations)
        assert(implicit == normal)

    other_implicit = ['x y', 'x sin x', '2x', 'sin x',
                      'cos 2*x', 'sin cos x']
    for case in other_implicit:
        raises(SyntaxError,
               lambda: parse_expr(case, transformations=transformations2))

    assert parse_expr('x**2', local_dict={ 'x': sympy.Symbol('x') },
                      transformations=transformations2) == parse_expr('x**2')


def test_symbol_splitting():
    # By default Greek letter names should not be split (lambda is a keyword
    # so skip it)
    transformations = standard_transformations + (split_symbols,)
    greek_letters = ('alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta',
                     'eta', 'theta', 'iota', 'kappa', 'mu', 'nu', 'xi',
                     'omicron', 'pi', 'rho', 'sigma', 'tau', 'upsilon',
                     'phi', 'chi', 'psi', 'omega')

    for letter in greek_letters:
        assert(parse_expr(letter, transformations=transformations) ==
               parse_expr(letter))

    # Make sure symbol splitting resolves names
    transformations += (implicit_multiplication,)
    local_dict = { 'e': sympy.E }
    cases = {
        'xe': 'E*x',
        'Iy': 'I*y',
        'ee': 'E*E',
    }
    for case, expected in cases.items():
        assert(parse_expr(case, local_dict=local_dict,
                          transformations=transformations) ==
               parse_expr(expected))

    # Make sure custom splitting works
    def can_split(symbol):
        if symbol not in ('unsplittable', 'names'):
            return _token_splittable(symbol)
        return False
    transformations = standard_transformations
    transformations += (split_symbols_custom(can_split),
                        implicit_multiplication)

    assert(parse_expr('unsplittable', transformations=transformations) ==
           parse_expr('unsplittable'))
    assert(parse_expr('names', transformations=transformations) ==
           parse_expr('names'))
    assert(parse_expr('xy', transformations=transformations) ==
           parse_expr('x*y'))
    for letter in greek_letters:
        assert(parse_expr(letter, transformations=transformations) ==
               parse_expr(letter))


def test_all_implicit_steps():
    cases = {
        '2x': '2*x',  # implicit multiplication
        'x y': 'x*y',
        'xy': 'x*y',
        'sin x': 'sin(x)',  # add parentheses
        '2sin x': '2*sin(x)',
        'x y z': 'x*y*z',
        'sin(2 * 3x)': 'sin(2 * 3 * x)',
        'sin(x) (1 + cos(x))': 'sin(x) * (1 + cos(x))',
        '(x + 2) sin(x)': '(x + 2) * sin(x)',
        '(x + 2) sin x': '(x + 2) * sin(x)',
        'sin(sin x)': 'sin(sin(x))',
        'sin x!': 'sin(factorial(x))',
        'sin x!!': 'sin(factorial2(x))',
        'factorial': 'factorial',  # don't apply a bare function
        'x sin x': 'x * sin(x)',  # both application and multiplication
        'xy sin x': 'x * y * sin(x)',
        '(x+2)(x+3)': '(x + 2) * (x+3)',
        'x**2 + 2xy + y**2': 'x**2 + 2 * x * y + y**2',  # split the xy
        'pi': 'pi',  # don't mess with constants
        'None': 'None',
        'ln sin x': 'ln(sin(x))',  # multiple implicit function applications
        'sin x**2': 'sin(x**2)',  # implicit application to an exponential
        'alpha': 'Symbol("alpha")',  # don't split Greek letters/subscripts
        'x_2': 'Symbol("x_2")',
        'sin^2 x**2': 'sin(x**2)**2',  # function raised to a power
        'sin**3(x)': 'sin(x)**3',
        '(factorial)': 'factorial',
        'tan 3x': 'tan(3*x)',
        'sin^2(3*E^(x))': 'sin(3*E**(x))**2',
        'sin**2(E^(3x))': 'sin(E**(3*x))**2',
        'sin^2 (3x*E^(x))': 'sin(3*x*E^x)**2',
        'pi sin x': 'pi*sin(x)',
    }
    transformations = standard_transformations + (convert_xor,)
    transformations2 = transformations + (implicit_multiplication_application,)
    for case in cases:
        implicit = parse_expr(case, transformations=transformations2)
        normal = parse_expr(cases[case], transformations=transformations)
        assert(implicit == normal)


def test_no_methods_implicit_multiplication():
    # Issue 21020
    u = sympy.Symbol('u')
    transformations = standard_transformations + \
                      (implicit_multiplication,)
    expr = parse_expr('x.is_polynomial(x)', transformations=transformations)
    assert expr == True
    expr = parse_expr('(exp(x) / (1 + exp(2x))).subs(exp(x), u)',
                      transformations=transformations)
    assert expr == u/(u**2 + 1)
