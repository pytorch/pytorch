from sympy.testing.pytest import warns_deprecated_sympy

# See https://github.com/sympy/sympy/pull/18095

def test_deprecated_utilities():
    with warns_deprecated_sympy():
        import sympy.utilities.pytest  # noqa:F401
    with warns_deprecated_sympy():
        import sympy.utilities.runtests  # noqa:F401
    with warns_deprecated_sympy():
        import sympy.utilities.randtest  # noqa:F401
    with warns_deprecated_sympy():
        import sympy.utilities.tmpfiles  # noqa:F401
