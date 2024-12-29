from sympy.testing.pytest import warns_deprecated_sympy

def test_deprecated_testing_randtest():
    with warns_deprecated_sympy():
        import sympy.testing.randtest  # noqa:F401
