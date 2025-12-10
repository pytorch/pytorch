from sympy.testing.pytest import warns_deprecated_sympy

def test_compatibility_submodule():
    # Test the sympy.core.compatibility deprecation warning
    with warns_deprecated_sympy():
        import sympy.core.compatibility # noqa:F401
