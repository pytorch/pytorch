from sympy.testing.pytest import raises
from sympy.utilities.exceptions import sympy_deprecation_warning

# Only test exceptions here because the other cases are tested in the
# warns_deprecated_sympy tests
def test_sympy_deprecation_warning():
    raises(TypeError, lambda: sympy_deprecation_warning('test',
                                                        deprecated_since_version=1.10,
                                                        active_deprecations_target='active-deprecations'))

    raises(ValueError, lambda: sympy_deprecation_warning('test',
                                                            deprecated_since_version="1.10", active_deprecations_target='(active-deprecations)='))
