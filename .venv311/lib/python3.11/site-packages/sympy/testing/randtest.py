"""
.. deprecated:: 1.10

   ``sympy.testing.randtest`` functions have been moved to
   :mod:`sympy.core.random`.

"""
from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning("The sympy.testing.randtest submodule is deprecated. Use sympy.core.random instead.",
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-sympy-testing-randtest")

from sympy.core.random import (  # noqa:F401
    random_complex_number,
    verify_numerically,
    test_derivative_numerically,
    _randrange,
    _randint)
