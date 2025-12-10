"""
.. deprecated:: 1.6

   sympy.utilities.pytest has been renamed to sympy.testing.pytest.
"""
from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning("The sympy.utilities.pytest submodule is deprecated. Use sympy.testing.pytest instead.",
    deprecated_since_version="1.6",
    active_deprecations_target="deprecated-sympy-utilities-submodules")

from sympy.testing.pytest import *  # noqa:F401,F403
