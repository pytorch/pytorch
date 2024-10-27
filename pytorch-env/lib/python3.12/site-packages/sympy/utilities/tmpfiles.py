"""
.. deprecated:: 1.6

   sympy.utilities.tmpfiles has been renamed to sympy.testing.tmpfiles.
"""
from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning("The sympy.utilities.tmpfiles submodule is deprecated. Use sympy.testing.tmpfiles instead.",
    deprecated_since_version="1.6",
    active_deprecations_target="deprecated-sympy-utilities-submodules")

from sympy.testing.tmpfiles import *  # noqa:F401
