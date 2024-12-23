from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning(
    """
    sympy.core.trace is deprecated. Use sympy.physics.quantum.trace
    instead.
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="sympy-core-trace-deprecated",
)

from sympy.physics.quantum.trace import Tr # noqa:F401
