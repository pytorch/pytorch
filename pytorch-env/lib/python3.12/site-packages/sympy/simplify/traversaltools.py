from sympy.core.traversal import use as _use
from sympy.utilities.decorator import deprecated

use = deprecated(
    """
    Using use from the sympy.simplify.traversaltools submodule is
    deprecated.

    Instead, use use from the top-level sympy namespace, like

        sympy.use
    """,
    deprecated_since_version="1.10",
    active_deprecations_target="deprecated-traversal-functions-moved"
)(_use)
