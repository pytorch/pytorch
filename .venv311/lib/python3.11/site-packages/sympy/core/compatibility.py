"""
.. deprecated:: 1.10

   ``sympy.core.compatibility`` is deprecated. See
   :ref:`sympy-core-compatibility`.

Reimplementations of constructs introduced in later versions of Python than
we support. Also some functions that are needed SymPy-wide and are located
here for easy import.

"""


from sympy.utilities.exceptions import sympy_deprecation_warning

sympy_deprecation_warning("""
The sympy.core.compatibility submodule is deprecated.

This module was only ever intended for internal use. Some of the functions
that were in this module are available from the top-level SymPy namespace,
i.e.,

    from sympy import ordered, default_sort_key

The remaining were only intended for internal SymPy use and should not be used
by user code.
""",
                          deprecated_since_version="1.10",
                          active_deprecations_target="deprecated-sympy-core-compatibility",
                          )


from .sorting import ordered, _nodes, default_sort_key # noqa:F401
from sympy.utilities.misc import as_int as _as_int # noqa:F401
from sympy.utilities.iterables import iterable, is_sequence, NotIterable # noqa:F401
