"""
Unified place for determining if external dependencies are installed or not.

You should import all external modules using the import_module() function.

For example

>>> from sympy.external import import_module
>>> numpy = import_module('numpy')

If the resulting library is not installed, or if the installed version
is less than a given minimum version, the function will return None.
Otherwise, it will return the library. See the docstring of
import_module() for more information.

"""

from sympy.external.importtools import import_module

__all__ = ['import_module']
