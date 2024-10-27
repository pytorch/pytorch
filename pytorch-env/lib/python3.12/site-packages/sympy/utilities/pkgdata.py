# This module is deprecated and will be removed.

import sys
import os
from io import StringIO

from sympy.utilities.decorator import deprecated


@deprecated(
    """
    The sympy.utilities.pkgdata module and its get_resource function are
    deprecated. Use the stdlib importlib.resources module instead.
    """,
    deprecated_since_version="1.12",
    active_deprecations_target="pkgdata",
)
def get_resource(identifier, pkgname=__name__):

    mod = sys.modules[pkgname]
    fn = getattr(mod, '__file__', None)
    if fn is None:
        raise OSError("%r has no __file__!")
    path = os.path.join(os.path.dirname(fn), identifier)
    loader = getattr(mod, '__loader__', None)
    if loader is not None:
        try:
            data = loader.get_data(path)
        except (OSError, AttributeError):
            pass
        else:
            return StringIO(data.decode('utf-8'))
    return open(os.path.normpath(path), 'rb')
