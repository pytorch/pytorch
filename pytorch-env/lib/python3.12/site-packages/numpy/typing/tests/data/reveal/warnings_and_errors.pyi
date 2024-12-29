import sys

import numpy.exceptions as ex

if sys.version_info >= (3, 11):
    from typing import assert_type
else:
    from typing_extensions import assert_type

assert_type(ex.ModuleDeprecationWarning(), ex.ModuleDeprecationWarning)
assert_type(ex.VisibleDeprecationWarning(), ex.VisibleDeprecationWarning)
assert_type(ex.ComplexWarning(), ex.ComplexWarning)
assert_type(ex.RankWarning(), ex.RankWarning)
assert_type(ex.TooHardError(), ex.TooHardError)
assert_type(ex.AxisError("test"), ex.AxisError)
assert_type(ex.AxisError(5, 1), ex.AxisError)
