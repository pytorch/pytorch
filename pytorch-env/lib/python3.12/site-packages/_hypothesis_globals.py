# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""
Module for globals shared between plugin(s) and the main hypothesis module, without
depending on either. This file should have no imports outside of stdlib.
"""

import os

in_initialization = 1
"""If >0, indicates that hypothesis is still initializing (importing or loading
the test environment). `import hypothesis` will cause this number to be decremented,
and the pytest plugin increments at load time, then decrements it just before each test
session starts. However, this leads to a hole in coverage if another pytest plugin
imports hypothesis before our plugin is loaded. HYPOTHESIS_EXTEND_INITIALIZATION may
be set to pre-increment the value on behalf of _hypothesis_pytestplugin, plugging the
hole."""

if os.environ.get("HYPOTHESIS_EXTEND_INITIALIZATION"):
    in_initialization += 1
