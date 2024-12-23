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
Stub for users who manually load our pytest plugin.

The plugin implementation is now located in a top-level module outside the main
hypothesis tree, so that Pytest can load the plugin without thereby triggering
the import of Hypothesis itself (and thus loading our own plugins).
"""

from _hypothesis_pytestplugin import *  # noqa
