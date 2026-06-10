# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Make ``test/comms`` importable as the root for test-local helper packages
(``helpers``, ``integration.helpers``, ``perf``) regardless of which test file
pytest collects first."""

import os
import sys


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
