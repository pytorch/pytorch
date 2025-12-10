# Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""Run unit tests. This is invoked by:
$ python3 -m psutil.tests.
"""

import sys

from psutil.tests import pytest

sys.exit(pytest.main())
