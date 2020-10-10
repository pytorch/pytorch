import re

import numpy as np
from numpy.testing import assert_


def test_valid_numpy_version():
    # Verify that the numpy version is a valid one (no .post suffix or other
    # nonsense).  See gh-6431 for an issue caused by an invalid version.
    version_pattern = r"^[0-9]+\.[0-9]+\.[0-9]+(|a[0-9]|b[0-9]|rc[0-9])"
    dev_suffix = r"(\.dev0\+([0-9a-f]{7}|Unknown))"
    if np.version.release:
        res = re.match(version_pattern, np.__version__)
    else:
        res = re.match(version_pattern + dev_suffix, np.__version__)

    assert_(res is not None, np.__version__)
