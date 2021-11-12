import os
import re
import site
import sys

import torch


def _prefix_regex() -> str:
    raw_paths = (
        site.getsitepackages() +
        sys.path +
        [site.getuserbase()] +
        [site.getusersitepackages()] +
        [os.path.dirname(os.path.dirname(torch.__file__))]
    )

    path_prefixes = sorted({os.path.abspath(i) for i in raw_paths})
    assert all(isinstance(i, str) for i in path_prefixes)
    return f"^({'|'.join([re.escape(i + os.sep) for i in path_prefixes])})"
