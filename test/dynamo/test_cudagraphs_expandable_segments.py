# Owner(s): ["module: cuda"]
# run time cuda tests, but with the allocator using expandable segments

import os
import pathlib
import sys

import torch
from torch.testing._internal.common_cuda import IS_JETSON, IS_WINDOWS
from torch.testing._internal.common_utils import run_tests


pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from dynamo.test_cudagraphs import TestAotCudagraphs  # noqa: F401


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT))
from tools.stats.import_test_stats import get_disabled_tests


# Make sure to remove REPO_ROOT after import is done
sys.path.remove(str(REPO_ROOT))

if __name__ == "__main__":
    if torch.cuda.is_available() and not IS_JETSON and not IS_WINDOWS:
        get_disabled_tests(".")

        torch.cuda.memory._set_allocator_settings("expandable_segments:True")

        run_tests()
