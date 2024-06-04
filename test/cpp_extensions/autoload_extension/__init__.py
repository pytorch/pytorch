"""
This is a device backend extension used for testing.
See this RFC: https://github.com/pytorch/pytorch/issues/122468
"""

import os

# When importing this package, set this environment variable to true
os.environ["IS_CUSTOM_DEVICE_BACKEND_IMPORTED"] = "true"


def apply_patch():
    # Do something here
    return "success"


def _autoload():
    # Do nothing in this entry point
    pass
