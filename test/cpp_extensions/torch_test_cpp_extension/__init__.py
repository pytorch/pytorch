"""
This is a device backend extension used for testing.
See this RFC: https://github.com/pytorch/pytorch/issues/122468
"""

import os


def _autoload():
    # Set the environment variable to true in this entrypoint
    os.environ["IS_CUSTOM_DEVICE_BACKEND_IMPORTED"] = "1"
