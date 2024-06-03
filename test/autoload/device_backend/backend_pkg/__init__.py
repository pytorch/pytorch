import os

# when importing this package, set this environment variable to true
os.environ["IS_CUSTOM_DEVICE_BACKEND_IMPORTED"] = "true"


def apply_patch():
    # Do something here
    return "success"


def _autoload():
    # Do nothing in this entry point
    pass
