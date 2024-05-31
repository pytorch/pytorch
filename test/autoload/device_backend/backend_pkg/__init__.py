import os

# when loading this package, set this environment variable to true
os.environ['IS_CUSTOM_DEVICE_BACKEND_IMPORTED'] = 'true'


def autoload():
    # do nothing in this entry point
    pass
