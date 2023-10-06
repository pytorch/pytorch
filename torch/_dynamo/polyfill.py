"""
Python polyfills for common builtins.
"""


def all(iterator):
    for elem in iterator:
        if not elem:
            return False
    return True
