"""
Python polyfills for common builtins.
"""


def all_polyfill(iterator):
    for elem in iterator:
        if not bool(elem):
            return False
    return True
