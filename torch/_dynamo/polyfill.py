"""
Python polyfills for common builtins.
"""


def all(iterator):
    for elem in iterator:
        if not elem:
            return False
    return True

def index(iterator, match, start=0, end=-1):
    for i, elem in enumerate(list(iterator))[start:end]:
        if match == elem:
            return i
    raise ValueError(f"{match} is not in {type(iterator)}")
