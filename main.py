def foo(x):
    y: int = x  # should error - x is Unknown
    return y
