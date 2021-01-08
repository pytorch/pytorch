try:
    import collections.abc
    container_abcs = collections.abc
except ImportError:
    # This fallback applies for all versions of Python before 3.3
    import collections
    container_abcs = collections
