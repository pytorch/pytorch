from six import PY2, PY3

if PY2:
    import collections
    container_abcs = collections
elif PY3:
    import collections.abc
    container_abcs = collections.abc
