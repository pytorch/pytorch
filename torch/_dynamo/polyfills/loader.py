# Used to load and initialize polyfill handlers when importing torch._dynamo
# Please add a new import when adding a new polyfill module.

from . import builtins, functools, itertools, os  # noqa: F401
