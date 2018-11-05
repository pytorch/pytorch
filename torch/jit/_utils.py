# Python equivalents for the empty list construction builtins. We need
# these otherwise the tests won't execute in regular Python mode.

# These can't live in jit/__init__.py because we use the module replacement
# trick on torch.jit (see JITModule), torch.jit becomes an object, and attribute
# look up is not yet supported on regular object.


def _construct_empty_int_list():
    return []


def _construct_empty_float_list():
    return []


def _construct_empty_tensor_list():
    return []
