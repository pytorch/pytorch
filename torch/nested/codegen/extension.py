import torch
from functools import wraps
from collections import namedtuple

# This entire file is half hack, half useful information.
#
# The function classifications (unary, binary, comparison)
# are useful to generate generic code based on certaion assumptions
# such as arity. For example, someone might implement a single function
# to efficiently implement a pointwise unary function such as cos
# and then generalize it using the list of unary functions.
#
# The hacky part of this file overwrites a module function via set_function
# and adds a dispatch mechanism via isinstance. The user can specify
# a function to be called if the overwritten function is called
# with a new object of type cls (based on the first argument).
# This dispatch mechanism is inherently inefficient and should be replaced.
# In fact, no release should include this mechanism and it is solely
# to support incremental development.
#
# Stores the relationship between a torch module function
# and a torch.Tensor method. For example torch.add
# maps to torch.Tensor.add and torch.Tensor.add can either
# be inplace or not.
Signature = namedtuple('Signature', ['torch', 'tensor', 'inplace'])

# These functions only take two inputs. An input and a output.
# No scalars etc.
def get_unary_functions():
    return [
        'abs',
        'acos',
        'asin',
        'atan',
        'ceil',
        # 'clamp', # TODO: Requires extra kwargs
        'cos',
        'cosh',
        'digamma',
        'erf',
        'erfc',
        'erfinv',
        'exp',
        'expm1',
        # 'exponential_', # TODO: Method only
        'floor',
        # 'fmod',
        'frac',
        # 'hardshrink', # TODO: Not part of aten
        'lgamma',
        'log',
        'log10',
        'log1p',
        'log2',
        # 'mvlgamma',
        'neg',
        # 'nonzero', #TODO: Special case because it modifies dtype
        # 'polygamma',
        # 'prelu', # TODO: no prelu_out in aten
        'reciprocal',
        # 'relu', # TODO: no relu_out in aten
        # 'renorm', # TODO: Requires extra kwargs
        'round',
        'rsqrt',
        'sigmoid',
        'sign',
        'sin',
        'sinh',
        'sqrt',
        'tan',
        'tanh',
        'tril',
        'triu',
        'trunc']


# These functions take exactly two Tensor arguments.
# It might be that they support scalar arguments as well,
# but we assume that the user will not use it in that way.
def get_binary_functions():
    return [
        'add',
        'mul',
        'sub',
        'div',
        'pow',
        'atan2',
        'remainder',
    ]


def get_comparison_functions():
    return [
        'eq',
        'ge',
        'gt',
        'le',
        'ne',
        'lt'
    ]


# Adds dispatch based on isinstance to torch function
# tfunc by wrapping the current torch.tfunc function
# and overwriting it with a function that dispatches
# to func(tfunc, reference_to_tfunc, *args, **kwargs)
# if the first argument is of instance cls.
# NOTE: This function is a hack. This dispatch mechanism
# is slow and should not be used in release code, but for
# development only.
def set_function(module, cls, tfunc, func):
    def _gen_func(tfunc):
        orig_tfunc = getattr(torch, tfunc)

        @wraps(orig_tfunc)
        def _func(*args, **kwargs):
            if isinstance(args[0], cls):
                return func(*((tfunc, orig_tfunc,) + args), **kwargs)
            else:
                return orig_tfunc(*args, **kwargs)
        return _func
    setattr(module, tfunc, _gen_func(tfunc))


def _check_meaningful_overwrite(cls, method_name):
    class DefaultClass(object):
        pass

    if getattr(cls, method_name, False) and not getattr(DefaultClass, method_name, False):
        raise Exception("WARNING: " + method_name + " already exists "
                        "and not part of default class")


def set_unary_method(cls, tfunc, method_name, inplace):
    def _gen_func(method_name):
        def _func(self):
            assert isinstance(self, cls)
            if inplace:
                return getattr(torch, tfunc)(self, out=self)
            else:
                return getattr(torch, tfunc)(self)
        return _func
    _check_meaningful_overwrite(cls, method_name)
    setattr(cls, method_name, _gen_func(method_name))


def set_binary_method(cls, tfunc, method_name, inplace):
    def _gen_func(tfunc):
        def _func(self, other):
            if inplace:
                return getattr(torch, tfunc)(self, other, out=self)
            else:
                return getattr(torch, tfunc)(self, other)
        return _func
    _check_meaningful_overwrite(cls, method_name)
    setattr(cls, method_name, _gen_func(tfunc))


def get_unary_method_signatures():
    signatures = []
    for method_name in get_unary_functions():
        signatures.append(Signature(method_name, method_name, False))
        signatures.append(Signature(method_name, method_name + "_", True))
    return signatures


def get_binary_method_signatures():
    signatures = []
    for method_name in get_binary_functions():
        signatures.append(Signature(method_name, method_name, False))
        signatures.append(Signature(method_name, method_name + "_", True))

    for method_name in ['add', 'mul', 'sub', 'div']:
        signatures.append(Signature(method_name, "__" + method_name + "__", False))
        signatures.append(Signature(method_name, "__i" + method_name + "__", True))
    return signatures


def get_comparison_method_signatures():
    signatures = []
    for method_name in get_comparison_functions():
        signatures.append(Signature(method_name, method_name, False))
        signatures.append(Signature(method_name, method_name + "_", True))
        signatures.append(Signature(method_name, "__" + method_name + "__", False))
    return signatures


# Add cls dispatch for unary functions and setting the unary methods of
# cls to call into the new unary functions.
def add_pointwise_unary_functions(module, cls, func):
    for function_name in get_unary_functions():
        set_function(module, cls, function_name, func)
    for signature in get_unary_method_signatures():
        set_unary_method(cls, signature.torch, signature.tensor, signature.inplace)
    return module, cls


# Add cls dispatch for binary functions and setting the binary methods of
# cls to call into the new binary functions.
def add_pointwise_binary_functions(module, cls, func):
    for function_name in get_binary_functions():
        set_function(module, cls, function_name, func)
    for signature in get_binary_method_signatures():
        set_binary_method(cls, signature.torch, signature.tensor, signature.inplace)
    return module, cls


# Add cls dispatch for comparison functions and setting the comparison methods of
# cls to call into the new comparison functions.
# It's up to the user to make sure that output is of type torch.uint8
def add_pointwise_comparison_functions(module, cls, func):
    for function_name in get_comparison_functions():
        set_function(module, cls, function_name, func)
    for signature in get_comparison_method_signatures():
        set_binary_method(cls, signature.torch, signature.tensor, signature.inplace)
    return module, cls
