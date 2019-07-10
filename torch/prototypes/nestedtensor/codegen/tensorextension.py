import torch

# These functions only take two inputs. An input and a output.
# No scalars etc.
def get_unary_functions():
    return [
        'abs',
        'acos',
        'asin',
        'atan',
        # 'atan2',
        # 'byte',
        'ceil',
        # 'char',
        'clamp',
        # 'clone',
        # 'contiguous',
        'cos',
        'cosh',
        # 'cpu',
        # 'cuda',
        'digamma',
        # 'div',
        # 'double',
        # 'dtype',
        'erf',
        'erfc',
        'erfinv',
        'exp',
        'expm1',
        # 'exponential_',
        # 'float',
        'floor',
        # 'fmod',
        'frac',
        # 'half',
        # 'hardshrink', #TODO: Not part of aten
        # 'int',
        'lgamma',
        'log',
        'log10',
        'log1p',
        'log2',
        # 'long',
        # 'lt',
        # 'mvlgamma',
        'neg',
        'nonzero',
        # 'polygamma',
        # 'pow',
        # 'prelu', # TODO: no prelu_out
        'reciprocal',
        # 'relu', # TODO: no relu_out
        # 'remainder',
        # 'renorm',
        'round',
        'rsqrt',
        'sigmoid',
        'sign',
        'sin',
        'sinh',
        'sqrt',
        # 'sub',
        'tan',
        'tanh',
        'tril',
        'triu',
        'trunc']


def get_binary_functions():
    funcs = [
        'add',
        'mul',
        'sub',
        'div',
    ]
    return funcs


def get_comparison_functions():
    funcs = [
        'eq',
        'ge',
        'gt',
        'le',
        'ne',
        'ge'
    ]
    return funcs


def set_function(module, cls, tfunc, func):
    def _gen_func(tfunc):
        orig_tfunc = getattr(torch, tfunc)
        def _func(*args, **kwargs):
            if isinstance(args[0], cls):
                return func(*((orig_tfunc,) + args), **kwargs)
            else:
                return orig_tfunc(*args, **kwargs)
        return _func
    setattr(module, tfunc, _gen_func(tfunc))


def set_unary_method(cls, tfunc, pbf, inplace):
    def _gen_func(pbf):
        def _func(self: cls):
            assert isinstance(self, cls)
            if inplace:
                return getattr(torch, tfunc)(self, out=self)
            else:
                return getattr(torch, tfunc)(self)
        return _func
    if getattr(cls, pbf, False):
        print("WARNING: " + pbf + " already exists")
    setattr(cls, pbf, _gen_func(pbf))


def set_binary_method(cls, tfunc, pbf, inplace):
    def _gen_func(tfunc):
        def _func(self: cls, other: cls):
            if inplace:
                return getattr(torch, tfunc)(self, other, out=self)
            else:
                return getattr(torch, tfunc)(self, other)
        return _func
    if getattr(cls, pbf, False):
        print("WARNING: " + pbf + " already exists")
    setattr(cls, pbf, _gen_func(tfunc))


def get_unary_method_signatures():
    signatures = []
    for pbf in get_unary_functions():
        signatures.append({'torch': pbf, 'Tensor': pbf,       'inplace': False})
        signatures.append({'torch': pbf, 'Tensor': pbf + "_", 'inplace': True})
    return signatures


def get_binary_method_signatures():
    signatures = []
    for pbf in ['add', 'mul', 'sub']:
        signatures.append({'torch': pbf, 'Tensor':         pbf,        'inplace': False})
        signatures.append({'torch': pbf, 'Tensor':         pbf + "_",  'inplace': True})
        signatures.append({'torch': pbf, 'Tensor':  "__" + pbf + "__", 'inplace': False})
        signatures.append({'torch': pbf, 'Tensor': "__i" + pbf + "__", 'inplace': True})
    signatures.append({'torch': 'div', 'Tensor':         'div',        'inplace': False})
    signatures.append({'torch': 'div', 'Tensor':         'div' + "_",  'inplace': True})
    signatures.append({'torch': 'div', 'Tensor':  "__" + 'div' + "__", 'inplace': False})
    signatures.append({'torch': 'div', 'Tensor': "__i" + 'div' + "__", 'inplace': True})
    return signatures


def get_comparison_method_signatures():
    signatures = []
    for pbf in get_comparison_functions():
        signatures.append({'torch': pbf, 'Tensor':         pbf,        'inplace': False})
        signatures.append({'torch': pbf, 'Tensor':         pbf + "_",  'inplace': True})
        signatures.append({'torch': pbf, 'Tensor':  "__" + pbf + "__", 'inplace': False})
    return signatures


def add_pointwise_unary_functions(module, cls, func):
    for function_name in get_unary_functions():
        set_function(module, cls, function_name, func)
    for signature in get_unary_method_signatures():
        set_unary_method(cls, signature['torch'], signature['Tensor'], signature['inplace'])
    return module, cls


def add_pointwise_binary_functions(module, cls, func):
    for function_name in get_binary_functions():
        set_function(module, cls, function_name, func)
    for signature in get_binary_method_signatures():
        set_binary_method(cls, signature['torch'], signature['Tensor'], signature['inplace'])
    return module, cls


# It's up to the user to make sure that output is of type torch.uint8
def add_pointwise_comparison_functions(module, cls, func):
    for function_name in get_comparison_functions():
        set_function(module, cls, function_name, func)
    for signature in get_comparison_method_signatures():
        set_binary_method(cls, signature['torch'], signature['Tensor'], signature['inplace'])
    return module, cls
