import torch


def set_binary_method(cls, tfunc, pbf, func, inplace):
    def _gen_func(pbf):
        def _func(self: cls, other: cls):
            def _t_func(output: torch.Tensor,
                        input1: torch.Tensor,
                        input2: torch.Tensor):
                getattr(torch, tfunc)(input1, input2, out=output)
            if inplace:
                return func(_t_func, self, self, other)
            else:
                output = self.clone()
                return func(_t_func, output, self, other)
        return _func
    setattr(cls, pbf, _gen_func(pbf))


def set_binary_function(module, cls, tfunc, pbf, func):
    def _gen_func(pbf):
        orig_tfunc = getattr(torch, tfunc)
        def _func(*args, **kwargs):
            if isinstance(args[0], cls):
                def _t_func(output: torch.Tensor,
                            input1: torch.Tensor,
                            input2: torch.Tensor):
                    orig_tfunc(input1, input2, out=output)
                assert len(args) == 2
                if len(kwargs):
                    assert list(kwargs.keys()) == ['out']
                    output = kwargs['out']
                else:
                    output = args[0].clone()
                return func(_t_func, output, args[0], args[1])
            else:
                return orig_tfunc(*args, **kwargs)
        return _func
    setattr(module, pbf, _gen_func(pbf))


def add_pointwise_binary_functions(module, cls, func):
    funcs = [
        'add',
        'mul',
        'sub',
    ]
    for pbf in funcs:
        set_binary_method(cls, pbf, pbf, func, False)
        set_binary_method(cls, pbf, pbf + "_", func, True)
        set_binary_method(cls, pbf, "__" + pbf + "__", func, False)
        set_binary_method(cls, pbf, "__i" + pbf + "__", func, True)
        set_binary_function(module, cls, pbf, pbf, func)
    set_binary_method(cls, "div", "div", func, False)
    set_binary_method(cls, "div", "div_", func, True)
    set_binary_method(cls, "div", "__truediv__", func, False)
    set_binary_method(cls, "div", "__itruediv__", func, True)
    set_binary_function(module, cls, "div", "div", func)
    return module, cls


# It's up to the user to make sure that output is of type torch.uint8
def add_pointwise_comparison_functions(cls, func):
    funcs = [
        'eq',
        'ge',
        'gt',
        'le',
        'ne',
        'ge'
    ]
    for pbf in funcs:
        set_binary_method(cls, pbf, pbf, func, False)
        set_binary_method(cls, pbf, pbf + "_", func, True)
        set_binary_method(cls, pbf, "__" + pbf + "__", func, False)
    return cls


def set_unary_method(cls, tfunc, pbf, func, inplace):
    def _gen_func(pbf):
        def _func(self: cls):
            assert isinstance(self, cls)

            def _t_func(output: torch.Tensor,
                        input1: torch.Tensor):
                # NOTE: We are disabling broadcasting for now
                output.size() == input1.size()
                getattr(torch, tfunc)(input1, out=output)
            if inplace:
                func(_t_func, self, self)
                return self
            else:
                output = self.clone()
                func(_t_func, output, self)
                return output
        return _func
    if getattr(cls, pbf, False):
        print("WARNING: " + pbf + " already exists")
    setattr(cls, pbf, _gen_func(pbf))


def add_pointwise_unary_functions(cls, func):
    funcs = [
        'abs',
        'acos',
        'asin',
        'atan',
        'atan2',
        # 'byte',
        'ceil',
        # 'char',
        'clamp',
        # 'clone',
        'contiguous',
        'cos',
        'cosh',
        # 'cpu',
        # 'cuda',
        'digamma',
        # 'div',
        'double',
        # 'dtype',
        'erf',
        'erfc',
        'erfinv',
        'exp',
        'expm1',
        'exponential_',
        # 'float',
        'floor',
        'fmod',
        'frac',
        # 'half',
        'hardshrink',
        # 'int',
        'lgamma',
        'log',
        'log10',
        'log1p',
        'log2',
        'long',
        'lt',
        'mvlgamma',
        'neg',
        'nonzero',
        'polygamma',
        'pow',
        'prelu',
        'reciprocal',
        'relu',
        'remainder',
        'renorm',
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
    for pbf in funcs:
        set_unary_method(cls, pbf, pbf, func, False)
        set_unary_method(cls, pbf, pbf + '_', func, True)
    return cls
