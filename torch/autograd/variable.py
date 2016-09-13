from .engine import ExecutionEngine


class Variable(object):

    _execution_engine = ExecutionEngine()

    _fallthrough_methods = [
        'size',
        'stride',
        'nElement',
        'numel',
        'dim',
        # TODO: add more
    ]

    def __init__(self, tensor, creator=None, volatile=False, requires_grad=True):
        if volatile:
            requires_grad = False
        if not volatile and creator is None:
            creator = Leaf(self, requires_grad)
        self.creator = creator
        self.volatile = volatile
        self.dirty = False
        self._requires_grad = None
        self._data = tensor
        self._grad = None

    @property
    def grad(self):
        if not self.volatile and self.requires_grad:
            # TODO: this won't have to be zeroed in the future
            self._grad = self._grad or self.data.new(self.data.size()).zero_()
        return self._grad

    @property
    def data(self):
        if self.dirty:
            raise RuntimeError('Accessing data of a dirty variable!')
        return self._data

    @property
    def requires_grad(self):
        if self.volatile:
            return False
        if self._requires_grad is not None:
            return self._requires_grad
        return self.creator.requires_grad

    def mark_dirty(self):
        self.dirty = True
        self._data = None

    def __getattr__(self, name):
        if name in self._fallthrough_methods:
            return getattr(self.data, name)
        raise AttributeError(name)

    def __getitem__(self, key):
        return Index(key)(self)

    def __deepcopy__(self, memo):
        if isinstance(self.creator, Leaf):
            return Variable(self.data.clone(), requires_grad=self.requires_grad,
                    volatile=self.volatile)
        raise RuntimeError("Only Variables created explicitly by the user "
                "(graph leaves) support the deepcopy protocol at the moment")

    def backward(self, gradient=None):
        if self.volatile:
            raise RuntimeError('calling backward on a volatile variable')
        if not self.requires_grad:
            raise RuntimeError("calling backward on a variable that doesn't require gradient")
        if gradient is None:
            if self.data.numel() != 1:
                raise RuntimeError('backward should be called only on a scalar (i.e. 1-element tensor) or with gradient w.r.t. the variable')
            gradient = self.data.new(1).fill_(1)
        self._execution_engine.run_backward(self, gradient)

    def __repr__(self):
        if self.dirty:
            return 'Variable used in an in-place operation'
        return 'Variable containing:' + self.data.__repr__()

    def register_hook(self, name, hook):
        if self.volatile:
            raise RuntimeError('registering hook on a volatile variable')
        if not self.requires_grad:
            raise RuntimeError("registering hook on a variable that doesn't require gradient")
        self.creator.register_hook(name, hook, self)

    def remove_hook(self, name):
        if self.volatile:
            raise RuntimeError("volatile variables don't support hooks")
        self.creator.remove_hook(name)

    def contiguous_(self):
        self._data = self.data.contiguous()
        return self

    def type(self, t):
        if t != type(self.data):
            return Copy(t)(self)
        return self

    def add(self, other, inplace=False):
        if isinstance(other, Variable):
            return Add(inplace)(self, other)
        else:
            assert not torch.isTensor(other)
            return AddConstant(other, inplace)(self)

    def add_(self, other):
        return self.add(other, inplace=True)

    def sub(self, other):
        if isinstance(other, Variable):
            return Sub()(self, other)
        else:
            assert not torch.isTensor(other)
            return SubConstant(other)(self)

    def sub_(self, other):
        return self.add(other, inplace=True)

    def mul(self, other):
        if isinstance(other, Variable):
            return Mul()(self, other)
        else:
            assert not torch.isTensor(other)
            return MulConstant(other)(self)

    def mul_(self, other):
        if not isinstance(other, Variable) and not torch.isTensor(other):
            return MulConstant(other, inplace=True)(self)

    def div(self, other):
        if isinstance(other, Variable):
            return Div()(self, other)
        else:
            assert not torch.isTensor(other)
            return DivConstant(other)(self)

    def div_(self, other):
        if not isinstance(other, Variable) and not torch.isTensor(other):
            return DivConstant(other, inplace=True)(self)

    def pow(self, other):
        if isinstance(other, Variable):
            return Pow()(self, other)
        else:
            assert not torch.isTensor(other)
            return PowConstant(other)(self)

    def exp(self):
        return Exp()(self)

    def exp_(self):
        return Exp(inplace=True)(self)

    def log(self):
        return Log()(self)

    def log1p(self):
        return Log1p()(self)

    def neg(self):
        return Negate()(self)

    def neg_(self):
        return Negate(inplace=True)(self)

    def view(self, *sizes):
        return View(*sizes)(self)

    def t(self):
        return Transpose(0, 1)(self)

    def transpose(self, dim1, dim2):
        return Transpose(dim1, dim2)(self)

    def __add__(self, other):
        return self.add(other)
    __radd__ = __add__

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return SubConstant(other, sub_tensor=True)(self)

    def __mul__(self, other):
        return self.mul(other)
    __rmul__ = __mul__

    def __div__(self, other):
        return self.div(other)
    __truediv__ = __div__

    def __rdiv__(self, other):
        return DivConstant(other, div_by_tensor=True)(self)
    __rtruediv__ = __rdiv__

    def __pow__(self, other):
        return self.pow(other)

    def __rpow__(self, other):
        return PowConstant(other, tensor_power=True)(self)

    def __neg__(self):
        return Negate()(self)


from .leaf import Leaf
from .functions import *
