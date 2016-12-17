import torch._C as _C
from collections import OrderedDict

from .functions import *


class Variable(_C._VariableBase):

    _fallthrough_methods = {
        'size',
        'stride',
        'nelement',
        'ndimension',
        'element_size',
        'is_contiguous',
        'is_same_size',
        'is_set_to',
        'is_signed',
        'numel',
        'dim',
        'get_device',
        'is_cuda',
    }

    @property
    def grad(self):
        if self.requires_grad:
            # TODO: this won't have to be zeroed in the future
            self._grad = self._grad or self.data.new(self.data.size()).zero_()
        return self._grad

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        if self.creator is not None:
            if value is False:
                hint = (" If you want to use a computed variable in a subgraph "
                    "that doesn't require differentiation use "
                    "var_no_grad = var.no_grad().")
            else:
                hint = ''
            raise RuntimeError("you can only change requires_grad flags of "
                    "leaf variables." + hint)
        self._requires_grad = value

    def __getattr__(self, name):
        if name in self._fallthrough_methods:
            return getattr(self.data, name)
        raise AttributeError(name)

    def __getitem__(self, key):
        if (isinstance(key, Variable) and
            type(key.data).__name__ == 'ByteTensor'):
            return MaskedSelect()(self, key)
        return Index(key)(self)

    def __setitem__(self, key, value):
        if (isinstance(key, Variable) and
            type(key.data).__name__ == 'ByteTensor'):
            if isinstance(value, Variable):
                return MaskedCopy(inplace=True)(self, key, value)
            else:
                return MaskedFill(value, inplace=True)(self, key)
        else:
            if isinstance(value, Variable):
                return SetItem(key)(self, value)
            else:
                return SetItem(key, value)(self)

    def __iter__(self):
        return iter(map(lambda i: self[i], range(self.size(0))))

    def __deepcopy__(self, memo):
        if self.creator is None:
            return Variable(self.data.clone(), requires_grad=self.requires_grad,
                    volatile=self.volatile)
        raise RuntimeError("Only Variables created explicitly by the user "
                "(graph leaves) support the deepcopy protocol at the moment")

    def backward(self, gradient=None, retain_variables=False):
        if self.volatile:
            raise RuntimeError('calling backward on a volatile variable')
        if gradient is None and self.requires_grad:
            if self.data.numel() != 1:
                raise RuntimeError('backward should be called only on a scalar (i.e. 1-element tensor) or with gradient w.r.t. the variable')
            gradient = self.data.new(1).fill_(1)
        self._execution_engine.run_backward(self, gradient, retain_variables)

    def __repr__(self):
        return 'Variable containing:' + self.data.__repr__()

    def register_hook(self, name, hook):
        if self.volatile:
            raise RuntimeError("registering hook on a volatile variable")
        if not self.requires_grad:
            raise RuntimeError("registering hook on a variable that doesn't require gradient")
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.creator is not None:
                self.creator._register_hook_dict(self)
        assert name not in self._backward_hooks, \
            "Trying to register a second hook with name {}".format(name)
        self._backward_hooks[name] = hook

    def remove_hook(self, name):
        if self.volatile:
            raise RuntimeError("volatile variables don't support hooks")
        assert self._backward_hooks and name in self._backward_hooks, \
            "Trying to remove an inexistent hook with name {}".format(name)
        del self._backward_hooks[name]

    def _do_backward(self, grad_output, retain_variables):
        assert len(grad_output) == 1
        assert self._version == 0 and self.creator is None, \
            "leaf variable was used in an inplace operation"
        unpacked_grad = grad_output[0]
        if self._backward_hooks:
            for hook in self._backward_hooks.values():
                result = hook(unpacked_grad)
                if result is not None:
                    unpacked_grad = result
        self.grad.add_(unpacked_grad)
        return tuple()

    def reinforce(self, reward):
        if not isinstance(self.creator, StochasticFunction):
            raise RuntimeError("reinforce() can be only called on outputs "
                    "of stochastic functions")
        self.creator._reinforce(reward)

    def no_grad(self):
        return NoGrad()(self)

    def contiguous(self):
        self.data = self.data.contiguous()
        return self

    def clone(self):
        return Clone()(self)

    def type(self, t):
        if t != type(self.data):
            return Type(t)(self)
        return self

    def _get_type(self, name):
        module = torch._import_dotted_name(self.data.__module__)
        return getattr(module, name)

    def cuda(self, device_id=None, async=False):
        return CudaTransfer(device_id, async)(self)

    def cpu(self):
        return self.type(getattr(torch, type(self.data).__name__))

    def double(self):
        return self.type(self._get_type('DoubleTensor'))

    def float(self):
        return self.type(self._get_type('FloatTensor'))

    def long(self):
        return self.type(self._get_type('LongTensor'))

    def int(self):
        return self.type(self._get_type('IntTensor'))

    def short(self):
        return self.type(self._get_type('ShortTensor'))

    def char(self):
        return self.type(self._get_type('CharTensor'))

    def byte(self):
        return self.type(self._get_type('ByteTensor'))

    def _add(self, other, inplace):
        if isinstance(other, Variable):
            return Add(inplace)(self, other)
        else:
            assert not torch.is_tensor(other)
            return AddConstant(other, inplace)(self)

    def add(self, other):
        return self._add(other, False)

    def add_(self, other):
        return self._add(other, True)

    def _sub(self, other, inplace):
        if isinstance(other, Variable):
            return Sub(inplace=inplace)(self, other)
        else:
            assert not torch.is_tensor(other)
            return SubConstant(other, inplace=inplace)(self)

    def sub(self, other):
        return self._sub(other, False)

    def sub_(self, other):
        return self._sub(other, True)

    def mul(self, other):
        if isinstance(other, Variable):
            return Mul()(self, other)
        else:
            assert not torch.is_tensor(other)
            return MulConstant(other)(self)

    def mul_(self, other):
        if not isinstance(other, Variable) and not torch.is_tensor(other):
            return MulConstant(other, inplace=True)(self)
        raise RuntimeError("mul_ only supports scalar multiplication")

    def div(self, other):
        if isinstance(other, Variable):
            return Div()(self, other)
        else:
            assert not torch.is_tensor(other)
            return DivConstant(other)(self)

    def div_(self, other):
        if not isinstance(other, Variable) and not torch.is_tensor(other):
            return DivConstant(other, inplace=True)(self)
        raise RuntimeError("div_ only supports scalar multiplication")

    def pow(self, other):
        if isinstance(other, Variable):
            return Pow()(self, other)
        else:
            assert not torch.is_tensor(other)
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

    def tanh(self):
        return Tanh()(self)

    def tanh_(self):
        return Tanh(True)(self)

    def sigmoid(self):
        return Sigmoid()(self)

    def sigmoid_(self):
        return Sigmoid(True)(self)

    def sin(self):
        return Sin()(self)

    def cos(self):
        return Cos()(self)

    def tan(self):
        return Tan()(self)

    def asin(self):
        return Asin()(self)

    def acos(self):
        return Acos()(self)

    def atan(self):
        return Atan()(self)

    def sinh(self):
        return Sinh()(self)

    def cosh(self):
        return Cosh()(self)

    def abs(self):
        return Abs()(self)

    def clamp(self, min_val, max_val):
        return Clamp(min_val, max_val)(self)

    def cinv(self):
        return Cinv()(self)

    def cmax(self, other):
        if isinstance(other, Variable):
            return Cmax()(self, other)
        else:
            return CmaxConstant(other)(self)

    def cmin(self, other):
        if isinstance(other, Variable):
            return Cmin()(self, other)
        else:
            return CminConstant(other)(self)

    def floor(self):
        return Floor()(self)

    def ceil(self):
        return Ceil()(self)

    def frac(self):
        return Frac()(self)

    def sqrt(self):
        return Sqrt()(self)

    def round(self):
        return Round()(self)

    def sign(self):
        return Sign()(self)

    def trunc(self):
        return Trunc()(self)

    def floor(self):
        return Floor()(self)

    def ceil(self):
        return Ceil()(self)

    def fmod(self, value):
        return Fmod(value)(self)

    def remainder(self, value):
        return Remainder(value)(self)

    def lerp(self, tensor, weight):
        return Lerp(weight)(self, tensor)

    def rsqrt(self):
        return Rsqrt()(self)

    def sum(self, dim=None):
        return Sum(dim)(self)

    def prod(self, dim=None):
        return Prod(dim)(self)

    def mean(self, dim=None):
        return Mean(dim)(self)

    def max(self, dim=None):
        return Max(dim)(self)

    def min(self, dim=None):
        return Min(dim)(self)

    def mode(self, dim):
        return Mode(dim)(self)

    def median(self, dim):
        return Median(dim)(self)

    def kthvalue(self, dim):
        return Kthvalue(dim)(self)

    def sort(self, dim=None, descending=False):
        return Sort(dim, descending)(self)

    def topk(self, k, dim=None, largest=True, sorted=True):
        return Topk(k, dim, largest, sorted)(self)

    def view(self, *sizes):
        return View(*sizes)(self)

    def view_as(self, tensor):
        return View(*tensor.size())(self)

    @staticmethod
    def _static_blas(cls, args, inplace):
        num_args = len(args)
        alpha = beta = 1
        if num_args > 5:
            raise RuntimeError("too many args")
        if num_args == 5:
            alpha, beta = args[1:3]
        if num_args == 4:
            alpha = args[1]
        return cls(alpha, beta, inplace)(*(args[:1] + args[-2:]))

    def _blas(self, cls, args, inplace):
        return self._static_blas(cls, (self,) + args, inplace)

    def mm(self, matrix):
        output = Variable(self.data.new(self.data.size(0), matrix.data.size(1)))
        return self._static_blas(Addmm, (output, 0, 1, self, matrix), False)

    def bmm(self, batch):
        output = Variable(self.data.new(self.data.size(0), self.data.size(1),
                batch.data.size(2)))
        return self._static_blas(Baddbmm, (output, 0, 1, self, batch), False)

    def mv(self, vector):
        output = Variable(self.data.new(self.data.size(0)))
        return self._static_blas(Addmv, (output, 0, 1, self, vector), False)

    def ger(self, vector):
        output = Variable(self.data.new(self.data.size(0), vector.data.size(0)))
        return self._static_blas(Addr, (output, 0, 1, self, vector), False)

    def resize(self, *sizes):
        return Resize(*sizes)(self)

    def resize_as(self, variable):
        return Resize(*variable.size())(self)

    def addmm(self, *args):
        return self._blas(Addmm, args, False)

    def addmm_(self, *args):
        return self._blas(Addmm, args, True)

    def addbmm(self, *args):
        return self._blas(Addbmm, args, False)

    def addbmm_(self, *args):
        return self._blas(Addbmm, args, True)

    def baddbmm(self, *args):
        return self._blas(Baddbmm, args, False)

    def baddbmm_(self, *args):
        return self._blas(Baddbmm, args, True)

    def addmv(self, *args):
        return self._blas(Addmv, args, False)

    def addmv_(self, *args):
        return self._blas(Addmv, args, True)

    def addr(self, *args):
        return self._blas(Addr, args, False)

    def addr_(self, *args):
        return self._blas(Addr, args, True)

    def dot(self, other):
        return Dot()(self, other)

    def _addcop(self, op, args):
        if len(args) == 3:
            # scale, tensor1, tensor2
            return op(args[0])(self, *args[1:])
        else:
            # tensor1, tensor2
            return op()(self, *args)

    def addcmul(self, *args):
        return self._addcop(Addcmul, args)

    def addcdiv(self, *args):
        return self._addcop(Addcdiv, args)

    def norm(self, norm_type=2, dim=None):
        return Norm(norm_type, dim)(self)

    def dist(self, tensor, norm_type=2):
        return Norm(norm_type)(self - tensor)

    def index_add(self, dim, index, tensor):
        return IndexAdd(dim)(self, index, tensor)

    def index_add_(self, dim, index, tensor):
        return IndexAdd(dim, True)(self, index, tensor)

    def index_copy(self, dim, index, tensor):
        return IndexCopy(dim)(self, index, tensor)

    def index_copy_(self, dim, index, tensor):
        return IndexCopy(dim, True)(self, index, tensor)

    def index_fill(self, dim, index, value):
        return IndexFill(dim, value)(self, index)

    def index_fill_(self, dim, index, value):
        return IndexFill(dim, value, True)(self, index)

    def index_select(self, dim, index):
        return IndexSelect(dim)(self, index)

    def masked_copy(self, mask, variable):
        return MaskedCopy()(self, mask, variable)

    def masked_copy_(self, mask, variable):
        return MaskedCopy(True)(self, mask, variable)

    def masked_fill(self, mask, value):
        return MaskedFill(value)(self, mask)

    def masked_fill_(self, mask, value):
        return MaskedFill(value, True)(self, mask)

    def masked_select(self, mask):
        return MaskedSelect()(self, mask)

    def expand(self, *sizes):
        if isinstance(sizes[0], torch.Size):
            if len(sizes) > 1:
                raise ValueError("expand expects a several ints or a single "
                        "torch.Size argument")
            sizes = sizes[0]
        return Expand(sizes)(self)

    def expand_as(self, tensor):
        return Expand(tensor.size())(self)

    def t(self):
        return Transpose(0, 1)(self)

    def transpose(self, dim1, dim2):
        return Transpose(dim1, dim2)(self)

    def select(self, dim, _index):
        index = tuple(slice(None, None) for _ in range(dim)) + (_index,)
        return Index(index)(self)

    def narrow(self, dim, start_index, length):
        index = tuple(slice(None, None) for _ in range(dim)) + \
                    (slice(start_index, start_index+length),)

        return Index(index)(self)

    def chunk(self, num_chunks, dim=0):
        return Chunk(num_chunks, dim)(self)

    def squeeze(self, dim=None):
        return Squeeze(dim)(self)

    def unsqueeze(self, dim):
        return Unsqueeze(dim)(self)

    def permute(self, *permutation):
        return Permute(permutation)(self)

    def diag(self, diagonal_idx=0):
        return Diag(diagonal_idx)(self)

    def tril(self, diagonal_idx=0):
        return Tril(diagonal_idx)(self)

    def triu(self, diagonal_idx=0):
        return Triu(diagonal_idx)(self)

    def multinomial(self, num_samples=1, with_replacement=False):
        return Multinomial(num_samples, with_replacement)(self)

    def bernoulli(self):
        return Bernoulli()(self)

    def __add__(self, other):
        return self.add(other)
    __radd__ = __add__

    def __iadd__(self, other):
        return self.add_(other)

    def __sub__(self, other):
        return self.sub(other)

    def __isub__(self, other):
        return self.sub_(other)

    def __rsub__(self, other):
        return SubConstant(other, sub_tensor=True)(self)

    def __mul__(self, other):
        return self.mul(other)
    __rmul__ = __mul__

    def __imul__(self, other):
        return self.mul_(other)

    def __div__(self, other):
        return self.div(other)
    __truediv__ = __div__

    def __rdiv__(self, other):
        return DivConstant(other, div_by_tensor=True)(self)
    __rtruediv__ = __rdiv__

    def __idiv__(self, other):
        return self.div_(other)

    def __pow__(self, other):
        return self.pow(other)

    def __ipow__(self, other):
        raise NotImplementedError("in-place pow not implemented")

    def __rpow__(self, other):
        return PowConstant(other, tensor_power=True)(self)

    def __neg__(self):
        return Negate()(self)

    class _torch(object):

        @staticmethod
        def cat(iterable, dim=0):
            return Concat(dim)(*iterable)

        @staticmethod
        def normal(means, stddev=1):
            if isinstance(stddev, Variable):
                return Normal()(means, stddev)
            else:
                return Normal(stddev)(means)

        @staticmethod
        def _blas(cls, args, inplace):
            num_args = len(args)
            alpha = beta = 1
            if num_args > 5:
                raise RuntimeError("too many args")
            if num_args == 5:
                alpha, beta = args[0], args[2]
                tensors = args[1:2] + args[3:]
            elif num_args == 4:
                alpha = args[0]
                tensors = args[1:]
            else:
                tensors = args
            return cls(alpha, beta, inplace)(*tensors)

        @classmethod
        def addmm(cls, *args):
            return cls._blas(Addmm, args, False)

        @classmethod
        def addbmm(cls, *args):
            return cls._blas(Addbmm, args, False)

        @classmethod
        def baddbmm(cls, *args):
            return cls._blas(Baddbmm, args, False)

        @classmethod
        def addmv(cls, *args):
            return cls._blas(Addmv, args, False)

        @classmethod
        def addr(cls, *args):
            return cls._blas(Addr, args, False)


for method in dir(Variable):
    # This will also wrap some methods that normally aren't part of the
    # funcitonal interface, but we don't care, as they won't ever be used
    if method.startswith('_') or method.endswith('_'):
        continue
    if hasattr(Variable._torch, method):
        continue
    as_static = staticmethod(getattr(Variable, method))
    setattr(Variable._torch, method, as_static)


from .engine import ImperativeEngine
Variable._execution_engine = ImperativeEngine()
