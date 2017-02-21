import sys
import torch._C as _C
from collections import OrderedDict
import torch.utils.hooks as hooks

from ._functions import *


class Variable(_C._VariableBase):
    """Wraps a tensor and records the operations applied to it.

    Variable is a thin wrapper around a Tensor object, that also holds
    the gradient w.r.t. to it, and a reference to a function that created it.
    This reference allows retracing the whole chain of operations that
    created the data. If the Variable has been created by the user, its creator
    will be ``None`` and we call such objects *leaf* Variables.

    Since autograd only supports scalar valued function differentiation, grad
    size always matches the data size. Also, grad is normally only allocated
    for leaf variables, and will be always zero otherwise.

    Attributes:
        data: Wrapped tensor of any type.
        grad: Variable holding the gradient of type and location matching
            the ``.data``.  This attribute is lazily allocated and can't
            be reassigned.
        requires_grad: Boolean indicating whether the Variable has been
            created by a subgraph containing any Variable, that requires it.
            See :ref:`excluding-subgraphs` for more details.
            Can be changed only on leaf Variables.
        volatile: Boolean indicating that the Variable should be used in
            inference mode, i.e. don't save the history. See
            :ref:`excluding-subgraphs` for more details.
            Can be changed only on leaf Variables.
        creator: Function of which the variable was an output. For leaf
            (user created) variables it's ``None``. Read-only attribute.

    Parameters:
        data (any tensor class): Tensor to wrap.
        requires_grad (bool): Value of the requires_grad flag. **Keyword only.**
        volatile (bool): Value of the volatile flag. **Keyword only.**
    """

    _fallthrough_methods = {
        'size',
        'stride',
        'nelement',
        'ndimension',
        'element_size',
        'is_contiguous',
        'is_set_to',
        'is_signed',
        'numel',
        'dim',
        'get_device',
        'is_cuda',
    }

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

    def __deepcopy__(self, memo):
        if self.creator is not None:
            raise RuntimeError("Only Variables created explicitly by the user "
                               "(graph leaves) support the deepcopy protocol at the moment")
        result = type(self)(self.data.clone())
        result.requires_grad = self.requires_grad
        result.volatile = self.volatile
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto):
        state = (self.requires_grad, self.volatile, self._backward_hooks)
        if proto > 1:
            return type(self), (self.data,), state
        if sys.version_info[0] == 2:
            from copy_reg import __newobj__
        else:
            from copyreg import __newobj__
        return __newobj__, (type(self), self.data), state

    def __setstate__(self, state):
        if len(state) == 5:
            # legacy serialization of Variable
            self.data = state[0]
            state = (state[3], state[4], state[2])
        if self.creator is not None:
            raise RuntimeError('__setstate__ can be only called on leaf variables')
        self.requires_grad, self.volatile, self._backward_hooks = state

    def __repr__(self):
        return 'Variable containing:' + self.data.__repr__()

    def backward(self, gradient=None, retain_variables=False):
        """Computes the gradient of current variable w.r.t. graph leaves.

        The graph is differentiated using the chain rule. If the variable is
        non-scalar (i.e. its data has more than one element) and requires
        gradient, the function additionaly requires specifying ``gradient``.
        It should be a tensor of matching type and location, that containins
        the gradient of the differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to zero
        them before calling it.

        Arguments:
            gradient (Tensor): Gradient of the differentiated function
                w.r.t. the data. Required only if the data has more than one
                element. Type and location should match these of ``self.data``.
            retain_variables (bool): If ``True``, buffers necessary for computing
                gradients won't be freed after use. It is only necessary to
                specify ``True`` if you want to differentiate some subgraph multiple
                times (in some cases it will be much more efficient to use
                `autograd.backward`).
        """
        if self.volatile:
            raise RuntimeError('calling backward on a volatile variable')
        if gradient is None and self.requires_grad:
            if self.data.numel() != 1:
                raise RuntimeError(
                    'backward should be called only on a scalar (i.e. 1-element tensor) '
                    'or with gradient w.r.t. the variable')
            gradient = self.data.new().resize_as_(self.data).fill_(1)
        self._execution_engine.run_backward((self,), (gradient,), retain_variables)

    def register_hook(self, hook):
        """Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        variable is computed. The hook should have the following signature::

            hook(grad) -> Tensor or None

        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        Example:
            >>> v = Variable(torch.Tensor([0, 0, 0]), requires_grad=True)
            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
            >>> v.backward(torch.Tensor([1, 1, 1]))
            >>> v.grad.data
             2
             2
             2
            [torch.FloatTensor of size 3]
            >>> h.remove()  # removes the hook
        """
        if self.volatile:
            raise RuntimeError("cannot register a hook on a volatile variable")
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a variable that "
                               "doesn't require gradient")
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.creator is not None:
                self.creator._register_hook_dict(self)
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[id(handle)] = hook
        return handle

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
        self.grad.data.add_(unpacked_grad)
        return tuple()

    def reinforce(self, reward):
        """Registers a reward obtained as a result of a stochastic process.

        Differentiating stochastic nodes requires providing them with reward
        value. If your graph contains any stochastic operations, you should
        call this function on their outputs. Otherwise an error will be raised.

        Parameters:
            reward(Tensor): Tensor with per-element rewards. It has to match
                the device location and shape of Variable's data.
        """
        if not isinstance(self.creator, StochasticFunction):
            raise RuntimeError("reinforce() can be only called on outputs "
                               "of stochastic functions")
        self.creator._reinforce(reward)

    def detach(self):
        """Returns a new Variable, detached from the current graph.

        Result will never require gradient. If the input is volatile, the output
        will be volatile too.

        .. note::

          Returned Variable uses the same data tensor, as the original one, and
          in-place modifications on either of them will be seen, and may trigger
          errors in correctness checks.
        """
        result = NoGrad()(self)  # this is needed, because it merges version counters
        result._creator = None
        return result

    def detach_(self):
        """Detaches the Variable from the graph that created it, making it a leaf."""
        self._creator = None
        self.requires_grad = False

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

    def half(self):
        return self.type(self._get_type('HalfTensor'))

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

    def is_same_size(self, other_var):
        return self.data.is_same_size(other_var.data)

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

    def clamp(self, min=None, max=None):
        if min is None and max is None:
            raise ValueError("clamp requires specifying at least one of "
                             "min and max arguments")
        elif min is None and max is not None:
            return CminConstant(max)(self)
        elif min is not None and max is None:
            return CmaxConstant(min)(self)
        else:
            return Clamp(min, max)(self)

    def reciprocal(self):
        return Reciprocal()(self)

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
        if isinstance(dim, Variable):
            return Cmax()(self, dim)
        return Max(dim)(self)

    def min(self, dim=None):
        if isinstance(dim, Variable):
            return Cmin()(self, dim)
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

    def split(self, split_size, dim=0):
        return torch.split(self, split_size, dim)

    def chunk(self, n_chunks, dim=0):
        return torch.chunk(self, n_chunks, dim)

    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], torch.Size):
            repeats = repeats[0]
        else:
            repeats = torch.Size(repeats)
        return Repeat(repeats)(self)

    def var(self, dim=None, unbiased=True):
        mean = self.mean(dim)
        if dim is None:
            mean = mean.view(*(1 for s in self.size()))
        mean_expanded = mean.expand_as(self)
        zero_centered = self.sub(mean_expanded)
        var = zero_centered.mul(zero_centered).sum(dim)
        numel = self.numel() if dim is None else self.size(dim)
        return var.div(numel - int(unbiased))

    def std(self, dim=None, unbiased=True):
        return self.var(dim, unbiased).sqrt()

    def renorm(self, norm_type, dim, maxnorm):
        t = self.transpose(dim, 0)
        flat = t.contiguous().view(self.size(0), -1)
        norms = flat.norm(norm_type, 1)
        norms = norms.clamp(max=maxnorm).div(norms.add(1e-7))
        flat_out = flat.mul(norms.expand_as(flat))
        return flat_out.view(t.size()).transpose(dim, 0)

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

    def gather(self, dim, index):
        return Gather(dim)(self, index)

    def scatter(self, dim, index, source):
        return Scatter(dim)(self, index, source)

    def scatter_(self, dim, index, source):
        return Scatter(dim, True)(self, index, source)

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
            (slice(start_index, start_index + length),)

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

    def eq(self, other):
        if isinstance(other, Variable):
            return Eq()(self, other)
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Eq(other)(self)

    def ne(self, other):
        if isinstance(other, Variable):
            return Ne()(self, other)
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Ne(other)(self)

    def gt(self, other):
        if isinstance(other, Variable):
            return Gt()(self, other)
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Gt(other)(self)

    def ge(self, other):
        if isinstance(other, Variable):
            return Ge()(self, other)
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Ge(other)(self)

    def lt(self, other):
        if isinstance(other, Variable):
            return Lt()(self, other)
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Lt(other)(self)

    def le(self, other):
        if isinstance(other, Variable):
            return Le()(self, other)
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Le(other)(self)

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

    def __matmul__(self, other):
        dim_self = self.dim()
        try:
            dim_other = other.dim()
        except AttributeError:  # not a Variable
            return NotImplemented
        if dim_self == 1 and dim_other == 1:
            return self.dot(other)
        if dim_self == 2 and dim_other == 1:
            return self.mv(other)
        if dim_self == 1 and dim_other == 2:
            return self.unsqueeze(0).mm(other).squeeze(0)
        elif dim_self == 2 and dim_other == 2:
            return self.mm(other)
        raise ValueError("both arguments to __matmul__ need to be 1D or 2D, "
                         "but they are {}D and {}D".format(dim_self, dim_other))

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

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(map(lambda i: self[i], range(self.size(0))))

    def __mod__(self, other):
        return self.remainder(other)

    def __eq__(self, other):
        return self.eq(other)

    def __ne__(self, other):
        return self.ne(other)

    def __lt__(self, other):
        return self.lt(other)

    def __le__(self, other):
        return self.le(other)

    def __gt__(self, other):
        return self.gt(other)

    def __ge__(self, other):
        return self.ge(other)

    def __hash__(self):
        return id(self)

    class _torch(object):

        @staticmethod
        def cat(iterable, dim=0):
            return Concat(dim)(*iterable)

        @staticmethod
        def normal(means, std=1):
            if isinstance(std, Variable):
                return Normal()(means, std)
            else:
                return Normal(std)(means)

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
