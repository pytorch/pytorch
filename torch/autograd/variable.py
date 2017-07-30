import sys
import torch
import torch._C as _C
from collections import OrderedDict
import torch.sparse as sparse
import torch.utils.hooks as hooks
import warnings
import weakref


class Variable(_C._VariableBase):
    """Wraps a tensor and records the operations applied to it.

    Variable is a thin wrapper around a Tensor object, that also holds
    the gradient w.r.t. to it, and a reference to a function that created it.
    This reference allows retracing the whole chain of operations that
    created the data. If the Variable has been created by the user, its grad_fn
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
        is_leaf: Boolean indicating if the Variable is a graph leaf (i.e
            if it was created by the user).
        grad_fn: Gradient function graph trace.

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
        'shape'
    }

    def __getattr__(self, name):
        if name in self._fallthrough_methods:
            return getattr(self.data, name)
        return object.__getattribute__(self, name)

    def __getitem__(self, key):
        if torch.is_tensor(key):
            key = Variable(key)  # auto-wrap tensors
        if isinstance(key, Variable):
            if type(key.data).__name__ == 'ByteTensor':
                return MaskedSelect.apply(self, key)
            elif type(key.data).__name__ == 'LongTensor':
                return IndexSelect.apply(self, 0, key)
            # else fall through and raise an error in Index
        return Index.apply(self, key)

    def __setitem__(self, key, value):
        if isinstance(key, Variable) and type(key.data).__name__ == 'ByteTensor':
            if isinstance(value, Variable):
                return MaskedScatter.apply(self, key, value, True)
            else:
                return MaskedFill.apply(self, key, value, True)
        else:
            return SetItem.apply(self, key, value)

    def __deepcopy__(self, memo):
        if not self.is_leaf:
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
        if not self.is_leaf:
            raise RuntimeError('__setstate__ can be only called on leaf variables')
        self.requires_grad, self.volatile, self._backward_hooks = state

    def __repr__(self):
        return 'Variable containing:' + self.data.__repr__()

    def __bool__(self):
        if self.data.numel() == 0:
            return False
        raise RuntimeError("bool value of Variable objects containing non-empty " +
                           torch.typename(self.data) + " is ambiguous")

    __nonzero__ = __bool__

    def backward(self, gradient=None, retain_graph=None, create_graph=None, retain_variables=None):
        """Computes the gradient of current variable w.r.t. graph leaves.

        The graph is differentiated using the chain rule. If the variable is
        non-scalar (i.e. its data has more than one element) and requires
        gradient, the function additionally requires specifying ``gradient``.
        It should be a tensor of matching type and location, that contains
        the gradient of the differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to
        zero them before calling it.

        Arguments:
            grad_variables (Tensor, Variable or None): Gradient w.r.t. the
                variable. If it is a tensor, it will be automatically converted
                to a Variable that is volatile unless ``create_graph`` is True.
                None values can be specified for scalar Variables or ones that
                don't require grad. If a None value would be acceptable then
                this argument is optional.
            retain_graph (bool, optional): If False, the graph used to compute
                the grads will be freed. Note that in nearly all cases setting
                this option to True is not needed and often can be worked around
                in a much more efficient way. Defaults to the value of
                ``create_graph``.
            create_graph (bool, optional): If true, graph of the derivative will
                be constructed, allowing to compute higher order derivative
                products. Defaults to False, unless ``gradient`` is a volatile
                Variable.
        """
        torch.autograd.backward(self, gradient, retain_graph, create_graph, retain_variables)

    def register_hook(self, hook):
        """Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        variable is computed. The hook should have the following signature::

            hook(grad) -> Variable or None

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
            if self.grad_fn is not None:
                self.grad_fn._register_hook_dict(self)
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def reinforce(self, reward):
        """Registers a reward obtained as a result of a stochastic process.

        Differentiating stochastic nodes requires providing them with reward
        value. If your graph contains any stochastic operations, you should
        call this function on their outputs. Otherwise an error will be raised.

        Parameters:
            reward(Tensor): Tensor with per-element rewards. It has to match
                the device location and shape of Variable's data.
        """
        if not isinstance(self.grad_fn, StochasticFunction):
            raise RuntimeError("reinforce() can be only called on outputs "
                               "of stochastic functions")
        self.grad_fn._reinforce(reward)

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
        result._grad_fn = None
        return result

    def detach_(self):
        """Detaches the Variable from the graph that created it, making it a
        leaf.
        """
        self._grad_fn = None
        self.requires_grad = False

    def retain_grad(self):
        """Enables .grad attribute for non-leaf Variables."""
        if self.grad_fn is None:  # no-op for leaves
            return
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Variable that has requires_grad=False")
        if hasattr(self, 'retains_grad'):
            return
        weak_self = weakref.ref(self)

        def retain_grad_hook(grad):
            var = weak_self()
            if var is None:
                return
            if var._grad is None:
                var._grad = grad.clone()
            else:
                var._grad = var._grad + grad

        self.register_hook(retain_grad_hook)
        self.retains_grad = True

    def contiguous(self):
        self.data = self.data.contiguous()
        return self

    def clone(self):
        return Clone.apply(self)

    def type(self, t):
        if t != type(self.data):
            return Type.apply(self, t)
        return self

    def type_as(self, t):
        if isinstance(t, Variable):
            t = t.data
        return self.type(type(t))

    def _get_type(self, name):
        module = torch._import_dotted_name(self.data.__module__)
        return getattr(module, name)

    def cuda(self, device_id=None, async=False):
        return CudaTransfer.apply(self, device_id, async)

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
            return Add.apply(self, other, inplace)
        else:
            assert not torch.is_tensor(other)
            return AddConstant.apply(self, other, inplace)

    def add(self, other):
        return self._add(other, False)

    def add_(self, other):
        return self._add(other, True)

    def _sub(self, other, inplace):
        if isinstance(other, Variable):
            return Sub.apply(self, other, inplace)
        else:
            assert not torch.is_tensor(other)
            return SubConstant.apply(self, other, inplace)

    def sub(self, other):
        return self._sub(other, False)

    def sub_(self, other):
        return self._sub(other, True)

    def mul(self, other):
        if isinstance(other, Variable):
            return Mul.apply(self, other)
        else:
            assert not torch.is_tensor(other)
            return MulConstant.apply(self, other)

    def mul_(self, other):
        if not isinstance(other, Variable) and not torch.is_tensor(other):
            return MulConstant.apply(self, other, True)
        raise RuntimeError("mul_ only supports scalar multiplication")

    def div(self, other):
        if isinstance(other, Variable):
            return Div.apply(self, other)
        else:
            assert not torch.is_tensor(other)
            return DivConstant.apply(self, other)

    def div_(self, other):
        if not isinstance(other, Variable) and not torch.is_tensor(other):
            return DivConstant.apply(self, other, True)
        raise RuntimeError("div_ only supports scalar multiplication")

    def pow(self, other):
        if isinstance(other, Variable):
            return Pow.apply(self, other)
        else:
            assert not torch.is_tensor(other)
            return PowConstant.apply(self, other)

    def exp(self):
        return Exp.apply(self)

    def exp_(self):
        return Exp.apply(self, True)

    def log(self):
        return Log.apply(self)

    def log1p(self):
        return Log1p.apply(self)

    def neg(self):
        return Negate.apply(self)

    def neg_(self):
        return Negate.apply(self, True)

    def tanh(self):
        return Tanh.apply(self)

    def tanh_(self):
        return Tanh.apply(self, True)

    def sigmoid(self):
        return Sigmoid.apply(self)

    def sigmoid_(self):
        return Sigmoid.apply(self, True)

    def sin(self):
        return Sin.apply(self)

    def cos(self):
        return Cos.apply(self)

    def tan(self):
        return Tan.apply(self)

    def asin(self):
        return Asin.apply(self)

    def acos(self):
        return Acos.apply(self)

    def atan(self):
        return Atan.apply(self)

    def atan2(self, x):
        return Atan2.apply(self, x)

    def sinh(self):
        return Sinh.apply(self)

    def cosh(self):
        return Cosh.apply(self)

    def abs(self):
        return Abs.apply(self)

    def clamp(self, min=None, max=None):
        if min is None and max is None:
            raise ValueError("clamp requires specifying at least one of "
                             "min and max arguments")
        elif min is None and max is not None:
            return CminConstant.apply(self, max)
        elif min is not None and max is None:
            return CmaxConstant.apply(self, min)
        else:
            return Clamp.apply(self, min, max)

    def reciprocal(self):
        return Reciprocal.apply(self)

    def floor(self):
        return Floor.apply(self)

    def ceil(self):
        return Ceil.apply(self)

    def frac(self):
        return Frac.apply(self)

    def sqrt(self):
        return Sqrt.apply(self)

    def round(self):
        return Round.apply(self)

    def sign(self):
        return Sign.apply(self)

    def trunc(self):
        return Trunc.apply(self)

    def fmod(self, value):
        return Fmod.apply(self, value)

    def remainder(self, value):
        return Remainder.apply(self, value)

    def lerp(self, tensor, weight):
        return Lerp.apply(self, tensor, weight)

    def rsqrt(self):
        return Rsqrt.apply(self)

    def sum(self, dim=None, keepdim=None):
        return Sum.apply(self, dim, keepdim)

    def prod(self, dim=None, keepdim=None):
        return Prod.apply(self, dim, keepdim)

    def mean(self, dim=None, keepdim=None):
        return Mean.apply(self, dim, keepdim)

    def max(self, dim=None, keepdim=None):
        if isinstance(dim, Variable):
            return Cmax.apply(self, dim)
        return Max.apply(self, dim, keepdim)

    def min(self, dim=None, keepdim=None):
        if isinstance(dim, Variable):
            return Cmin.apply(self, dim)
        return Min.apply(self, dim, keepdim)

    def mode(self, dim=None, keepdim=None):
        return Mode.apply(self, dim, keepdim)

    def median(self, dim=None, keepdim=None):
        return Median.apply(self, dim, keepdim)

    def kthvalue(self, k, dim=None, keepdim=None):
        return Kthvalue.apply(self, k, dim, keepdim)

    def sort(self, dim=None, descending=False):
        return Sort.apply(self, dim, descending, True)

    def topk(self, k, dim=None, largest=True, sorted=True):
        return Topk.apply(self, k, dim, largest, sorted, True)

    def view(self, *sizes):
        return View.apply(self, sizes)

    def view_as(self, tensor):
        return View.apply(self, tensor.size())

    def split(self, split_size, dim=0):
        return torch.split(self, split_size, dim)

    def repeat(self, *repeats):
        if len(repeats) == 1 and isinstance(repeats[0], torch.Size):
            repeats = repeats[0]
        else:
            repeats = torch.Size(repeats)
        return Repeat.apply(self, repeats)

    def cumsum(self, dim):
        return Cumsum.apply(self, dim)

    def cumprod(self, dim):
        return Cumprod.apply(self, dim)

    def unfold(self, dim, size, step):
        return Unfold.apply(self, dim, size, step)

    def var(self, dim=None, keepdim=None, unbiased=True):
        keepdim_ = False if keepdim is None else keepdim
        mean = self.mean(dim, keepdim)
        if dim is None:
            mean = mean.view(*(1 for s in self.size()))
        # we could just set keepdim to True, but this preserves some fidelity
        elif keepdim_ is False and self.dim() != 1:
            mean = mean.unsqueeze(dim)
        mean_expanded = mean.expand_as(self)
        zero_centered = self.sub(mean_expanded)
        var = zero_centered.mul(zero_centered).sum(dim, keepdim=keepdim_)
        numel = self.numel() if dim is None else self.size(dim)
        return var.div(numel - int(unbiased))

    def std(self, dim=None, keepdim=None, unbiased=True):
        return self.var(dim, keepdim, unbiased).sqrt()

    def renorm(self, p, dim, maxnorm):
        t = self.transpose(dim, 0)
        flat = t.contiguous().view(self.size(0), -1)
        norms = flat.norm(p, 1, True)
        norms = norms.clamp(max=maxnorm).div(norms.add(1e-7))
        flat_out = flat.mul(norms.expand_as(flat))
        return flat_out.view(t.size()).transpose(dim, 0)

    def matmul(self, other):
        return torch.matmul(self, other)

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
        return cls.apply(*(args[:1] + args[-2:] + (alpha, beta, inplace)))

    def _blas(self, cls, args, inplace):
        return self._static_blas(cls, (self,) + args, inplace)

    def mm(self, matrix):
        output = Variable(self.data.new(self.data.size(0), matrix.data.size(1)))
        return Addmm.apply(output, self, matrix, 0, 1, True)

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
        return Resize.apply(self, sizes)

    def resize_as(self, variable):
        return Resize.apply(self, variable.size())

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
        return Dot.apply(self, other)

    def _addcop(self, op, args, inplace):
        if len(args) == 3:
            # args == [scale, tensor1, tensor2]
            return op.apply(self, args[1], args[2], args[0], inplace)
        else:
            # args == [tensor1, tensor2]
            return op.apply(self, args[0], args[1], 1.0, inplace)

    def addcmul(self, *args):
        return self._addcop(Addcmul, args, False)

    def addcdiv(self, *args):
        return self._addcop(Addcdiv, args, False)

    def addcmul_(self, *args):
        return self._addcop(Addcmul, args, True)

    def addcdiv_(self, *args):
        return self._addcop(Addcdiv, args, True)

    def norm(self, p=2, dim=None, keepdim=None):
        return Norm.apply(self, p, dim, keepdim)

    def dist(self, tensor, p=2):
        return Norm.apply(self - tensor, p)

    def index_add(self, dim, index, tensor):
        return IndexAdd.apply(self, dim, index, tensor)

    def _advanced_index_add(self, index, tensor):
        return AdvancedIndexAdd.apply(self, index, tensor)

    def index_add_(self, dim, index, tensor):
        return IndexAdd.apply(self, dim, index, tensor, True)

    def index_copy(self, dim, index, tensor):
        return IndexCopy.apply(self, dim, index, tensor)

    def index_copy_(self, dim, index, tensor):
        return IndexCopy.apply(self, dim, index, tensor, True)

    def index_fill(self, dim, index, value):
        return IndexFill.apply(self, dim, index, value)

    def index_fill_(self, dim, index, value):
        return IndexFill.apply(self, dim, index, value, True)

    def index_select(self, dim, index):
        return IndexSelect.apply(self, dim, index)

    def gather(self, dim, index):
        return Gather.apply(self, dim, index)

    def scatter(self, dim, index, source):
        return Scatter.apply(self, dim, index, source)

    def scatter_(self, dim, index, source):
        return Scatter.apply(self, dim, index, source, True)

    def scatter_add(self, dim, index, source):
        return ScatterAdd.apply(self, dim, index, source)

    def scatter_add_(self, dim, index, source):
        return ScatterAdd.apply(self, dim, index, source, True)

    def masked_copy(self, mask, variable):
        warnings.warn("masked_copy is deprecated and renamed to masked_scatter, and will be removed in v0.3")
        return MaskedScatter.apply(self, mask, variable)

    def masked_copy_(self, mask, variable):
        warnings.warn("masked_copy_ is deprecated and renamed to masked_scatter_, and will be removed in v0.3")
        return MaskedScatter.apply(self, mask, variable, True)

    def masked_scatter(self, mask, variable):
        return MaskedScatter.apply(self, mask, variable)

    def masked_scatter_(self, mask, variable):
        return MaskedScatter.apply(self, mask, variable, True)

    def masked_fill(self, mask, value):
        return MaskedFill.apply(self, mask, value)

    def masked_fill_(self, mask, value):
        return MaskedFill.apply(self, mask, value, True)

    def masked_select(self, mask):
        return MaskedSelect.apply(self, mask)

    def expand(self, *sizes):
        return Expand.apply(self, sizes)

    def expand_as(self, tensor):
        return Expand.apply(self, (tensor.size(),))

    def t(self):
        if self.dim() != 2:
            raise RuntimeError("t() expects a 2D Variable, but self is {}D".format(self.dim()))
        return Transpose.apply(self, 0, 1)

    def transpose(self, dim1, dim2):
        return Transpose.apply(self, dim1, dim2)

    def select(self, dim, _index):
        dim = dim if dim >= 0 else dim + self.dim()
        index = tuple(slice(None, None) for _ in range(dim)) + (_index,)
        return Index.apply(self, index)

    def narrow(self, dim, start_index, length):
        dim = dim if dim >= 0 else dim + self.dim()
        index = tuple(slice(None, None) for _ in range(dim)) + \
            (slice(start_index, start_index + length),)
        return Index.apply(self, index)

    def chunk(self, num_chunks, dim=0):
        return Chunk.apply(self, num_chunks, dim)

    def squeeze(self, dim=None):
        return Squeeze.apply(self, dim)

    def squeeze_(self, dim=None):
        return Squeeze.apply(self, dim, True)

    def unsqueeze(self, dim):
        return Unsqueeze.apply(self, dim)

    def permute(self, *permutation):
        return Permute.apply(self, permutation)

    def diag(self, diagonal=0):
        return Diag.apply(self, diagonal)

    def tril(self, diagonal=0):
        return Tril.apply(self, diagonal)

    def triu(self, diagonal=0):
        return Triu.apply(self, diagonal)

    def trace(self):
        return Trace.apply(self)

    def cross(self, other, dim=-1):
        return Cross.apply(self, other)

    def inverse(self):
        return Inverse.apply(self)

    def gesv(self, a):
        return Gesv.apply(self, a)

    def multinomial(self, num_samples=1, replacement=False):
        return Multinomial(num_samples, replacement)(self)

    def bernoulli(self):
        return Bernoulli()(self)

    def eq(self, other):
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Eq.apply(self, other)

    def ne(self, other):
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Ne.apply(self, other)

    def gt(self, other):
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Gt.apply(self, other)

    def ge(self, other):
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Ge.apply(self, other)

    def lt(self, other):
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Lt.apply(self, other)

    def le(self, other):
        assert not torch.is_tensor(other), "can't compare Variable and tensor"
        return Le.apply(self, other)

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
        return SubConstant.apply(other, self)

    def __mul__(self, other):
        return self.mul(other)
    __rmul__ = __mul__

    def __imul__(self, other):
        return self.mul_(other)

    def __matmul__(self, other):
        if not isinstance(other, Variable):
            return NotImplemented
        return self.matmul(other)

    def __div__(self, other):
        return self.div(other)
    __truediv__ = __div__

    def __rdiv__(self, other):
        return DivConstant.apply(other, self)
    __rtruediv__ = __rdiv__

    def __idiv__(self, other):
        return self.div_(other)

    def __pow__(self, other):
        return self.pow(other)

    def __ipow__(self, other):
        raise NotImplementedError("in-place pow not implemented")

    def __rpow__(self, other):
        return PowConstant.apply(other, self)

    def __neg__(self):
        return Negate.apply(self)

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
            return Concat.apply(dim, *iterable)

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
            return cls.apply(*(tensors + (alpha, beta, inplace)))

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


from ._functions import *
from torch._C import _ImperativeEngine as ImperativeEngine
Variable._execution_engine = ImperativeEngine()
