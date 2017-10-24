import sys
import torch
import torch._C as _C
from collections import OrderedDict
import torch.sparse as sparse
import torch.utils.hooks as hooks
import warnings
import weakref
from torch._six import imap


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
            gradient (Tensor, Variable or None): Gradient w.r.t. the
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

    def cuda(self, device=None, async=False):
        return CudaTransfer.apply(self, device, async)

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

    def prod(self, dim=None, keepdim=None):
        return Prod.apply(self, dim, keepdim)

    def view_as(self, tensor):
        return self.view(tensor.size())

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

    def var(self, dim=None, keepdim=False, unbiased=True):
        if dim is None:
            mean = self.mean().view(*(1 for s in self.size()))
        else:
            mean = self.mean(dim, keepdim)
            # we could just set keepdim to True, but this preserves some fidelity
            if keepdim is False and self.dim() != 1:
                mean = mean.unsqueeze(dim)
        mean_expanded = mean.expand_as(self)
        zero_centered = self.sub(mean_expanded)
        if dim is None:
            var = zero_centered.mul(zero_centered).sum()
        else:
            var = zero_centered.mul(zero_centered).sum(dim, keepdim=keepdim)
        numel = self.numel() if dim is None else self.size(dim)
        return var.div(numel - int(unbiased))

    def std(self, dim=None, keepdim=False, unbiased=True):
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

    def resize(self, *sizes):
        return Resize.apply(self, sizes)

    def resize_as(self, variable):
        return Resize.apply(self, variable.size())

    def norm(self, p=2, dim=None, keepdim=False):
        if dim is None:
            return super(Variable, self).norm(p)
        else:
            return super(Variable, self).norm(p, dim, keepdim)

    def index_add(self, dim, index, tensor):
        return self.clone().index_add_(dim, index, tensor)

    def _advanced_index_add(self, index, tensor):
        return AdvancedIndexAdd.apply(self, index, tensor)

    def index_copy(self, dim, index, tensor):
        return self.clone().index_copy_(dim, index, tensor)

    def index_fill(self, dim, index, value):
        return self.clone().index_fill_(dim, index, value)

    def scatter(self, dim, index, source):
        return self.clone().scatter_(dim, index, source)

    def scatter_add(self, dim, index, source):
        return self.clone().scatter_add_(dim, index, source)

    def masked_copy(self, mask, variable):
        warnings.warn("masked_copy is deprecated and renamed to masked_scatter, and will be removed in v0.3")
        return self.masked_scatter(mask, variable)

    def masked_copy_(self, mask, variable):
        warnings.warn("masked_copy_ is deprecated and renamed to masked_scatter_, and will be removed in v0.3")
        return self.masked_scatter_(mask, variable)

    def masked_scatter(self, mask, variable):
        return self.clone().masked_scatter_(mask, variable)

    def masked_fill(self, mask, value):
        return self.clone().masked_fill_(mask, value)

    def expand_as(self, tensor):
        return self.expand(tensor.size())

    def select(self, dim, _index):
        dim = dim if dim >= 0 else dim + self.dim()
        index = tuple(slice(None, None) for _ in range(dim)) + (_index,)
        return Index.apply(self, index)

    def permute(self, *permutation):
        return Permute.apply(self, permutation)

    def multinomial(self, num_samples=1, replacement=False):
        return Multinomial.apply(self, num_samples, replacement)

    def bernoulli(self):
        return Bernoulli.apply(self)

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
        return -self + other

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
        return self.reciprocal() * other
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
        # NB: we use 'imap' and not 'map' here, so that in Python 2 we get a
        # generator and don't eagerly perform all the indexes.  This could
        # save us work, and also helps keep trace ordering deterministic
        # (e.g., if you zip(*hiddens), the eager map will force all the
        # indexes of hiddens[0] before hiddens[1], while the generator
        # map will interleave them.)
        return iter(imap(lambda i: self[i], range(self.size(0))))

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
            return Normal.apply(means, std)

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
    # functional interface, but we don't care, as they won't ever be used
    if method.startswith('_') or method.endswith('_'):
        continue
    if hasattr(Variable._torch, method):
        continue
    as_static = staticmethod(getattr(Variable, method))
    setattr(Variable._torch, method, as_static)


from ._functions import *
from torch._C import _ImperativeEngine as ImperativeEngine
Variable._execution_engine = ImperativeEngine()
