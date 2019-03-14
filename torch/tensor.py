import sys
import numpy
import torch
import torch._C as _C
from collections import OrderedDict
import torch.utils.hooks as hooks
import warnings
import weakref
from torch._six import imap
from torch._C import _add_docstr
from numbers import Number


def _sort_args(*args):
    """Sorts Arguments so that the most derived subclass moved to the first argument"""
    index = 0
    for ind, arg in enumerate(args):
        if not torch.is_tensor(args[index]) or arg.__class__ in args[index].__class__.__subclasses__():
            index = ind
    return args[index:index + 1] + args[:index] + args[index + 1:]


def _call_ufunc(func, method, *args, **kwargs):
    inputs = _sort_args(*args)
    return inputs[0].__tensor_ufunc__(func, method, *inputs, **kwargs)


def _nonsense_(*args, **kwargs):
    return NotImplemented


def _nonsense(*args, **kwargs):
    inputs = _sort_args(*args)
    return inputs[0].__tensor_ufunc__(_nonsense_, "__call__", *inputs, **kwargs)


# NB: If you subclass Tensor, and want to share the subclassed class
# across processes, you must also update torch/multiprocessing/reductions.py
# to define a ForkingPickler serialization mode for the class.
#
# NB: If you add a new method to Tensor, you must update
# torch/__init__.py.in to add a type annotation for your method;
# otherwise, it will not show up in autocomplete.
class Tensor(torch._C._TensorBase):

    def __deepcopy__(self, memo):
        if not self.is_leaf:
            raise RuntimeError("Only Tensors created explicitly by the user "
                               "(graph leaves) support the deepcopy protocol at the moment")
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            if self.is_sparse:
                new_tensor = self.clone()
            else:
                new_storage = self.storage().__deepcopy__(memo)
                new_tensor = self.new()
                new_tensor.set_(new_storage, self.storage_offset(), self.size(), self.stride())
            memo[id(self)] = new_tensor
            new_tensor.requires_grad = self.requires_grad
            return new_tensor

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        torch.utils.hooks.warn_if_has_hooks(self)
        args = (self.storage(),
                self.storage_offset(),
                tuple(self.size()),
                self.stride(),
                self.requires_grad,
                OrderedDict())  # previously was self._backward_hooks
        return (torch._utils._rebuild_tensor_v2, args)

    def __setstate__(self, state):
        # Warning: this method is NOT called when you torch.load() a tensor;
        # that is managed by _rebuild_tensor_v2
        if not self.is_leaf:
            raise RuntimeError('__setstate__ can be only called on leaf Tensors')
        if len(state) == 4:
            # legacy serialization of Tensor
            self.set_(*state)
            return
        elif len(state) == 5:
            # legacy serialization of Variable
            self.data = state[0]
            state = (state[3], state[4], state[2])
        # The setting of _backward_hooks is expected to be a no-op.
        # See Note [Don't serialize hooks]
        self.requires_grad, _, self._backward_hooks = state

    def __repr__(self):
        # All strings are unicode in Python 3, while we have to encode unicode
        # strings in Python2. If we can't, let python decide the best
        # characters to replace unicode characters with.
        if sys.version_info > (3,):
            return torch._tensor_str._str(self)
        else:
            if hasattr(sys.stdout, 'encoding'):
                return torch._tensor_str._str(self).encode(
                    sys.stdout.encoding or 'UTF-8', 'replace')
            else:
                return torch._tensor_str._str(self).encode('UTF-8', 'replace')

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        r"""Computes the gradient of current tensor w.r.t. graph leaves.

        The graph is differentiated using the chain rule. If the tensor is
        non-scalar (i.e. its data has more than one element) and requires
        gradient, the function additionally requires specifying ``gradient``.
        It should be a tensor of matching type and location, that contains
        the gradient of the differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to
        zero them before calling it.

        Arguments:
            gradient (Tensor or None): Gradient w.r.t. the
                tensor. If it is a tensor, it will be automatically converted
                to a Tensor that does not require grad unless ``create_graph`` is True.
                None values can be specified for scalar Tensors or ones that
                don't require grad. If a None value would be acceptable then
                this argument is optional.
            retain_graph (bool, optional): If ``False``, the graph used to compute
                the grads will be freed. Note that in nearly all cases setting
                this option to True is not needed and often can be worked around
                in a much more efficient way. Defaults to the value of
                ``create_graph``.
            create_graph (bool, optional): If ``True``, graph of the derivative will
                be constructed, allowing to compute higher order derivative
                products. Defaults to ``False``.
        """
        torch.autograd.backward(self, gradient, retain_graph, create_graph)

    def register_hook(self, hook):
        r"""Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        Tensor is computed. The hook should have the following signature::

            hook(grad) -> Tensor or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        Example::

            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v.grad

             2
             4
             6
            [torch.FloatTensor of size (3,)]

            >>> h.remove()  # removes the hook
        """
        if not self.requires_grad:
            raise RuntimeError("cannot register a hook on a tensor that "
                               "doesn't require gradient")
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.grad_fn is not None:
                self.grad_fn._register_hook_dict(self)
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def reinforce(self, reward):
        def trim(str):
            return '\n'.join([line.strip() for line in str.split('\n')])

        raise RuntimeError(trim(r"""reinforce() was removed.
            Use torch.distributions instead.
            See https://pytorch.org/docs/master/distributions.html

            Instead of:

            probs = policy_network(state)
            action = probs.multinomial()
            next_state, reward = env.step(action)
            action.reinforce(reward)
            action.backward()

            Use:

            probs = policy_network(state)
            # NOTE: categorical is equivalent to what used to be called multinomial
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            next_state, reward = env.step(action)
            loss = -m.log_prob(action) * reward
            loss.backward()
        """))

    detach = _add_docstr(_C._TensorBase.detach, r"""
    Returns a new Tensor, detached from the current graph.

    The result will never require gradient.

    .. note::

      Returned Tensor shares the same storage with the original one.
      In-place modifications on either of them will be seen, and may trigger
      errors in correctness checks.
      IMPORTANT NOTE: Previously, in-place size / stride / storage changes
      (such as `resize_` / `resize_as_` / `set_` / `transpose_`) to the returned tensor
      also update the original tensor. Now, these in-place changes will not update the
      original tensor anymore, and will instead trigger an error.
      For sparse tensors:
      In-place indices / values changes (such as `zero_` / `copy_` / `add_`) to the
      returned tensor will not update the original tensor anymore, and will instead
      trigger an error.
    """)

    detach_ = _add_docstr(_C._TensorBase.detach_, r"""
    Detaches the Tensor from the graph that created it, making it a leaf.
    Views cannot be detached in-place.
    """)

    def retain_grad(self):
        r"""Enables .grad attribute for non-leaf Tensors."""
        if self.grad_fn is None:  # no-op for leaves
            return
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Tensor that has requires_grad=False")
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

    def is_pinned(self):
        r"""Returns true if this tensor resides in pinned memory"""
        storage = self.storage()
        return storage.is_pinned() if storage else False

    def is_shared(self):
        r"""Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        """
        return self.storage().is_shared()

    def share_memory_(self):
        r"""Moves the underlying storage to shared memory.

        This is a no-op if the underlying storage is already in shared memory
        and for CUDA tensors. Tensors in shared memory cannot be resized.
        """
        self.storage().share_memory_()
        return self

    def __reversed__(self):
        r"""Reverses the tensor along dimension 0."""
        if self.dim() == 0:
            return self
        else:
            return self.flip(0)

    def argmax(self, dim=None, keepdim=False):
        r"""See :func:`torch.argmax`"""
        return _call_ufunc(torch.argmax, "__call__", self, dim, keepdim)

    def argmin(self, dim=None, keepdim=False):
        r"""See :func:`torch.argmin`"""
        return _call_ufunc(torch.argmin, "__call__", self, dim, keepdim)

    def norm(self, p="fro", dim=None, keepdim=False, dtype=None):
        r"""See :func:`torch.norm`"""
        return _call_ufunc(torch.norm, "__call__", self, p, dim, keepdim, dtype=dtype)
        # return torch.norm(self, p, dim, keepdim, dtype=dtype)

    def potrf(self, upper=True):
        r"""See :func:`torch.cholesky`"""
        warnings.warn("torch.potrf is deprecated in favour of torch.cholesky and will be removed "
                      "in the next release. Please use torch.cholesky instead and note that the "
                      ":attr:`upper` argument in torch.cholesky defaults to ``False``.", stacklevel=2)
        return super(Tensor, self).cholesky(upper=upper)

    def pstrf(self, upper=True):
        r"""See :func:`torch.pstrf`"""
        warnings.warn("torch.pstrf is deprecated in favour of torch.cholesky and will be removed "
                      "in the next release.", stacklevel=2)
        return super(Tensor, self).pstrf(upper=upper)

    def potrs(self, u, upper=True):
        r"""See :func:`torch.cholesky_solve`"""
        warnings.warn("torch.potrs is deprecated in favour of torch.cholesky_solve and "
                      "will be removed in the next release. Please use torch.cholesky_solve instead "
                      "and note that the :attr:`upper` argument in torch.cholesky_solve defaults "
                      "to ``False``.", stacklevel=2)
        return super(Tensor, self).cholesky_solve(u, upper=upper)

    def stft(self, n_fft, hop_length=None, win_length=None, window=None,
             center=True, pad_mode='reflect', normalized=False, onesided=True):
        r"""See :func:`torch.stft`

        .. warning::
          This function changed signature at version 0.4.1. Calling with
          the previous signature may cause error or return incorrect result.
        """
        return _call_ufunc(torch.stft, "__call__", self,
                           n_fft, hop_length, win_length, window, center,
                          pad_mode, normalized, onesided)

    def resize(self, *sizes):
        warnings.warn("non-inplace resize is deprecated")
        from torch.autograd._functions import Resize
        new_tensor = Resize.apply(self, sizes)
        return torch.Tensor._make_subclass(self.__class__, new_tensor.data, new_tensor.requires_grad)

    def resize_as(self, tensor):
        warnings.warn("non-inplace resize_as is deprecated")
        from torch.autograd._functions import Resize
        new_tensor = Resize.apply(self, tensor.size())
        return torch.Tensor._make_subclass(self.__class__, new_tensor.data, new_tensor.requires_grad)

    def to_sparse(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.to_sparse, "__call__", *(self, *args), **kwargs)

    def to(self, *args, **kwargs):
        new_tensor = _C._TensorBase.to(*(self, *args), **kwargs)
        return torch.Tensor._make_subclass(self.__class__, new_tensor.data, new_tensor.requires_grad)

    def detach(self, *args, **kwargs):
        new_tensor = _C._TensorBase.detach(self)
        new_tensor = torch.Tensor._make_subclass(self.__class__, new_tensor.data, new_tensor.requires_grad)
        new_tensor._metadata = None
        return new_tensor

    def split(self, split_size, dim=0):
        r"""See :func:`torch.split`
        """
        if isinstance(split_size, int):
            return super(Tensor, self).split(split_size, dim)
        else:
            return super(Tensor, self).split_with_sizes(split_size, dim)

    def unique(self, sorted=True, return_inverse=False, dim=None):
        r"""Returns the unique scalar elements of the tensor as a 1-D tensor.

        See :func:`torch.unique`
        """
        if dim is not None:
            output, inverse_indices = _call_ufunc(torch._unique_dim, "__call__",
                self,
                sorted=sorted,
                return_inverse=return_inverse,
                dim=dim
            )
        else:
            output, inverse_indices = _call_ufunc(torch._unique, "__call__",
                self,
                sorted=sorted,
                return_inverse=return_inverse
            )
        if return_inverse:
            return output, inverse_indices
        else:
            return output

    def __abs__(self, *args, **kwargs):
        return _call_ufunc(torch.abs, "__call__", *(self, *args), **kwargs)

    def __add__(self, *args, **kwargs):
        return _call_ufunc(torch.add, "__call__", *(self, *args), **kwargs)

    def __mul__(self, *args, **kwargs):
        return _call_ufunc(torch.mul, "__call__", *(self, *args), **kwargs)

    def __and__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__and__, "__call__", *(self, *args), **kwargs)

    def __div__(self, *args, **kwargs):
        return _call_ufunc(torch.div, "__call__", *(self, *args), **kwargs)

    def __eq__(self, *args, **kwargs):
        return _call_ufunc(torch.eq, "__call__", *(self, *args), **kwargs)

    def __ge__(self, *args, **kwargs):
        return _call_ufunc(torch.ge, "__call__", *(self, *args), **kwargs)

    def __gt__(self, *args, **kwargs):
        return _call_ufunc(torch.gt, "__call__", *(self, *args), **kwargs)

    def __floordiv__(self, other):
        result = self / other
        if result.dtype.is_floating_point:
            result = result.trunc()
        return result

    def __ipow__(self, other):
        raise NotImplementedError("in-place pow not implemented")

    def __le__(self, *args, **kwargs):
        return _call_ufunc(torch.le, "__call__", *(self, *args), **kwargs)

    def __lt__(self, *args, **kwargs):
        return _call_ufunc(torch.lt, "__call__", *(self, *args), **kwargs)

    def __ne__(self, *args, **kwargs):
        return _call_ufunc(torch.ne, "__call__", *(self, *args), **kwargs)

    def __neg__(self, *args, **kwargs):
        return _call_ufunc(torch.neg, "__call__", *(self, *args), **kwargs)

    def __pow__(self, *args, **kwargs):
        return _call_ufunc(torch.pow, "__call__", *(self, *args), **kwargs)

    def __rdiv__(self, other):
        if self.dtype.is_floating_point:
            return self.reciprocal() * other
        else:
            return (self.double().reciprocal() * other).type_as(self)

    def __rfloordiv__(self, other):
        result = other / self
        if result.dtype.is_floating_point:
            result = result.trunc()
        return result

    def __rpow__(self, other):
        return self.new_tensor(other) ** self

    def __rsub__(self, *args, **kwargs):
        return _call_ufunc(torch.rsub, "__call__", *(self, *args), **kwargs)

    def nonsense(self, *args, **kwargs):
        return _nonsense(*(self, *args), **kwargs)

    multiply = __mul__
    __itruediv__ = _C._TensorBase.__idiv__
    __rtruediv__ = __rdiv__

    # auto generated

    def abs(self, *args, **kwargs):
        return _call_ufunc(torch.abs, "__call__", *(self, *args), **kwargs)

    def acos(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.acos, "__call__", *(self, *args), **kwargs)

    def add(self, *args, **kwargs):
        return _call_ufunc(torch.add, "__call__", *(self, *args), **kwargs)

    def addmv(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.addmv, "__call__", *(self, *args), **kwargs)

    def addr(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.addr, "__call__", *(self, *args), **kwargs)

    def all(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.all, "__call__", *(self, *args), **kwargs)

    def any(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.any, "__call__", *(self, *args), **kwargs)

    def as_strided(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.as_strided, "__call__", *(self, *args), **kwargs)

    def asin(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.asin, "__call__", *(self, *args), **kwargs)

    def atan(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.atan, "__call__", *(self, *args), **kwargs)

    def baddbmm(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.baddbmm, "__call__", *(self, *args), **kwargs)

    def bernoulli(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.bernoulli, "__call__", *(self, *args), **kwargs)

    def bincount(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.bincount, "__call__", *(self, *args), **kwargs)

    def bmm(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.bmm, "__call__", *(self, *args), **kwargs)

    def ceil(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.ceil, "__call__", *(self, *args), **kwargs)

    def clamp(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.clamp, "__call__", *(self, *args), **kwargs)

    def clamp_max(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.clamp_max, "__call__", *(self, *args), **kwargs)

    def clamp_min(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.clamp_min, "__call__", *(self, *args), **kwargs)

    def contiguous(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.contiguous, "__call__", *(self, *args), **kwargs)

    def cos(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.cos, "__call__", *(self, *args), **kwargs)

    def cosh(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.cosh, "__call__", *(self, *args), **kwargs)

    def cumsum(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.cumsum, "__call__", *(self, *args), **kwargs)

    def cumprod(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.cumprod, "__call__", *(self, *args), **kwargs)

    def det(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.det, "__call__", *(self, *args), **kwargs)

    def diag_embed(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.diag_embed, "__call__", *(self, *args), **kwargs)

    def diagflat(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.diagflat, "__call__", *(self, *args), **kwargs)

    def diagonal(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.diagonal, "__call__", *(self, *args), **kwargs)

    def div(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.div, "__call__", *(self, *args), **kwargs)

    def dot(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.dot, "__call__", *(self, *args), **kwargs)

    # def resize(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.resize_, "__call__", *(self, *args), **kwargs)

    def erf(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.erf, "__call__", *(self, *args), **kwargs)

    def erfc(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.erfc, "__call__", *(self, *args), **kwargs)

    def exp(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.exp, "__call__", *(self, *args), **kwargs)

    def expm1(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.expm1, "__call__", *(self, *args), **kwargs)

    def expand(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.expand, "__call__", *(self, *args), **kwargs)

    def expand_as(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.expand_as, "__call__", *(self, *args), **kwargs)

    def flatten(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.flatten, "__call__", *(self, *args), **kwargs)

    # def fill(self, *args, **kwargs):
    #     return _call_ufunc(torch.fill, "__call__", *(self, *args), **kwargs)

    def floor(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.floor, "__call__", *(self, *args), **kwargs)

    def ger(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.ger, "__call__", *(self, *args), **kwargs)

    def gesv(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.gesv, "__call__", *(self, *args), **kwargs)

    def fft(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.fft, "__call__", *(self, *args), **kwargs)

    def ifft(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.ifft, "__call__", *(self, *args), **kwargs)

    def rfft(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.rfft, "__call__", *(self, *args), **kwargs)

    def irfft(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.irfft, "__call__", *(self, *args), **kwargs)

    # def index(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.index, "__call__", *(self, *args), **kwargs)

    # def index_copy_(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.index_copy_, "__call__", *(self, *args), **kwargs)

    def index_put(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.index_put, "__call__", *(self, *args), **kwargs)

    def inverse(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.inverse, "__call__", *(self, *args), **kwargs)

    def isclose(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.isclose, "__call__", *(self, *args), **kwargs)

    def kthvalue(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.kthvalue, "__call__", *(self, *args), **kwargs)

    def log(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.log, "__call__", *(self, *args), **kwargs)

    def log10(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.log10, "__call__", *(self, *args), **kwargs)

    def log1p(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.log1p, "__call__", *(self, *args), **kwargs)

    def log2(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.log2, "__call__", *(self, *args), **kwargs)

    def logdet(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.logdet, "__call__", *(self, *args), **kwargs)

    def log_softmax(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.log_softmax, "__call__", *(self, *args), **kwargs)

    def logsumexp(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.logsumexp, "__call__", *(self, *args), **kwargs)

    def matmul(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.matmul, "__call__", *(self, *args), **kwargs)

    def matrix_power(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.matrix_power, "__call__", *(self, *args), **kwargs)

    def max(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.max, "__call__", *(self, *args), **kwargs)\

    # def max_values(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.max_values, "__call__", *(self, *args), **kwargs)

    def mean(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.mean, "__call__", *(self, *args), **kwargs)

    def median(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.median, "__call__", *(self, *args), **kwargs)

    def min(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.min, "__call__", *(self, *args), **kwargs)

    # def min_values(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.min_values, "__call__", *(self, *args), **kwargs)

    def mm(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.mm, "__call__", *(self, *args), **kwargs)

    def mode(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.mode, "__call__", *(self, *args), **kwargs)

    def mul(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.mul, "__call__", *(self, *args), **kwargs)

    def mv(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.mv, "__call__", *(self, *args), **kwargs)

    def mvlgamma(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.mvlgamma, "__call__", *(self, *args), **kwargs)

    def narrow_copy(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.narrow_copy, "__call__", *(self, *args), **kwargs)

    def narrow(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.narrow, "__call__", *(self, *args), **kwargs)

    def permute(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.permute, "__call__", *(self, *args), **kwargs)

    # def pin_memory(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.pin_memory, "__call__", *(self, *args), **kwargs)

    def pinverse(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.pinverse, "__call__", *(self, *args), **kwargs)

    def repeat(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.repeat, "__call__", *(self, *args), **kwargs)

    def reshape(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.reshape, "__call__", *(self, *args), **kwargs)

    def reshape_as(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.reshape_as, "__call__", *(self, *args), **kwargs)

    def round(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.round, "__call__", *(self, *args), **kwargs)

    def relu(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.relu, "__call__", *(self, *args), **kwargs)

    def prelu(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.prelu, "__call__", *(self, *args), **kwargs)

    def prelu_backward(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.prelu_backward, "__call__", *(self, *args), **kwargs)

    def hardshrink(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.hardshrink, "__call__", *(self, *args), **kwargs)

    # def hardshrink_backward(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.hardshrink_backward, "__call__", *(self, *args), **kwargs)

    def rsqrt(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.rsqrt, "__call__", *(self, *args), **kwargs)

    def select(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.select, "__call__", *(self, *args), **kwargs)

    def sigmoid(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sigmoid, "__call__", *(self, *args), **kwargs)

    def sin(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sin, "__call__", *(self, *args), **kwargs)

    def sinh(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sinh, "__call__", *(self, *args), **kwargs)

    # def slice(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.slice, "__call__", *(self, *args), **kwargs)

    def slogdet(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.slogdet, "__call__", *(self, *args), **kwargs)

    def smm(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.smm, "__call__", *(self, *args), **kwargs)

    def softmax(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.softmax, "__call__", *(self, *args), **kwargs)

    def squeeze(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.squeeze, "__call__", *(self, *args), **kwargs)

    def sspaddmm(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sspaddmm, "__call__", *(self, *args), **kwargs)

    def sum(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sum, "__call__", *(self, *args), **kwargs)

    # def sum_to_size(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.sum_to_size, "__call__", *(self, *args), **kwargs)

    def sqrt(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sqrt, "__call__", *(self, *args), **kwargs)

    def std(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.std, "__call__", *(self, *args), **kwargs)

    def prod(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.prod, "__call__", *(self, *args), **kwargs)

    def t(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.t, "__call__", *(self, *args), **kwargs)

    def tan(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.tan, "__call__", *(self, *args), **kwargs)

    def tanh(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.tanh, "__call__", *(self, *args), **kwargs)

    def transpose(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.transpose, "__call__", *(self, *args), **kwargs)

    def flip(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.flip, "__call__", *(self, *args), **kwargs)

    def roll(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.roll, "__call__", *(self, *args), **kwargs)

    def rot90(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.rot90, "__call__", *(self, *args), **kwargs)

    def trunc(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.trunc, "__call__", *(self, *args), **kwargs)

    def type_as(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.type_as, "__call__", *(self, *args), **kwargs)

    def unsqueeze(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.unsqueeze, "__call__", *(self, *args), **kwargs)

    def var(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.var, "__call__", *(self, *args), **kwargs)

    def view_as(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.view_as, "__call__", *(self, *args), **kwargs)

    def where(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.where, "__call__", *(self, *args), **kwargs)

    def clone(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.clone, "__call__", *(self, *args), **kwargs)

    def pow(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.pow, "__call__", *(self, *args), **kwargs)

    # def zero(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.zero, "__call__", *(self, *args), **kwargs)

    def sub(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sub, "__call__", *(self, *args), **kwargs)

    def addmm(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.addmm, "__call__", *(self, *args), **kwargs)

    def sparse_resize(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sparse_resize_, "__call__", *(self, *args), **kwargs)

    def sparse_resize_and_clear(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sparse_resize_and_clear_, "__call__", *(self, *args), **kwargs)

    def sparse_mask(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sparse_mask, "__call__", *(self, *args), **kwargs)

    def to_dense(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.to_dense, "__call__", *(self, *args), **kwargs)

    def coalesce(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.coalesce, "__call__", *(self, *args), **kwargs)

    def indices(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.indices, "__call__", *(self, *args), **kwargs)

    def values(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.values, "__call__", *(self, *args), **kwargs)

    def set(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.set_, "__call__", *(self, *args), **kwargs)

    def masked_fill(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.masked_fill, "__call__", *(self, *args), **kwargs)

    def masked_scatter(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.masked_scatter, "__call__", *(self, *args), **kwargs)

    def view(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.view, "__call__", *(self, *args), **kwargs)

    def put(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.put, "__call__", *(self, *args), **kwargs)

    def index_add(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.index_add, "__call__", *(self, *args), **kwargs)

    def index_fill(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.index_fill, "__call__", *(self, *args), **kwargs)

    def scatter(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.scatter, "__call__", *(self, *args), **kwargs)

    def scatter_add(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.scatter_add, "__call__", *(self, *args), **kwargs)

    def lt(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.lt, "__call__", *(self, *args), **kwargs)

    def gt(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.gt, "__call__", *(self, *args), **kwargs)

    def le(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.le, "__call__", *(self, *args), **kwargs)

    def ge(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.ge, "__call__", *(self, *args), **kwargs)

    def eq(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.eq, "__call__", *(self, *args), **kwargs)

    def ne(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.ne, "__call__", *(self, *args), **kwargs)

    def __and__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__and__, "__call__", *(self, *args), **kwargs)

    def __iand__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__iand__, "__call__", *(self, *args), **kwargs)

    def __or__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__or__, "__call__", *(self, *args), **kwargs)

    def __ior__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__ior__, "__call__", *(self, *args), **kwargs)

    def __xor__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__xor__, "__call__", *(self, *args), **kwargs)

    def __ixor__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__ixor__, "__call__", *(self, *args), **kwargs)

    def __lshift__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__lshift__, "__call__", *(self, *args), **kwargs)

    def __ilshift__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__ilshift__, "__call__", *(self, *args), **kwargs)

    def __rshift__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__rshift__, "__call__", *(self, *args), **kwargs)

    def __irshift__(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.__irshift__, "__call__", *(self, *args), **kwargs)

    def lgamma(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.lgamma, "__call__", *(self, *args), **kwargs)

    def atan2(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.atan2, "__call__", *(self, *args), **kwargs)

    def tril(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.tril, "__call__", *(self, *args), **kwargs)

    def triu(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.triu, "__call__", *(self, *args), **kwargs)

    def digamma(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.digamma, "__call__", *(self, *args), **kwargs)

    def polygamma(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.polygamma, "__call__", *(self, *args), **kwargs)

    def erfinv(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.erfinv, "__call__", *(self, *args), **kwargs)

    def frac(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.frac, "__call__", *(self, *args), **kwargs)

    def renorm(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.renorm, "__call__", *(self, *args), **kwargs)

    def reciprocal(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.reciprocal, "__call__", *(self, *args), **kwargs)

    def neg(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.neg, "__call__", *(self, *args), **kwargs)

    def lerp(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.lerp, "__call__", *(self, *args), **kwargs)

    def sign(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sign, "__call__", *(self, *args), **kwargs)

    def fmod(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.fmod, "__call__", *(self, *args), **kwargs)

    def remainder(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.remainder, "__call__", *(self, *args), **kwargs)

    def addbmm(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.addbmm, "__call__", *(self, *args), **kwargs)

    def addcmul(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.addcmul, "__call__", *(self, *args), **kwargs)

    def addcdiv(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.addcdiv, "__call__", *(self, *args), **kwargs)

    # def random(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.random, "__call__", *(self, *args), **kwargs)

    # def uniform(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.uniform, "__call__", *(self, *args), **kwargs)

    # def normal(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.normal, "__call__", *(self, *args), **kwargs)

    def cauchy(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.cauchy_, "__call__", *(self, *args), **kwargs)

    # def log_normal(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.log_normal_, "__call__", *(self, *args), **kwargs)

    def exponential(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.exponential_, "__call__", *(self, *args), **kwargs)

    # def geometric(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.geometric_, "__call__", *(self, *args), **kwargs)

    def diag(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.diag, "__call__", *(self, *args), **kwargs)

    def cross(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.cross, "__call__", *(self, *args), **kwargs)

    def trace(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.trace, "__call__", *(self, *args), **kwargs)

    def take(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.take, "__call__", *(self, *args), **kwargs)

    def index_select(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.index_select, "__call__", *(self, *args), **kwargs)

    def masked_select(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.masked_select, "__call__", *(self, *args), **kwargs)

    def nonzero(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.nonzero, "__call__", *(self, *args), **kwargs)

    def gather(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.gather, "__call__", *(self, *args), **kwargs)

    def gels(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.gels, "__call__", *(self, *args), **kwargs)

    def trtrs(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.trtrs, "__call__", *(self, *args), **kwargs)

    def symeig(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.symeig, "__call__", *(self, *args), **kwargs)

    def eig(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.eig, "__call__", *(self, *args), **kwargs)

    def svd(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.svd, "__call__", *(self, *args), **kwargs)

    def cholesky(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.cholesky, "__call__", *(self, *args), **kwargs)

    def cholesky_solve(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.cholesky_solve, "__call__", *(self, *args), **kwargs)

    def potri(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.potri, "__call__", *(self, *args), **kwargs)

    def pstrf(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.pstrf, "__call__", *(self, *args), **kwargs)

    def qr(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.qr, "__call__", *(self, *args), **kwargs)

    def geqrf(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.geqrf, "__call__", *(self, *args), **kwargs)

    def orgqr(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.orgqr, "__call__", *(self, *args), **kwargs)

    def ormqr(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.ormqr, "__call__", *(self, *args), **kwargs)

    def btrifact(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.btrifact, "__call__", *(self, *args), **kwargs)

    def btrifact_with_info(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.btrifact_with_info, "__call__", *(self, *args), **kwargs)

    def btrisolve(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.btrisolve, "__call__", *(self, *args), **kwargs)

    def multinomial(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.multinomial, "__call__", *(self, *args), **kwargs)

    def dist(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.dist, "__call__", *(self, *args), **kwargs)

    def histc(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.histc, "__call__", *(self, *args), **kwargs)

    def sort(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.sort, "__call__", *(self, *args), **kwargs)

    def argsort(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.argsort, "__call__", *(self, *args), **kwargs)

    def topk(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.topk, "__call__", *(self, *args), **kwargs)

    def unfold(self, *args, **kwargs):
        return _call_ufunc(_C._TensorBase.unfold, "__call__", *(self, *args), **kwargs)

    # def alias(self, *args, **kwargs):
    #     return _call_ufunc(_C._TensorBase.alias, "__call__", *(self, *args), **kwargs)

    def __format__(self, format_spec):
        if self.dim() == 0:
            return self.item().__format__(format_spec)
        return object.__format__(self, format_spec)

    def __len__(self):
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def __iter__(self):
        # NB: we use 'imap' and not 'map' here, so that in Python 2 we get a
        # generator and don't eagerly perform all the indexes.  This could
        # save us work, and also helps keep trace ordering deterministic
        # (e.g., if you zip(*hiddens), the eager map will force all the
        # indexes of hiddens[0] before hiddens[1], while the generator
        # map will interleave them.)
        if self.dim() == 0:
            raise TypeError('iteration over a 0-d tensor')
        if torch._C._get_tracing_state():
            warnings.warn('Iterating over a tensor might cause the trace to be incorrect. '
                          'Passing a tensor of different shape won\'t change the number of '
                          'iterations executed (and might lead to errors or silently give '
                          'incorrect results).', category=RuntimeWarning)
        return iter(imap(lambda i: self[i], range(self.size(0))))

    def __hash__(self):
        return id(self)

    def __dir__(self):
        tensor_methods = dir(self.__class__)
        tensor_methods.remove('volatile')  # deprecated
        attrs = list(self.__dict__.keys())
        keys = tensor_methods + attrs

        # property only available dense, cuda tensors
        if (not self.is_cuda) or self.is_sparse:
            keys.remove("__cuda_array_interface__")

        return sorted(keys)

    # Numpy array interface, to support `numpy.asarray(tensor) -> ndarray`
    __array_priority__ = 1000    # prefer Tensor ops over numpy ones

    def __array__(self, dtype=None):
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    # Wrap Numpy array again in a suitable tensor when done, to support e.g.
    # `numpy.sin(tensor) -> tensor` or `numpy.greater(tensor, 0) -> ByteTensor`
    def __array_wrap__(self, array):
        if array.dtype == bool:
            # Workaround, torch has no built-in bool tensor
            array = array.astype('uint8')
        return torch.Tensor._make_subclass(self.__class__, torch.as_tensor(array.astype('uint8')).data, False)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, self.__class__):
                in_no.append(i)
                args.append(input_.detach().numpy())
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, self.__class__):
                    out_no.append(j)
                    out_args.append(output.detach().numpy())
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = getattr(ufunc, method)(*args, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            return

        if ufunc.nout == 1:
            if results.dtype == bool:
                # Workaround, torch has no built-in bool tensor
                results = torch.as_tensor(results.astype('uint8'))
            results = (results,)

        results = tuple((torch.Tensor._make_subclass(self.__class__, torch.as_tensor(result).data, self.requires_grad)
                         if output is None else output)
                        for result, output in zip(results, outputs))

        return results[0] if len(results) == 1 else results

    def __tensor_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, self.__class__):
                in_no.append(i)
                args.append(torch.Tensor._make_subclass(Tensor, input_.data, input_.requires_grad))
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        out_no = []
        if outputs is not None:
            if not isinstance(outputs, (list, tuple)):
                outputs = (outputs,)
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, self.__class__):
                    out_no.append(j)
                    out_args.append(torch.Tensor._make_subclass(Tensor, output.data, output.requires_grad))
                else:
                    out_args.append(output)
            kwargs['out'] = out_args[0] if len(out_args) == 1 else tuple(out_args)

        results = getattr(ufunc, method)(*args, **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if not isinstance(results, tuple):
            results = (results,)

        if outputs is None:
            outputs = (None,) * len(results)


        results = results.__class__((torch.Tensor._make_subclass(self.__class__, result.data, result.requires_grad)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        return results[0] if len(results) == 1 else results

    def __contains__(self, element):
        r"""Check if `element` is present in tensor

        Arguments:
            element (Tensor or scalar): element to be checked
                for presence in current tensor"
        """
        if isinstance(element, (torch.Tensor, Number)):
            return (element == self).any().item()
        return NotImplemented

    @property
    def __cuda_array_interface__(self):
        """Array view description for cuda tensors.

        See:
        https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
        """

        # raise AttributeError for unsupported tensors, so that
        # hasattr(cpu_tensor, "__cuda_array_interface__") is False.
        if not self.is_cuda:
            raise AttributeError(
                "Can't get __cuda_array_interface__ on non-CUDA tensor type: %s "
                "If CUDA data is required use tensor.cuda() to copy tensor to device memory." %
                self.type()
            )

        if self.is_sparse:
            raise AttributeError(
                "Can't get __cuda_array_interface__ on sparse type: %s "
                "Use Tensor.to_dense() to convert to a dense tensor first." %
                self.type()
            )

        # RuntimeError, matching tensor.__array__() behavior.
        if self.requires_grad:
            raise RuntimeError(
                "Can't get __cuda_array_interface__ on Variable that requires grad. "
                "If gradients aren't required, use var.detach() to get Variable that doesn't require grad."
            )

        # CUDA devices are little-endian and tensors are stored in native byte
        # order. 1-byte entries are endian-agnostic.
        typestr = {
            torch.float16: "<f2",
            torch.float32: "<f4",
            torch.float64: "<f8",
            torch.uint8: "|u1",
            torch.int8: "|i1",
            torch.int16: "<i2",
            torch.int32: "<i4",
            torch.int64: "<i8",
        }[self.dtype]

        itemsize = self.storage().element_size()

        shape = self.shape
        strides = tuple(s * itemsize for s in self.stride())
        data = (self.data_ptr(), False)  # read-only is false

        return dict(typestr=typestr, shape=shape, strides=strides, data=data, version=0)

    __module__ = 'torch'
