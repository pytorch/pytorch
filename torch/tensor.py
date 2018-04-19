import sys
import torch
import torch._C as _C
from collections import OrderedDict
import torch.utils.hooks as hooks
import warnings
import weakref
from torch._six import imap
from torch._C import _add_docstr


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
        args = (self.storage(),
                self.storage_offset(),
                tuple(self.size()),
                self.stride(),
                self.requires_grad,
                self._backward_hooks)
        return (torch._utils._rebuild_tensor_v2, args)

    def __setstate__(self, state):
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

        Example:
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
            See http://pytorch.org/docs/master/distributions.html

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

      Returned Tensor uses the same data tensor as the original one.
      In-place modifications on either of them will be seen, and may trigger
      errors in correctness checks.
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

    def view_as(self, tensor):
        r"""view_as(other) -> Tensor

        View this tensor as the same size as :attr:`other`.
        ``self.view_as(other)`` is equivalent to ``self.view(other.size())``.

        Args:
            other (:class:`torch.Tensor`): The result tensor has the same size
                as :attr:`other.size()`.
        """
        return self.view(tensor.size())

    def argmax(self, dim=None, keepdim=False):
        r"""See :func:`torch.argmax`"""
        return torch.argmax(self, dim, keepdim)

    def argmin(self, dim=None, keepdim=False):
        r"""See :func:`torch.argmin`"""
        return torch.argmin(self, dim, keepdim)

    def btrifact(self, info=None, pivot=True):
        r"""See :func:`torch.btrifact`
        """
        if info is not None:
            warnings.warn("info option in btrifact is deprecated and will be removed in v0.4, "
                          "consider using btrifact_with_info instead", stacklevel=2)
            factorization, pivots, _info = super(Tensor, self).btrifact_with_info(pivot=pivot)
            if info.type() != _info.type():
                raise ValueError('btrifact expects info to be an IntTenor')
            info.resize_as_(_info).copy_(_info)
            return factorization, pivots
        else:
            return super(Tensor, self).btrifact(pivot=pivot)

    def resize(self, *sizes):
        warnings.warn("non-inplace resize is deprecated")
        from torch.autograd._functions import Resize
        return Resize.apply(self, sizes)

    def resize_as(self, tensor):
        warnings.warn("non-inplace resize_as is deprecated")
        from torch.autograd._functions import Resize
        return Resize.apply(self, tensor.size())

    def split(self, split_size, dim=0):
        r"""See :func:`torch.split`
        """
        if isinstance(split_size, int):
            return super(Tensor, self).split(split_size, dim)
        else:
            return super(Tensor, self).split_with_sizes(split_size, dim)

    def index_add(self, dim, index, tensor):
        return self.clone().index_add_(dim, index, tensor)

    def index_copy(self, dim, index, tensor):
        return self.clone().index_copy_(dim, index, tensor)

    def index_fill(self, dim, index, value):
        return self.clone().index_fill_(dim, index, value)

    def scatter(self, dim, index, source):
        return self.clone().scatter_(dim, index, source)

    def scatter_add(self, dim, index, source):
        return self.clone().scatter_add_(dim, index, source)

    def masked_copy(self, mask, tensor):
        warnings.warn("masked_copy is deprecated and renamed to masked_scatter, and will be removed in v0.3")
        return self.masked_scatter(mask, tensor)

    def masked_copy_(self, mask, tensor):
        warnings.warn("masked_copy_ is deprecated and renamed to masked_scatter_, and will be removed in v0.3")
        return self.masked_scatter_(mask, tensor)

    def masked_scatter(self, mask, tensor):
        return self.clone().masked_scatter_(mask, tensor)

    def masked_fill(self, mask, value):
        return self.clone().masked_fill_(mask, value)

    def expand_as(self, tensor):
        return self.expand(tensor.size())

    def unique(self, sorted=False, return_inverse=False):
        r"""Returns the unique scalar elements of the tensor as a 1-D tensor.

        See :func:`torch.unique`
        """
        output, inverse_indices = self._unique(
            sorted=sorted, return_inverse=return_inverse)
        if return_inverse:
            return output, inverse_indices
        else:
            return output

    def __rsub__(self, other):
        return -self + other

    def __rdiv__(self, other):
        return self.reciprocal() * other
    __rtruediv__ = __rdiv__
    __itruediv__ = _C._TensorBase.__idiv__

    __pow__ = _C._TensorBase.pow

    def __format__(self, format_spec):
        if self.dim() == 0:
            return self.item().__format__(format_spec)
        return object.__format__(self, format_spec)

    def __ipow__(self, other):
        raise NotImplementedError("in-place pow not implemented")

    def __rpow__(self, other):
        return self.new([other]) ** self

    __neg__ = _C._TensorBase.neg

    __eq__ = _C._TensorBase.eq
    __ne__ = _C._TensorBase.ne
    __lt__ = _C._TensorBase.lt
    __le__ = _C._TensorBase.le
    __gt__ = _C._TensorBase.gt
    __ge__ = _C._TensorBase.ge
    __abs__ = _C._TensorBase.abs

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
        return iter(imap(lambda i: self[i], range(self.size(0))))

    def __hash__(self):
        return id(self)

    def __dir__(self):
        tensor_methods = dir(self.__class__)
        tensor_methods.remove('volatile')  # deprecated
        attrs = list(self.__dict__.keys())
        keys = tensor_methods + attrs
        return sorted(keys)

    # Numpy array interface, to support `numpy.asarray(tensor) -> ndarray`
    def __array__(self, dtype=None):
        if dtype is None:
            return self.cpu().numpy()
        else:
            return self.cpu().numpy().astype(dtype, copy=False)

    # Wrap Numpy array again in a suitable tensor when done, to support e.g.
    # `numpy.sin(tensor) -> tensor` or `numpy.greater(tensor, 0) -> ByteTensor`
    def __array_wrap__(self, array):
        if array.dtype == bool:
            # Workaround, torch has no built-in bool tensor
            array = array.astype('uint8')
        return torch.from_numpy(array)

    __module__ = 'torch'
