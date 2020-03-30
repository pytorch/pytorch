import sys
import torch
import torch._C as _C
from torch._namedtensor_internals import update_names, check_serializing_named_tensor, resolve_ellipsis
from torch._namedtensor_internals import unzip_namedshape, single_ellipsis_index, is_ellipsis
from collections import OrderedDict
import torch.utils.hooks as hooks
import warnings
import weakref
from torch._six import imap
from torch._C import _add_docstr
from numbers import Number
import functools


def _wrap_type_error_to_not_implemented(f):
    from torch import _six
    import inspect

    # functools.wraps doesn't work well with methods in python 2
    method_assignments = ('__name__', '__doc__')
    assigned = (method_assignments if _six.PY2 and inspect.ismethoddescriptor(f)
                else functools.WRAPPER_ASSIGNMENTS)

    @functools.wraps(f, assigned=assigned)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except TypeError:
            return NotImplemented
    return wrapped


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
            if self.is_sparse or self.device.type == 'xla':
                new_tensor = self.clone()
            else:
                new_storage = self.storage().__deepcopy__(memo)
                if self.is_quantized:
                    if self.qscheme() == torch.per_tensor_affine:
                        quantizer_params = self.qscheme(), self.q_scale(), self.q_zero_point()
                    elif self.qscheme() == torch.per_channel_affine:
                        quantizer_params = self.qscheme(), \
                            self.q_per_channel_scales(), \
                            self.q_per_channel_zero_points(), \
                            self.q_per_channel_axis()
                    else:
                        raise RuntimeError("Unsupported qscheme {} in deepcopy".format(self.qscheme()))
                    new_tensor = torch._utils._rebuild_qtensor(
                        new_storage,
                        self.storage_offset(),
                        self.size(),
                        self.stride(),
                        quantizer_params,
                        self.requires_grad,
                        self._backward_hooks)
                else:
                    new_tensor = self.new()
                    new_tensor.set_(new_storage, self.storage_offset(), self.size(), self.stride())
                    new_tensor.requires_grad = self.requires_grad
            memo[id(self)] = new_tensor
            return new_tensor

    def __reduce_ex__(self, proto):
        check_serializing_named_tensor(self)
        # See Note [Don't serialize hooks]
        torch.utils.hooks.warn_if_has_hooks(self)
        # Note: Numpy array is chosen to be the rebuild component for XLA Tensor.
        # We considered a few options:
        # 1. CPU tensor can't be used here.
        #    Otherwise in torch.load CPU storage is reconstructed with randomly
        #    initialized data, moved onto XLA device, and then storage is updated
        #    to the serialized content. This works perfectly for CPU/CUDA but not XLA.
        #    XLA tensor is disconnected with storage so it doesn't get the update.
        # 2. Python list is not a good fit due to performance reason.
        #    `tolist()` converts every single element in the tensor into python objects
        #    and serialize them one by one.
        if self.device.type == 'xla':
            args = (self.cpu().numpy(),
                    self.dtype,
                    str(self.device),
                    self.requires_grad)
            return (torch._utils._rebuild_xla_tensor, args)
        if self.is_quantized:
            if self.qscheme() == torch.per_tensor_affine:
                quantizer_params = (torch.per_tensor_affine,
                                    self.q_scale(),
                                    self.q_zero_point())
            elif self.qscheme() == torch.per_channel_affine:
                # convert scales and zero points to tuple to avoid recursive calls
                # when/if we get multi-axis quantized tensors in the future, the shape
                # is recoverable from the main tensor shape
                quantizer_params = (torch.per_channel_affine,
                                    self.q_per_channel_scales(),
                                    self.q_per_channel_zero_points(),
                                    self.q_per_channel_axis())
            else:
                raise RuntimeError("Serialization is not supported for tensors of type {}".format(self.qscheme()))
            args = (self.storage(),
                    self.storage_offset(),
                    tuple(self.size()),
                    self.stride(),
                    quantizer_params,
                    self.requires_grad,
                    OrderedDict())
            return (torch._utils._rebuild_qtensor, args)
        elif self.is_sparse:
            if self.layout == torch.sparse_coo:
                args = (self.layout,
                        (self._indices(),
                         self._values(),
                         self.size()))
            else:
                raise NotImplementedError(
                    'sparse tensor __reduce_ex__ for layout `%s`' % (self.layout))
            return (torch._utils._rebuild_sparse_tensor, args)
        else:
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
        if not self.requires_grad:
            raise RuntimeError("can't retain_grad on Tensor that has requires_grad=False")
        if self.is_leaf:  # no-op for leaves
            return
        if hasattr(self, 'retains_grad'):
            return
        weak_self = weakref.ref(self)

        def retain_grad_hook(grad):
            var = weak_self()
            if var is None:
                return
            if var._grad is None:
                if grad.is_sparse:
                    var._grad = grad.clone()
                else:
                    var._grad = grad.clone(memory_format=torch.contiguous_format)
            else:
                var._grad = var._grad + grad

        self.register_hook(retain_grad_hook)
        self.retains_grad = True

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

    def norm(self, p="fro", dim=None, keepdim=False, dtype=None):
        r"""See :func:`torch.norm`"""
        return torch.norm(self, p, dim, keepdim, dtype=dtype)

    def lu(self, pivot=True, get_infos=False):
        r"""See :func:`torch.lu`"""
        # If get_infos is True, then we don't need to check for errors and vice versa
        LU, pivots, infos = torch._lu_with_info(self, pivot=pivot, check_errors=(not get_infos))
        if get_infos:
            return LU, pivots, infos
        else:
            return LU, pivots

    def stft(self, n_fft, hop_length=None, win_length=None, window=None,
             center=True, pad_mode='reflect', normalized=False, onesided=True):
        r"""See :func:`torch.stft`

        .. warning::
          This function changed signature at version 0.4.1. Calling with
          the previous signature may cause error or return incorrect result.
        """
        return torch.stft(self, n_fft, hop_length, win_length, window, center,
                          pad_mode, normalized, onesided)

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
        elif isinstance(split_size, Tensor):
            try:
                split_size = int(split_size)
                return super(Tensor, self).split(split_size, dim)
            except ValueError:
                return super(Tensor, self).split_with_sizes(split_size, dim)
        else:
            return super(Tensor, self).split_with_sizes(split_size, dim)

    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        r"""Returns the unique elements of the input tensor.

        See :func:`torch.unique`
        """
        return torch.unique(self, sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)

    def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
        r"""Eliminates all but the first element from every consecutive group of equivalent elements.

        See :func:`torch.unique_consecutive`
        """
        return torch.unique_consecutive(self, return_inverse=return_inverse, return_counts=return_counts, dim=dim)

    def __rsub__(self, other):
        return _C._VariableFunctions.rsub(self, other)

    def __rdiv__(self, other):
        if self.dtype.is_floating_point:
            return self.reciprocal() * other
        else:
            return (self.double().reciprocal() * other).type_as(self)

    __rtruediv__ = __rdiv__
    __itruediv__ = _C._TensorBase.__idiv__

    __pow__ = _C._TensorBase.pow

    def __format__(self, format_spec):
        if self.dim() == 0:
            return self.item().__format__(format_spec)
        return object.__format__(self, format_spec)

    def __ipow__(self, other):
        return NotImplemented

    @_wrap_type_error_to_not_implemented
    def __rpow__(self, other):
        return self.new_tensor(other) ** self

    @_wrap_type_error_to_not_implemented
    def __floordiv__(self, other):
        return torch.floor_divide(self, other)

    @_wrap_type_error_to_not_implemented
    def __rfloordiv__(self, other):
        result = other / self
        if result.dtype.is_floating_point:
            result = result.trunc()
        return result

    __neg__ = _C._TensorBase.neg

    __eq__ = _wrap_type_error_to_not_implemented(_C._TensorBase.eq)
    __ne__ = _wrap_type_error_to_not_implemented(_C._TensorBase.ne)
    __lt__ = _wrap_type_error_to_not_implemented(_C._TensorBase.lt)
    __le__ = _wrap_type_error_to_not_implemented(_C._TensorBase.le)
    __gt__ = _wrap_type_error_to_not_implemented(_C._TensorBase.gt)
    __ge__ = _wrap_type_error_to_not_implemented(_C._TensorBase.ge)
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
        if torch._C._get_tracing_state():
            warnings.warn('Iterating over a tensor might cause the trace to be incorrect. '
                          'Passing a tensor of different shape won\'t change the number of '
                          'iterations executed (and might lead to errors or silently give '
                          'incorrect results).', category=RuntimeWarning)
        return iter(imap(lambda i: self[i], range(self.size(0))))

    def __hash__(self):
        return id(self)

    def __dir__(self):
        if self.is_quantized:
            warnings.warn('Only a small subset of methods are supported for quantized tensors.')
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
        return torch.from_numpy(array)

    def __contains__(self, element):
        r"""Check if `element` is present in tensor

        Arguments:
            element (Tensor or scalar): element to be checked
                for presence in current tensor"
        """
        if isinstance(element, (torch.Tensor, Number)):
            return (element == self).any().item()

        raise RuntimeError(
            "Tensor.__contains__ only supports Tensor or scalar, but you passed in a %s." %
            type(element)
        )

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

        shape = tuple(self.shape)
        if self.is_contiguous():
            # __cuda_array_interface__ v2 requires the strides to be omitted
            # (either not set or set to None) for C-contiguous arrays.
            strides = None
        else:
            strides = tuple(s * itemsize for s in self.stride())
        data_ptr = self.data_ptr() if self.numel() > 0 else 0
        data = (data_ptr, False)  # read-only is false

        return dict(typestr=typestr, shape=shape, strides=strides, data=data, version=2)

    def refine_names(self, *names):
        r"""Refines the dimension names of :attr:`self` according to :attr:`names`.

        Refining is a special case of renaming that "lifts" unnamed dimensions.
        A ``None`` dim can be refined to have any name; a named dim can only be
        refined to have the same name.

        Because named tensors can coexist with unnamed tensors, refining names
        gives a nice way to write named-tensor-aware code that works with both
        named and unnamed tensors.

        :attr:`names` may contain up to one Ellipsis (``...``).
        The Ellipsis is expanded greedily; it is expanded in-place to fill
        :attr:`names` to the same length as ``self.dim()`` using names from the
        corresponding indices of ``self.names``.

        Python 2 does not support Ellipsis but one may use a string literal
        instead (``'...'``).

        Arguments:
            names (iterable of str): The desired names of the output tensor. May
                contain up to one Ellipsis.

        Examples::

            >>> imgs = torch.randn(32, 3, 128, 128)
            >>> named_imgs = imgs.refine_names('N', 'C', 'H', 'W')
            >>> named_imgs.names
            ('N', 'C', 'H', 'W')

            >>> tensor = torch.randn(2, 3, 5, 7, 11)
            >>> tensor = tensor.refine_names('A', ..., 'B', 'C')
            >>> tensor.names
            ('A', None, None, 'B', 'C')

        .. warning::
            The named tensor API is experimental and subject to change.

        """
        names = resolve_ellipsis(names, self.names, 'refine_names')
        return super(Tensor, self).refine_names(names)

    def align_to(self, *names):
        r"""Permutes the dimensions of the :attr:`self` tensor to match the order
        specified in :attr:`names`, adding size-one dims for any new names.

        All of the dims of :attr:`self` must be named in order to use this method.
        The resulting tensor is a view on the original tensor.

        All dimension names of :attr:`self` must be present in :attr:`names`.
        :attr:`names` may contain additional names that are not in ``self.names``;
        the output tensor has a size-one dimension for each of those new names.

        :attr:`names` may contain up to one Ellipsis (``...``).
        The Ellipsis is expanded to be equal to all dimension names of :attr:`self`
        that are not mentioned in :attr:`names`, in the order that they appear
        in :attr:`self`.

        Python 2 does not support Ellipsis but one may use a string literal
        instead (``'...'``).

        Arguments:
            names (iterable of str): The desired dimension ordering of the
                output tensor. May contain up to one Ellipsis that is expanded
                to all unmentioned dim names of :attr:`self`.

        Examples::

            >>> tensor = torch.randn(2, 2, 2, 2, 2, 2)
            >>> named_tensor = tensor.refine_names('A', 'B', 'C', 'D', 'E', 'F')

            # Move the F and E dims to the front while keeping the rest in order
            >>> named_tensor.align_to('F', 'E', ...)

        .. warning::
            The named tensor API is experimental and subject to change.

        """
        ellipsis_idx = single_ellipsis_index(names, 'align_to')
        if ellipsis_idx is None:
            return super(Tensor, self).align_to(names)
        return super(Tensor, self).align_to(
            [name for name in names if not is_ellipsis(name)],
            ellipsis_idx)

    def unflatten(self, dim, namedshape):
        r"""Unflattens the named dimension :attr:`dim`, viewing it in the shape
        specified by :attr:`namedshape`.

        Arguments:
            namedshape: (iterable of ``(name, size)`` tuples).

        Examples::

            >>> flat_imgs = torch.rand(32, 3 * 128 * 128, names=('N', 'features'))
            >>> imgs = flat_imgs.unflatten('features', (('C', 3), ('H', 128), ('W', 128)))
            >>> imgs.names, images.shape
            (('N', 'C', 'H', 'W'), torch.Size([32, 3, 128, 128]))

        .. warning::
            The named tensor API is experimental and subject to change.

        """
        names, sizes = unzip_namedshape(namedshape)
        return super(Tensor, self).unflatten(dim, sizes, names)

    def rename_(self, *names, **rename_map):
        """In-place version of :meth:`~Tensor.rename`."""

        # Note [rename_ / rename API]
        # The Python API for these is different from the C++ API. In Python:
        # 1) tensor.rename(*names) takes a vararglist of names
        # 2) tensor.rename(**rename_map) takes a map of names to rename.
        # C++ is static, making it difficult to implement similar behavior.
        return update_names(self, names, rename_map, inplace=True)

    def rename(self, *names, **rename_map):
        """Renames dimension names of :attr:`self`.

        There are two main usages:

        ``self.rename(**rename_map)`` returns a view on tensor that has dims
        renamed as specified in the mapping :attr:`rename_map`.

        ``self.rename(*names)`` returns a view on tensor, renaming all
        dimensions positionally using :attr:`names`.
        Use ``self.rename(None)`` to drop names on a tensor.

        One cannot specify both positional args :attr:`names` and keyword args
        :attr:`rename_map`.

        Examples::

            >>> imgs = torch.rand(2, 3, 5, 7, names=('N', 'C', 'H', 'W'))
            >>> renamed_imgs = imgs.rename(N='batch', C='channels')
            >>> renamed_imgs.names
            ('batch', 'channels', 'H', 'W')

            >>> renamed_imgs = imgs.rename(None)
            >>> renamed_imgs.names
            (None,)

            >>> renamed_imgs = imgs.rename('batch', 'channel', 'height', 'width')
            >>> renamed_imgs.names
            ('batch', 'channel', 'height', 'width')

        .. warning::
            The named tensor API is experimental and subject to change.

        """
        # See Note [rename_ / rename API]
        return update_names(self, names, rename_map, inplace=False)

    def _update_names(self, names, inplace):
        # See Note [rename_ / rename API]
        if inplace:
            return super(Tensor, self).rename_(names)
        else:
            return super(Tensor, self).rename(names)

    @property
    def grad(self):
        """
        This attribute is ``None`` by default and becomes a Tensor the first time a call to
        :func:`backward` computes gradients for ``self``.
        The attribute will then contain the gradients computed and future calls to
        :func:`backward` will accumulate (add) gradients into it.
        """
        if self.requires_grad and not hasattr(self, "retains_grad") and not self.is_leaf and self._grad is None:
            warnings.warn("The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad "
                          "attribute won't be populated during autograd.backward(). If you indeed want the gradient "
                          "for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the "
                          "non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See "
                          "github.com/pytorch/pytorch/pull/30531 for more informations.")
        return self._grad

    @grad.setter
    def grad(self, new_grad):
        self._grad = new_grad

    @grad.deleter
    def grad(self):
        del self._grad

    __module__ = 'torch'
