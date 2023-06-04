import copyreg
import enum
import functools
import warnings
from collections import OrderedDict
from copy import deepcopy
from numbers import Number
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._namedtensor_internals import (
    check_serializing_named_tensor,
    is_ellipsis,
    resolve_ellipsis,
    single_ellipsis_index,
    unzip_namedshape,
    update_names,
)
from torch.overrides import (
    get_default_nowrap_functions,
    handle_torch_function,
    has_torch_function,
    has_torch_function_unary,
    has_torch_function_variadic,
)
from torch.utils.dlpack import DLDeviceType


def _handle_torch_function_and_wrap_type_error_to_not_implemented(f):
    assigned = functools.WRAPPER_ASSIGNMENTS

    @functools.wraps(f, assigned=assigned)
    def wrapped(*args, **kwargs):
        try:
            # See https://github.com/pytorch/pytorch/issues/75462
            if has_torch_function(args):
                return handle_torch_function(wrapped, args, *args, **kwargs)
            return f(*args, **kwargs)
        except TypeError:
            return NotImplemented

    return wrapped


# Should not be used, this is kept only for BC of loading old serialized Tensor subclasses
def _rebuild_from_type(func, type, args, dict):
    if type is Tensor:
        return func(*args)

    ret = func(*args).as_subclass(type)
    ret.__dict__ = dict
    return ret


def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    if type(ret) is not new_type:
        ret = ret.as_subclass(new_type)
    # Tensor does define __setstate__ even though it doesn't define
    # __getstate__. So only use __setstate__ if it is NOT the one defined
    # on Tensor
    if (
        getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
        is not Tensor.__setstate__
    ):
        ret.__setstate__(state)
    else:
        ret = torch._utils._set_obj_state(ret, state)
    return ret


# NB: If you subclass Tensor, and want to share the subclassed class
# across processes, you must also update torch/multiprocessing/reductions.py
# to define a ForkingPickler serialization mode for the class.
#
# NB: If you add a new method to Tensor, you must update
# torch/__init__.py.in to add a type annotation for your method;
# otherwise, it will not show up in autocomplete.
class Tensor(torch._C._TensorBase):
    def __deepcopy__(self, memo):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__deepcopy__, (self,), self, memo)
        if not self.is_leaf:
            raise RuntimeError(
                "Only Tensors created explicitly by the user "
                "(graph leaves) support the deepcopy protocol at the moment"
            )
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            # TODO: skipping storage copy is wrong for meta, as meta
            # does accurate alias tracking; however, the code below
            # doesn't work because of
            # https://github.com/pytorch/pytorch/issues/47442
            # Update the test in test_serialization if you remove 'meta' from here
            if (
                self.is_sparse
                or self.device.type in ["lazy", "xla", "mps", "ort", "meta", "ipu"]
                or (
                    not torch._C._has_storage(self)
                    and self.device.type == torch._C._get_privateuse1_backend_name()
                )
                or (type(self) is not Tensor and self.data_ptr() == 0)
            ):
                new_tensor = self.clone()
                if type(new_tensor) is not type(self):
                    raise RuntimeError(
                        "The default implementation of __deepcopy__() for wrapper subclasses "
                        "only works for subclass types that implement clone() and for which "
                        "cloning returns another instance of the same subclass. You should either "
                        "properly implement clone() for your subclass or override __deepcopy__() "
                        "if it is intended behavior for clone() to return an instance of a "
                        "different type."
                    )
            else:
                new_storage = self._typed_storage()._deepcopy(memo)
                if self.is_quantized:
                    # quantizer_params can be different type based on torch attribute
                    quantizer_params: Union[
                        Tuple[torch.qscheme, float, int],
                        Tuple[torch.qscheme, Tensor, Tensor, int],
                    ]
                    if self.qscheme() == torch.per_tensor_affine:
                        quantizer_params = (
                            self.qscheme(),
                            self.q_scale(),
                            self.q_zero_point(),
                        )
                    elif self.qscheme() in (
                        torch.per_channel_affine,
                        torch.per_channel_affine_float_qparams,
                    ):
                        quantizer_params = (
                            self.qscheme(),
                            self.q_per_channel_scales(),
                            self.q_per_channel_zero_points(),
                            self.q_per_channel_axis(),
                        )
                    else:
                        raise RuntimeError(
                            f"Unsupported qscheme {self.qscheme()} in deepcopy"
                        )
                    # TODO: Once we decide to break serialization FC, no longer
                    # need to wrap with TypedStorage
                    new_tensor = torch._utils._rebuild_qtensor(
                        torch.storage.TypedStorage(
                            wrap_storage=new_storage._untyped_storage,
                            dtype=self.dtype,
                            _internal=True,
                        ),
                        self.storage_offset(),
                        self.size(),
                        self.stride(),
                        quantizer_params,
                        self.requires_grad,
                        self._backward_hooks,
                    )
                    if type(new_tensor) is not type(self):
                        raise RuntimeError(
                            "The default implementation of __deepcopy__() for quantized tensors "
                            "expects the tensor returned by torch._utils._rebuild_qtensor() to "
                            "match the type of the instance being copied. If you encounter this, "
                            "please open an issue on PyTorch's GitHub."
                        )
                else:
                    new_tensor = self.new_empty([])
                    if type(new_tensor) is not type(self):
                        raise RuntimeError(
                            "The default implementation of __deepcopy__() for non-wrapper subclasses "
                            "only works for subclass types that implement new_empty() and for which "
                            "that function returns another instance of the same subclass. You should "
                            "either properly implement new_empty() for your subclass or override "
                            "__deepcopy__() if it is intended behavior for new_empty() to return "
                            "an instance of a different type."
                        )
                    new_tensor.set_(
                        new_storage, self.storage_offset(), self.size(), self.stride()
                    )
                    if self.is_conj():
                        new_tensor = new_tensor.conj_physical()
                    if self.is_neg():
                        new_tensor = new_tensor.neg()
            if self.requires_grad:
                new_tensor.requires_grad_()
            if self.grad is not None:
                new_tensor.grad = self.grad.__deepcopy__(memo)

            if type(self) is not Tensor:
                if type(new_tensor) is not type(self):
                    raise RuntimeError(
                        "Type of deepcopy result does not match the type of the source tensor. "
                        "If you encounter this, please open an issue on PyTorch's GitHub."
                    )

                # Plain Tensors don't have slots
                slots_to_save = copyreg._slotnames(self.__class__)  # type: ignore[attr-defined]
                for slot in slots_to_save:
                    if hasattr(self, slot):
                        setattr(new_tensor, slot, deepcopy(getattr(self, slot), memo))

            new_tensor.__dict__ = deepcopy(self.__dict__, memo)

            memo[id(self)] = new_tensor
            return new_tensor

    def __reduce_ex__(self, proto):
        state = torch._utils._get_obj_state(self)
        if type(self) is Tensor and not state:
            # Fast path for regular tensor without Python state.
            return self._reduce_ex_internal(proto)
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__reduce_ex__, (self,), self, proto)
        func, args = self._reduce_ex_internal(proto)
        return (_rebuild_from_type_v2, (func, type(self), args, state))

    def storage(self):
        r"""
        storage() -> torch.TypedStorage

        Returns the underlying :class:`TypedStorage`.

        .. warning::

            :class:`TypedStorage` is deprecated. It will be removed in the future, and
            :class:`UntypedStorage` will be the only storage class. To access the
            :class:`UntypedStorage` directly, use :attr:`Tensor.untyped_storage()`.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.storage, (self,), self)

        torch.storage._warn_typed_storage_removal(stacklevel=2)
        return self._typed_storage()

    # For internal use only, to avoid raising deprecation warning
    def _typed_storage(self):
        untyped_storage = self.untyped_storage()
        return torch.TypedStorage(
            wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
        )

    def _reduce_ex_internal(self, proto):
        check_serializing_named_tensor(self)
        # See Note [Don't serialize hooks]
        torch.utils.hooks.warn_if_has_hooks(self)
        backward_hooks: Dict[Any, Any] = OrderedDict()
        # Note: Numpy array is chosen to be the rebuild component for XLA, ORT Tensors.
        # We considered a few options:
        # 1. CPU tensor can't be used here.
        #    Otherwise in torch.load CPU storage is reconstructed with randomly
        #    initialized data, moved onto backend device, and then storage is updated
        #    to the serialized content. This works perfectly for CPU/CUDA but not these backends;
        #    their tensors are disconnected with storage so they don't get the update.
        # 2. Python list is not a good fit due to performance reason.
        #    `tolist()` converts every single element in the tensor into python objects
        #    and serialize them one by one.
        if self.device.type in ["xla", "ort"] or (
            not torch._C._has_storage(self)
            and self.device.type == torch._C._get_privateuse1_backend_name()
        ):
            # Convert BFloat16 tesors to Float32 before conversion to numpy, as numpy doesn't
            # support BFloat16. The rebuild tensor from numpy takes in the original self.dtype,
            # this would reconstruct the BFloat16 tensor from numpy.
            numpy_tensor = (
                self.cpu().numpy()
                if self.dtype != torch.bfloat16
                else self.cpu().to(torch.float32).numpy()
            )
            return (
                torch._utils._rebuild_device_tensor_from_numpy,
                (numpy_tensor, self.dtype, str(self.device), self.requires_grad),
            )
        if self.device.type == "meta":
            # NB: This implementation BREAKS storage sharing.  Current
            # hypothesis is that no one cares for meta tensors.
            arg_meta = (
                self.dtype,
                tuple(self.size()),
                self.stride(),
                self.requires_grad,
            )
            return (torch._utils._rebuild_meta_tensor_no_storage, arg_meta)
        if self.is_quantized:
            # quantizer_params can be different type based on torch attribute
            quantizer_params: Union[
                Tuple[torch.qscheme, float, int], Tuple[Any, Tensor, Tensor, int]
            ]
            if self.qscheme() == torch.per_tensor_affine:
                quantizer_params = (
                    torch.per_tensor_affine,
                    self.q_scale(),
                    self.q_zero_point(),
                )
            elif self.qscheme() in (
                torch.per_channel_affine,
                torch.per_channel_affine_float_qparams,
            ):
                # convert scales and zero points to tuple to avoid recursive calls
                # when/if we get multi-axis quantized tensors in the future, the shape
                # is recoverable from the main tensor shape
                quantizer_params = (
                    torch.per_channel_affine,
                    self.q_per_channel_scales(),
                    self.q_per_channel_zero_points(),
                    self.q_per_channel_axis(),
                )
            else:
                raise RuntimeError(
                    f"Serialization is not supported for tensors of type {self.qscheme()}"
                )
            # TODO: Once we decide to break serialization FC, no longer
            # need to wrap with TypedStorage
            args_qtensor = (
                torch.storage.TypedStorage(
                    wrap_storage=self._typed_storage()._untyped_storage,
                    dtype=self.dtype,
                    _internal=True,
                ),
                self.storage_offset(),
                tuple(self.size()),
                self.stride(),
                quantizer_params,
                self.requires_grad,
                backward_hooks,
            )
            return (torch._utils._rebuild_qtensor, args_qtensor)
        elif self.is_sparse:
            if self.layout == torch.sparse_coo:
                args_sparse = (
                    self.layout,
                    (self._indices(), self._values(), self.size(), self.is_coalesced()),
                )
            else:
                raise NotImplementedError(
                    "sparse tensor __reduce_ex__ for layout `%s`" % (self.layout)
                )
            return (torch._utils._rebuild_sparse_tensor, args_sparse)
        elif self.layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices, plain_indices = (
                    self.crow_indices(),
                    self.col_indices(),
                )
            else:
                compressed_indices, plain_indices = (
                    self.ccol_indices(),
                    self.row_indices(),
                )
            args_sparse_compressed = (
                self.layout,
                (
                    compressed_indices,
                    plain_indices,
                    self.values(),
                    self.size(),
                ),
            )
            return (torch._utils._rebuild_sparse_tensor, args_sparse_compressed)
        elif (
            self.data_ptr() == 0
            and type(self) is not torch.Tensor
            and type(self).__torch_dispatch__ is not torch.Tensor.__torch_dispatch__
        ):
            arg_wrapper_subclass = (
                type(self),
                self.dtype,
                tuple(self.size()),
                self.stride(),
                self.storage_offset(),
                self.layout,
                self.device,
                self.requires_grad,
            )
            return (torch._utils._rebuild_wrapper_subclass, arg_wrapper_subclass)
        else:
            # TODO: Once we decide to break serialization FC, no longer
            # need to wrap with TypedStorage
            args = (
                torch.storage.TypedStorage(
                    wrap_storage=self._typed_storage()._untyped_storage,
                    dtype=self.dtype,
                    _internal=True,
                ),
                self.storage_offset(),
                tuple(self.size()),
                self.stride(),
                self.requires_grad,
                backward_hooks,
            )  # previously was self._backward_hooks

            metadata = torch._utils.get_tensor_metadata(self)
            if metadata:
                args = args + (metadata,)  # type: ignore[assignment]
            return (torch._utils._rebuild_tensor_v2, args)

    def __setstate__(self, state):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__setstate__, (self,), self, state)
        # Warning: this method is NOT called when you torch.load() a tensor;
        # that is managed by _rebuild_tensor_v2
        if not self.is_leaf:
            raise RuntimeError("__setstate__ can be only called on leaf Tensors")
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

    def __repr__(self, *, tensor_contents=None):
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.__repr__, (self,), self, tensor_contents=tensor_contents
            )
        # All strings are unicode in Python 3.
        return torch._tensor_str._str(self, tensor_contents=tensor_contents)

    def backward(
        self, gradient=None, retain_graph=None, create_graph=False, inputs=None
    ):
        r"""Computes the gradient of current tensor wrt graph leaves.

        The graph is differentiated using the chain rule. If the tensor is
        non-scalar (i.e. its data has more than one element) and requires
        gradient, the function additionally requires specifying ``gradient``.
        It should be a tensor of matching type and location, that contains
        the gradient of the differentiated function w.r.t. ``self``.

        This function accumulates gradients in the leaves - you might need to zero
        ``.grad`` attributes or set them to ``None`` before calling it.
        See :ref:`Default gradient layouts<default-grad-layouts>`
        for details on the memory layout of accumulated gradients.

        .. note::

            If you run any forward ops, create ``gradient``, and/or call ``backward``
            in a user-specified CUDA stream context, see
            :ref:`Stream semantics of backward passes<bwd-cuda-stream-semantics>`.

        .. note::

            When ``inputs`` are provided and a given input is not a leaf,
            the current implementation will call its grad_fn (though it is not strictly needed to get this gradients).
            It is an implementation detail on which the user should not rely.
            See https://github.com/pytorch/pytorch/pull/60521#issuecomment-867061780 for more details.

        Args:
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
            inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be
                accumulated into ``.grad``. All other Tensors will be ignored. If not
                provided, the gradient is accumulated into all the leaf Tensors that were
                used to compute the attr::tensors.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.backward,
                (self,),
                self,
                gradient=gradient,
                retain_graph=retain_graph,
                create_graph=create_graph,
                inputs=inputs,
            )
        torch.autograd.backward(
            self, gradient, retain_graph, create_graph, inputs=inputs
        )

    def register_hook(self, hook):
        r"""Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        Tensor is computed. The hook should have the following signature::

            hook(grad) -> Tensor or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

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
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.register_hook, (self,), self, hook)
        if not self.requires_grad:
            raise RuntimeError(
                "cannot register a hook on a tensor that " "doesn't require gradient"
            )
        if self._backward_hooks is None:
            self._backward_hooks = OrderedDict()
            if self.grad_fn is not None:
                self.grad_fn._register_hook_dict(self)
        handle = hooks.RemovableHandle(self._backward_hooks)
        self._backward_hooks[handle.id] = hook
        return handle

    def reinforce(self, reward):
        def trim(str):
            return "\n".join([line.strip() for line in str.split("\n")])

        raise RuntimeError(
            trim(
                r"""reinforce() was removed.
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
        """
            )
        )

    detach = _C._add_docstr(
        _C._TensorBase.detach,
        r"""
    Returns a new Tensor, detached from the current graph.

    The result will never require gradient.

    This method also affects forward mode AD gradients and the result will never
    have forward mode AD gradients.

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
    """,
    )

    detach_ = _C._add_docstr(
        _C._TensorBase.detach_,
        r"""
    Detaches the Tensor from the graph that created it, making it a leaf.
    Views cannot be detached in-place.

    This method also affects forward mode AD gradients and the result will never
    have forward mode AD gradients.
    """,
    )

    def is_shared(self):
        r"""Checks if tensor is in shared memory.

        This is always ``True`` for CUDA tensors.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.is_shared, (self,), self)
        return self._typed_storage()._is_shared()

    def share_memory_(self):
        r"""Moves the underlying storage to shared memory.

        This is a no-op if the underlying storage is already in shared memory
        and for CUDA tensors. Tensors in shared memory cannot be resized.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.share_memory_, (self,), self)
        self._typed_storage()._share_memory_()
        return self

    def __reversed__(self):
        r"""Reverses the tensor along dimension 0."""
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__reversed__, (self,), self)
        if self.dim() == 0:
            return self
        else:
            return self.flip(0)

    def norm(
        self,
        p: Optional[Union[float, str]] = "fro",
        dim=None,
        keepdim=False,
        dtype=None,
    ):
        r"""See :func:`torch.norm`"""
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.norm, (self,), self, p=p, dim=dim, keepdim=keepdim, dtype=dtype
            )
        return torch.norm(self, p, dim, keepdim, dtype=dtype)

    def solve(self, other):
        from ._linalg_utils import solve

        return solve(self, other)

    def lstsq(self, other):
        from ._linalg_utils import lstsq

        return lstsq(self, other)

    def eig(self, eigenvectors=False):
        from ._linalg_utils import eig

        return eig(self, eigenvectors=eigenvectors)

    def symeig(self, eigenvectors=False):
        from ._linalg_utils import _symeig

        return _symeig(self, eigenvectors=eigenvectors)

    def lu(self, pivot=True, get_infos=False):
        r"""See :func:`torch.lu`"""
        # If get_infos is True, then we don't need to check for errors and vice versa
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.lu, (self,), self, pivot=pivot, get_infos=get_infos
            )

        LU, pivots, infos = torch._lu_with_info(
            self, pivot=pivot, check_errors=(not get_infos)
        )
        if get_infos:
            return LU, pivots, infos
        else:
            return LU, pivots

    def stft(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "Optional[Tensor]" = None,
        center: bool = True,
        pad_mode: str = "reflect",
        normalized: bool = False,
        onesided: Optional[bool] = None,
        return_complex: Optional[bool] = None,
    ):
        r"""See :func:`torch.stft`

        .. warning::
          This function changed signature at version 0.4.1. Calling with
          the previous signature may cause error or return incorrect result.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.stft,
                (self,),
                self,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                pad_mode=pad_mode,
                normalized=normalized,
                onesided=onesided,
                return_complex=return_complex,
            )
        return torch.stft(
            self,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            pad_mode,
            normalized,
            onesided,
            return_complex=return_complex,
        )

    def istft(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: "Optional[Tensor]" = None,
        center: bool = True,
        normalized: bool = False,
        onesided: Optional[bool] = None,
        length: Optional[int] = None,
        return_complex: bool = False,
    ):
        r"""See :func:`torch.istft`"""
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.istft,
                (self,),
                self,
                n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=center,
                normalized=normalized,
                onesided=onesided,
                length=length,
                return_complex=return_complex,
            )
        return torch.istft(
            self,
            n_fft,
            hop_length,
            win_length,
            window,
            center,
            normalized,
            onesided,
            length,
            return_complex=return_complex,
        )

    def resize(self, *sizes):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.resize, (self,), self, *sizes)
        warnings.warn("non-inplace resize is deprecated")
        from torch.autograd._functions import Resize

        return Resize.apply(self, sizes)

    def resize_as(self, tensor):
        if has_torch_function_variadic(self, tensor):
            return handle_torch_function(Tensor.resize_as, (self, tensor), self, tensor)
        warnings.warn("non-inplace resize_as is deprecated")
        from torch.autograd._functions import Resize

        return Resize.apply(self, tensor.size())

    def split(self, split_size, dim=0):
        r"""See :func:`torch.split`"""
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.split, (self,), self, split_size, dim=dim
            )
        if isinstance(split_size, Tensor):
            try:
                split_size = int(split_size)
            except ValueError:
                pass

        if isinstance(split_size, (int, torch.SymInt)):
            return torch._VF.split(self, split_size, dim)  # type: ignore[attr-defined]
        else:
            return torch._VF.split_with_sizes(self, split_size, dim)

    def unique(self, sorted=True, return_inverse=False, return_counts=False, dim=None):
        r"""Returns the unique elements of the input tensor.

        See :func:`torch.unique`
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.unique,
                (self,),
                self,
                sorted=sorted,
                return_inverse=return_inverse,
                return_counts=return_counts,
                dim=dim,
            )
        return torch.unique(
            self,
            sorted=sorted,
            return_inverse=return_inverse,
            return_counts=return_counts,
            dim=dim,
        )

    def unique_consecutive(self, return_inverse=False, return_counts=False, dim=None):
        r"""Eliminates all but the first element from every consecutive group of equivalent elements.

        See :func:`torch.unique_consecutive`
        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.unique_consecutive,
                (self,),
                self,
                return_inverse=return_inverse,
                return_counts=return_counts,
                dim=dim,
            )
        return torch.unique_consecutive(
            self, return_inverse=return_inverse, return_counts=return_counts, dim=dim
        )

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rsub__(self, other):
        return _C._VariableFunctions.rsub(self, other)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rdiv__(self, other):
        return self.reciprocal() * other

    __rtruediv__ = __rdiv__
    __itruediv__ = _C._TensorBase.__idiv__

    __pow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented(
        _C._TensorBase.pow
    )
    __ipow__ = _handle_torch_function_and_wrap_type_error_to_not_implemented(
        _C._TensorBase.pow_
    )

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmod__(self, other):
        return torch.remainder(other, self)

    def __format__(self, format_spec):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__format__, (self,), self, format_spec)
        if self.dim() == 0 and not self.is_meta and type(self) is Tensor:
            return self.item().__format__(format_spec)
        return object.__format__(self, format_spec)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rpow__(self, other):
        dtype = torch.result_type(other, self)
        return torch.tensor(other, dtype=dtype, device=self.device) ** self

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __floordiv__(self, other):
        return torch.floor_divide(self, other)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rfloordiv__(self, other):
        return torch.floor_divide(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rlshift__(self, other):
        return torch.bitwise_left_shift(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rrshift__(self, other):
        return torch.bitwise_right_shift(other, self)

    @_handle_torch_function_and_wrap_type_error_to_not_implemented
    def __rmatmul__(self, other):
        return torch.matmul(other, self)

    __pos__ = _C._TensorBase.positive
    __neg__ = _C._TensorBase.neg
    __abs__ = _C._TensorBase.abs

    def __len__(self):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__len__, (self,), self)
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        if torch._C._get_tracing_state():
            warnings.warn(
                "Using len to get tensor shape might cause the trace to be incorrect. "
                "Recommended usage would be tensor.shape[0]. "
                "Passing a tensor of different shape might lead to errors or silently give "
                "incorrect results.",
                category=torch.jit.TracerWarning,
                stacklevel=2,
            )
        return self.shape[0]

    def __iter__(self):
        # NB: we use 'imap' and not 'map' here, so that in Python 2 we get a
        # generator and don't eagerly perform all the indexes.  This could
        # save us work, and also helps keep trace ordering deterministic
        # (e.g., if you zip(*hiddens), the eager map will force all the
        # indexes of hiddens[0] before hiddens[1], while the generator
        # map will interleave them.)
        # NB: We have intentionally skipped __torch_function__ dispatch here.
        # See gh-54457
        if self.dim() == 0:
            raise TypeError("iteration over a 0-d tensor")
        if torch._C._get_tracing_state():
            warnings.warn(
                "Iterating over a tensor might cause the trace to be incorrect. "
                "Passing a tensor of different shape won't change the number of "
                "iterations executed (and might lead to errors or silently give "
                "incorrect results).",
                category=torch.jit.TracerWarning,
                stacklevel=2,
            )
        return iter(self.unbind(0))

    def __hash__(self):
        # Do NOT handle __torch_function__ here as user's default
        # implementation that handle most functions will most likely do it wrong.
        # It can be easily overridden by defining this method on the user
        # subclass if needed.
        return id(self)

    def __dir__(self):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dir__, (self,), self)
        tensor_methods = dir(self.__class__)
        tensor_methods.remove("volatile")  # deprecated
        attrs = list(self.__dict__.keys())
        keys = tensor_methods + attrs

        # property only available dense, cuda tensors
        if (not self.is_cuda) or self.is_sparse:
            keys.remove("__cuda_array_interface__")

        return sorted(keys)

    # Numpy array interface, to support `numpy.asarray(tensor) -> ndarray`
    __array_priority__ = 1000  # prefer Tensor ops over numpy ones

    def __array__(self, dtype=None):
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__array__, (self,), self, dtype=dtype)
        if dtype is None:
            return self.numpy()
        else:
            return self.numpy().astype(dtype, copy=False)

    # Wrap Numpy array again in a suitable tensor when done, to support e.g.
    # `numpy.sin(tensor) -> tensor` or `numpy.greater(tensor, 0) -> ByteTensor`
    def __array_wrap__(self, array):
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.__array_wrap__, (self,), self, array=array
            )
        if array.dtype == bool:
            # Workaround, torch has no built-in bool tensor
            array = array.astype("uint8")
        return torch.from_numpy(array)

    def __contains__(self, element):
        r"""Check if `element` is present in tensor

        Args:
            element (Tensor or scalar): element to be checked
                for presence in current tensor"
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__contains__, (self,), self, element)
        if isinstance(
            element, (torch.Tensor, Number, torch.SymInt, torch.SymFloat, torch.SymBool)
        ):
            # type hint doesn't understand the __contains__ result array
            return (element == self).any().item()  # type: ignore[union-attr]

        raise RuntimeError(
            "Tensor.__contains__ only supports Tensor or scalar, but you passed in a %s."
            % type(element)
        )

    @property
    def __cuda_array_interface__(self):
        """Array view description for cuda tensors.

        See:
        https://numba.pydata.org/numba-doc/latest/cuda/cuda_array_interface.html
        """
        if has_torch_function_unary(self):
            # TODO mypy doesn't support @property, see: https://github.com/python/mypy/issues/6185
            return handle_torch_function(Tensor.__cuda_array_interface__.__get__, (self,), self)  # type: ignore[attr-defined]

        # raise AttributeError for unsupported tensors, so that
        # hasattr(cpu_tensor, "__cuda_array_interface__") is False.
        if not self.is_cuda:
            raise AttributeError(
                "Can't get __cuda_array_interface__ on non-CUDA tensor type: %s "
                "If CUDA data is required use tensor.cuda() to copy tensor to device memory."
                % self.type()
            )

        if self.is_sparse:
            raise AttributeError(
                "Can't get __cuda_array_interface__ on sparse type: %s "
                "Use Tensor.to_dense() to convert to a dense tensor first."
                % self.type()
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
            torch.complex64: "<c8",
            torch.complex128: "<c16",
            torch.float16: "<f2",
            torch.float32: "<f4",
            torch.float64: "<f8",
            torch.uint8: "|u1",
            torch.int8: "|i1",
            torch.int16: "<i2",
            torch.int32: "<i4",
            torch.int64: "<i8",
        }[self.dtype]

        itemsize = self.element_size()

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

    def storage_type(self):
        r"""storage_type() -> type

        Returns the type of the underlying storage.

        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.storage_type, (self,), self)

        torch.storage._warn_typed_storage_removal()

        return self._typed_storage()._get_legacy_storage_class()

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

        Args:
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
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.refine_names, (self,), self, *names)
        names = resolve_ellipsis(names, self.names, "refine_names")
        return super().refine_names(names)

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

        Args:
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
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.align_to, (self,), self, *names)
        ellipsis_idx = single_ellipsis_index(names, "align_to")
        if ellipsis_idx is None:
            return super().align_to(names)
        return super().align_to(
            [name for name in names if not is_ellipsis(name)], ellipsis_idx
        )

    def unflatten(self, dim, sizes):
        r"""
        unflatten(dim, sizes) -> Tensor

        See :func:`torch.unflatten`.

        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.unflatten, (self,), self, dim, sizes)

        if not sizes:
            raise RuntimeError("unflatten: sizes must be non-empty")

        names = None
        if isinstance(sizes, OrderedDict) or (
            isinstance(sizes, (tuple, list)) and isinstance(sizes[0], (tuple, list))
        ):
            names, sizes = unzip_namedshape(sizes)
            return super().unflatten(dim, sizes, names)
        else:
            return super().unflatten(dim, sizes)

    def rename_(self, *names, **rename_map):
        """In-place version of :meth:`~Tensor.rename`."""

        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.rename_, (self,), self, *names, **rename_map
            )

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
            (None, None, None, None)

            >>> renamed_imgs = imgs.rename('batch', 'channel', 'height', 'width')
            >>> renamed_imgs.names
            ('batch', 'channel', 'height', 'width')

        .. warning::
            The named tensor API is experimental and subject to change.

        """
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor.rename, (self,), self, *names, **rename_map
            )

        # See Note [rename_ / rename API]
        return update_names(self, names, rename_map, inplace=False)

    def to_sparse_coo(self):
        """Convert a tensor to :ref:`coordinate format <sparse-coo-docs>`.

        Examples::

             >>> dense = torch.randn(5, 5)
             >>> sparse = dense.to_sparse_coo()
             >>> sparse._nnz()
             25

        """
        return self.to_sparse()

    def _update_names(self, names, inplace):
        if has_torch_function_unary(self):
            return handle_torch_function(
                Tensor._update_names, (self,), self, names, inplace
            )

        # See Note [rename_ / rename API]
        if inplace:
            return super().rename_(names)
        else:
            return super().rename(names)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        This __torch_function__ implementation wraps subclasses such that
        methods called on subclasses return a subclass instance instead of
        a ``torch.Tensor`` instance.

        One corollary to this is that you need coverage for torch.Tensor
        methods if implementing __torch_function__ for subclasses.

        We recommend always calling ``super().__torch_function__`` as the base
        case when doing the above.

        While not mandatory, we recommend making `__torch_function__` a classmethod.
        """
        if kwargs is None:
            kwargs = {}

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        with _C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return _convert(ret, cls)

    __torch_dispatch__ = _C._disabled_torch_dispatch_impl

    def __dlpack__(self, stream=None):
        """
        Creates a DLpack `capsule https://data-apis.org/array-api/latest/design_topics/data_interchange.html#data-interchange`_
        of the current tensor to be exported to other libraries.

        This function will be called from the `from_dlpack` method
        of the library that will consume the capsule. `from_dlpack` passes the current
        stream to this method as part of the specification.

        Args:
            stream (integer or None): An optional Python integer representing a
            pointer to a CUDA stream. The current stream is synchronized with
            this stream before the capsule is created, and since the capsule
            shares its storage with the tensor this make it safe to access from
            both streams.  If None or -1 is passed then no synchronization is performed.
            If 1 (on CUDA) or 0 (on ROCM) then the default stream is used for
            synchronization.
        """
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dlpack__, (self,), self, stream)

        # DLPack capsules can't capture all of PyTorch's semantics,
        # so we prohibit exporting tensors that would lose their properties like
        # requires_grad and having the conjugate bit set.
        if self.requires_grad:
            raise RuntimeError(
                "Can't export tensors that require gradient, use tensor.detach()"
            )
        if self.is_conj():
            raise RuntimeError("Can't export tensors with the conjugate bit set")
        if self.layout != torch.strided:
            raise RuntimeError(
                "Can't export tensors with layout other than torch.strided"
            )

        if stream is not None and type(stream) is not int:
            # Stream pointers in CUDA/ROCm are uniquely numbered and can
            # be retrieved from their integer value.
            raise TypeError("stream must be ``int`` or ``none``")
        elif stream is not None and stream != -1:
            if self.device.type == "cuda":
                # NB: This logic handles the special case values for default
                # streams and must be kept in sync with from_dlpack in
                # torch/utils/dlpack.py
                if stream == 1 and torch.version.hip is None:
                    stream = torch.cuda.default_stream()
                elif stream == 0 and torch.version.hip is not None:
                    stream = torch.cuda.default_stream()
                else:
                    stream = torch.cuda.ExternalStream(stream)
                # Only synchronize on different streams
                sync_stream = torch.cuda.current_stream()
                if stream != sync_stream:
                    event = torch.cuda.Event()
                    event.record(sync_stream)
                    stream.wait_event(event)
        return torch.to_dlpack(self)

    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]:
        if has_torch_function_unary(self):
            return handle_torch_function(Tensor.__dlpack_device__, (self,), self)
        device = self.device
        idx = device.index if device.index is not None else 0
        torch_device_type = device.type
        if torch_device_type == "cuda" and torch.version.hip is not None:
            device_type = DLDeviceType.kDLROCM
        elif torch_device_type == "cpu" and self.is_pinned():
            device_type = DLDeviceType.kDLCPUPinned
        elif torch_device_type == "cuda":
            device_type = DLDeviceType.kDLGPU
        elif torch_device_type == "cpu":
            device_type = DLDeviceType.kDLCPU
        elif self.device.type == "xpu":
            device_type = DLDeviceType.kDLOneAPI
        else:
            raise ValueError(
                "Unknown device type {} for Dlpack".format(torch_device_type)
            )
        return (device_type, idx)

    __module__ = "torch"


def _convert(ret, cls):
    if cls is Tensor:
        return ret

    if isinstance(ret, Tensor) and not isinstance(ret, cls):
        ret = ret.as_subclass(cls)

    if isinstance(ret, (tuple, list)):
        # Also handles things like namedtuples
        ret = type(ret)(_convert(r, cls) for r in ret)

    return ret
