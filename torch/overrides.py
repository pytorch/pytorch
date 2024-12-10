"""
Python implementation of ``__torch_function__``

While most of the torch API and handling for ``__torch_function__`` happens
at the C++ level, some of the torch API is written in Python so we need
python-level handling for ``__torch_function__`` overrides as well. The main
developer-facing functionality in this file are handle_torch_function and
has_torch_function. See torch/functional.py and test/test_overrides.py
for usage examples.

Note
----
heavily inspired by NumPy's ``__array_function__`` (see:
https://github.com/pytorch/pytorch/issues/24015 and
https://www.numpy.org/neps/nep-0018-array-function-protocol.html
)

If changing this file in a way that can affect ``__torch_function__`` overhead,
please report the benchmarks in ``benchmarks/overrides_benchmark``. See the
instructions in the ``README.md`` in that directory.
"""

import __future__  # noqa: F404

import collections
import contextlib
import functools
import types
import warnings
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type

import torch
from torch._C import (
    _add_docstr,
    _get_function_stack_at,
    _has_torch_function,
    _has_torch_function_unary,
    _has_torch_function_variadic,
    _is_torch_function_mode_enabled,
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)


__all__ = [
    "get_ignored_functions",
    "get_overridable_functions",
    "get_testing_overrides",
    "handle_torch_function",
    "has_torch_function",
    "resolve_name",
    "is_tensor_like",
    "is_tensor_method_or_property",
    "wrap_torch_function",
    "enable_reentrant_dispatch",
]


def _disable_user_warnings(
    func: Callable,
    regex: str = ".*is deprecated, please use.*",
    module: str = "torch",
) -> Callable:
    """
    Decorator that temporarily disables ``UserWarning``s for the given ``module`` if the warning message matches the
    given ``regex`` pattern.

    Arguments
    ---------
    func : function
        Function to disable the warnings for.
    regex : str
        A regex pattern compilable by ``re.compile``. This is used to match the ``UserWarning`` message.
    module : str
        The python module to which the filtering should be restricted.

    Returns
    -------
    function
        The wrapped function.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, message=regex, module=module
            )
            return func(*args, **kwargs)

    return wrapper


@functools.lru_cache(None)
@_disable_user_warnings
def get_ignored_functions() -> Set[Callable]:
    """
    Return public functions that cannot be overridden by ``__torch_function__``.

    Returns
    -------
    Set[Callable]
        A tuple of functions that are publicly available in the torch API but cannot
        be overridden with ``__torch_function__``. Mostly this is because none of the
        arguments of these functions are tensors or tensor-likes.

    Examples
    --------
    >>> torch.Tensor.as_subclass in torch.overrides.get_ignored_functions()
    True
    >>> torch.add in torch.overrides.get_ignored_functions()
    False
    """
    Tensor = torch.Tensor
    return {
        torch.typename,
        torch.is_tensor,
        torch.is_storage,
        torch.set_default_tensor_type,
        torch.set_default_device,
        torch.get_default_device,
        torch.set_rng_state,
        torch.get_rng_state,
        torch.manual_seed,
        torch.initial_seed,
        torch.seed,
        torch.save,
        torch.load,
        torch.set_printoptions,
        torch.fork,
        torch.get_default_dtype,
        torch.get_num_interop_threads,
        torch.get_num_threads,
        torch.init_num_threads,
        torch.import_ir_module,
        torch.import_ir_module_from_buffer,
        torch.is_anomaly_enabled,
        torch.is_anomaly_check_nan_enabled,
        torch.is_grad_enabled,
        torch.merge_type_from_type_comment,
        torch.parse_ir,
        torch.parse_schema,
        torch.parse_type_comment,
        torch.set_anomaly_enabled,
        torch.set_flush_denormal,
        torch.set_num_interop_threads,
        torch.set_num_threads,
        torch.wait,
        torch.as_tensor,
        torch.from_numpy,
        torch.tensor,
        torch.default_generator,
        torch.has_cuda,
        torch.has_cudnn,
        torch.has_lapack,
        torch.device,
        torch.dtype,
        torch.finfo,
        torch.has_mkl,
        torch.has_mps,
        torch.has_mkldnn,
        torch.has_openmp,
        torch.iinfo,
        torch.memory_format,
        torch.qscheme,
        torch.set_grad_enabled,
        torch.no_grad,
        torch.enable_grad,
        torch.inference_mode,
        torch.is_inference_mode_enabled,
        torch.layout,
        torch.align_tensors,
        torch.arange,
        torch.as_strided,
        torch.bartlett_window,
        torch.blackman_window,
        torch.broadcast_shapes,
        torch.can_cast,
        torch.compile,
        torch.cudnn_affine_grid_generator,
        torch.cudnn_batch_norm,
        torch.cudnn_convolution,
        torch.cudnn_convolution_transpose,
        torch.cudnn_convolution_relu,
        torch.cudnn_convolution_add_relu,
        torch.cudnn_grid_sampler,
        torch.cudnn_is_acceptable,
        torch.empty,
        torch.empty_permuted,
        torch.empty_strided,
        torch.empty_quantized,
        torch.export.export,
        torch.export.load,
        torch.export.register_dataclass,
        torch.export.save,
        torch.eye,
        torch.fft.fftfreq,
        torch.fft.rfftfreq,
        torch.from_file,
        torch.full,
        torch.fill,
        torch.hamming_window,
        torch.hann_window,
        torch.kaiser_window,
        torch.linspace,
        torch.logspace,
        torch.mkldnn_adaptive_avg_pool2d,
        torch.mkldnn_convolution,
        torch.mkldnn_max_pool2d,
        torch.mkldnn_max_pool3d,
        torch.mkldnn_linear_backward_weights,
        torch.mkldnn_rnn_layer,
        torch.normal,
        torch.ones,
        torch.promote_types,
        torch.rand,
        torch.randn,
        torch.randint,
        torch.randperm,
        torch.range,
        torch.result_type,
        torch.scalar_tensor,
        torch.sparse_coo_tensor,
        torch.sparse_compressed_tensor,
        torch.sparse_csr_tensor,
        torch.sparse_csc_tensor,
        torch.sparse_bsr_tensor,
        torch.sparse_bsc_tensor,
        torch.sym_constrain_range,
        torch.sym_constrain_range_for_size,
        torch.sym_fresh_size,
        torch.tril_indices,
        torch.triu_indices,
        torch.vander,
        torch.zeros,
        torch._jit_internal.boolean_dispatch,
        torch.nn.functional.assert_int_or_pair,
        torch.nn.functional.upsample,
        torch.nn.functional.upsample_bilinear,
        torch.nn.functional.upsample_nearest,
        torch.nn.functional.has_torch_function,
        torch.nn.functional.has_torch_function_unary,
        torch.nn.functional.has_torch_function_variadic,
        torch.nn.functional.handle_torch_function,
        torch.nn.functional.sigmoid,
        torch.nn.functional.hardsigmoid,
        torch.nn.functional.tanh,
        torch.nn.functional._canonical_mask,
        torch.nn.functional._none_or_dtype,
        # Doesn't actually take or return tensor arguments
        torch.nn.init.calculate_gain,
        # These are deprecated; don't test them
        torch.nn.init.uniform,
        torch.nn.init.normal,
        torch.nn.init.constant,
        torch.nn.init.eye,
        torch.nn.init.dirac,
        torch.nn.init.xavier_uniform,
        torch.nn.init.xavier_normal,
        torch.nn.init.kaiming_uniform,
        torch.nn.init.kaiming_normal,
        torch.nn.init.orthogonal,
        torch.nn.init.sparse,
        torch.nested.to_padded_tensor,
        has_torch_function,
        handle_torch_function,
        torch.set_autocast_enabled,
        torch.is_autocast_enabled,
        torch.set_autocast_dtype,
        torch.get_autocast_dtype,
        torch.clear_autocast_cache,
        torch.set_autocast_cpu_enabled,
        torch.is_autocast_cpu_enabled,
        torch.set_autocast_xla_enabled,
        torch.is_autocast_xla_enabled,
        torch.set_autocast_ipu_enabled,
        torch.is_autocast_ipu_enabled,
        torch.set_autocast_cpu_dtype,
        torch.get_autocast_cpu_dtype,
        torch.set_autocast_ipu_dtype,
        torch.get_autocast_ipu_dtype,
        torch.get_autocast_gpu_dtype,
        torch.set_autocast_gpu_dtype,
        torch.get_autocast_xla_dtype,
        torch.set_autocast_xla_dtype,
        torch.autocast_increment_nesting,
        torch.autocast_decrement_nesting,
        torch.is_autocast_cache_enabled,
        torch.set_autocast_cache_enabled,
        torch.nn.functional.hardswish,
        torch.is_vulkan_available,
        torch.are_deterministic_algorithms_enabled,
        torch.use_deterministic_algorithms,
        torch.is_deterministic_algorithms_warn_only_enabled,
        torch.set_deterministic_debug_mode,
        torch.get_device_module,
        torch.get_deterministic_debug_mode,
        torch.set_float32_matmul_precision,
        torch.get_float32_matmul_precision,
        torch.unify_type_list,
        torch.is_warn_always_enabled,
        torch.set_warn_always,
        torch.vitals_enabled,
        torch.set_vital,
        torch.read_vitals,
        torch.vmap,
        torch.cond,
        torch.frombuffer,
        torch.asarray,
        torch._functional_sym_constrain_range,
        torch._make_dep_token,
        Tensor.__delitem__,
        Tensor.__dir__,
        Tensor.__getattribute__,
        Tensor.__init__,
        Tensor.__iter__,
        Tensor.__init_subclass__,
        Tensor.__delattr__,
        Tensor.__setattr__,
        Tensor.__torch_function__,
        Tensor.__torch_dispatch__,
        Tensor.__new__,
        Tensor.__class__,
        Tensor.__subclasshook__,
        Tensor.__hash__,
        Tensor.as_subclass,
        Tensor.eig,
        Tensor.lstsq,
        Tensor.reinforce,
        Tensor.new,
        Tensor.new_tensor,
        Tensor.new_empty,
        Tensor.new_empty_strided,
        Tensor.new_zeros,
        Tensor.new_ones,
        Tensor.new_full,
        Tensor._make_subclass,
        Tensor.solve,
        Tensor.symeig,
        Tensor.stride,
        Tensor.unflatten,
        Tensor.to_sparse_coo,
        Tensor.to_sparse_csr,
        Tensor.to_sparse_csc,
        Tensor.to_sparse_bsr,
        Tensor.to_sparse_bsc,
        Tensor._to_sparse,
        Tensor._to_sparse_csr,
        Tensor._to_sparse_csc,
        Tensor._to_sparse_bsr,
        Tensor._to_sparse_bsc,
        Tensor._typed_storage,
        Tensor._reduce_ex_internal,
        Tensor._fix_weakref,
        Tensor._view_func,
        Tensor._view_func_unsafe,
        Tensor._rev_view_func_unsafe,
        Tensor._make_wrapper_subclass,
        Tensor._python_dispatch.__get__,
        Tensor._has_symbolic_sizes_strides.__get__,
        Tensor._conj,
        Tensor._conj_physical,
        Tensor._lazy_clone,
        Tensor._neg_view,
        Tensor._is_zerotensor,
        Tensor._is_all_true,
        Tensor._is_any_true,
        Tensor._addmm_activation,
        Tensor.to_padded_tensor,
        Tensor._use_count,
    }


@functools.lru_cache(None)
def get_default_nowrap_functions() -> Set[Callable]:
    """
    Return public functions that do not wrap in a subclass when invoked by
    the default ``Tensor.__torch_function__`` that preserves subclasses.  Typically,
    these functions represent field accesses (i.e., retrieving a Tensor that
    is stored somewhere on the Tensor) as opposed to computation.  Users of
    these functions expect object identity to be preserved over multiple accesses
    (e.g., ``a.grad is a.grad``) which cannot be upheld if we're wrapping on
    the fly every time (furthermore, the tensor stored here might already be
    the subclass, in which case wrapping really ought not to happen).

    Not ALL property accessors have this property; for example ``Tensor.T`` actually
    just creates a new transposed tensor on the fly, and so we SHOULD interpose on
    these calls (you need to check the implementation of the function to see if
    this is the case or not).  Additionally, if a property accessor doesn't return a Tensor,
    it doesn't have to be on this list (though it is harmless if it is).
    """
    Tensor = torch.Tensor
    return {
        Tensor._base.__get__,
        Tensor.grad.__get__,
        Tensor._grad.__get__,
    }


@functools.lru_cache(None)
@_disable_user_warnings
def get_testing_overrides() -> Dict[Callable, Callable]:
    """Return a dict containing dummy overrides for all overridable functions

    Returns
    -------
    Dict[Callable, Callable]
        A dictionary that maps overridable functions in the PyTorch API to
        lambda functions that have the same signature as the real function
        and unconditionally return -1. These lambda functions are useful
        for testing API coverage for a type that defines ``__torch_function__``.

    Examples
    --------
    >>> import inspect
    >>> my_add = torch.overrides.get_testing_overrides()[torch.add]
    >>> inspect.signature(my_add)
    <Signature (input, other, out=None)>
    """
    # Every function in the PyTorchAPI that can be overriden needs an entry
    # in this dict.
    #
    # Optimally we would use inspect to get the function signature and define
    # the lambda function procedurally but that is blocked by generating
    # function signatures for native kernels that can be consumed by inspect.
    # See Issue #28233.
    Tensor = torch.Tensor
    ret: Dict[Callable, Callable] = {
        torch.abs: lambda *args, **kwargs: -1,
        torch.absolute: lambda *args, **kwargs: -1,
        torch.adaptive_avg_pool1d: lambda *args, **kwargs: -1,
        torch.adaptive_max_pool1d: lambda *args, **kwargs: -1,
        torch.acos: lambda *args, **kwargs: -1,
        torch.adjoint: lambda *args, **kwargs: -1,
        torch.arccos: lambda *args, **kwargs: -1,
        torch.acosh: lambda *args, **kwargs: -1,
        torch.arccosh: lambda *args, **kwargs: -1,
        torch.add: lambda *args, **kwargs: -1,
        torch.addbmm: lambda *args, **kwargs: -1,
        torch.addcdiv: lambda *args, **kwargs: -1,
        torch.addcmul: lambda *args, **kwargs: -1,
        torch.addmm: lambda *args, **kwargs: -1,
        torch.addmv: lambda *args, **kwargs: -1,
        torch.addr: lambda *args, **kwargs: -1,
        torch.affine_grid_generator: lambda *args, **kwargs: -1,
        torch.all: lambda *args, **kwargs: -1,
        torch.allclose: lambda *args, **kwargs: -1,
        torch.alpha_dropout: lambda *args, **kwargs: -1,
        torch.amax: lambda *args, **kwargs: -1,
        torch.amin: lambda *args, **kwargs: -1,
        torch.aminmax: lambda *args, **kwargs: -1,
        torch.angle: lambda *args, **kwargs: -1,
        torch.any: lambda *args, **kwargs: -1,
        torch.argmax: lambda *args, **kwargs: -1,
        torch.argmin: lambda *args, **kwargs: -1,
        torch.argsort: lambda *args, **kwargs: -1,
        torch.asin: lambda *args, **kwargs: -1,
        torch._assert_async: lambda *args, **kwargs: -1,
        torch.arcsin: lambda *args, **kwargs: -1,
        torch.asinh: lambda *args, **kwargs: -1,
        torch.arcsinh: lambda *args, **kwargs: -1,
        torch.atan: lambda *args, **kwargs: -1,
        torch.arctan: lambda *args, **kwargs: -1,
        torch.atan2: lambda *args, **kwargs: -1,
        torch.arctan2: lambda *args, **kwargs: -1,
        torch.atanh: lambda *args, **kwargs: -1,
        torch.arctanh: lambda *args, **kwargs: -1,
        torch.atleast_1d: lambda *args, **kwargs: -1,
        torch.atleast_2d: lambda *args, **kwargs: -1,
        torch.atleast_3d: lambda *args, **kwargs: -1,
        torch.avg_pool1d: lambda *args, **kwargs: -1,
        torch.baddbmm: lambda *args, **kwargs: -1,
        torch.batch_norm: lambda *args, **kwargs: -1,
        torch.batch_norm_backward_elemt: lambda *args, **kwargs: -1,
        torch.batch_norm_backward_reduce: lambda *args, **kwargs: -1,
        torch.batch_norm_elemt: lambda *args, **kwargs: -1,
        torch.batch_norm_gather_stats: lambda *args, **kwargs: -1,
        torch.batch_norm_gather_stats_with_counts: lambda *args, **kwargs: -1,
        torch.batch_norm_stats: lambda *args, **kwargs: -1,
        torch.batch_norm_update_stats: lambda *args, **kwargs: -1,
        torch.bernoulli: lambda *args, **kwargs: -1,
        torch.bilinear: lambda *args, **kwargs: -1,
        torch.binary_cross_entropy_with_logits: (
            lambda *args, **kwargs: -1
        ),
        torch.bincount: lambda *args, **kwargs: -1,
        torch.binomial: lambda *args, **kwargs: -1,
        torch.bitwise_and: lambda *args, **kwargs: -1,
        torch.bitwise_not: lambda *args, **kwargs: -1,
        torch.bitwise_or: lambda *args, **kwargs: -1,
        torch.bitwise_xor: lambda *args, **kwargs: -1,
        torch.bitwise_left_shift: lambda *args, **kwargs: -1,
        torch.bitwise_right_shift: lambda *args, **kwargs: -1,
        torch.block_diag: lambda *args, **kwargs: -1,
        torch.bmm: lambda *args, **kwargs: -1,
        torch.broadcast_tensors: lambda *args, **kwargs: -1,
        torch.broadcast_to: lambda *args, **kwargs: -1,
        torch.bucketize: lambda *args, **kwargs: -1,
        torch.cartesian_prod: lambda *args, **kwargs: -1,
        torch.cat: lambda *args, **kwargs: -1,
        torch.concat: lambda *args, **kwargs: -1,  # alias for torch.cat
        torch.concatenate: lambda *args, **kwargs: -1,  # alias for torch.concatenate
        torch.cdist: lambda *args, **kwargs: -1,
        torch.ceil: lambda *args, **kwargs: -1,
        torch.celu: lambda *args, **kwargs: -1,
        torch.chain_matmul: lambda *args, **kwargs: -1,
        torch.channel_shuffle: lambda *args, **kwargs: -1,
        torch.cholesky: lambda *args, **kwargs: -1,
        torch.linalg.cholesky: lambda *args, **kwargs: -1,
        torch.linalg.cholesky_ex: lambda *args, **kwargs: -1,
        torch.cholesky_inverse: lambda *args, **kwargs: -1,
        torch.cholesky_solve: lambda *args, **kwargs: -1,
        torch.choose_qparams_optimized: lambda *args, **kwargs: -1,
        torch.chunk: lambda *args, **kwargs: -1,
        torch.clamp: lambda *args, **kwargs: -1,
        torch.clip: lambda *args, **kwargs: -1,
        torch.clamp_min: lambda *args, **kwargs: -1,
        torch.clamp_max: lambda *args, **kwargs: -1,
        torch.column_stack: lambda *args, **kwargs: -1,
        torch.cov: lambda *args, **kwargs: -1,
        torch.clone: lambda *args, **kwargs: -1,
        torch.combinations: lambda *args, **kwargs: -1,
        torch.complex: lambda *args, **kwargs: -1,
        torch.copysign: lambda *args, **kwargs: -1,
        torch.polar: lambda *args, **kwargs: -1,
        torch.linalg.cond: lambda *args, **kwargs: -1,
        torch.conj: lambda *args, **kwargs: -1,
        torch.conj_physical: lambda *args, **kwargs: -1,
        torch.resolve_conj: lambda *args, **kwargs: -1,
        torch.resolve_neg: lambda *args, **kwargs: -1,
        torch.constant_pad_nd: lambda *args, **kwargs: -1,
        torch.conv1d: lambda *args, **kwargs: -1,
        torch.conv2d: lambda *args, **kwargs: -1,
        torch.conv3d: lambda *args, **kwargs: -1,
        torch.convolution: lambda *args, **kwargs: -1,
        torch.conv_tbc: lambda *args, **kwargs: -1,
        torch.conv_transpose1d: lambda *args, **kwargs: -1,
        torch.conv_transpose2d: lambda *args, **kwargs: -1,
        torch.conv_transpose3d: lambda *args, **kwargs: -1,
        torch.corrcoef: lambda *args, **kwargs: -1,
        torch.cos: lambda *args, **kwargs: -1,
        torch.cosine_embedding_loss: lambda *args, **kwargs: -1,
        torch.cosh: lambda *args, **kwargs: -1,
        torch.cosine_similarity: lambda *args, **kwargs: -1,
        torch.count_nonzero: lambda *args, **kwargs: -1,
        torch.cross: lambda *args, **kwargs: -1,
        torch.linalg.cross: lambda *args, **kwargs: -1,
        torch.ctc_loss: (
            lambda *args, **kwargs: -1
        ),
        torch.cummax: lambda *args, **kwargs: -1,
        torch.cummin: lambda *args, **kwargs: -1,
        torch.cumprod: lambda *args, **kwargs: -1,
        torch.cumsum: lambda *args, **kwargs: -1,
        torch.cumulative_trapezoid: lambda *args, **kwargs: -1,
        torch.logcumsumexp: lambda *args, **kwargs: -1,
        torch.deg2rad: lambda *args, **kwargs: -1,
        torch.dequantize: lambda *args, **kwargs: -1,
        torch.det: lambda *args, **kwargs: -1,
        torch.linalg.det: lambda *args, **kwargs: -1,  # alias for torch.det  # type: ignore[attr-defined]
        torch.detach: lambda *args, **kwargs: -1,
        torch.diag: lambda *args, **kwargs: -1,
        torch.diag_embed: lambda *args, **kwargs: -1,
        torch.diagflat: lambda *args, **kwargs: -1,
        torch.diff: lambda *args, **kwargs: -1,
        torch.diagonal: lambda *args, **kwargs: -1,
        torch.linalg.diagonal: lambda *args, **kwargs: -1,
        torch.diagonal_scatter: lambda *args, **kwargs: -1,
        torch.as_strided_scatter: lambda *args, **kwargs: -1,
        torch.digamma: lambda *args, **kwargs: -1,
        torch.dist: lambda *args, **kwargs: -1,
        torch.div: lambda *args, **kwargs: -1,
        torch.divide: lambda *args, **kwargs: -1,
        torch.dot: lambda *args, **kwargs: -1,
        torch.dropout: lambda *args, **kwargs: -1,
        torch.dsmm: lambda *args, **kwargs: -1,
        torch.hsmm: lambda *args, **kwargs: -1,
        torch.dsplit: lambda *args, **kwargs: -1,
        torch.dstack: lambda *args, **kwargs: -1,
        torch.linalg.eig: lambda *args, **kwargs: -1,
        torch.linalg.eigvals: lambda *args, **kwargs: -1,
        torch.linalg.eigh: lambda *args, **kwargs: -1,
        torch.linalg.eigvalsh: lambda *args, **kwargs: -1,
        torch.einsum: lambda *args, **kwargs: -1,
        torch.embedding: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.embedding_bag: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.empty_like: lambda *args, **kwargs: -1,
        torch.eq: lambda *args, **kwargs: -1,
        torch.equal: lambda *args, **kwargs: -1,
        torch.erf: lambda *args, **kwargs: -1,
        torch.erfc: lambda *args, **kwargs: -1,
        torch.erfinv: lambda *args, **kwargs: -1,
        torch.exp: lambda *args, **kwargs: -1,
        torch.exp2: lambda *args, **kwargs: -1,
        torch.expm1: lambda *args, **kwargs: -1,
        torch.fake_quantize_per_channel_affine: lambda *args, **kwargs: -1,
        torch.fake_quantize_per_tensor_affine: lambda *args, **kwargs: -1,
        torch.fused_moving_avg_obs_fake_quant: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.fbgemm_linear_fp16_weight: lambda *args, **kwargs: -1,
        torch.fbgemm_linear_fp16_weight_fp32_activation: lambda *args, **kwargs: -1,
        torch.fbgemm_linear_int8_weight: lambda *args, **kwargs: -1,  # noqa: B950
        torch.fbgemm_linear_int8_weight_fp32_activation: (
            lambda *args, **kwargs: -1
        ),
        torch.fbgemm_linear_quantize_weight: lambda *args, **kwargs: -1,
        torch.fbgemm_pack_gemm_matrix_fp16: lambda *args, **kwargs: -1,
        torch.fbgemm_pack_quantized_matrix: lambda *args, **kwargs: -1,
        torch.feature_alpha_dropout: lambda *args, **kwargs: -1,
        torch.feature_dropout: lambda *args, **kwargs: -1,
        torch.fft.ifft: lambda *args, **kwargs: -1,
        torch.fft.rfft: lambda *args, **kwargs: -1,
        torch.fft.irfft: lambda *args, **kwargs: -1,
        torch.fft.hfft: lambda *args, **kwargs: -1,
        torch.fft.ihfft: lambda *args, **kwargs: -1,
        torch.fft.hfft2: lambda *args, **kwargs: -1,
        torch.fft.ihfft2: lambda *args, **kwargs: -1,
        torch.fft.hfftn: lambda *args, **kwargs: -1,
        torch.fft.ihfftn: lambda *args, **kwargs: -1,
        torch.fft.fftn: lambda *args, **kwargs: -1,
        torch.fft.ifftn: lambda *args, **kwargs: -1,
        torch.fft.rfftn: lambda *args, **kwargs: -1,
        torch.fft.irfftn: lambda *args, **kwargs: -1,
        torch.fft.fft2: lambda *args, **kwargs: -1,
        torch.fft.ifft2: lambda *args, **kwargs: -1,
        torch.fft.rfft2: lambda *args, **kwargs: -1,
        torch.fft.irfft2: lambda *args, **kwargs: -1,
        torch.fft.fftshift: lambda *args, **kwargs: -1,
        torch.fft.ifftshift: lambda *args, **kwargs: -1,
        torch.fft.fft: lambda *args, **kwargs: -1,
        torch.fix: lambda *args, **kwargs: -1,
        torch.flatten: lambda *args, **kwargs: -1,
        torch.flip: lambda *args, **kwargs: -1,
        torch.fliplr: lambda *args, **kwargs: -1,
        torch.flipud: lambda *args, **kwargs: -1,
        torch.frobenius_norm: lambda *args, **kwargs: -1,
        torch.floor: lambda *args, **kwargs: -1,
        torch.floor_divide: lambda *args, **kwargs: -1,
        torch.float_power: lambda *args, **kwargs: -1,
        torch.fmod: lambda *args, **kwargs: -1,
        torch.frac: lambda *args, **kwargs: -1,
        torch.frexp: lambda *args, **kwargs: -1,
        torch.full_like: lambda *args, **kwargs: -1,  # noqa: B950
        torch._functional_assert_async: lambda *args, **kwargs: -1,
        torch.lu_unpack: lambda *args, **kwargs: -1,
        torch.gather: lambda *args, **kwargs: -1,
        torch.gcd: lambda *args, **kwargs: -1,
        torch.ge: lambda *args, **kwargs: -1,
        torch.get_device: lambda *args, **kwargs: -1,
        torch.greater_equal: lambda *args, **kwargs: -1,
        torch.geqrf: lambda *args, **kwargs: -1,
        torch.i0: lambda *args, **kwargs: -1,
        torch.inner: lambda *args, **kwargs: -1,
        torch.outer: lambda *args, **kwargs: -1,
        torch.ger: lambda *args, **kwargs: -1,  # alias for torch.outer
        torch.gradient: lambda *args, **kwargs: -1,
        torch.grid_sampler: lambda *args, **kwargs: -1,
        torch.grid_sampler_2d: lambda *args, **kwargs: -1,
        torch.grid_sampler_3d: lambda *args, **kwargs: -1,
        torch.group_norm: lambda *args, **kwargs: -1,
        torch.gru: lambda *args, **kwargs: -1,
        torch.gru_cell: lambda *args, **kwargs: -1,
        torch.gt: lambda *args, **kwargs: -1,
        torch.greater: lambda *args, **kwargs: -1,
        torch.hardshrink: lambda *args, **kwargs: -1,
        torch.heaviside: lambda *args, **kwargs: -1,
        torch.hinge_embedding_loss: lambda *args, **kwargs: -1,  # noqa: B950
        torch.histc: lambda *args, **kwargs: -1,
        torch.histogram: lambda *args, **kwargs: -1,
        torch.histogramdd: lambda *args, **kwargs: -1,
        torch.linalg.householder_product: lambda *args, **kwargs: -1,
        torch.hspmm: lambda *args, **kwargs: -1,
        torch.hsplit: lambda *args, **kwargs: -1,
        torch.hstack: lambda *args, **kwargs: -1,
        torch.hypot: lambda *args, **kwargs: -1,
        torch.igamma: lambda *args, **kwargs: -1,
        torch.igammac: lambda *args, **kwargs: -1,
        torch.imag: lambda *args, **kwargs: -1,
        torch.index_add: lambda *args, **kwargs: -1,
        torch.index_copy: lambda *args, **kwargs: -1,
        torch.index_put: lambda *args, **kwargs: -1,
        torch.index_select: lambda *args, **kwargs: -1,
        torch.index_fill: lambda *args, **kwargs: -1,
        torch.index_reduce: lambda *args, **kwargs: -1,
        torch.isfinite: lambda *args, **kwargs: -1,
        torch.isin: lambda *args, **kwargs: -1,
        torch.isinf: lambda *args, **kwargs: -1,
        torch.isreal: lambda *args, **kwargs: -1,
        torch.isposinf: lambda *args, **kwargs: -1,
        torch.isneginf: lambda *args, **kwargs: -1,
        torch.instance_norm: (
            lambda *args, **kwargs: -1
        ),
        torch.int_repr: lambda *args, **kwargs: -1,
        torch.inverse: lambda *args, **kwargs: -1,
        torch.linalg.inv: lambda *args, **kwargs: -1,
        torch.linalg.inv_ex: lambda *args, **kwargs: -1,
        torch.is_complex: lambda *args, **kwargs: -1,
        torch.is_conj: lambda *args, **kwargs: -1,
        torch.is_neg: lambda *args, **kwargs: -1,
        torch.is_distributed: lambda *args, **kwargs: -1,
        torch.is_inference: lambda *args, **kwargs: -1,
        torch.is_floating_point: lambda *args, **kwargs: -1,
        torch.is_nonzero: lambda *args, **kwargs: -1,
        torch.is_same_size: lambda *args, **kwargs: -1,
        torch.is_signed: lambda *args, **kwargs: -1,
        torch.isclose: lambda *args, **kwargs: -1,
        torch.isnan: lambda *args, **kwargs: -1,
        torch.istft: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.kl_div: lambda *args, **kwargs: -1,
        torch.kron: lambda *args, **kwargs: -1,
        torch.kthvalue: lambda *args, **kwargs: -1,
        torch.linalg.ldl_factor_ex: lambda *args, **kwargs: -1,
        torch.linalg.ldl_factor: lambda *args, **kwargs: -1,
        torch.linalg.ldl_solve: lambda *args, **kwargs: -1,
        torch.layer_norm: lambda *args, **kwargs: -1,
        torch.lcm: lambda *args, **kwargs: -1,
        torch.ldexp: lambda *args, **kwargs: -1,
        torch.le: lambda *args, **kwargs: -1,
        torch.less_equal: lambda *args, **kwargs: -1,
        torch.lerp: lambda *args, **kwargs: -1,
        torch.lgamma: lambda *args, **kwargs: -1,
        torch.lobpcg: lambda *args, **kwargs: -1,  # noqa: B950
        torch.log: lambda *args, **kwargs: -1,
        torch.log_softmax: lambda *args, **kwargs: -1,
        torch.log10: lambda *args, **kwargs: -1,
        torch.log1p: lambda *args, **kwargs: -1,
        torch.log2: lambda *args, **kwargs: -1,
        torch.logaddexp: lambda *args, **kwargs: -1,
        torch.logaddexp2: lambda *args, **kwargs: -1,
        torch.logdet: lambda *args, **kwargs: -1,
        torch.xlogy: lambda *args, **kwargs: -1,
        torch.logical_and: lambda *args, **kwargs: -1,
        torch.logical_not: lambda *args, **kwargs: -1,
        torch.logical_or: lambda *args, **kwargs: -1,
        torch.logical_xor: lambda *args, **kwargs: -1,
        torch.logit: lambda *args, **kwargs: -1,
        torch.logsumexp: lambda *args, **kwargs: -1,
        torch.lstm: lambda *args, **kwargs: -1,
        torch.lstm_cell: lambda *args, **kwargs: -1,
        torch.lt: lambda *args, **kwargs: -1,
        torch.less: lambda *args, **kwargs: -1,
        torch.lu: lambda *args, **kwargs: -1,
        torch.lu_solve: lambda *args, **kwargs: -1,
        torch.margin_ranking_loss: lambda *args, **kwargs: -1,  # type: ignore[attr-defined]  # noqa: B950
        torch.masked_fill: lambda *args, **kwargs: -1,
        torch.masked_scatter: lambda *args, **kwargs: -1,
        torch.masked_select: lambda *args, **kwargs: -1,
        torch.matmul: lambda *args, **kwargs: -1,
        torch.linalg.lu: lambda *args, **kwargs: -1,
        torch.linalg.lu_factor: lambda *args, **kwargs: -1,
        torch.linalg.lu_factor_ex: lambda *args, **kwargs: -1,
        torch.linalg.lu_solve: lambda *args, **kwargs: -1,
        torch.linalg.matmul: lambda *args, **kwargs: -1,  # alias for torch.matmul
        torch.matrix_power: lambda *args, **kwargs: -1,
        torch.linalg.matrix_power: lambda *args, **kwargs: -1,
        torch.linalg.matrix_rank: lambda *args, **kwargs: -1,
        torch.linalg.multi_dot: lambda *args, **kwargs: -1,
        torch.matrix_exp: lambda *args, **kwargs: -1,
        torch.linalg.matrix_exp: lambda *args, **kwargs: -1,
        torch.max: lambda *args, **kwargs: -1,
        torch.maximum: lambda *args, **kwargs: -1,
        torch.fmax: lambda *args, **kwargs: -1,
        torch.max_pool1d: lambda *args, **kwargs: -1,
        torch.max_pool2d: lambda *args, **kwargs: -1,
        torch.max_pool3d: lambda *args, **kwargs: -1,
        torch.max_pool1d_with_indices: (
            lambda *args, **kwargs: -1
        ),
        torch.mean: lambda *args, **kwargs: -1,
        torch.nanmean: lambda *args, **kwargs: -1,
        torch.median: lambda *args, **kwargs: -1,
        torch.nanmedian: lambda *args, **kwargs: -1,
        torch.meshgrid: lambda *args, **kwargs: -1,
        torch.min: lambda *args, **kwargs: -1,
        torch.minimum: lambda *args, **kwargs: -1,
        torch.fmin: lambda *args, **kwargs: -1,
        torch.miopen_batch_norm: (
            lambda *args, **kwargs: -1
        ),
        torch.miopen_convolution: lambda *args, **kwargs: -1,  # noqa: B950
        torch.miopen_convolution_add_relu: lambda *args, **kwargs: -1,
        torch.miopen_convolution_relu: lambda *args, **kwargs: -1,
        torch.miopen_convolution_transpose: (
            lambda *args, **kwargs: -1
        ),
        torch.miopen_depthwise_convolution: (
            lambda *args, **kwargs: -1
        ),
        torch.miopen_rnn: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.mm: lambda *args, **kwargs: -1,
        torch.mode: lambda *args, **kwargs: -1,
        torch.movedim: lambda *args, **kwargs: -1,
        torch.moveaxis: lambda *args, **kwargs: -1,
        torch.msort: lambda *args, **kwargs: -1,
        torch.mul: lambda *args, **kwargs: -1,
        torch.multiply: lambda *args, **kwargs: -1,
        torch.multinomial: lambda *args, **kwargs: -1,
        torch.mv: lambda *args, **kwargs: -1,
        torch.mvlgamma: lambda *args, **kwargs: -1,
        torch.narrow: lambda *args, **kwargs: -1,
        torch.nan_to_num: lambda *args, **kwargs: -1,
        torch.native_batch_norm: lambda *args, **kwargs: -1,
        torch._native_batch_norm_legit: lambda *args, **kwargs: -1,
        torch.native_dropout: lambda *args, **kwargs: -1,
        torch.native_layer_norm: lambda *args, **kwargs: -1,
        torch.native_group_norm: lambda *args, **kwargs: -1,
        torch.native_norm: lambda *args, **kwargs: -1,
        torch.native_channel_shuffle: lambda *args, **kwargs: -1,
        torch.ne: lambda *args, **kwargs: -1,
        torch.not_equal: lambda *args, **kwargs: -1,
        torch.neg: lambda *args, **kwargs: -1,
        torch.negative: lambda *args, **kwargs: -1,
        torch.nextafter: lambda *args, **kwargs: -1,
        torch.nn.functional.adaptive_avg_pool2d: lambda *args, **kwargs: -1,
        torch.nn.functional.adaptive_avg_pool3d: lambda *args, **kwargs: -1,
        torch.nn.functional.adaptive_max_pool1d: lambda *args, **kwargs: -1,
        torch.nn.functional.adaptive_max_pool1d_with_indices: lambda *args, **kwargs: -1,
        torch.nn.functional.adaptive_max_pool2d: lambda *args, **kwargs: -1,
        torch.nn.functional.adaptive_max_pool2d_with_indices: lambda *args, **kwargs: -1,
        torch.nn.functional.adaptive_max_pool3d: lambda *args, **kwargs: -1,
        torch.nn.functional.adaptive_max_pool3d_with_indices: lambda *args, **kwargs: -1,
        torch.nn.functional.affine_grid: lambda *args, **kwargs: -1,
        torch.nn.functional.alpha_dropout: lambda *args, **kwargs: -1,
        torch.nn.functional.avg_pool2d: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.avg_pool3d: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.batch_norm: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.bilinear: lambda *args, **kwargs: -1,
        torch.nn.functional.binary_cross_entropy: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.binary_cross_entropy_with_logits: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.celu: lambda *args, **kwargs: -1,
        torch.nn.functional.cosine_embedding_loss: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.cross_entropy: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.ctc_loss: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.dropout: lambda *args, **kwargs: -1,
        torch.nn.functional.dropout1d: lambda *args, **kwargs: -1,
        torch.nn.functional.dropout2d: lambda *args, **kwargs: -1,
        torch.nn.functional.dropout3d: lambda *args, **kwargs: -1,
        torch.nn.functional.elu: lambda *args, **kwargs: -1,
        torch.nn.functional.embedding: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.embedding_bag: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.feature_alpha_dropout: lambda *args, **kwargs: -1,
        torch.nn.functional.fold: lambda *args, **kwargs: -1,
        torch.nn.functional.fractional_max_pool2d: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.fractional_max_pool2d_with_indices: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.fractional_max_pool3d: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.fractional_max_pool3d_with_indices: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.gaussian_nll_loss: lambda *args, **kwargs: -1,
        torch.nn.functional.gelu: lambda *args, **kwargs: -1,
        torch.nn.functional.glu: lambda *args, **kwargs: -1,
        torch.nn.functional.grid_sample: lambda *args, **kwargs: -1,  # noqa: B950
        torch.nn.functional.group_norm: lambda *args, **kwargs: -1,
        torch.nn.functional.gumbel_softmax: lambda *args, **kwargs: -1,
        torch.nn.functional.hardshrink: lambda *args, **kwargs: -1,
        torch.nn.functional.hardtanh: lambda *args, **kwargs: -1,
        torch.nn.functional.hinge_embedding_loss: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.instance_norm: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.interpolate: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.kl_div: lambda *args, **kwargs: -1,  # noqa: B950
        torch.nn.functional.l1_loss: lambda *args, **kwargs: -1,
        torch.nn.functional.layer_norm: lambda *args, **kwargs: -1,
        torch.nn.functional.leaky_relu: lambda *args, **kwargs: -1,
        torch.nn.functional.linear: lambda *args, **kwargs: -1,
        torch.nn.functional.local_response_norm: lambda *args, **kwargs: -1,
        torch.nn.functional.log_softmax: lambda *args, **kwargs: -1,
        torch.nn.functional.logsigmoid: lambda *args, **kwargs: -1,
        torch.nn.functional.lp_pool1d: lambda *args, **kwargs: -1,
        torch.nn.functional.lp_pool2d: lambda *args, **kwargs: -1,
        torch.nn.functional.lp_pool3d: lambda *args, **kwargs: -1,
        torch.nn.functional.margin_ranking_loss: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.max_pool1d: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.max_pool1d_with_indices: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.max_pool2d: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.max_pool2d_with_indices: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.max_pool3d: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.max_pool3d_with_indices: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.max_unpool1d: lambda *args, **kwargs: -1,  # noqa: B950
        torch.nn.functional.max_unpool2d: lambda *args, **kwargs: -1,  # noqa: B950
        torch.nn.functional.max_unpool3d: lambda *args, **kwargs: -1,  # noqa: B950
        torch.nn.functional.mse_loss: lambda *args, **kwargs: -1,
        torch.nn.functional.multi_head_attention_forward: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.multi_margin_loss: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.multilabel_margin_loss: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.multilabel_soft_margin_loss: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.nll_loss: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.normalize: lambda *args, **kwargs: -1,
        torch.nn.functional.one_hot: lambda *args, **kwargs: -1,
        torch.nn.functional.pad: lambda *args, **kwargs: -1,
        torch.nn.functional.pairwise_distance: lambda *args, **kwargs: -1,
        torch.nn.functional.poisson_nll_loss: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.prelu: lambda *args, **kwargs: -1,
        torch.nn.functional.relu: lambda *args, **kwargs: -1,
        torch.nn.functional.relu6: lambda *args, **kwargs: -1,
        torch.nn.functional.rms_norm: lambda *args, **kwargs: -1,
        torch.nn.functional.rrelu: lambda *args, **kwargs: -1,  # noqa: B950
        torch.nn.functional.selu: lambda *args, **kwargs: -1,
        torch.nn.functional.silu: lambda *args, **kwargs: -1,
        torch.nn.functional.mish: lambda *args, **kwargs: -1,
        torch.nn.functional.scaled_dot_product_attention: lambda *args, **kwargs: -1,
        torch.nn.functional.smooth_l1_loss: lambda *args, **kwargs: -1,  # noqa: B950
        torch.nn.functional.huber_loss: lambda *args, **kwargs: -1,
        torch.nn.functional.soft_margin_loss: lambda *args, **kwargs: -1,  # noqa: B950
        torch.nn.functional.softmax: lambda *args, **kwargs: -1,
        torch.nn.functional.softmin: lambda *args, **kwargs: -1,
        torch.nn.functional.softplus: lambda *args, **kwargs: -1,
        torch.nn.functional.softshrink: lambda *args, **kwargs: -1,
        torch.nn.functional.softsign: lambda *args, **kwargs: -1,
        torch.nn.functional.tanhshrink: lambda *args, **kwargs: -1,
        torch.nn.functional.threshold: lambda *args, **kwargs: -1,
        torch.nn.functional.triplet_margin_loss: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.nn.functional.triplet_margin_with_distance_loss: (
            lambda *args, **kwargs: -1
        ),
        torch.nn.functional.unfold: lambda *args, **kwargs: -1,
        torch.nn.init.uniform_: lambda *args, **kwargs: -1,
        torch.nn.init.normal_: lambda *args, **kwargs: -1,
        torch.nn.init.constant_: lambda *args, **kwargs: -1,
        torch.nn.init.kaiming_uniform_: lambda *args, **kwargs: -1,  # noqa: B950
        torch.nonzero: lambda *args, **kwargs: -1,
        torch.nonzero_static: lambda *args, **kwargs: -1,
        torch.argwhere: lambda *args, **kwargs: -1,
        torch.norm: lambda *args, **kwargs: -1,
        torch.linalg.norm: lambda *args, **kwargs: -1,
        torch.linalg.vector_norm: lambda *args, **kwargs: -1,
        torch.linalg.matrix_norm: lambda input, ord="fro", dim=(
            -2,
            -1,
        ), keepdim=False, out=None, dtype=None: -1,
        torch.norm_except_dim: lambda *args, **kwargs: -1,
        torch.nuclear_norm: lambda *args, **kwargs: -1,
        torch.numel: lambda *args, **kwargs: -1,
        torch.orgqr: lambda *args, **kwargs: -1,
        torch.ormqr: lambda *args, **kwargs: -1,
        torch.pairwise_distance: lambda *args, **kwargs: -1,
        torch.permute: lambda *args, **kwargs: -1,
        torch.pca_lowrank: lambda *args, **kwargs: -1,
        torch.pdist: lambda *args, **kwargs: -1,
        torch.pinverse: lambda *args, **kwargs: -1,
        torch.linalg.pinv: lambda *args, **kwargs: -1,
        torch.pixel_shuffle: lambda *args, **kwargs: -1,
        torch.pixel_unshuffle: lambda *args, **kwargs: -1,
        torch.poisson: lambda *args, **kwargs: -1,
        torch.poisson_nll_loss: lambda *args, **kwargs: -1,
        torch.polygamma: lambda *args, **kwargs: -1,
        torch.positive: lambda *args, **kwargs: -1,
        torch.prelu: lambda *args, **kwargs: -1,
        torch.ones_like: lambda *args, **kwargs: -1,
        torch.pow: lambda *args, **kwargs: -1,
        torch.prod: lambda *args, **kwargs: -1,
        torch.put: lambda *args, **kwargs: -1,
        torch.q_per_channel_axis: lambda *args, **kwargs: -1,
        torch.q_per_channel_scales: lambda *args, **kwargs: -1,
        torch.q_per_channel_zero_points: lambda *args, **kwargs: -1,
        torch.q_scale: lambda *args, **kwargs: -1,
        torch.q_zero_point: lambda *args, **kwargs: -1,
        torch.qr: lambda *args, **kwargs: -1,
        torch.linalg.qr: lambda *args, **kwargs: -1,
        torch.quantile: lambda *args, **kwargs: -1,
        torch.nanquantile: lambda *args, **kwargs: -1,
        torch.quantize_per_channel: lambda *args, **kwargs: -1,
        torch.quantize_per_tensor: lambda *args, **kwargs: -1,
        torch.quantize_per_tensor_dynamic: lambda *args, **kwargs: -1,
        torch.quantized_batch_norm: lambda *args, **kwargs: -1,
        torch.quantized_gru_cell: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.quantized_lstm_cell: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.quantized_max_pool1d: (
            lambda input, kernel_size, stride=(), padding=(0,), dilation=(
                1,
            ), ceil_mode=False: -1
        ),
        torch.quantized_max_pool2d: (
            lambda input, kernel_size, stride=(), padding=(0, 0), dilation=(
                1,
                1,
            ), ceil_mode=False: -1
        ),
        torch.quantized_max_pool3d: (
            lambda input, kernel_size, stride=(), padding=(0, 0, 0), dilation=(
                1,
                1,
                1,
            ), ceil_mode=False: -1
        ),
        torch.quantized_rnn_relu_cell: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.quantized_rnn_tanh_cell: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.rad2deg: lambda *args, **kwargs: -1,
        torch.rand_like: lambda *args, **kwargs: -1,
        torch.randint_like: lambda *args, **kwargs: -1,
        torch.randn_like: lambda *args, **kwargs: -1,
        torch.ravel: lambda *args, **kwargs: -1,
        torch.real: lambda *args, **kwargs: -1,
        torch.vdot: lambda *args, **kwargs: -1,
        torch.linalg.vecdot: lambda *args, **kwargs: -1,
        torch.view_as_real: lambda *args, **kwargs: -1,
        torch.view_as_complex: lambda *args, **kwargs: -1,
        torch.reciprocal: lambda *args, **kwargs: -1,
        torch.relu: lambda *args, **kwargs: -1,
        torch.remainder: lambda *args, **kwargs: -1,
        torch.renorm: lambda *args, **kwargs: -1,
        torch.repeat_interleave: lambda *args, **kwargs: -1,
        torch.reshape: lambda *args, **kwargs: -1,
        torch.rms_norm: lambda *args, **kwargs: -1,
        torch.rnn_relu: lambda *args, **kwargs: -1,  # noqa: B950
        torch.rnn_relu_cell: lambda *args, **kwargs: -1,
        torch.rnn_tanh: lambda *args, **kwargs: -1,  # noqa: B950
        torch.rnn_tanh_cell: lambda *args, **kwargs: -1,
        torch.roll: lambda *args, **kwargs: -1,
        torch.rot90: lambda *args, **kwargs: -1,
        torch.round: lambda *args, **kwargs: -1,
        torch.row_stack: lambda *args, **kwargs: -1,  # alias for torch.vstack
        torch._rowwise_prune: (lambda *args, **kwargs: -1),
        torch.rrelu: lambda *args, **kwargs: -1,
        torch.rsqrt: lambda *args, **kwargs: -1,
        torch.rsub: lambda *args, **kwargs: -1,
        torch.saddmm: lambda *args, **kwargs: -1,
        torch.scatter: lambda *args, **kwargs: -1,
        torch.scatter_add: lambda *args, **kwargs: -1,
        torch.scatter_reduce: lambda *args, **kwargs: -1,
        torch.searchsorted: lambda *args, **kwargs: -1,
        torch._segment_reduce: lambda *args, **kwargs: -1,  # noqa: B950
        torch.select: lambda *args, **kwargs: -1,
        torch.select_scatter: lambda *args, **kwargs: -1,
        torch.slice_inverse: lambda *args, **kwargs: -1,
        torch.slice_scatter: lambda *args, **kwargs: -1,
        torch.selu: lambda *args, **kwargs: -1,
        torch.sigmoid: lambda *args, **kwargs: -1,
        torch.sign: lambda *args, **kwargs: -1,
        torch.signbit: lambda *args, **kwargs: -1,
        torch.sgn: lambda *args, **kwargs: -1,
        torch.sin: lambda *args, **kwargs: -1,
        torch.sinc: lambda *args, **kwargs: -1,
        torch.sinh: lambda *args, **kwargs: -1,
        torch.slogdet: lambda *args, **kwargs: -1,
        torch.linalg.slogdet: lambda *args, **kwargs: -1,
        torch.smm: lambda *args, **kwargs: -1,
        torch.spmm: lambda *args, **kwargs: -1,
        torch.softmax: lambda *args, **kwargs: -1,
        torch.linalg.solve: lambda *args, **kwargs: -1,
        torch.linalg.solve_ex: lambda *args, **kwargs: -1,
        torch.sort: lambda *args, **kwargs: -1,
        torch.split: lambda *args, **kwargs: -1,
        torch.split_with_sizes: lambda *args, **kwargs: -1,
        torch.sqrt: lambda *args, **kwargs: -1,
        torch.square: lambda *args, **kwargs: -1,
        torch.squeeze: lambda *args, **kwargs: -1,
        torch.sspaddmm: lambda *args, **kwargs: -1,
        torch.stack: lambda *args, **kwargs: -1,
        torch.std: lambda *args, **kwargs: -1,
        torch.std_mean: lambda *args, **kwargs: -1,
        torch.stft: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.sub: lambda *args, **kwargs: -1,
        torch.subtract: lambda *args, **kwargs: -1,
        torch.sum: lambda *args, **kwargs: -1,
        torch.sym_float: lambda *args, **kwargs: -1,
        torch.sym_int: lambda *args, **kwargs: -1,
        torch.sym_max: lambda *args, **kwargs: -1,
        torch.sym_min: lambda *args, **kwargs: -1,
        torch.sym_not: lambda *args, **kwargs: -1,
        torch.sym_ite: lambda *args, **kwargs: -1,
        torch.sym_sum: lambda *args, **kwargs: -1,
        torch._sym_sqrt: lambda *args, **kwargs: -1,
        torch._sym_cos: lambda *args, **kwargs: -1,
        torch._sym_cosh: lambda *args, **kwargs: -1,
        torch._sym_sin: lambda *args, **kwargs: -1,
        torch._sym_sinh: lambda *args, **kwargs: -1,
        torch._sym_tan: lambda *args, **kwargs: -1,
        torch._sym_tanh: lambda *args, **kwargs: -1,
        torch._sym_asin: lambda *args, **kwargs: -1,
        torch._sym_acos: lambda *args, **kwargs: -1,
        torch._sym_atan: lambda *args, **kwargs: -1,
        torch.nansum: lambda *args, **kwargs: -1,
        torch.svd: lambda *args, **kwargs: -1,
        torch.svd_lowrank: lambda *args, **kwargs: -1,
        torch.linalg.svd: lambda *args, **kwargs: -1,
        torch.linalg.svdvals: lambda *args, **kwargs: -1,
        torch.swapaxes: lambda *args, **kwargs: -1,
        torch.swapdims: lambda *args, **kwargs: -1,
        torch.special.airy_ai: lambda *args, **kwargs: -1,
        torch.special.bessel_j0: lambda *args, **kwargs: -1,
        torch.special.bessel_j1: lambda *args, **kwargs: -1,
        torch.special.bessel_y0: lambda *args, **kwargs: -1,
        torch.special.bessel_y1: lambda *args, **kwargs: -1,
        torch.special.chebyshev_polynomial_t: lambda *args, **kwargs: -1,
        torch.special.chebyshev_polynomial_u: lambda *args, **kwargs: -1,
        torch.special.chebyshev_polynomial_v: lambda *args, **kwargs: -1,
        torch.special.chebyshev_polynomial_w: lambda *args, **kwargs: -1,
        torch.special.digamma: lambda *args, **kwargs: -1,
        torch.special.entr: lambda *args, **kwargs: -1,
        torch.special.erf: lambda *args, **kwargs: -1,
        torch.special.erfc: lambda *args, **kwargs: -1,
        torch.special.erfcx: lambda *args, **kwargs: -1,
        torch.special.erfinv: lambda *args, **kwargs: -1,
        torch.special.exp2: lambda *args, **kwargs: -1,
        torch.special.expit: lambda *args, **kwargs: -1,
        torch.special.expm1: lambda *args, **kwargs: -1,
        torch.special.gammainc: lambda *args, **kwargs: -1,
        torch.special.gammaincc: lambda *args, **kwargs: -1,
        torch.special.gammaln: lambda *args, **kwargs: -1,
        torch.special.hermite_polynomial_h: lambda *args, **kwargs: -1,
        torch.special.hermite_polynomial_he: lambda *args, **kwargs: -1,
        torch.special.i0: lambda *args, **kwargs: -1,
        torch.special.i0e: lambda *args, **kwargs: -1,
        torch.special.i1: lambda *args, **kwargs: -1,
        torch.special.i1e: lambda *args, **kwargs: -1,
        torch.special.laguerre_polynomial_l: lambda *args, **kwargs: -1,
        torch.special.legendre_polynomial_p: lambda *args, **kwargs: -1,
        torch.special.log1p: lambda *args, **kwargs: -1,
        torch.special.log_ndtr: lambda *args, **kwargs: -1,
        torch.special.log_softmax: lambda *args, **kwargs: -1,
        torch.special.logit: lambda *args, **kwargs: -1,
        torch.special.logsumexp: lambda *args, **kwargs: -1,
        torch.special.modified_bessel_i0: lambda *args, **kwargs: -1,
        torch.special.modified_bessel_i1: lambda *args, **kwargs: -1,
        torch.special.modified_bessel_k0: lambda *args, **kwargs: -1,
        torch.special.modified_bessel_k1: lambda *args, **kwargs: -1,
        torch.special.multigammaln: lambda *args, **kwargs: -1,
        torch.special.ndtr: lambda *args, **kwargs: -1,
        torch.special.ndtri: lambda *args, **kwargs: -1,
        torch.special.polygamma: lambda *args, **kwargs: -1,
        torch.special.psi: lambda *args, **kwargs: -1,
        torch.special.round: lambda *args, **kwargs: -1,
        torch.special.scaled_modified_bessel_k0: lambda *args, **kwargs: -1,
        torch.special.scaled_modified_bessel_k1: lambda *args, **kwargs: -1,
        torch.special.shifted_chebyshev_polynomial_t: lambda *args, **kwargs: -1,
        torch.special.shifted_chebyshev_polynomial_u: lambda *args, **kwargs: -1,
        torch.special.shifted_chebyshev_polynomial_v: lambda *args, **kwargs: -1,
        torch.special.shifted_chebyshev_polynomial_w: lambda *args, **kwargs: -1,
        torch.special.sinc: lambda *args, **kwargs: -1,
        torch.special.softmax: lambda *args, **kwargs: -1,
        torch.special.spherical_bessel_j0: lambda *args, **kwargs: -1,
        torch.special.xlog1py: lambda *args, **kwargs: -1,
        torch.special.xlogy: lambda *args, **kwargs: -1,
        torch.special.zeta: lambda *args, **kwargs: -1,
        torch.t: lambda *args, **kwargs: -1,
        torch.take: lambda *args, **kwargs: -1,
        torch.take_along_dim: lambda *args, **kwargs: -1,
        torch.tan: lambda *args, **kwargs: -1,
        torch.tanh: lambda *args, **kwargs: -1,
        torch.linalg.tensorinv: lambda *args, **kwargs: -1,
        torch.linalg.tensorsolve: lambda *args, **kwargs: -1,
        torch.tensordot: lambda *args, **kwargs: -1,
        torch.tensor_split: lambda *args, **kwargs: -1,
        torch.threshold: lambda *args, **kwargs: -1,
        torch.tile: lambda *args, **kwargs: -1,
        torch.topk: lambda *args, **kwargs: -1,
        torch.trace: lambda *args, **kwargs: -1,
        torch.transpose: lambda *args, **kwargs: -1,
        torch.trapz: lambda *args, **kwargs: -1,
        torch.trapezoid: lambda *args, **kwargs: -1,
        torch.triangular_solve: lambda *args, **kwargs: -1,
        torch.linalg.solve_triangular: lambda *args, **kwargs: -1,
        torch.tril: lambda *args, **kwargs: -1,
        torch.triplet_margin_loss: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.triu: lambda *args, **kwargs: -1,
        torch.true_divide: lambda *args, **kwargs: -1,
        torch.trunc: lambda *args, **kwargs: -1,
        torch.unbind: lambda *args, **kwargs: -1,
        torch.unflatten: lambda *args, **kwargs: -1,
        torch.unique: lambda *args, **kwargs: -1,
        torch.unique_consecutive: lambda *args, **kwargs: -1,
        torch.unravel_index: lambda *args, **kwargs: -1,
        torch.unsafe_chunk: lambda *args, **kwargs: -1,
        torch.unsafe_split: lambda *args, **kwargs: -1,
        torch.unsafe_split_with_sizes: lambda *args, **kwargs: -1,
        torch.unsqueeze: lambda *args, **kwargs: -1,
        torch.linalg.vander: lambda *args, **kwargs: -1,
        torch.var: lambda *args, **kwargs: -1,
        torch.var_mean: lambda *args, **kwargs: -1,
        torch.vsplit: lambda *args, **kwargs: -1,
        torch.vstack: lambda *args, **kwargs: -1,
        torch.where: lambda *args, **kwargs: -1,
        torch._wrapped_linear_prepack: lambda *args, **kwargs: -1,
        torch._wrapped_quantized_linear_prepacked: (
            lambda *args, **kwargs: -1  # noqa: B950
        ),
        torch.zeros_like: lambda *args, **kwargs: -1,
        torch._fw_primal_copy: lambda *args, **kwargs: -1,
        torch._make_dual_copy: lambda *args, **kwargs: -1,
        torch.view_as_real_copy: lambda *args, **kwargs: -1,
        torch.view_as_complex_copy: lambda *args, **kwargs: -1,
        torch._conj_copy: lambda *args, **kwargs: -1,
        torch._neg_view_copy: lambda *args, **kwargs: -1,
        torch.as_strided_copy: lambda *args, **kwargs: -1,
        torch._sparse_broadcast_to_copy: lambda *args, **kwargs: -1,
        torch.diagonal_copy: lambda *args, **kwargs: -1,
        torch.expand_copy: lambda *args, **kwargs: -1,
        torch.narrow_copy: lambda *args, **kwargs: -1,
        torch.permute_copy: lambda *args, **kwargs: -1,
        torch._reshape_alias_copy: lambda *args, **kwargs: -1,
        torch.select_copy: lambda *args, **kwargs: -1,
        torch.detach_copy: lambda *args, **kwargs: -1,
        torch.slice_copy: lambda *args, **kwargs: -1,
        torch.split_copy: lambda *args, **kwargs: -1,
        torch.split_with_sizes_copy: lambda *args, **kwargs: -1,
        torch.squeeze_copy: lambda *args, **kwargs: -1,
        torch.t_copy: lambda *args, **kwargs: -1,
        torch.transpose_copy: lambda *args, **kwargs: -1,
        torch.unsqueeze_copy: lambda *args, **kwargs: -1,
        torch._indices_copy: lambda *args, **kwargs: -1,
        torch._values_copy: lambda *args, **kwargs: -1,
        torch.indices_copy: lambda *args, **kwargs: -1,
        torch.values_copy: lambda *args, **kwargs: -1,
        torch.crow_indices_copy: lambda *args, **kwargs: -1,
        torch.col_indices_copy: lambda *args, **kwargs: -1,
        torch.ccol_indices_copy: lambda *args, **kwargs: -1,
        torch.row_indices_copy: lambda *args, **kwargs: -1,
        torch.unbind_copy: lambda *args, **kwargs: -1,
        torch.view_copy: lambda *args, **kwargs: -1,
        torch.unfold_copy: lambda *args, **kwargs: -1,
        torch.alias_copy: lambda *args, **kwargs: -1,
        Tensor.__floordiv__: lambda *args, **kwargs: -1,
        Tensor.__rfloordiv__: lambda *args, **kwargs: -1,
        Tensor.__ifloordiv__: lambda *args, **kwargs: -1,
        Tensor.__truediv__: lambda *args, **kwargs: -1,
        Tensor.__rtruediv__: lambda *args, **kwargs: -1,
        Tensor.__itruediv__: lambda *args, **kwargs: -1,
        Tensor.__lshift__: lambda *args, **kwargs: -1,
        Tensor.__rlshift__: lambda *args, **kwargs: -1,
        Tensor.__ilshift__: lambda *args, **kwargs: -1,
        Tensor.__rshift__: lambda *args, **kwargs: -1,
        Tensor.__rrshift__: lambda *args, **kwargs: -1,
        Tensor.__irshift__: lambda *args, **kwargs: -1,
        Tensor.__and__: lambda *args, **kwargs: -1,
        Tensor.__or__: lambda *args, **kwargs: -1,
        Tensor.__xor__: lambda *args, **kwargs: -1,
        Tensor.__float__: lambda *args, **kwargs: -1,
        Tensor.__complex__: lambda *args, **kwargs: -1,
        Tensor.__array__: lambda *args, **kwargs: -1,
        Tensor.__bool__: lambda *args, **kwargs: -1,
        Tensor.__contains__: lambda *args, **kwargs: -1,
        Tensor.__neg__: lambda *args, **kwargs: -1,
        Tensor.__invert__: lambda *args, **kwargs: -1,
        Tensor.__mod__: lambda *args, **kwargs: -1,
        Tensor.__rmod__: lambda *args, **kwargs: -1,
        Tensor.__imod__: lambda *args, **kwargs: -1,
        Tensor.__array_wrap__: lambda *args, **kwargs: -1,
        Tensor.__getitem__: lambda *args, **kwargs: -1,
        Tensor.__deepcopy__: lambda *args, **kwargs: -1,
        Tensor.__int__: lambda *args, **kwargs: -1,
        Tensor.__long__: lambda *args, **kwargs: -1,
        Tensor.__index__: lambda *args, **kwargs: -1,
        Tensor.__len__: lambda *args, **kwargs: -1,
        Tensor.__format__: lambda *args, **kwargs: -1,
        Tensor.__reduce_ex__: lambda *args, **kwargs: -1,
        Tensor.__reversed__: lambda *args, **kwargs: -1,
        Tensor.__repr__: lambda *args, **kwargs: -1,
        Tensor.__setitem__: lambda *args, **kwargs: -1,
        Tensor.__setstate__: lambda *args, **kwargs: -1,
        Tensor.T.__get__: lambda *args, **kwargs: -1,
        Tensor.H.__get__: lambda *args, **kwargs: -1,
        Tensor.mT.__get__: lambda *args, **kwargs: -1,
        Tensor.mH.__get__: lambda *args, **kwargs: -1,
        Tensor._backward_hooks.__get__: lambda *args, **kwargs: -1,
        Tensor._post_accumulate_grad_hooks.__get__: lambda *args, **kwargs: -1,
        Tensor._base.__get__: lambda *args, **kwargs: -1,
        Tensor._cdata.__get__: lambda *args, **kwargs: -1,
        Tensor.grad.__get__: lambda *args, **kwargs: -1,
        Tensor._grad.__get__: lambda *args, **kwargs: -1,
        Tensor._grad_fn.__get__: lambda *args, **kwargs: -1,
        Tensor.grad_fn.__get__: lambda *args, **kwargs: -1,
        Tensor._version.__get__: lambda *args, **kwargs: -1,
        Tensor._autocast_to_reduced_precision: lambda *args, **kwargs: -1,
        Tensor._autocast_to_full_precision: lambda *args, **kwargs: -1,
        Tensor._clear_non_serializable_cached_data: lambda *args, **kwargs: -1,
        Tensor.data.__get__: lambda *args, **kwargs: -1,
        Tensor.device.__get__: lambda *args, **kwargs: -1,
        Tensor.dtype.__get__: lambda *args, **kwargs: -1,
        Tensor.is_cuda.__get__: lambda *args, **kwargs: -1,
        Tensor.is_cpu.__get__: lambda *args, **kwargs: -1,
        Tensor.is_xla.__get__: lambda *args, **kwargs: -1,
        Tensor.is_xpu.__get__: lambda *args, **kwargs: -1,
        Tensor.is_ipu.__get__: lambda *args, **kwargs: -1,
        Tensor.is_leaf.__get__: lambda *args, **kwargs: -1,
        Tensor.retains_grad.__get__: lambda *args, **kwargs: -1,
        Tensor.is_meta.__get__: lambda *args, **kwargs: -1,
        Tensor.is_mps.__get__: lambda *args, **kwargs: -1,
        Tensor.is_mtia.__get__: lambda *args, **kwargs: -1,
        Tensor.is_nested.__get__: lambda *args, **kwargs: -1,
        Tensor.is_maia.__get__: lambda *args, **kwargs: -1,
        Tensor.is_mkldnn.__get__: lambda *args, **kwargs: -1,
        Tensor.is_quantized.__get__: lambda *args, **kwargs: -1,
        Tensor.is_sparse.__get__: lambda *args, **kwargs: -1,
        Tensor.is_sparse_csr.__get__: lambda *args, **kwargs: -1,
        Tensor.is_vulkan.__get__: lambda *args, **kwargs: -1,
        Tensor.itemsize.__get__: lambda *args, **kwargs: -1,
        Tensor.layout.__get__: lambda *args, **kwargs: -1,
        Tensor.name.__get__: lambda *args, **kwargs: -1,
        Tensor.names.__get__: lambda *args, **kwargs: -1,
        Tensor.nbytes.__get__: lambda *args, **kwargs: -1,
        Tensor.ndim.__get__: lambda *args, **kwargs: -1,
        Tensor.output_nr.__get__: lambda *args, **kwargs: -1,
        Tensor.requires_grad.__get__: lambda *args, **kwargs: -1,
        Tensor.shape.__get__: lambda *args, **kwargs: -1,
        Tensor.volatile.__get__: lambda *args, **kwargs: -1,
        Tensor.real.__get__: lambda *args, **kwargs: -1,
        Tensor.imag.__get__: lambda *args, **kwargs: -1,
        Tensor.__cuda_array_interface__.__get__: lambda *args, **kwargs: -1,
        Tensor.type: lambda *args, **kwargs: -1,
        Tensor._dimI: lambda *args, **kwargs: -1,
        Tensor._dimV: lambda *args, **kwargs: -1,
        Tensor._indices: lambda *args, **kwargs: -1,
        Tensor._is_view: lambda *args, **kwargs: -1,
        Tensor._nnz: lambda *args, **kwargs: -1,
        Tensor.crow_indices: lambda *args, **kwargs: -1,
        Tensor.col_indices: lambda *args, **kwargs: -1,
        Tensor.ccol_indices: lambda *args, **kwargs: -1,
        Tensor.row_indices: lambda *args, **kwargs: -1,
        Tensor._update_names: lambda *args, **kwargs: -1,
        Tensor._values: lambda *args, **kwargs: -1,
        Tensor.adjoint: lambda *args, **kwargs: -1,
        Tensor.align_as: lambda *args, **kwargs: -1,
        Tensor.align_to: lambda *args, **kwargs: -1,
        Tensor.apply_: lambda *args, **kwargs: -1,
        Tensor.as_strided: lambda *args, **kwargs: -1,
        Tensor.as_strided_: lambda *args, **kwargs: -1,
        Tensor.backward: lambda *args, **kwargs: -1,
        Tensor.bfloat16: lambda *args, **kwargs: -1,
        Tensor.bool: lambda *args, **kwargs: -1,
        Tensor.byte: lambda *args, **kwargs: -1,
        Tensor.char: lambda *args, **kwargs: -1,
        Tensor.cauchy_: lambda *args, **kwargs: -1,
        Tensor.coalesce: lambda *args, **kwargs: -1,
        Tensor._coalesced_: lambda *args, **kwargs: -1,
        Tensor.contiguous: lambda *args, **kwargs: -1,
        Tensor.copy_: lambda *args, **kwargs: -1,
        Tensor.cpu: lambda *args, **kwargs: -1,
        Tensor.cuda: lambda *args, **kwargs: -1,
        Tensor.mtia: lambda *args, **kwargs: -1,
        Tensor.xpu: lambda *args, **kwargs: -1,
        Tensor.ipu: lambda *args, **kwargs: -1,
        Tensor.data_ptr: lambda *args, **kwargs: -1,
        Tensor.dense_dim: lambda *args, **kwargs: -1,
        Tensor.diagonal_scatter: lambda *args, **kwargs: -1,
        Tensor.dim: lambda *args, **kwargs: -1,
        Tensor.dim_order: lambda *args, **kwargs: -1,
        Tensor.double: lambda *args, **kwargs: -1,
        Tensor.cdouble: lambda *args, **kwargs: -1,
        Tensor.element_size: lambda *args, **kwargs: -1,
        Tensor.expand: lambda *args, **kwargs: -1,
        Tensor.expand_as: lambda *args, **kwargs: -1,
        Tensor.exponential_: lambda *args, **kwargs: -1,
        Tensor.fill_: lambda *args, **kwargs: -1,
        Tensor.fill_diagonal_: lambda *args, **kwargs: -1,
        Tensor.float: lambda *args, **kwargs: -1,
        Tensor.cfloat: lambda *args, **kwargs: -1,
        Tensor.geometric_: lambda *args, **kwargs: -1,
        Tensor.get_device: lambda *args, **kwargs: -1,
        Tensor.half: lambda *args, **kwargs: -1,
        Tensor.chalf: lambda *args, **kwargs: -1,
        Tensor.has_names: lambda *args, **kwargs: -1,
        Tensor.indices: lambda *args, **kwargs: -1,
        Tensor.int: lambda *args, **kwargs: -1,
        Tensor.is_coalesced: lambda *args, **kwargs: -1,
        Tensor.is_contiguous: lambda *args, **kwargs: -1,
        Tensor.is_inference: lambda *args, **kwargs: -1,
        Tensor.is_pinned: lambda *args, **kwargs: -1,
        Tensor.is_set_to: lambda *args, **kwargs: -1,
        Tensor.is_shared: lambda *args, **kwargs: -1,
        Tensor.item: lambda *args, **kwargs: -1,
        Tensor.log_normal_: lambda *args, **kwargs: -1,
        Tensor.log_softmax: lambda *args, **kwargs: -1,
        Tensor.long: lambda *args, **kwargs: -1,
        Tensor.map_: lambda *args, **kwargs: -1,
        Tensor.map2_: lambda *args, **kwargs: -1,
        Tensor.mm: lambda *args, **kwargs: -1,
        Tensor.module_load: lambda *args, **kwargs: -1,
        Tensor.narrow_copy: lambda *args, **kwargs: -1,
        Tensor.ndimension: lambda *args, **kwargs: -1,
        Tensor.nelement: lambda *args, **kwargs: -1,
        Tensor._nested_tensor_size: lambda *args, **kwargs: -1,
        Tensor._nested_tensor_storage_offsets: lambda *args, **kwargs: -1,
        Tensor._nested_tensor_strides: lambda *args, **kwargs: -1,
        Tensor.normal_: lambda *args, **kwargs: -1,
        Tensor.numpy: lambda *args, **kwargs: -1,
        Tensor.permute: lambda *args, **kwargs: -1,
        Tensor.pin_memory: lambda *args, **kwargs: -1,
        Tensor.put_: lambda *args, **kwargs: -1,
        Tensor.qscheme: lambda *args, **kwargs: -1,
        Tensor.random_: lambda *args, **kwargs: -1,
        Tensor.record_stream: lambda *args, **kwargs: -1,
        Tensor.refine_names: lambda *args, **kwargs: -1,
        Tensor.register_hook: lambda *args, **kwargs: -1,
        Tensor.register_post_accumulate_grad_hook: lambda *args, **kwargs: -1,
        Tensor.rename: lambda *args, **kwargs: -1,
        Tensor.repeat: lambda *args, **kwargs: -1,
        Tensor.requires_grad_: lambda *args, **kwargs: -1,
        Tensor.reshape_as: lambda *args, **kwargs: -1,
        Tensor.resize: lambda *args, **kwargs: -1,
        Tensor.resize_: lambda *args, **kwargs: -1,
        Tensor.resize_as: lambda *args, **kwargs: -1,
        Tensor.resize_as_sparse_: lambda *args, **kwargs: -1,
        Tensor.retain_grad: lambda *args, **kwargs: -1,
        Tensor.set_: lambda *args, **kwargs: -1,
        Tensor.select_scatter: lambda *args, **kwargs: -1,
        Tensor.share_memory_: lambda *args, **kwargs: -1,
        Tensor.short: lambda *args, **kwargs: -1,
        Tensor.size: lambda *args, **kwargs: -1,
        Tensor.slice_scatter: lambda *args, **kwargs: -1,
        Tensor.sparse_dim: lambda *args, **kwargs: -1,
        Tensor.sparse_mask: lambda *args, **kwargs: -1,
        Tensor._sparse_mask_projection: lambda *args, **kwargs: -1,
        Tensor.sparse_resize_: lambda *args, **kwargs: -1,
        Tensor.sparse_resize_and_clear_: lambda *args, **kwargs: -1,
        Tensor.sspaddmm: lambda *args, **kwargs: -1,
        Tensor.storage: lambda *args, **kwargs: -1,
        Tensor.untyped_storage: lambda *args, **kwargs: -1,
        Tensor.storage_offset: lambda *args, **kwargs: -1,
        Tensor.storage_type: lambda *args, **kwargs: -1,
        Tensor.sum_to_size: lambda *args, **kwargs: -1,
        Tensor.tile: lambda *args, **kwargs: -1,
        Tensor.to: lambda *args, **kwargs: -1,
        Tensor.to_dense: lambda *args, **kwargs: -1,
        Tensor._to_dense: lambda *args, **kwargs: -1,
        Tensor.to_sparse: lambda *args, **kwargs: -1,
        Tensor.tolist: lambda *args, **kwargs: -1,
        Tensor.to_mkldnn: lambda *args, **kwargs: -1,
        Tensor.type_as: lambda *args, **kwargs: -1,
        Tensor.unfold: lambda *args, **kwargs: -1,
        Tensor.uniform_: lambda *args, **kwargs: -1,
        Tensor.values: lambda *args, **kwargs: -1,
        Tensor.view: lambda *args, **kwargs: -1,
        Tensor.view_as: lambda *args, **kwargs: -1,
        Tensor.zero_: lambda *args, **kwargs: -1,
        Tensor.__dlpack__: lambda *args, **kwargs: -1,
        Tensor.__dlpack_device__: lambda *args, **kwargs: -1,
        torch.linalg.lstsq: lambda *args, **kwargs: -1,
    }  # fmt: skip

    privateuse1_backend_name = (
        torch.utils.backend_registration._privateuse1_backend_name
    )
    if hasattr(Tensor, privateuse1_backend_name):
        ret[getattr(Tensor, privateuse1_backend_name)] = lambda *args, **kwargs: -1
        ret[getattr(Tensor, f"is_{privateuse1_backend_name}").__get__] = (
            lambda *args, **kwargs: -1
        )

    ret2 = {}
    ignored = get_ignored_functions()

    for k, v in ret.items():
        # Generate methods like __add__ and add_ by default from add
        names = [
            k.__name__,  # Default method
            k.__name__ + "_",  # Inplace variant
            "__" + k.__name__ + "__",  # Dunder method
            "__i" + k.__name__ + "__",  # Inplace dunder method
            "__r" + k.__name__ + "__",  # Reverse dunder method
        ]

        if k.__name__.startswith("bitwise_"):
            # bitwise_<op> have dunder methods of the form __<op>__
            # And so on.
            subname = k.__name__[len("bitwise_") :]
            names.extend(
                ["__" + subname + "__", "__i" + subname + "__", "__r" + subname + "__"]
            )

        for name in names:
            func = getattr(Tensor, name, None)
            if callable(func) and func not in ret and func not in ignored:
                ret2[func] = v

    ret.update(ret2)
    return ret


def wrap_torch_function(dispatcher: Callable):
    """Wraps a given function with ``__torch_function__`` -related functionality.

    Parameters
    ----------
    dispatcher: Callable
        A callable that returns an iterable of Tensor-likes passed into the function.

    Note
    ----
    This decorator may reduce the performance of your code. Generally, it's enough to express
    your code as a series of functions that, themselves, support __torch_function__. If you
    find yourself in the rare situation where this is not the case, e.g. if you're wrapping a
    low-level library and you also need it to work for Tensor-likes, then this function is available.

    Examples
    --------
    >>> def dispatcher(a):  # Must have the same signature as func
    ...     return (a,)
    >>> @torch.overrides.wrap_torch_function(dispatcher)
    >>> def func(a):  # This will make func dispatchable by __torch_function__
    ...     return a + 0
    """

    def inner(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            relevant_args = dispatcher(*args, **kwargs)
            if has_torch_function(relevant_args):
                return handle_torch_function(wrapped, relevant_args, *args, **kwargs)

            return func(*args, **kwargs)

        return wrapped

    return inner


def _get_overloaded_args(
    relevant_args: Iterable[Any],
    get_type_fn: Optional[Callable[[Any], Type]] = None,
) -> List[Any]:
    """Returns a list of arguments on which to call __torch_function__.

    Checks arguments in relevant_args for __torch_function__ implementations,
    storing references to the arguments and their types in overloaded_args and
    overloaded_types in order of calling precedence. Only distinct types are
    considered. If a type is a subclass of another type it will have higher
    precedence, otherwise the precedence order is the same as the order of
    arguments in relevant_args, that is, from left-to-right in the argument list.

    The precedence-determining algorithm implemented in this function is
    described in `NEP-0018`_.

    See torch::append_overloaded_arg for the equivalent function in the C++
    implementation.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of array-like arguments to check for __torch_function__
        methods.

    get_type_fn : callable, optional
        Function to call on each argument in relevant_args to get its type.

    Returns
    -------
    overloaded_args : list
        Arguments from relevant_args on which to call __torch_function__
        methods, in the order in which they should be called.

    .. _NEP-0018:
       https://numpy.org/neps/nep-0018-array-function-protocol.html
    """
    if get_type_fn is None:
        get_type_fn = type

    # If torch function is not enabled, there are no overloaded types
    if not torch._C._is_torch_function_enabled():
        return []
    # Runtime is O(num_arguments * num_unique_types)
    overloaded_types: Set[Type] = set()
    overloaded_args: List[Any] = []
    for arg in relevant_args:
        arg_type = get_type_fn(arg)
        # We only collect arguments if they have a unique type, which ensures
        # reasonable performance even with a long list of possibly overloaded
        # arguments.
        #
        # NB: Important to exclude _disabled_torch_function_impl, otherwise
        # https://github.com/pytorch/pytorch/issues/64687
        if (
            arg_type not in overloaded_types
            and hasattr(arg_type, "__torch_function__")
            and arg_type.__torch_function__ != torch._C._disabled_torch_function_impl
        ):
            # Create lists explicitly for the first type (usually the only one
            # done) to avoid setting up the iterator for overloaded_args.
            if overloaded_types:
                overloaded_types.add(arg_type)
                # By default, insert argument at the end, but if it is
                # subclass of another argument, insert it before that argument.
                # This ensures "subclasses before superclasses".
                index = len(overloaded_args)
                for i, old_arg in enumerate(overloaded_args):
                    if issubclass(arg_type, get_type_fn(old_arg)):
                        index = i
                        break
                overloaded_args.insert(index, arg)
            else:
                overloaded_types = {arg_type}
                overloaded_args = [arg]
    return overloaded_args


def handle_torch_function(
    public_api: Callable,
    relevant_args: Iterable[Any],
    *args,
    **kwargs,
) -> Any:
    """Implement a function with checks for ``__torch_function__`` overrides.

    See torch::autograd::handle_torch_function for the equivalent of this
    function in the C++ implementation.

    Arguments
    ---------
    public_api : function
        Function exposed by the public torch API originally called like
        ``public_api(*args, **kwargs)`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __torch_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : tuple
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    object
        Result from calling ``implementation`` or an ``__torch_function__``
        method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.

    Example
    -------
    >>> def func(a):
    ...     if has_torch_function_unary(a):
    ...         return handle_torch_function(func, (a,), a)
    ...     return a + 0
    """
    # Check for __torch_function__ methods.
    overloaded_args = _get_overloaded_args(relevant_args)
    # overloaded_args already have unique types.
    types = tuple(map(type, overloaded_args))

    # Check for __torch_function__ mode.
    if _is_torch_function_mode_enabled():
        # if we're here, the mode must be set to a TorchFunctionStackMode
        # this unsets it and calls directly into TorchFunctionStackMode's torch function
        with _pop_mode_temporarily() as mode:
            result = mode.__torch_function__(public_api, types, args, kwargs)
        if result is not NotImplemented:
            return result

    # Call overrides
    for overloaded_arg in overloaded_args:
        # This call needs to become a classmethod call in the future.
        # See https://github.com/pytorch/pytorch/issues/63767
        torch_func_method = overloaded_arg.__torch_function__
        if (
            hasattr(torch_func_method, "__self__")
            and torch_func_method.__self__ is overloaded_arg
            and torch_func_method is not torch._C._disabled_torch_function_impl
        ):
            warnings.warn(
                "Defining your `__torch_function__ as a plain method is deprecated and "
                "will be an error in future, please define it as a classmethod.",
                DeprecationWarning,
            )

        # Use `public_api` instead of `implementation` so __torch_function__
        # implementations can do equality/identity comparisons.
        result = torch_func_method(public_api, types, args, kwargs)

        if result is not NotImplemented:
            return result

    func_name = f"{public_api.__module__}.{public_api.__name__}"
    msg = (
        f"no implementation found for '{func_name}' on types that implement "
        f"__torch_function__: {[type(arg) for arg in overloaded_args]}"
    )
    if _is_torch_function_mode_enabled():
        msg += f" nor in mode {_get_current_function_mode()}"
    raise TypeError(msg)


has_torch_function = _add_docstr(
    _has_torch_function,
    r"""Check for __torch_function__ implementations in the elements of an iterable
    or if a __torch_function__ mode is enabled.  Considers exact ``Tensor`` s
    and ``Parameter`` s non-dispatchable.  Use this to guard a call to
    :func:`handle_torch_function`; don't use it to test if something
    is Tensor-like, use :func:`is_tensor_like` instead.
    Arguments
    ---------
    relevant_args : iterable
        Iterable or arguments to check for __torch_function__ methods.
    Returns
    -------
    bool
        True if any of the elements of relevant_args have __torch_function__
        implementations, False otherwise.
    See Also
    ________
    torch.is_tensor_like
        Checks if something is a Tensor-like, including an exact ``Tensor``.
    """,
)

has_torch_function_unary = _add_docstr(
    _has_torch_function_unary,
    r"""Special case of `has_torch_function` for single inputs.
    Instead of:
      `has_torch_function((t,))`
    call:
      `has_torch_function_unary(t)`
    which skips unnecessary packing and unpacking work.
    """,
)

has_torch_function_variadic = _add_docstr(
    _has_torch_function_variadic,
    r"""Special case of `has_torch_function` that skips tuple creation.

    This uses the METH_FASTCALL protocol introduced in Python 3.7

    Instead of:
      `has_torch_function((a, b))`
    call:
      `has_torch_function_variadic(a, b)`
    which skips unnecessary packing and unpacking work.
    """,
)


@functools.lru_cache(None)
def _get_overridable_functions() -> (
    Tuple[Dict[Any, List[Callable]], Dict[Callable, str]]
):
    overridable_funcs = collections.defaultdict(list)
    index = {}
    tested_namespaces = [
        ("torch", torch, torch.__all__),
        ("torch.functional", torch.functional, torch.functional.__all__),
        ("torch.nn.functional", torch.nn.functional, dir(torch.nn.functional)),
        ("torch.nn.init", torch.nn.init, dir(torch.nn.init)),
        ("torch.Tensor", torch.Tensor, dir(torch.Tensor)),
        ("torch.linalg", torch.linalg, dir(torch.linalg)),
        ("torch.fft", torch.fft, dir(torch.fft)),
        ("torch.special", torch.special, dir(torch.special)),
    ]
    for namespace_str, namespace, ns_funcs in tested_namespaces:
        for func_name in ns_funcs:
            ignore = False
            # ignore private functions or functions that are deleted in torch.__init__
            if namespace is not torch.Tensor:
                if func_name.startswith("__"):
                    continue
                elif func_name.startswith("_"):
                    ignore = True
                elif func_name.endswith("_"):
                    ignore = True
                elif not func_name[0].islower():
                    ignore = True
                elif func_name == "unique_dim":
                    continue
            else:
                func = getattr(namespace, func_name)
                if getattr(object, func_name, None) == func:
                    continue
                if func_name == "__weakref__":
                    continue
            func = getattr(namespace, func_name)
            if namespace is torch.Tensor and getattr(object, func_name, None) == func:
                continue
            # ignore re-exported modules
            if isinstance(func, types.ModuleType):
                continue
            # ignore __future__ imports
            if isinstance(func, __future__._Feature):
                continue

            if not callable(func) and hasattr(func, "__get__"):
                index[func.__get__] = f"{namespace_str}.{func_name}.__get__"
                index[func.__set__] = f"{namespace_str}.{func_name}.__set__"
                if ignore:
                    continue
                if func.__get__ in get_ignored_functions():
                    msg = (
                        "{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                        "but still has an explicit override"
                    )
                    assert func.__get__ not in get_testing_overrides(), msg.format(
                        namespace, func.__name__
                    )
                    continue
                else:
                    overridable_funcs[func].append(func.__get__)
                    continue

            if not callable(func):
                continue

            index[func] = f"{namespace_str}.{func_name}"

            if ignore:
                continue

            # cannot be overriden by __torch_function__
            if func in get_ignored_functions():
                msg = (
                    "{}.{} is in the tuple returned by torch._overrides.get_ignored_functions "
                    "but still has an explicit override"
                )
                assert func not in get_testing_overrides(), msg.format(
                    namespace, func.__name__
                )
                continue
            overridable_funcs[namespace].append(func)
    return overridable_funcs, index


@_disable_user_warnings
def get_overridable_functions() -> Dict[Any, List[Callable]]:
    """List functions that are overridable via __torch_function__

    Returns
    -------
    Dict[Any, List[Callable]]
        A dictionary that maps namespaces that contain overridable functions
        to functions in that namespace that can be overridden.
    """
    return _get_overridable_functions()[0]


@_disable_user_warnings
def resolve_name(f):
    """Get a human readable string name for a function passed to
    __torch_function__

    Arguments
    ---------
    f : Callable
        Function to resolve the name of.

    Returns
    -------
    str
        Name of the function; if eval'ed it should give back the input
        function.
    """
    if isinstance(f, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return str(f)
    return _get_overridable_functions()[1].get(f)


@functools.lru_cache(None)
def _get_tensor_methods() -> Set[Callable]:
    """Returns a set of the overridable methods on ``torch.Tensor``"""
    overridable_funcs = get_overridable_functions()
    methods = set(overridable_funcs[torch.Tensor])
    return methods


@_disable_user_warnings
def is_tensor_method_or_property(func: Callable) -> bool:
    """
    Returns True if the function passed in is a handler for a
    method or property belonging to ``torch.Tensor``, as passed
    into ``__torch_function__``.

    .. note::
       For properties, their ``__get__`` method must be passed in.

    This may be needed, in particular, for the following reasons:

    1. Methods/properties sometimes don't contain a `__module__` slot.
    2. They require that the first passed-in argument is an instance
       of ``torch.Tensor``.

    Examples
    --------
    >>> is_tensor_method_or_property(torch.Tensor.add)
    True
    >>> is_tensor_method_or_property(torch.add)
    False
    """
    return func in _get_tensor_methods() or func.__name__ == "__get__"


def is_tensor_like(inp):
    """
    Returns ``True`` if the passed-in input is a Tensor-like.

    Currently, this occurs whenever there's a ``__torch_function__``
    attribute on the type of the input.

    Examples
    --------
    A subclass of tensor is generally a Tensor-like.

    >>> class SubTensor(torch.Tensor): ...
    >>> is_tensor_like(SubTensor([0]))
    True

    Built-in or user types aren't usually Tensor-like.

    >>> is_tensor_like(6)
    False
    >>> is_tensor_like(None)
    False
    >>> class NotATensor: ...
    >>> is_tensor_like(NotATensor())
    False

    But, they can be made Tensor-like by implementing __torch_function__.

    >>> class TensorLike:
    ...     @classmethod
    ...     def __torch_function__(cls, func, types, args, kwargs):
    ...         return -1
    >>> is_tensor_like(TensorLike())
    True
    """
    return type(inp) is torch.Tensor or hasattr(inp, "__torch_function__")


class TorchFunctionMode:
    """
    A ``TorchFunctionMode`` allows you to override the meaning of all
    ``__torch_function__`` overrideable functions within a dynamic scope,
    without having to actually create a tensor subclass or manually
    monkey-patch functions in the PyTorch API.  Some common situations
    where you should use a mode:

        * You want to override the meaning of factory functions, or other
          functions that do not otherwise take a tensor as an argument
          (these cannot be overridden with tensor subclasses).

        * You want to override the behavior of all functions without needing
          to wrap your inputs in tensor subclasses; e.g., if you are just
          interested in logging intermediate computations.

        * You want to control the order of execution of various tensor
          subclasses explicitly, rather than implicitly via the return of
          ``NotImplemented``.

    Independent subclasses of :class:`TorchFunctionMode` are compositional:
    modes can be pushed onto a stack using ``with MyMode():``.
    When you call functions in the PyTorch API inside your
    ``__torch_function__`` implementation, by default, they will forward on to
    the next mode on the mode stack.  If you want recursively call back into
    your current ``__torch_function__`` implementation, either explicitly
    invoke ``self.__torch_function__(...)``, or use the context manager
    ``enable_torch_function_mode(self, replace=self.inner)`` to make PyTorch
    API self-referential (beware of infinite loops, in this case!)
    """

    inner: "TorchFunctionMode"

    # Force metaclass to generate constructor at the base of the hierarchy
    def __init__(self) -> None:
        pass

    def __torch_function__(self, func, types, args=(), kwargs=None):
        raise NotImplementedError

    def __enter__(self):
        _push_mode(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _pop_mode()

    @classmethod
    def push(cls, *args, **kwargs):
        warnings.warn(
            "`Mode.push()` is no longer necessary and can be replaced with just `with Mode()`"
        )
        instance = cls(*args, **kwargs)
        return instance


def _get_current_function_mode():
    stack_len = _len_torch_function_stack()
    return _get_function_stack_at(stack_len - 1) if stack_len > 0 else None


def _get_current_function_mode_stack():
    stack_len = _len_torch_function_stack()
    return [_get_function_stack_at(i) for i in range(stack_len)]


def _push_mode(mode):
    _push_on_torch_function_stack(mode)


def _pop_mode():
    old = _pop_torch_function_stack()
    return old


@contextlib.contextmanager
def _pop_mode_temporarily():
    old = _pop_mode()
    try:
        yield old
    finally:
        _push_mode(old)


class BaseTorchFunctionMode(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)


@contextlib.contextmanager
def _enable_torch_function():
    old_state = torch._C._get_torch_function_state()
    try:
        torch._C._set_torch_function_state(torch._C._TorchFunctionState.ENABLED)
        yield
    finally:
        torch._C._set_torch_function_state(old_state)


@contextlib.contextmanager
def enable_reentrant_dispatch():
    # NB: this can't simply be
    # `enable_reentrant_dispatch = torch._C._RestorePythonTLSSnapshot`
    # because:
    # 1. torch._C._RestorePythonTLSSnapshot is unavailable when this file
    #    initially gets imported. Probably an import order thing.
    # 2. enable_reentrant_dispatch is technically public API; assigning
    #    it the object would change the __module__ to look private.
    with torch._C._RestorePythonTLSSnapshot():
        try:
            yield
        finally:
            pass
