# mypy: allow-untyped-defs
import _collections_abc
import _weakrefset
import abc
import builtins
import collections
import contextlib
import copy
import copyreg
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import linecache
import logging
import multiprocessing
import operator
import os
import posixpath
import random
import re
import selectors
import signal
import sys
import tempfile
import threading
import tokenize
import traceback
import types
import typing
import unittest
import weakref
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union

np: Optional[types.ModuleType] = None
try:
    import numpy as np
except ModuleNotFoundError:
    pass

import torch
import torch._inductor.test_operators
import torch.distributed
import torch.utils._content_store
from ..utils import _config_module
from .resume_execution import TORCH_DYNAMO_RESUME_IN_PREFIX
from .utils import getfile, hashable, NP_SUPPORTED_MODULES, unwrap_if_wrapper

from .variables import (
    BuiltinVariable,
    FunctionalCallVariable,
    FunctorchHigherOrderVariable,
    NestedUserFunctionVariable,
    SkipFunctionVariable,
    TorchInGraphFunctionVariable,
    UserFunctionVariable,
    UserMethodVariable,
)


if typing.TYPE_CHECKING:
    from .variables.base import VariableTracker


"""
A note on skip/inline rules:

Dynamo consults this file to determine whether function should be inlined or skipped.

A skip applies at the frame boundary, meaning dynamo either triggers a graph break
at the beginning of the frame or attempts to trace/inline the whole frame. When skipping
a frame, recursively called frames are still traced by dynamo unless also skipped.

Skipfiles (skipped at the file level instead of function level) still apply on a
frame-by-frame boundary as dynamo traces, but apply to all functions in that file.

@skip is a helper decorator that can be applied to your function to cause it to be
included here.

Dynamo skip/inline rules & priorities are defined as follows:
* Inline is the default behavior and will be used unless explicitly skipped.
* Dynamo has two SKIPLIST: BUILTIN_SKIPLIST and THIRDPARTY_SKIPLIST.
    * BUILTIN_SKIPLIST contains builtin python modules, such as abc, collections, etc.
    * THIRDPARTY_SKIPLIST contains common third party libraries, such as numpy, pandas, etc.
* Functions in these two SKIPLISTs are always skipped, except:
    * They have explicitly defined rule in `manual_torch_name_rule_map`;
    * The corresponding python module has been put into MOD_INLINELIST.
* PyTorch(torch) is in the BUILTIN_SKIPLIST by default, but there are many cases
    where we want inline the functions under torch namespace.
    We should specify inline for the functions in `manual_torch_name_rule_map` or
    put the corresponding python module into MOD_INLINELIST to make dynamo inline them.
* If you call functions under skipped modules/files, Dynamo will wrap these functions
    as SkipFunctionVariable. There are a few functions(e.g, collections.OrderedDict) that
    we have special handling at SkipFunctionVariable.call_function.

Overall: *_INLINELIST has precedence over *_SKIPLIST has precedence over DEFAULT (inline)

To figure out what the behavior is, check the following list in order:
* `manual_torch_name_rule_map` (Inline if YES)
* MOD_INLINELIST (Inline if YES)
* BUILTIN_SKIPLIST & THIRDPARTY_SKIPLIST (Skip if YES)
* Inline by default

In general, if you want to force inline a function or module, please consider adding
the function's python module to MOD_INLINELIST first.
Use the `manual_torch_name_rule_map` only when there are other functions under the same module that
you don't want to inline them.
"""

"""
Map of function objects to their tracing rules (Dynamo variables).
* TorchInGraphFunctionVariable: The functions should be put into the FX graph or can be constant folded. E.g.,
  - torch.add: should be put into the FX graph.
  - torch.is_floating_point: constant folded.
* SkipFunctionVariable: The objects should be skipped from tracing.
* UserFunctionVariable: The functions should be inlined.

For developers: If you add/remove a torch level API, it may trigger failures from
test/dynamo/test_trace_rules.py:test_torch_name_rule_map_updated. To fix the failures:
If you are adding a new torch level API or Dynamo implementation:
* Add the name with the corresponding tracing rule to this map
  if you are adding a new in graph function or Dynamo implementation for an existing function.
* Remove the object name from test/dynamo/test_trace_rules.ignored_c_binding_in_graph_function_names if it's there.

If you are removing an existing torch level API:
* Remove the entry represented the API from this map or test/dynamo/test_trace_rules.ignored_c_binding_in_graph_function_names
  depends on where it is.


"""
manual_torch_name_rule_map = {
    "torch.onnx.is_in_onnx_export": TorchInGraphFunctionVariable,
    "torch.onnx.operators.shape_as_tensor": TorchInGraphFunctionVariable,
    "torch.overrides.is_tensor_like": TorchInGraphFunctionVariable,
    "torch.jit.is_scripting": TorchInGraphFunctionVariable,
    "torch.jit.is_tracing": TorchInGraphFunctionVariable,
    "torch.jit.annotate": TorchInGraphFunctionVariable,
    "torch.distributed.is_available": TorchInGraphFunctionVariable,
    "torch.distributed.is_initialized": TorchInGraphFunctionVariable,
    "torch.distributed.get_rank": TorchInGraphFunctionVariable,
    "torch.distributed.get_world_size": TorchInGraphFunctionVariable,
    "torch.distributed._tensor.api.DTensor#from_local": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d._get_group_size_by_name": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d._resolve_group_name_by_ranks_and_tag": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d._get_group_tag": TorchInGraphFunctionVariable,
    "torch.distributed.distributed_c10d.get_process_group_ranks": TorchInGraphFunctionVariable,
    "torch._utils.is_compiling": TorchInGraphFunctionVariable,
    "torch.fx._symbolic_trace.is_fx_tracing": TorchInGraphFunctionVariable,
    "torch._dynamo.external_utils.is_compiling": TorchInGraphFunctionVariable,
    "torch.compiler.is_compiling": TorchInGraphFunctionVariable,
    "torch.compiler.is_dynamo_compiling": TorchInGraphFunctionVariable,
    "torch.autograd._profiler_enabled": SkipFunctionVariable,
    "torch._C._to_dlpack": SkipFunctionVariable,
    "torch.to_dlpack": SkipFunctionVariable,
    # We graph break on RNG state setters or getters like
    # `torch.get_rng_state` or `torch.set_rng_state`. These functions
    # are not aten operations and therefore they are completely ignored
    # by the AOT dispatcher. As a result, the AOT graph does not have
    # these setter or getter functions, producing an incorrect graph
    # when it comes to rng states.
    "torch.default_generator#get_state": SkipFunctionVariable,
    "torch._C.Generator#get_state": SkipFunctionVariable,
    "torch.get_rng_state": SkipFunctionVariable,
    "torch.cuda.get_rng_state": SkipFunctionVariable,
    "torch.default_generator#set_state": SkipFunctionVariable,
    "torch._C.Generator#set_state": SkipFunctionVariable,
    "torch.set_rng_state": SkipFunctionVariable,
    "torch.cuda.set_rng_state": SkipFunctionVariable,
    # https://github.com/pytorch/pytorch/issues/107187
    "torch.manual_seed": SkipFunctionVariable,
    # https://github.com/pytorch/pytorch/issues/93501
    "torch.nn.utils.rnn.pack_padded_sequence": SkipFunctionVariable,
    "torch.nn.Parameter": TorchInGraphFunctionVariable,
    "torch._nested_tensor_from_mask": SkipFunctionVariable,
    "torch._nested_from_padded": SkipFunctionVariable,
    "torch.nested.nested_tensor_from_jagged": UserFunctionVariable,
    # symbol operators implemented in Python
    "torch.sym_not": TorchInGraphFunctionVariable,
    "torch.sym_float": TorchInGraphFunctionVariable,
    "torch.sym_int": TorchInGraphFunctionVariable,
    "torch.sym_max": TorchInGraphFunctionVariable,
    "torch.sym_min": TorchInGraphFunctionVariable,
    "torch.sym_sqrt": TorchInGraphFunctionVariable,
    "torch.sym_ite": TorchInGraphFunctionVariable,
    "torch.Tensor#_make_wrapper_subclass": SkipFunctionVariable,
    "torch.Tensor#__init__": SkipFunctionVariable,
    "torch.cuda.set_device": SkipFunctionVariable,
    "torch.cuda.current_device": SkipFunctionVariable,
    "torch._C.autocast_decrement_nesting": SkipFunctionVariable,
    "torch._C.autocast_increment_nesting": SkipFunctionVariable,
    "torch.autograd.grad": SkipFunctionVariable,
    "torch.autograd.backward": SkipFunctionVariable,
    "torch._C.clear_autocast_cache": SkipFunctionVariable,
    "torch.distributions.constraints.is_dependent": SkipFunctionVariable,
    "torch.jit.isinstance": SkipFunctionVariable,
    "torch._C.set_anomaly_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_cache_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_cpu_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_cpu_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_gpu_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_ipu_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_ipu_enabled": SkipFunctionVariable,
    "torch._C.set_autocast_xla_dtype": SkipFunctionVariable,
    "torch._C.set_autocast_xla_enabled": SkipFunctionVariable,
    "torch.resize_as_": SkipFunctionVariable,
    "torch.resize_as_sparse_": SkipFunctionVariable,
    "torch.get_default_device": TorchInGraphFunctionVariable,
    # functorch/vmap
    "torch._functorch.vmap._check_int_or_none": UserFunctionVariable,
    "torch._functorch.vmap._check_out_dims_is_int_or_int_pytree": UserFunctionVariable,
    "torch._functorch.vmap._check_randomness_arg": UserFunctionVariable,
    "torch._functorch.vmap._chunked_vmap": UserFunctionVariable,
    "torch._functorch.vmap._concat_chunked_outputs": UserFunctionVariable,
    "torch._functorch.vmap._create_batched_inputs": UserFunctionVariable,
    "torch._functorch.vmap._flat_vmap": UserFunctionVariable,
    "torch._functorch.vmap._flatten_chunks_output": UserFunctionVariable,
    "torch._functorch.vmap._get_chunked_inputs": UserFunctionVariable,
    "torch._functorch.vmap._get_name": UserFunctionVariable,
    "torch._functorch.vmap._maybe_remove_batch_dim": UserFunctionVariable,
    "torch._functorch.vmap._num_outputs": UserFunctionVariable,
    "torch._functorch.vmap._process_batched_inputs": UserFunctionVariable,
    "torch._functorch.vmap._unwrap_batched": UserFunctionVariable,
    "torch._functorch.vmap._validate_and_get_batch_size": UserFunctionVariable,
    "torch._functorch.vmap.doesnt_support_saved_tensors_hooks": UserFunctionVariable,
    "torch._functorch.vmap.get_chunk_sizes": UserFunctionVariable,
    # lazy_load_decompositions uses a lock that is not supported yet in dynamo
    # "torch._functorch.vmap.lazy_load_decompositions": UserFunctionVariable,
    "torch._functorch.vmap.restore_vmap": UserFunctionVariable,
    "torch._functorch.apis.vmap": UserFunctionVariable,
    "torch._functorch.vmap.unwrap_batched": UserFunctionVariable,
    "torch._functorch.vmap.vmap_impl": FunctorchHigherOrderVariable,
    "torch._functorch.vmap.wrap_batched": UserFunctionVariable,
    # functorch/grad
    "torch._functorch.eager_transforms.grad_impl": FunctorchHigherOrderVariable,
    "torch._functorch.apis.grad_and_value": UserFunctionVariable,
    "torch._functorch.eager_transforms._as_tuple": UserFunctionVariable,
    "torch._functorch.eager_transforms._check_unique_non_empty": UserFunctionVariable,
    "torch._functorch.eager_transforms._create_differentiable": UserFunctionVariable,
    "torch._functorch.eager_transforms._slice_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms._undo_create_differentiable": UserFunctionVariable,
    "torch._functorch.eager_transforms._validate_and_wrap_argnum": UserFunctionVariable,
    "torch._functorch.eager_transforms._validate_and_wrap_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms._wrap_all_tensors": UserFunctionVariable,
    "torch._functorch.eager_transforms._wrap_tensor_for_grad": UserFunctionVariable,
    # functorch/jacrev
    "torch._functorch.eager_transforms.jacrev": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms.error_if_complex": UserFunctionVariable,
    "torch._functorch.eager_transforms._chunked_standard_basis_for_": UserFunctionVariable,
    "torch._functorch.eager_transforms._safe_zero_index": UserFunctionVariable,
    # functorch/vjp
    "torch._functorch.eager_transforms.vjp": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms._vjp_with_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms.assert_non_empty_tensor_output": UserFunctionVariable,
    # functorch/jvp
    "torch._functorch.eager_transforms._jvp_with_argnums": UserFunctionVariable,
    "torch._functorch.eager_transforms.jvp": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms._replace_args": UserFunctionVariable,
    "torch._functorch.eager_transforms.safe_unpack_dual": UserFunctionVariable,
    "torch._functorch.eager_transforms.assert_non_empty_list_of_tensors": UserFunctionVariable,
    "torch._functorch.eager_transforms.assert_output_is_tensor_or_tensors": UserFunctionVariable,
    "torch.autograd.forward_ad.enter_dual_level": UserFunctionVariable,
    "torch.autograd.forward_ad.exit_dual_level": UserFunctionVariable,
    "torch.autograd.forward_ad.make_dual": UserFunctionVariable,
    "torch.autograd.forward_ad.unpack_dual": UserFunctionVariable,
    # functorch/linearize
    "torch._functorch.eager_transforms.linearize": FunctorchHigherOrderVariable,
    # functorch/jacfwd
    "torch._functorch.eager_transforms.jacfwd": FunctorchHigherOrderVariable,
    "torch._functorch.eager_transforms._construct_standard_basis_for": UserFunctionVariable,
    "torch._functorch.eager_transforms.safe_unflatten": UserFunctionVariable,
    # functorch/hessian
    "torch._functorch.eager_transforms.hessian": FunctorchHigherOrderVariable,
    # functional_call
    "torch._functorch.functional_call.functional_call": FunctionalCallVariable,
    "torch.nn.utils.stateless._groupby_tensor": TorchInGraphFunctionVariable,
    # functorch/deprecated
    "torch._functorch.deprecated.jvp": UserFunctionVariable,
    "torch._functorch.deprecated.hessian": UserFunctionVariable,
    "torch._functorch.deprecated.jacfwd": UserFunctionVariable,
    "torch._functorch.deprecated.jacrev": UserFunctionVariable,
    "torch._functorch.deprecated.grad": UserFunctionVariable,
    "torch._functorch.deprecated.grad_and_value": UserFunctionVariable,
    "torch._functorch.deprecated.vjp": UserFunctionVariable,
    # everything else
    "torch._constrain_as_size": UserFunctionVariable,
    "torch._tensor._convert": UserFunctionVariable,
    "torch.jit._unwrap_optional": UserFunctionVariable,
    "torch.backends.mha.get_fastpath_enabled": UserFunctionVariable,
    "torch._C._functorch._add_batch_dim": TorchInGraphFunctionVariable,
    "torch._C._functorch._remove_batch_dim": TorchInGraphFunctionVariable,
    "torch._C._functorch._wrap_for_grad": TorchInGraphFunctionVariable,
    "torch._C._functorch._unwrap_for_grad": TorchInGraphFunctionVariable,
    "torch._C._functorch.maybe_current_level": TorchInGraphFunctionVariable,
    "torch._C._functorch.is_batchedtensor": TorchInGraphFunctionVariable,
    "torch._dynamo.mark_static": UserFunctionVariable,
    "torch.fx.experimental.symbolic_shapes.guard_size_oblivious": TorchInGraphFunctionVariable,
    "torch.cuda._get_device_properties": TorchInGraphFunctionVariable,
    "torch.utils.hooks.BackwardHook": TorchInGraphFunctionVariable,
    "torch.sparse_bsc_tensor": SkipFunctionVariable,
    "torch.sparse_bsr_tensor": SkipFunctionVariable,
    "torch.sparse_csc_tensor": SkipFunctionVariable,
    "torch.sparse_csr_tensor": SkipFunctionVariable,
    "torch.sparse_compressed_tensor": SkipFunctionVariable,
    "torch._C._autograd._unsafe_set_version_counter": TorchInGraphFunctionVariable,
    # avoid skipping user defined modules in distributed unit tests
    "torch/testing/_internal/common_fsdp.py#forward": UserFunctionVariable,
    f"torch/testing/_internal/common_fsdp.py#{TORCH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
    "torch/testing/_internal/distributed/_tensor/common_dtensor.py#forward": UserFunctionVariable,
    f"torch/testing/_internal/distributed/_tensor/common_dtensor.py#{TORCH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
    "torch/testing/_internal/common_distributed.py#forward": UserFunctionVariable,
    f"torch/testing/_internal/common_distributed.py#{TORCH_DYNAMO_RESUME_IN_PREFIX}": UserFunctionVariable,
}


# In graph functions (including constant folding) that are C bindings
torch_c_binding_in_graph_functions = dict.fromkeys(
    [
        "math.acos",
        "math.acosh",
        "math.asin",
        "math.asinh",
        "math.atan",
        "math.atan2",
        "math.atanh",
        "math.ceil",
        "math.comb",
        "math.copysign",
        "math.cos",
        "math.cosh",
        "math.degrees",
        "math.dist",
        "math.erf",
        "math.erfc",
        "math.exp",
        "math.expm1",
        "math.fabs",
        "math.factorial",
        "math.floor",
        "math.fmod",
        "math.frexp",
        "math.fsum",
        "math.gamma",
        "math.gcd",
        "math.hypot",
        "math.isclose",
        "math.isfinite",
        "math.isinf",
        "math.isnan",
        "math.isqrt",
        "math.ldexp",
        "math.lgamma",
        "math.log",
        "math.log10",
        "math.log1p",
        "math.log2",
        "math.modf",
        "math.nextafter",
        "math.perm",
        "math.pow",
        "math.prod",
        "math.radians",
        "math.remainder",
        "math.sin",
        "math.sinh",
        "math.tan",
        "math.tanh",
        "math.trunc",
        "math.ulp",
        "torch._adaptive_avg_pool2d",
        "torch._adaptive_avg_pool3d",
        "torch._add_batch_dim",
        "torch._add_relu_",
        "torch._add_relu",
        "torch._addmm_activation",
        "torch._aminmax",
        "torch._amp_foreach_non_finite_check_and_unscale_",
        "torch._amp_update_scale_",
        "torch._assert_async",
        "torch._assert_tensor_metadata",
        "torch._batch_norm_impl_index",
        "torch._C._activate_gpu_trace",
        "torch._C._add_cached_tensor",
        "torch._C._add_docstr",
        "torch._C._are_functorch_transforms_active",
        "torch._C._autograd_init",
        "torch._C._awaitable_nowait",
        "torch._C._awaitable_wait",
        "torch._C._awaitable",
        "torch._C._backport_for_mobile_from_buffer_to_buffer",
        "torch._C._backport_for_mobile_from_buffer",
        "torch._C._backport_for_mobile_to_buffer",
        "torch._C._backport_for_mobile",
        "torch._C._broadcast_coalesced",
        "torch._C._broadcast_out",
        "torch._C._broadcast",
        "torch._C._c10d_init",
        "torch._C._calculate_package_version_based_on_upgraders",
        "torch._C._can_use_flash_attention",
        "torch._C._can_use_mem_efficient_attention",
        "torch._C._can_use_cudnn_attention",
        "torch._C._check_onnx_proto",
        "torch._C._check_sparse_tensor_invariants",
        "torch._C._collect_all",
        "torch._C._commit_update",
        "torch._C._compile_graph_to_code_table",
        "torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata",
        "torch._C._construct_storage_from_data_pointer",
        "torch._C._conv_determine_backend_memory_format",
        "torch._C._cpu._is_cpu_support_avx2",
        "torch._C._cpu._is_cpu_support_avx512",
        "torch._C._cpu._is_cpu_support_avx512_vnni",
        "torch._C._cpu._is_cpu_support_amx_tile",
        "torch._C._cpu._init_amx",
        "torch._C._crash_if_aten_asan",
        "torch._C._crash_if_csrc_asan",
        "torch._C._crash_if_csrc_ubsan",
        "torch._C._crash_if_debug_asserts_fail",
        "torch._C._crash_if_vptr_ubsan",
        "torch._C._create_function_from_graph",
        "torch._C._create_function_from_trace_with_dict",
        "torch._C._create_function_from_trace",
        "torch._C._create_graph_by_tracing",
        "torch._C._create_module_with_type",
        "torch._C._create_object_with_type",
        "torch._C._cuda_attach_out_of_memory_observer",
        "torch._C._cuda_beginAllocateCurrentStreamToPool",
        "torch._C._cuda_canDeviceAccessPeer",
        "torch._C._cuda_changeCurrentAllocator",
        "torch._C._cuda_checkPoolLiveAllocations",
        "torch._C._cuda_clearCublasWorkspaces",
        "torch._C._cuda_cudaCachingAllocator_raw_alloc",
        "torch._C._cuda_cudaCachingAllocator_raw_delete",
        "torch._C._cuda_cudaCachingAllocator_set_allocator_settings",
        "torch._C._cuda_cudaHostAllocator",
        "torch._C._cuda_customAllocator",
        "torch._C._cuda_emptyCache",
        "torch._C._cuda_endAllocateCurrentStreamToPool",
        "torch._C._cuda_exchangeDevice",
        "torch._C._cuda_get_conv_benchmark_empty_cache",
        "torch._C._cuda_get_cudnn_benchmark_limit",
        "torch._C._cuda_get_sync_debug_mode",
        "torch._C._cuda_getAllocator",
        "torch._C._cuda_getAllocatorBackend",
        "torch._C._cuda_getArchFlags",
        "torch._C._cuda_getCheckpointState",
        "torch._C._cuda_getCompiledVersion",
        "torch._C._cuda_getCurrentBlasHandle",
        "torch._C._cuda_getCurrentRawStream",
        "torch._C._cuda_getCurrentStream",
        "torch._C._cuda_getDefaultStream",
        "torch._C._cuda_getDevice",
        "torch._C._cuda_getDeviceCount",
        "torch._C._cuda_hasPrimaryContext",
        "torch._C._cuda_init",
        "torch._C._cuda_ipc_collect",
        "torch._C._cuda_isCurrentStreamCapturing",
        "torch._C._cuda_isHistoryEnabled",
        "torch._C._cuda_isInBadFork",
        "torch._C._cuda_jiterator_compile_and_launch_kernel",
        "torch._C._cuda_lock_mutex",
        "torch._C._cuda_maybeExchangeDevice",
        "torch._C._cuda_memorySnapshot",
        "torch._C._cuda_memoryStats",
        "torch._C._cuda_record_memory_history_legacy",
        "torch._C._cuda_record_memory_history",
        "torch._C._cuda_releasePool",
        "torch._C._cuda_resetAccumulatedMemoryStats",
        "torch._C._cuda_resetPeakMemoryStats",
        "torch._C._cuda_set_cudnn_benchmark_limit",
        "torch._C._cuda_set_sync_debug_mode",
        "torch._C._cuda_setCheckpointPoolState",
        "torch._C._cuda_setDevice",
        "torch._C._cuda_setMemoryFraction",
        "torch._C._cuda_setStream",
        "torch._C._cuda_sleep",
        "torch._C._cuda_synchronize",
        "torch._C._cuda_unlock_mutex",
        "torch._C._cudnn_set_conv_benchmark_empty_cache",
        "torch._C._cudnn.getCompileVersion",
        "torch._C._cudnn.getRuntimeVersion",
        "torch._C._cudnn.getVersionInt",
        "torch._C._current_autograd_node",
        "torch._C._current_graph_task_execution_order",
        "torch._C._current_graph_task_id",
        "torch._C._cxx_flags",
        "torch._C._debug_get_fusion_group_inlining",
        "torch._C._debug_only_are_vmap_fallback_warnings_enabled",
        "torch._C._debug_only_display_vmap_fallback_warnings",
        "torch._C._debug_set_autodiff_subgraph_inlining",
        "torch._C._debug_set_fusion_group_inlining",
        "torch._C._demangle",
        "torch._C._disabled_torch_dispatch_impl",
        "torch._C._disabled_torch_function_impl",
        "torch._C._dispatch_call_boxed",
        "torch._C._dispatch_check_all_invariants",
        "torch._C._dispatch_check_invariants",
        "torch._C._dispatch_dump_table",
        "torch._C._dispatch_dump",
        "torch._C._dispatch_find_dangling_impls",
        "torch._C._dispatch_find_schema_or_throw",
        "torch._C._dispatch_get_all_op_names",
        "torch._C._dispatch_get_backend_keyset_from_autograd",
        "torch._C._dispatch_get_registrations_for_dispatch_key",
        "torch._C._dispatch_has_backend_fallback",
        "torch._C._dispatch_has_computed_kernel_for_dispatch_key",
        "torch._C._dispatch_has_kernel_for_any_dispatch_key",
        "torch._C._dispatch_has_kernel_for_dispatch_key",
        "torch._C._dispatch_has_kernel",
        "torch._C._dispatch_is_alias_key",
        "torch._C._dispatch_is_included_in_alias",
        "torch._C._dispatch_is_main_interpreter",
        "torch._C._dispatch_isTensorSubclassLike",
        "torch._C._dispatch_key_for_device",
        "torch._C._dispatch_key_name",
        "torch._C._dispatch_key_parse",
        "torch._C._dispatch_key_set",
        "torch._C._dispatch_keys",
        "torch._C._dispatch_keyset_full_after",
        "torch._C._dispatch_keyset_full",
        "torch._C._dispatch_keyset_to_string",
        "torch._C._dispatch_library",
        "torch._C._dispatch_num_backends",
        "torch._C._dispatch_print_registrations_for_dispatch_key",
        "torch._C._dispatch_pystub",
        "torch._C._dispatch_set_report_error_callback",
        "torch._C._dispatch_tls_is_dispatch_key_excluded",
        "torch._C._dispatch_tls_is_dispatch_key_included",
        "torch._C._dispatch_tls_local_exclude_set",
        "torch._C._dispatch_tls_local_include_set",
        "torch._C._dispatch_tls_set_dispatch_key_excluded",
        "torch._C._dispatch_tls_set_dispatch_key_included",
        "torch._C._dist_autograd_init",
        "torch._C._dump_local_tls_set",
        "torch._C._dump_upgraders_map",
        "torch._C._enable_mobile_interface_call_export",
        "torch._C._enter_dual_level",
        "torch._C._error_if_any_worker_fails",
        "torch._C._exit_dual_level",
        "torch._C._export_operator_list",
        "torch._C._export_opnames",
        "torch._C._faulty_agent_init",
        "torch._C._fft.fft_fft",
        "torch._C._fft.fft_fft2",
        "torch._C._fft.fft_fftfreq",
        "torch._C._fft.fft_fftn",
        "torch._C._fft.fft_fftshift",
        "torch._C._fft.fft_hfft",
        "torch._C._fft.fft_hfft2",
        "torch._C._fft.fft_hfftn",
        "torch._C._fft.fft_ifft",
        "torch._C._fft.fft_ifft2",
        "torch._C._fft.fft_ifftn",
        "torch._C._fft.fft_ifftshift",
        "torch._C._fft.fft_ihfft",
        "torch._C._fft.fft_ihfft2",
        "torch._C._fft.fft_ihfftn",
        "torch._C._fft.fft_irfft",
        "torch._C._fft.fft_irfft2",
        "torch._C._fft.fft_irfftn",
        "torch._C._fft.fft_rfft",
        "torch._C._fft.fft_rfft2",
        "torch._C._fft.fft_rfftfreq",
        "torch._C._fft.fft_rfftn",
        "torch._C._free_And_Remove_DeleterFn",
        "torch._C._freeze_module",
        "torch._C._from_dlpack",
        "torch._C._functionality_to_backend_keys",
        "torch._C._functionalization_reapply_views_tls",
        "torch._C._fuse_to_static_module",
        "torch._C._gather_out",
        "torch._C._gather",
        "torch._C._generate_upgraders_graph",
        "torch._C._get_autograd_fallback_mode",
        "torch._C._get_backcompat_broadcast_warn",
        "torch._C._get_backcompat_keepdim_warn",
        "torch._C._get_blas_preferred_backend",
        "torch._C._get_caught_jit_exception_class_name",
        "torch._C._get_caught_jit_exception_original_msg",
        "torch._C._get_constant_bool_symnode",
        "torch._C._get_cpp_backtrace",
        "torch._C._get_cpu_capability",
        "torch._C._get_cublas_allow_bf16_reduced_precision_reduction",
        "torch._C._get_cublas_allow_fp16_reduced_precision_reduction",
        "torch._C._get_cublas_allow_tf32",
        "torch._C._get_cudnn_allow_tf32",
        "torch._C._get_cudnn_benchmark",
        "torch._C._get_cudnn_deterministic",
        "torch._C._get_cudnn_enabled",
        "torch._C._get_custom_class_python_wrapper",
        "torch._C._get_default_device",
        "torch._C._get_deterministic_algorithms_warn_only",
        "torch._C._get_deterministic_algorithms",
        "torch._C._get_deterministic_fill_uninitialized_memory",
        "torch._C._get_dispatch_mode",
        "torch._C._get_dispatch_stack_at",
        "torch._C._get_file_format",
        "torch._C._get_flash_sdp_enabled",
        "torch._C._get_float32_matmul_precision",
        "torch._C._get_function_stack_at",
        "torch._C._get_graph_executor_optimize",
        "torch._C._get_linalg_preferred_backend",
        "torch._C._get_math_sdp_enabled",
        "torch._C._get_max_operator_version",
        "torch._C._get_mem_efficient_sdp_enabled",
        "torch._C._get_mkldnn_enabled",
        "torch._C._get_cudnn_sdp_enabled",
        "torch._C._set_sdp_use_cudnn",
        "torch._C._get_mobile_model_contained_types_from_buffer",
        "torch._C._get_mobile_model_contained_types",
        "torch._C._get_model_bytecode_version_from_buffer",
        "torch._C._get_model_bytecode_version",
        "torch._C._get_model_extra_files_from_buffer",
        "torch._C._get_model_extra_files",
        "torch._C._get_model_ops_and_info_from_buffer",
        "torch._C._get_model_ops_and_info",
        "torch._C._get_module_info_from_flatbuffer",
        "torch._C._get_nnpack_enabled",
        "torch._C._get_obj_in_tls",
        "torch._C._get_operation_overload",
        "torch._C._get_operator_version_map",
        "torch._C._get_privateuse1_backend_name",
        "torch._C._get_qengine",
        "torch._C._get_schema",
        "torch._C._get_nested_int",
        "torch._C._get_tensor_metadata",
        "torch._C._get_tracing_state",
        "torch._C._get_upgrader_ranges",
        "torch._C._get_upgraders_entry_map",
        "torch._C._get_upgraders_map_size",
        "torch._C._get_value_trace",
        "torch._C._get_version_calculator_flag",
        "torch._C._get_warnAlways",
        "torch._C._graph_pool_handle",
        "torch._C._group_tensors_by_device_and_dtype",
        "torch._C._hack_do_not_use_clone_module_with_class",
        "torch._C._has_distributed",
        "torch._C._has_Standard_Deleter",
        "torch._C._has_storage",
        "torch._C._has_tensorexpr_cpp_tests",
        "torch._C._run_tensorexpr_cpp_tests",
        "torch._C._has_torch_function_unary",
        "torch._C._has_torch_function_variadic",
        "torch._C._has_torch_function",
        "torch._C._import_ir_module_from_package",
        "torch._C._increment_version",
        "torch._C._infer_size",
        "torch._C._init_names",
        "torch._C._initExtension",
        "torch._C._is_alias_of",
        "torch._C._is_any_autocast_enabled",
        "torch._C._is_cached_tensor",
        "torch._C._is_fwd_grad_enabled",
        "torch._C._is_key_in_tls",
        "torch._C._is_multithreading_enabled",
        "torch._C._is_torch_function_enabled",
        "torch._C._is_torch_function_mode_enabled",
        "torch._C._is_tracing",
        "torch._C._is_view_replay_enabled",
        "torch._C._is_xnnpack_enabled",
        "torch._C._itt.is_available",
        "torch._C._itt.mark",
        "torch._C._itt.rangePop",
        "torch._C._itt.rangePush",
        "torch._C._ivalue_debug_python_object",
        "torch._C._ivalue_tags_match",
        "torch._C._jit_assert_is_instance",
        "torch._C._jit_can_fuse_on_cpu_legacy",
        "torch._C._jit_can_fuse_on_cpu",
        "torch._C._jit_can_fuse_on_gpu",
        "torch._C._jit_cat_wo_conditionals",
        "torch._C._jit_check_alias_annotation",
        "torch._C._jit_clear_class_registry",
        "torch._C._jit_debug_fuser_num_cached_kernel_specs",
        "torch._C._jit_debug_module_iterators",
        "torch._C._jit_decay_packed_param_input_types",
        "torch._C._jit_decomposition_graph_for_node",
        "torch._C._jit_differentiate",
        "torch._C._jit_erase_non_input_shape_information",
        "torch._C._jit_flatten",
        "torch._C._jit_fuser_get_fused_kernel_code",
        "torch._C._jit_get_all_schemas",
        "torch._C._jit_get_custom_class_schemas",
        "torch._C._jit_get_emit_hooks",
        "torch._C._jit_get_inline_everything_mode",
        "torch._C._jit_get_logging_option",
        "torch._C._jit_get_num_profiled_runs",
        "torch._C._jit_get_operation",
        "torch._C._jit_get_schemas_for_operator",
        "torch._C._jit_get_te_cuda_pointwise_block_count",
        "torch._C._jit_get_te_cuda_pointwise_block_size",
        "torch._C._jit_get_te_cuda_pointwise_loop_levels",
        "torch._C._jit_get_te_generate_block_code",
        "torch._C._jit_get_te_must_use_llvm_cpu",
        "torch._C._jit_get_tracer_state_warn",
        "torch._C._jit_has_cpp_tests",
        "torch._C._jit_init",
        "torch._C._jit_interpret_graph",
        "torch._C._jit_is_onnx_log_enabled",
        "torch._C._jit_is_script_object",
        "torch._C._jit_llga_enabled",
        "torch._C._jit_nvfuser_can_be_enabled",
        "torch._C._jit_nvfuser_clear_comparison_callback",
        "torch._C._jit_nvfuser_enabled",
        "torch._C._jit_nvfuser_horizontal_mode",
        "torch._C._jit_nvfuser_set_comparison_callback",
        "torch._C._jit_nvfuser_single_node_mode",
        "torch._C._jit_object_is_non_holding",
        "torch._C._jit_onnx_convert_pattern_from_subblock",
        "torch._C._jit_onnx_create_full_scope_name",
        "torch._C._jit_onnx_list_model_parameters",
        "torch._C._jit_onnx_log",
        "torch._C._jit_opt_conditionals",
        "torch._C._jit_override_can_fuse_on_cpu_legacy",
        "torch._C._jit_override_can_fuse_on_cpu",
        "torch._C._jit_override_can_fuse_on_gpu",
        "torch._C._jit_pass_autocast",
        "torch._C._jit_pass_batch_mm",
        "torch._C._jit_pass_canonicalize_graph_fuser_ops",
        "torch._C._jit_pass_canonicalize",
        "torch._C._jit_pass_complete_shape_analysis",
        "torch._C._jit_pass_concat_frozen_linear",
        "torch._C._jit_pass_constant_loop_unrolling",
        "torch._C._jit_pass_constant_pooling",
        "torch._C._jit_pass_constant_propagation_immutable_types",
        "torch._C._jit_pass_constant_propagation",
        "torch._C._jit_pass_convert_frozen_ops_to_mkldnn",
        "torch._C._jit_pass_create_autodiff_subgraphs",
        "torch._C._jit_pass_create_functional_graphs",
        "torch._C._jit_pass_cse",
        "torch._C._jit_pass_custom_pattern_based_rewrite_graph",
        "torch._C._jit_pass_custom_pattern_based_rewrite",
        "torch._C._jit_pass_dbr_quant_remove_redundant_aliases",
        "torch._C._jit_pass_dce_allow_deleting_nodes_with_side_effects",
        "torch._C._jit_pass_dce",
        "torch._C._jit_pass_decompose_ops",
        "torch._C._jit_pass_dedup_module_uses",
        "torch._C._jit_pass_erase_number_types",
        "torch._C._jit_pass_erase_shape_information",
        "torch._C._jit_pass_filter_non_tensor_arguments",
        "torch._C._jit_pass_fixup_onnx_controlflow_node",
        "torch._C._jit_pass_fold_convbn",
        "torch._C._jit_pass_fold_frozen_conv_add_or_sub",
        "torch._C._jit_pass_fold_frozen_conv_bn",
        "torch._C._jit_pass_fold_frozen_conv_mul_or_div",
        "torch._C._jit_pass_fold_frozen_linear_bn",
        "torch._C._jit_pass_fold_prepacking_ops",
        "torch._C._jit_pass_functional_to_inplace_activation",
        "torch._C._jit_pass_fuse_add_relu",
        "torch._C._jit_pass_fuse_addmm",
        "torch._C._jit_pass_fuse_clamp_w_prepacked_linear_conv",
        "torch._C._jit_pass_fuse_frozen_conv_add_relu",
        "torch._C._jit_pass_fuse_linear",
        "torch._C._jit_pass_fuse_quantized_add_relu",
        "torch._C._jit_pass_fuse_tensorexprs",
        "torch._C._jit_pass_fuse",
        "torch._C._jit_pass_inline_fork_wait",
        "torch._C._jit_pass_inline_functional_graphs",
        "torch._C._jit_pass_inline",
        "torch._C._jit_pass_inplace_to_functional_activation",
        "torch._C._jit_pass_insert_observer_method_for_ondevice_ptq",
        "torch._C._jit_pass_insert_observers",
        "torch._C._jit_pass_insert_prepack_unpack",
        "torch._C._jit_pass_insert_prepacked_ops",
        "torch._C._jit_pass_insert_quant_dequant_for_ondevice_ptq",
        "torch._C._jit_pass_insert_quant_dequant",
        "torch._C._jit_pass_integer_value_refinement",
        "torch._C._jit_pass_lint",
        "torch._C._jit_pass_loop_unrolling",
        "torch._C._jit_pass_lower_all_tuples",
        "torch._C._jit_pass_lower_graph",
        "torch._C._jit_pass_metal_fold_prepacking_ops",
        "torch._C._jit_pass_metal_fuse_clamp_w_prepacked_conv",
        "torch._C._jit_pass_metal_insert_prepacked_ops",
        "torch._C._jit_pass_metal_optimize_for_mobile",
        "torch._C._jit_pass_onnx_assign_output_shape",
        "torch._C._jit_pass_onnx_assign_scoped_names_for_node_and_value",
        "torch._C._jit_pass_onnx_autograd_function_process",
        "torch._C._jit_pass_onnx_block",
        "torch._C._jit_pass_onnx_cast_all_constant_to_floating",
        "torch._C._jit_pass_onnx_clear_scope_records",
        "torch._C._jit_pass_onnx_constant_fold",
        "torch._C._jit_pass_onnx_deduplicate_initializers",
        "torch._C._jit_pass_onnx_eliminate_unused_items",
        "torch._C._jit_pass_onnx_eval_peephole",
        "torch._C._jit_pass_onnx_function_extraction",
        "torch._C._jit_pass_onnx_function_substitution",
        "torch._C._jit_pass_onnx_graph_shape_type_inference",
        "torch._C._jit_pass_onnx_lint",
        "torch._C._jit_pass_onnx_node_shape_type_inference",
        "torch._C._jit_pass_onnx_peephole",
        "torch._C._jit_pass_onnx_preprocess_caffe2",
        "torch._C._jit_pass_onnx_preprocess",
        "torch._C._jit_pass_onnx_quantization_insert_permutes",
        "torch._C._jit_pass_onnx_remove_inplace_ops_for_onnx",
        "torch._C._jit_pass_onnx_remove_print",
        "torch._C._jit_pass_onnx_scalar_type_analysis",
        "torch._C._jit_pass_onnx_set_dynamic_input_shape",
        "torch._C._jit_pass_onnx_track_scope_attributes",
        "torch._C._jit_pass_onnx_unpack_quantized_weights",
        "torch._C._jit_pass_onnx",
        "torch._C._jit_pass_optimize_for_inference",
        "torch._C._jit_pass_optimize_for_mobile",
        "torch._C._jit_pass_optimize_frozen_graph",
        "torch._C._jit_pass_pattern_based_rewrite",
        "torch._C._jit_pass_peephole_list_idioms",
        "torch._C._jit_pass_peephole",
        "torch._C._jit_pass_prepare_division_for_onnx",
        "torch._C._jit_pass_propagate_device",
        "torch._C._jit_pass_propagate_dtype",
        "torch._C._jit_pass_propagate_shapes_on_graph_and_build_compute",
        "torch._C._jit_pass_propagate_shapes_on_graph",
        "torch._C._jit_pass_quant_finalize_for_ondevice_ptq",
        "torch._C._jit_pass_quant_finalize",
        "torch._C._jit_pass_quant_fusion",
        "torch._C._jit_pass_refine_integer_values",
        "torch._C._jit_pass_refine_tuple_types",
        "torch._C._jit_pass_remove_dropout",
        "torch._C._jit_pass_remove_expands",
        "torch._C._jit_pass_remove_inplace_ops",
        "torch._C._jit_pass_remove_mutation",
        "torch._C._jit_pass_replace_old_ops_with_upgraders",
        "torch._C._jit_pass_replicate_dequantize",
        "torch._C._jit_pass_run_decompositions",
        "torch._C._jit_pass_specialize_autogradzero",
        "torch._C._jit_pass_swap_functional_linear",
        "torch._C._jit_pass_transform_conv1d_to_conv2d",
        "torch._C._jit_pass_transpose_frozen_linear",
        "torch._C._jit_pass_vulkan_fold_prepacking_ops",
        "torch._C._jit_pass_vulkan_fuse_clamp_w_prepacked_conv",
        "torch._C._jit_pass_vulkan_insert_prepacked_ops",
        "torch._C._jit_pass_vulkan_optimize_for_mobile",
        "torch._C._jit_register_decomposition_for_schema",
        "torch._C._jit_register_shape_compute_graph_for_node",
        "torch._C._jit_resolve_packet",
        "torch._C._jit_run_cpp_tests",
        "torch._C._jit_script_class_compile",
        "torch._C._jit_script_compile_overload",
        "torch._C._jit_script_compile",
        "torch._C._jit_script_interface_compile",
        "torch._C._jit_set_autocast_mode",
        "torch._C._jit_set_bailout_depth",
        "torch._C._jit_set_emit_hooks",
        "torch._C._jit_set_fusion_strategy",
        "torch._C._jit_set_inline_everything_mode",
        "torch._C._jit_set_llga_enabled",
        "torch._C._jit_set_logging_option",
        "torch._C._jit_set_logging_stream",
        "torch._C._jit_set_num_profiled_runs",
        "torch._C._jit_set_nvfuser_enabled",
        "torch._C._jit_set_nvfuser_guard_mode",
        "torch._C._jit_set_nvfuser_horizontal_mode",
        "torch._C._jit_set_nvfuser_single_node_mode",
        "torch._C._jit_set_nvfuser_skip_node_kind",
        "torch._C._jit_set_onnx_log_enabled",
        "torch._C._jit_set_onnx_log_output_stream",
        "torch._C._jit_set_profiling_executor",
        "torch._C._jit_set_profiling_mode",
        "torch._C._jit_set_symbolic_shapes_test_mode",
        "torch._C._jit_set_te_cuda_pointwise_block_count",
        "torch._C._jit_set_te_cuda_pointwise_block_size",
        "torch._C._jit_set_te_cuda_pointwise_loop_levels",
        "torch._C._jit_set_te_generate_block_code",
        "torch._C._jit_set_te_must_use_llvm_cpu",
        "torch._C._jit_set_texpr_dynamic_shape_enabled",
        "torch._C._jit_set_texpr_fuser_enabled",
        "torch._C._jit_set_texpr_reductions_enabled",
        "torch._C._jit_set_tracer_state_warn",
        "torch._C._jit_set_utf8_decoding_ignore",
        "torch._C._jit_shape_compute_graph_for_node",
        "torch._C._jit_symbolic_shapes_test_mode_enabled",
        "torch._C._jit_texpr_dynamic_shape_enabled",
        "torch._C._jit_texpr_fallback_allowed",
        "torch._C._jit_texpr_fuser_enabled",
        "torch._C._jit_texpr_reductions_enabled",
        "torch._C._jit_texpr_set_fallback_allowed",
        "torch._C._jit_to_backend_selective",
        "torch._C._jit_to_backend",
        "torch._C._jit_to_static_module",
        "torch._C._jit_trace_graph",
        "torch._C._jit_trace_module",
        "torch._C._jit_tree_views.FalseLiteral",
        "torch._C._jit_tree_views.NoneLiteral",
        "torch._C._jit_tree_views.TrueLiteral",
        "torch._C._jit_try_infer_type",
        "torch._C._jit_unflatten",
        "torch._C._last_executed_optimized_graph",
        "torch._C._len_torch_dispatch_stack",
        "torch._C._len_torch_function_stack",
        "torch._C._linalg._linalg_eigvals",
        "torch._C._linalg.linalg_cholesky_ex",
        "torch._C._linalg.linalg_cholesky",
        "torch._C._linalg.linalg_cond",
        "torch._C._linalg.linalg_cross",
        "torch._C._linalg.linalg_det",
        "torch._C._linalg.linalg_diagonal",
        "torch._C._linalg.linalg_eig",
        "torch._C._linalg.linalg_eigh",
        "torch._C._linalg.linalg_eigvals",
        "torch._C._linalg.linalg_eigvalsh",
        "torch._C._linalg.linalg_householder_product",
        "torch._C._linalg.linalg_inv_ex",
        "torch._C._linalg.linalg_inv",
        "torch._C._linalg.linalg_ldl_factor_ex",
        "torch._C._linalg.linalg_ldl_factor",
        "torch._C._linalg.linalg_ldl_solve",
        "torch._C._linalg.linalg_lstsq",
        "torch._C._linalg.linalg_lu_factor_ex",
        "torch._C._linalg.linalg_lu_factor",
        "torch._C._linalg.linalg_lu_solve",
        "torch._C._linalg.linalg_lu",
        "torch._C._linalg.linalg_matmul",
        "torch._C._linalg.linalg_matrix_exp",
        "torch._C._linalg.linalg_matrix_norm",
        "torch._C._linalg.linalg_matrix_power",
        "torch._C._linalg.linalg_matrix_rank",
        "torch._C._linalg.linalg_multi_dot",
        "torch._C._linalg.linalg_norm",
        "torch._C._linalg.linalg_pinv",
        "torch._C._linalg.linalg_qr",
        "torch._C._linalg.linalg_slogdet",
        "torch._C._linalg.linalg_solve_ex",
        "torch._C._linalg.linalg_solve_triangular",
        "torch._C._linalg.linalg_solve",
        "torch._C._linalg.linalg_svd",
        "torch._C._linalg.linalg_svdvals",
        "torch._C._linalg.linalg_tensorinv",
        "torch._C._linalg.linalg_tensorsolve",
        "torch._C._linalg.linalg_vander",
        "torch._C._linalg.linalg_vecdot",
        "torch._C._linalg.linalg_vector_norm",
        "torch._C._llvm_enabled",
        "torch._C._load_for_lite_interpreter_from_buffer",
        "torch._C._load_for_lite_interpreter",
        "torch._C._load_jit_module_from_bytes",
        "torch._C._load_jit_module_from_file",
        "torch._C._load_mobile_module_from_bytes",
        "torch._C._load_mobile_module_from_file",
        "torch._C._log_api_usage_metadata",
        "torch._C._log_api_usage_once",
        "torch._C._logging_set_logger",
        "torch._C._meta_in_tls_dispatch_include",
        "torch._C._mps_acquireEvent",
        "torch._C._mps_currentAllocatedMemory",
        "torch._C._mps_deviceSynchronize",
        "torch._C._mps_driverAllocatedMemory",
        "torch._C._mps_recommendedMaxMemory",
        "torch._C._mps_elapsedTimeOfEvents",
        "torch._C._mps_emptyCache",
        "torch._C._mps_get_default_generator",
        "torch._C._mps_is_available",
        "torch._C._mps_is_in_bad_fork",
        "torch._C._mps_is_on_macos_13_or_newer",
        "torch._C._mps_profilerStartTrace",
        "torch._C._mps_profilerStopTrace",
        "torch._C._mps_queryEvent",
        "torch._C._mps_recordEvent",
        "torch._C._mps_releaseEvent",
        "torch._C._mps_setMemoryFraction",
        "torch._C._mps_synchronizeEvent",
        "torch._C._mps_waitForEvent",
        "torch._C._multiprocessing_init",
        "torch._C._nccl_all_gather",
        "torch._C._nccl_all_reduce",
        "torch._C._nccl_broadcast",
        "torch._C._nccl_init_rank",
        "torch._C._nccl_reduce_scatter",
        "torch._C._nccl_reduce",
        "torch._C._nccl_unique_id",
        "torch._C._nccl_version_suffix",
        "torch._C._nccl_version",
        "torch._C._nested.nested_tensor",
        "torch._C._nested.nested_to_padded_tensor",
        "torch._C._new_symbolic_shape_symbol",
        "torch._C._nn_module_to_mobile",
        "torch._C._nn._conv_depthwise2d",
        "torch._C._nn._pad_circular",
        "torch._C._nn._pad_enum",
        "torch._C._nn._parse_to",
        "torch._C._nn._test_ambiguous_defaults",
        "torch._C._nn._test_optional_filled_intlist",
        "torch._C._nn._test_optional_floatlist",
        "torch._C._nn._test_optional_intlist",
        "torch._C._nn._test_string_default",
        "torch._C._nn._test_warn_in_autograd",
        "torch._C._nn._upsample_bicubic2d_aa",
        "torch._C._nn._upsample_bilinear2d_aa",
        "torch._C._nn._upsample_nearest_exact1d",
        "torch._C._nn._upsample_nearest_exact2d",
        "torch._C._nn._upsample_nearest_exact3d",
        "torch._C._nn.adaptive_avg_pool2d",
        "torch._C._nn.adaptive_avg_pool3d",
        "torch._C._nn.adaptive_max_pool2d",
        "torch._C._nn.adaptive_max_pool3d",
        "torch._C._nn.avg_pool2d",
        "torch._C._nn.avg_pool3d",
        "torch._C._nn.binary_cross_entropy",
        "torch._C._nn.col2im",
        "torch._C._nn.conv_depthwise3d",
        "torch._C._nn.cross_entropy_loss",
        "torch._C._nn.elu_",
        "torch._C._nn.elu",
        "torch._C._nn.flatten_dense_tensors",
        "torch._C._nn.fractional_max_pool2d",
        "torch._C._nn.fractional_max_pool3d",
        "torch._C._nn.gelu_",
        "torch._C._nn.gelu",
        "torch._C._nn.glu",
        "torch._C._nn.hardsigmoid_",
        "torch._C._nn.hardsigmoid",
        "torch._C._nn.hardswish_",
        "torch._C._nn.hardswish",
        "torch._C._nn.hardtanh_",
        "torch._C._nn.hardtanh",
        "torch._C._nn.huber_loss",
        "torch._C._nn.im2col",
        "torch._C._nn.l1_loss",
        "torch._C._nn.leaky_relu_",
        "torch._C._nn.leaky_relu",
        "torch._C._nn.linear",
        "torch._C._nn.log_sigmoid",
        "torch._C._nn.max_pool2d_with_indices",
        "torch._C._nn.max_pool3d_with_indices",
        "torch._C._nn.max_unpool2d",
        "torch._C._nn.max_unpool3d",
        "torch._C._nn.mish_",
        "torch._C._nn.mish",
        "torch._C._nn.mkldnn_linear",
        "torch._C._nn.mkldnn_reorder_conv2d_weight",
        "torch._C._nn.mkldnn_reorder_conv3d_weight",
        "torch._C._nn.mse_loss",
        "torch._C._nn.multi_margin_loss",
        "torch._C._nn.multilabel_margin_loss",
        "torch._C._nn.nll_loss_nd",
        "torch._C._nn.nll_loss",
        "torch._C._nn.nll_loss2d",
        "torch._C._nn.one_hot",
        "torch._C._nn.pad_sequence",
        "torch._C._nn.pad",
        "torch._C._nn.reflection_pad1d",
        "torch._C._nn.reflection_pad2d",
        "torch._C._nn.reflection_pad3d",
        "torch._C._nn.relu6_",
        "torch._C._nn.relu6",
        "torch._C._nn.replication_pad1d",
        "torch._C._nn.replication_pad2d",
        "torch._C._nn.replication_pad3d",
        "torch._C._nn.rrelu_with_noise_",
        "torch._C._nn.rrelu_with_noise",
        "torch._C._nn.scaled_dot_product_attention",
        "torch._C._nn.silu_",
        "torch._C._nn.silu",
        "torch._C._nn.slow_conv_dilated2d",
        "torch._C._nn.slow_conv_dilated3d",
        "torch._C._nn.slow_conv_transpose2d",
        "torch._C._nn.slow_conv_transpose3d",
        "torch._C._nn.slow_conv3d",
        "torch._C._nn.smooth_l1_loss",
        "torch._C._nn.soft_margin_loss",
        "torch._C._nn.softplus",
        "torch._C._nn.softshrink",
        "torch._C._nn.thnn_conv2d",
        "torch._C._nn.unflatten_dense_tensors",
        "torch._C._nn.upsample_bicubic2d",
        "torch._C._nn.upsample_bilinear2d",
        "torch._C._nn.upsample_linear1d",
        "torch._C._nn.upsample_nearest1d",
        "torch._C._nn.upsample_nearest2d",
        "torch._C._nn.upsample_nearest3d",
        "torch._C._nn.upsample_trilinear3d",
        "torch._C._non_sym_sizes",
        "torch._C._overlaps",
        "torch._C._parallel_info",
        "torch._C._parse_dispatch_key",
        "torch._C._parse_source_def",
        "torch._C._pop_torch_dispatch_stack",
        "torch._C._pop_torch_function_stack",
        "torch._C._propagate_and_assign_input_shapes",
        "torch._C._propagate_shapes",
        "torch._C._propagate_xla_data",
        "torch._C._push_on_torch_dispatch_stack",
        "torch._C._push_on_torch_function_stack",
        "torch._C._quantize_ondevice_ptq_dynamic",
        "torch._C._register_py_class_for_device",
        "torch._C._remove_cached_tensor",
        "torch._C._remove_worker_pids",
        "torch._C._rename_privateuse1_backend",
        "torch._C._replace_",
        "torch._C._replace_overloaded_method_decl",
        "torch._C._resolve_type_from_object",
        "torch._C._resolve_type",
        "torch._C._rocm_is_backward_pass",
        "torch._C._rpc_init",
        "torch._C._run_emit_module_hook",
        "torch._C._save_jit_module_to_bytes",
        "torch._C._save_jit_module",
        "torch._C._save_mobile_module_to_bytes",
        "torch._C._save_mobile_module",
        "torch._C._save_parameters",
        "torch._C._scatter_out",
        "torch._C._scatter",
        "torch._C._select_conv_backend",
        "torch._C._select_batch_norm_backend",
        "torch._C._set_autograd_fallback_mode",
        "torch._C._set_backcompat_broadcast_warn",
        "torch._C._set_backcompat_keepdim_warn",
        "torch._C._set_blas_preferred_backend",
        "torch._C._set_cached_tensors_enabled",
        "torch._C._set_check_sparse_tensor_invariants",
        "torch._C._set_conj",
        "torch._C._set_cublas_allow_bf16_reduced_precision_reduction",
        "torch._C._set_cublas_allow_fp16_reduced_precision_reduction",
        "torch._C._set_cublas_allow_tf32",
        "torch._C._set_cudnn_allow_tf32",
        "torch._C._set_cudnn_benchmark",
        "torch._C._set_cudnn_deterministic",
        "torch._C._set_cudnn_enabled",
        "torch._C._set_default_dtype",
        "torch._C._set_default_mobile_cpu_allocator",
        "torch._C._set_default_tensor_type",
        "torch._C._set_deterministic_algorithms",
        "torch._C._set_deterministic_fill_uninitialized_memory",
        "torch._C._set_dispatch_mode",
        "torch._C._set_float32_matmul_precision",
        "torch._C._set_fwd_grad_enabled",
        "torch._C._set_grad_enabled",
        "torch._C._set_graph_executor_optimize",
        "torch._C._set_linalg_preferred_backend",
        "torch._C._set_meta_in_tls_dispatch_include",
        "torch._C._set_mkldnn_enabled",
        "torch._C._set_multithreading_enabled",
        "torch._C._set_neg",
        "torch._C._set_nnpack_enabled",
        "torch._C._set_print_stack_traces_on_fatal_signal",
        "torch._C._set_qengine",
        "torch._C._set_sdp_use_flash",
        "torch._C._set_sdp_use_math",
        "torch._C._set_sdp_use_mem_efficient",
        "torch._C._set_should_use_format_with_string_table",
        "torch._C._set_storage_access_error_msg",
        "torch._C._set_tensor_metadata",
        "torch._C._set_tracing_state",
        "torch._C._set_value_trace",
        "torch._C._set_view_replay_enabled",
        "torch._C._set_warnAlways",
        "torch._C._set_worker_pids",
        "torch._C._set_worker_signal_handlers",
        "torch._C._should_allow_numbers_as_tensors",
        "torch._C._show_config",
        "torch._C._sparse._sparse_addmm",
        "torch._C._sparse._sparse_log_softmax",
        "torch._C._sparse._sparse_mm_reduce_impl",
        "torch._C._sparse._sparse_mm",
        "torch._C._sparse._sparse_softmax",
        "torch._C._sparse._spdiags",
        "torch._C._sparse.sparse_sampled_addmm",
        "torch._C._special.special_airy_ai",
        "torch._C._special.special_bessel_j0",
        "torch._C._special.special_bessel_j1",
        "torch._C._special.special_bessel_y0",
        "torch._C._special.special_bessel_y1",
        "torch._C._special.special_chebyshev_polynomial_t",
        "torch._C._special.special_chebyshev_polynomial_u",
        "torch._C._special.special_chebyshev_polynomial_v",
        "torch._C._special.special_chebyshev_polynomial_w",
        "torch._C._special.special_digamma",
        "torch._C._special.special_entr",
        "torch._C._special.special_erf",
        "torch._C._special.special_erfc",
        "torch._C._special.special_erfcx",
        "torch._C._special.special_erfinv",
        "torch._C._special.special_exp2",
        "torch._C._special.special_expit",
        "torch._C._special.special_expm1",
        "torch._C._special.special_gammainc",
        "torch._C._special.special_gammaincc",
        "torch._C._special.special_gammaln",
        "torch._C._special.special_hermite_polynomial_h",
        "torch._C._special.special_hermite_polynomial_he",
        "torch._C._special.special_i0",
        "torch._C._special.special_i0e",
        "torch._C._special.special_i1",
        "torch._C._special.special_i1e",
        "torch._C._special.special_laguerre_polynomial_l",
        "torch._C._special.special_legendre_polynomial_p",
        "torch._C._special.special_log_ndtr",
        "torch._C._special.special_log_softmax",
        "torch._C._special.special_log1p",
        "torch._C._special.special_logit",
        "torch._C._special.special_logsumexp",
        "torch._C._special.special_modified_bessel_i0",
        "torch._C._special.special_modified_bessel_i1",
        "torch._C._special.special_modified_bessel_k0",
        "torch._C._special.special_modified_bessel_k1",
        "torch._C._special.special_multigammaln",
        "torch._C._special.special_ndtr",
        "torch._C._special.special_ndtri",
        "torch._C._special.special_polygamma",
        "torch._C._special.special_psi",
        "torch._C._special.special_round",
        "torch._C._special.special_scaled_modified_bessel_k0",
        "torch._C._special.special_scaled_modified_bessel_k1",
        "torch._C._special.special_shifted_chebyshev_polynomial_t",
        "torch._C._special.special_shifted_chebyshev_polynomial_u",
        "torch._C._special.special_shifted_chebyshev_polynomial_v",
        "torch._C._special.special_shifted_chebyshev_polynomial_w",
        "torch._C._special.special_sinc",
        "torch._C._special.special_softmax",
        "torch._C._special.special_spherical_bessel_j0",
        "torch._C._special.special_xlog1py",
        "torch._C._special.special_xlogy",
        "torch._C._special.special_zeta",
        "torch._C._stash_obj_in_tls",
        "torch._C._storage_id",
        "torch._C._storage_Use_Count",
        "torch._C._supported_qengines",
        "torch._C._te.abs",
        "torch._C._te.acos",
        "torch._C._te.annotate_input_shapes",
        "torch._C._te.asin",
        "torch._C._te.atan",
        "torch._C._te.atan2",
        "torch._C._te.ceil",
        "torch._C._te.Compute",
        "torch._C._te.Compute2",
        "torch._C._te.construct_codegen",
        "torch._C._te.cos",
        "torch._C._te.cosh",
        "torch._C._te.erf",
        "torch._C._te.erfc",
        "torch._C._te.exp",
        "torch._C._te.expm1",
        "torch._C._te.fixup_missing_shape_info",
        "torch._C._te.floor",
        "torch._C._te.fmod",
        "torch._C._te.frac",
        "torch._C._te.ifThenElse",
        "torch._C._te.is_graph_compilable",
        "torch._C._te.isnan",
        "torch._C._te.lgamma",
        "torch._C._te.log",
        "torch._C._te.log10",
        "torch._C._te.log1p",
        "torch._C._te.log2",
        "torch._C._te.lower",
        "torch._C._te.make_shapes_symbolic",
        "torch._C._te.pow",
        "torch._C._te.Reduce",
        "torch._C._te.remainder",
        "torch._C._te.remove_graph_output",
        "torch._C._te.remove_unused_self_argument",
        "torch._C._te.replace_list_output_with_tuple",
        "torch._C._te.round",
        "torch._C._te.rsqrt",
        "torch._C._te.sigmoid",
        "torch._C._te.simplify",
        "torch._C._te.sin",
        "torch._C._te.sinh",
        "torch._C._te.sqrt",
        "torch._C._te.tan",
        "torch._C._te.tanh",
        "torch._C._te.trim_graph",
        "torch._C._te.trunc",
        "torch._C._tensor_impl_raw_handle",
        "torch._C._test_only_add_entry_to_op_version_map",
        "torch._C._test_only_populate_upgraders",
        "torch._C._test_only_remove_entry_to_op_version_map",
        "torch._C._test_only_remove_upgraders",
        "torch._C._to_functionality_key",
        "torch._C._tracer_set_force_outplace",
        "torch._C._tracer_set_get_unique_name_fn",
        "torch._C._tracer_warn_use_python",
        "torch._C._unset_default_mobile_cpu_allocator",
        "torch._C._unset_dispatch_mode",
        "torch._C._valgrind_supported_platform",
        "torch._C._valgrind_toggle_and_dump_stats",
        "torch._C._valgrind_toggle",
        "torch._C._verbose.mkl_set_verbose",
        "torch._C._verbose.mkldnn_set_verbose",
        "torch._C._vmapmode_decrement_nesting",
        "torch._C._vmapmode_increment_nesting",
        "torch._C._warn_deprecation",
        "torch._C._warn",
        "torch._C._will_engine_execute_node",
        "torch._C._wrap_tensor_impl",
        "torch._C.fork",
        "torch._C.get_autocast_cpu_dtype",
        "torch._C.get_autocast_dtype",
        "torch._C.get_autocast_gpu_dtype",
        "torch._C.get_autocast_ipu_dtype",
        "torch._C.get_autocast_xla_dtype",
        "torch._C.get_default_dtype",
        "torch._C.get_num_interop_threads",
        "torch._C.get_num_threads",
        "torch._C.import_ir_module_from_buffer",
        "torch._C.import_ir_module",
        "torch._C.init_num_threads",
        "torch._C.is_anomaly_check_nan_enabled",
        "torch._C.is_anomaly_enabled",
        "torch._C.is_autocast_cache_enabled",
        "torch._C.is_autocast_cpu_enabled",
        "torch._C.is_autocast_enabled",
        "torch._C.is_autocast_ipu_enabled",
        "torch._C.is_autocast_xla_enabled",
        "torch._C.is_grad_enabled",
        "torch._C.is_inference_mode_enabled",
        "torch._C.merge_type_from_type_comment",
        "torch._C.parse_ir",
        "torch._C.parse_schema",
        "torch._C.parse_type_comment",
        "torch._C.read_vitals",
        "torch._C.set_vital",
        "torch._C.unify_type_list",
        "torch._C.vitals_enabled",
        "torch._C.wait",
        "torch._cast_Byte",
        "torch._cast_Char",
        "torch._cast_Double",
        "torch._cast_Float",
        "torch._cast_Half",
        "torch._cast_Int",
        "torch._cast_Long",
        "torch._cast_Short",
        "torch._choose_qparams_per_tensor",
        "torch._chunk_cat",
        "torch._coalesce",
        "torch._compute_linear_combination",
        "torch._conj_copy",
        "torch._conj_physical",
        "torch._conj",
        "torch._convert_indices_from_coo_to_csr",
        "torch._convert_indices_from_csr_to_coo",
        "torch._convert_weight_to_int4pack",
        "torch._convolution_mode",
        "torch._convolution",
        "torch._copy_from_and_resize",
        "torch._copy_from",
        "torch._cslt_compress",
        "torch._cslt_sparse_mm",
        "torch._ctc_loss",
        "torch._cudnn_ctc_loss",
        "torch._cudnn_init_dropout_state",
        "torch._cudnn_rnn_flatten_weight",
        "torch._cudnn_rnn",
        "torch._cufft_clear_plan_cache",
        "torch._cufft_get_plan_cache_max_size",
        "torch._cufft_get_plan_cache_size",
        "torch._cufft_set_plan_cache_max_size",
        "torch._cummax_helper",
        "torch._cummin_helper",
        "torch._debug_has_internal_overlap",
        "torch._dim_arange",
        "torch._dirichlet_grad",
        "torch._disable_functionalization",
        "torch._efficientzerotensor",
        "torch._embedding_bag_forward_only",
        "torch._embedding_bag",
        "torch._empty_affine_quantized",
        "torch._empty_per_channel_affine_quantized",
        "torch._enable_functionalization",
        "torch._euclidean_dist",
        "torch._fake_quantize_learnable_per_channel_affine",
        "torch._fake_quantize_learnable_per_tensor_affine",
        "torch._fake_quantize_per_tensor_affine_cachemask_tensor_qparams",
        "torch._fft_c2c",
        "torch._fft_c2r",
        "torch._fft_r2c",
        "torch._fill_mem_eff_dropout_mask_",
        "torch._foobar",
        "torch._foreach_abs_",
        "torch._foreach_abs",
        "torch._foreach_acos_",
        "torch._foreach_acos",
        "torch._foreach_add_",
        "torch._foreach_add",
        "torch._foreach_addcdiv_",
        "torch._foreach_addcdiv",
        "torch._foreach_addcmul_",
        "torch._foreach_addcmul",
        "torch._foreach_asin_",
        "torch._foreach_asin",
        "torch._foreach_atan_",
        "torch._foreach_atan",
        "torch._foreach_ceil_",
        "torch._foreach_ceil",
        "torch._foreach_clamp_max_",
        "torch._foreach_clamp_max",
        "torch._foreach_clamp_min_",
        "torch._foreach_clamp_min",
        "torch._foreach_copy_",
        "torch._foreach_cos_",
        "torch._foreach_cos",
        "torch._foreach_cosh_",
        "torch._foreach_cosh",
        "torch._foreach_div_",
        "torch._foreach_div",
        "torch._foreach_erf_",
        "torch._foreach_erf",
        "torch._foreach_erfc_",
        "torch._foreach_erfc",
        "torch._foreach_exp_",
        "torch._foreach_exp",
        "torch._foreach_expm1_",
        "torch._foreach_expm1",
        "torch._foreach_floor_",
        "torch._foreach_floor",
        "torch._foreach_frac_",
        "torch._foreach_frac",
        "torch._foreach_lerp_",
        "torch._foreach_lerp",
        "torch._foreach_lgamma_",
        "torch._foreach_lgamma",
        "torch._foreach_log_",
        "torch._foreach_log",
        "torch._foreach_log10_",
        "torch._foreach_log10",
        "torch._foreach_log1p_",
        "torch._foreach_log1p",
        "torch._foreach_log2_",
        "torch._foreach_log2",
        "torch._foreach_maximum_",
        "torch._foreach_maximum",
        "torch._foreach_minimum_",
        "torch._foreach_minimum",
        "torch._foreach_mul_",
        "torch._foreach_mul",
        "torch._foreach_neg_",
        "torch._foreach_neg",
        "torch._foreach_norm",
        "torch._foreach_pow_",
        "torch._foreach_pow",
        "torch._foreach_reciprocal_",
        "torch._foreach_reciprocal",
        "torch._foreach_round_",
        "torch._foreach_round",
        "torch._foreach_sigmoid_",
        "torch._foreach_sigmoid",
        "torch._foreach_sign_",
        "torch._foreach_sign",
        "torch._foreach_sin_",
        "torch._foreach_sin",
        "torch._foreach_sinh_",
        "torch._foreach_sinh",
        "torch._foreach_sqrt_",
        "torch._foreach_sqrt",
        "torch._foreach_sub_",
        "torch._foreach_sub",
        "torch._foreach_tan_",
        "torch._foreach_tan",
        "torch._foreach_tanh_",
        "torch._foreach_tanh",
        "torch._foreach_trunc_",
        "torch._foreach_trunc",
        "torch._foreach_zero_",
        "torch._freeze_functional_tensor",
        "torch._from_functional_tensor",
        "torch._functional_assert_async",
        "torch._functional_sym_constrain_range_for_size",
        "torch._functional_sym_constrain_range",
        "torch._functionalize_are_all_mutations_hidden_from_autograd",
        "torch._functionalize_commit_update",
        "torch._functionalize_enable_reapply_views",
        "torch._functionalize_has_data_mutation",
        "torch._functionalize_has_metadata_mutation",
        "torch._functionalize_is_multi_output_view",
        "torch._functionalize_mark_mutation_hidden_from_autograd",
        "torch._functionalize_replace",
        "torch._functionalize_sync",
        "torch._functionalize_was_storage_changed",
        "torch._fused_adam_",
        "torch._fused_adamw_",
        "torch._fused_dropout",
        "torch._fused_moving_avg_obs_fq_helper",
        "torch._fused_sdp_choice",
        "torch._fw_primal_copy",
        "torch._grid_sampler_2d_cpu_fallback",
        "torch._has_compatible_shallow_copy_type",
        "torch._histogramdd_bin_edges",
        "torch._histogramdd_from_bin_cts",
        "torch._histogramdd_from_bin_tensors",
        "torch._index_put_impl_",
        "torch._indices_copy",
        "torch._int_mm",
        "torch._is_all_true",
        "torch._is_any_true",
        "torch._is_functional_tensor",
        "torch._is_zerotensor",
        "torch._linalg_check_errors",
        "torch._linalg_det",
        "torch._linalg_eigh",
        "torch._linalg_eigvals",
        "torch._linalg_slogdet",
        "torch._linalg_solve_ex",
        "torch._linalg_svd",
        "torch._log_softmax_backward_data",
        "torch._log_softmax",
        "torch._logcumsumexp",
        "torch._lstm_mps",
        "torch._lu_with_info",
        "torch._make_dep_token",
        "torch._make_dual_copy",
        "torch._make_dual",
        "torch._make_per_channel_quantized_tensor",
        "torch._make_per_tensor_quantized_tensor",
        "torch._masked_scale",
        "torch._masked_softmax",
        "torch._mirror_autograd_meta_to",
        "torch._mixed_dtypes_linear",
        "torch._mkldnn_reshape",
        "torch._mkldnn_transpose_",
        "torch._mkldnn_transpose",
        "torch._mps_convolution_transpose",
        "torch._mps_convolution",
        "torch._native_batch_norm_legit_no_training",
        "torch._native_batch_norm_legit",
        "torch._native_multi_head_attention",
        "torch._neg_view_copy",
        "torch._neg_view",
        "torch._nested_from_padded_and_nested_example",
        "torch._nested_tensor_from_mask_left_aligned",
        "torch._nested_tensor_from_tensor_list",
        "torch._nested_tensor_softmax_with_shape",
        "torch._nested_view_from_buffer_copy",
        "torch._nested_view_from_buffer",
        "torch._nnpack_available",
        "torch._nnpack_spatial_convolution",
        "torch._pack_padded_sequence",
        "torch._pad_packed_sequence",
        "torch._pin_memory",
        "torch._prelu_kernel",
        "torch._propagate_xla_data",
        "torch._remove_batch_dim",
        "torch._reshape_alias_copy",
        "torch._reshape_from_tensor",
        "torch._resize_output_",
        "torch._rowwise_prune",
        "torch._sample_dirichlet",
        "torch._saturate_weight_to_fp16",
        "torch._scaled_dot_product_attention_math",
        "torch._scaled_dot_product_efficient_attention",
        "torch._scaled_dot_product_flash_attention",
        "torch._scaled_dot_product_flash_attention_for_cpu",
        "torch._scaled_dot_product_cudnn_attention",
        "torch._scaled_mm",
        "torch._shape_as_tensor",
        "torch._sobol_engine_draw",
        "torch._sobol_engine_ff_",
        "torch._sobol_engine_initialize_state_",
        "torch._sobol_engine_scramble_",
        "torch._softmax_backward_data",
        "torch._softmax",
        "torch._sparse_broadcast_to_copy",
        "torch._sparse_broadcast_to",
        "torch._sparse_csr_prod",
        "torch._sparse_csr_sum",
        "torch._sparse_log_softmax_backward_data",
        "torch._sparse_semi_structured_addmm",
        "torch._sparse_semi_structured_linear",
        "torch._sparse_semi_structured_mm",
        "torch._sparse_softmax_backward_data",
        "torch._sparse_sparse_matmul",
        "torch._sparse_sum",
        "torch._stack",
        "torch._standard_gamma_grad",
        "torch._standard_gamma",
        "torch._test_autograd_multiple_dispatch_view_copy",
        "torch._test_autograd_multiple_dispatch_view",
        "torch._test_autograd_multiple_dispatch",
        "torch._test_check_tensor",
        "torch._test_functorch_fallback",
        "torch._test_serialization_subcmul",
        "torch._to_cpu",
        "torch._to_functional_tensor",
        "torch._to_sparse_semi_structured",
        "torch._transform_bias_rescale_qkv",
        "torch._transformer_encoder_layer_fwd",
        "torch._trilinear",
        "torch._triton_multi_head_attention",
        "torch._triton_scaled_dot_attention",
        "torch._unique",
        "torch._unique2",
        "torch._unpack_dual",
        "torch._unsafe_index_put",
        "torch._unsafe_index",
        "torch._unsafe_masked_index_put_accumulate",
        "torch._unsafe_masked_index",
        "torch._use_cudnn_ctc_loss",
        "torch._use_cudnn_rnn_flatten_weight",
        "torch._values_copy",
        "torch._weight_int4pack_mm",
        "torch._weight_int8pack_mm",
        "torch._weight_norm_interface",
        "torch._weight_norm",
        "torch.abs_",
        "torch.abs",
        "torch.absolute",
        "torch.acos_",
        "torch.acos",
        "torch.acosh_",
        "torch.acosh",
        "torch.adaptive_avg_pool1d",
        "torch.adaptive_max_pool1d",
        "torch.add",
        "torch.addbmm",
        "torch.addcdiv",
        "torch.addcmul",
        "torch.addmm",
        "torch.addmv_",
        "torch.addmv",
        "torch.addr",
        "torch.adjoint",
        "torch.affine_grid_generator",
        "torch.alias_copy",
        "torch.all",
        "torch.allclose",
        "torch.alpha_dropout_",
        "torch.alpha_dropout",
        "torch.amax",
        "torch.amin",
        "torch.aminmax",
        "torch.angle",
        "torch.any",
        "torch.arange",
        "torch.arccos_",
        "torch.arccos",
        "torch.arccosh_",
        "torch.arccosh",
        "torch.arcsin_",
        "torch.arcsin",
        "torch.arcsinh_",
        "torch.arcsinh",
        "torch.arctan_",
        "torch.arctan",
        "torch.arctan2",
        "torch.arctanh_",
        "torch.arctanh",
        "torch.argmax",
        "torch.argmin",
        "torch.argsort",
        "torch.argwhere",
        "torch.as_strided_",
        "torch.as_strided_copy",
        "torch.as_strided_scatter",
        "torch.as_strided",
        "torch.as_tensor",
        "torch.asarray",
        "torch.asin_",
        "torch.asin",
        "torch.asinh_",
        "torch.asinh",
        "torch.atan_",
        "torch.atan",
        "torch.atan2",
        "torch.atanh_",
        "torch.atanh",
        "torch.avg_pool1d",
        "torch.baddbmm",
        "torch.bartlett_window",
        "torch.batch_norm_backward_elemt",
        "torch.batch_norm_backward_reduce",
        "torch.batch_norm_elemt",
        "torch.batch_norm_gather_stats_with_counts",
        "torch.batch_norm_gather_stats",
        "torch.batch_norm_stats",
        "torch.batch_norm_update_stats",
        "torch.batch_norm",
        "torch.bernoulli",
        "torch.bilinear",
        "torch.binary_cross_entropy_with_logits",
        "torch.bincount",
        "torch.binomial",
        "torch.bitwise_and",
        "torch.bitwise_left_shift",
        "torch.bitwise_not",
        "torch.bitwise_or",
        "torch.bitwise_right_shift",
        "torch.bitwise_xor",
        "torch.blackman_window",
        "torch.bmm",
        "torch.broadcast_to",
        "torch.bucketize",
        "torch.can_cast",
        "torch.cat",
        "torch.ccol_indices_copy",
        "torch.ceil_",
        "torch.ceil",
        "torch.celu_",
        "torch.celu",
        "torch.channel_shuffle",
        "torch.cholesky_inverse",
        "torch.cholesky_solve",
        "torch.cholesky",
        "torch.choose_qparams_optimized",
        "torch.chunk",
        "torch.clamp_",
        "torch.clamp_max_",
        "torch.clamp_max",
        "torch.clamp_min_",
        "torch.clamp_min",
        "torch.clamp",
        "torch.clip_",
        "torch.clip",
        "torch.clone",
        "torch.col_indices_copy",
        "torch.column_stack",
        "torch.combinations",
        "torch.complex",
        "torch.concat",
        "torch.concatenate",
        "torch.conj_physical_",
        "torch.conj_physical",
        "torch.conj",
        "torch.constant_pad_nd",
        "torch.conv_tbc",
        "torch.conv_transpose1d",
        "torch.conv_transpose2d",
        "torch.conv_transpose3d",
        "torch.conv1d",
        "torch.conv2d",
        "torch.conv3d",
        "torch.convolution",
        "torch.copysign",
        "torch.corrcoef",
        "torch.cos_",
        "torch.cos",
        "torch.cosh_",
        "torch.cosh",
        "torch.cosine_embedding_loss",
        "torch.cosine_similarity",
        "torch.count_nonzero",
        "torch.cov",
        "torch.cross",
        "torch.crow_indices_copy",
        "torch.ctc_loss",
        "torch.cudnn_affine_grid_generator",
        "torch.cudnn_batch_norm",
        "torch.cudnn_convolution_add_relu",
        "torch.cudnn_convolution_relu",
        "torch.cudnn_convolution_transpose",
        "torch.cudnn_convolution",
        "torch.cudnn_grid_sampler",
        "torch.cudnn_is_acceptable",
        "torch.cummax",
        "torch.cummin",
        "torch.cumprod",
        "torch.cumsum",
        "torch.cumulative_trapezoid",
        "torch.deg2rad_",
        "torch.deg2rad",
        "torch.dequantize",
        "torch.det",
        "torch.detach_",
        "torch.detach_copy",
        "torch.detach",
        "torch.diag_embed",
        "torch.diag",
        "torch.diagflat",
        "torch.diagonal_copy",
        "torch.diagonal_scatter",
        "torch.diagonal",
        "torch.diff",
        "torch.digamma",
        "torch.dist",
        "torch.div",
        "torch.divide",
        "torch.dot",
        "torch.dropout_",
        "torch.dropout",
        "torch.dsmm",
        "torch.dsplit",
        "torch.dstack",
        "torch.embedding_bag",
        "torch.embedding_renorm_",
        "torch.embedding",
        "torch.empty_like",
        "torch.empty_permuted",
        "torch.empty_quantized",
        "torch.empty_strided",
        "torch.empty",
        "torch.eq",
        "torch.equal",
        "torch.erf_",
        "torch.erf",
        "torch.erfc_",
        "torch.erfc",
        "torch.erfinv",
        "torch.exp_",
        "torch.exp",
        "torch.exp2_",
        "torch.exp2",
        "torch.expand_copy",
        "torch.expm1_",
        "torch.expm1",
        "torch.eye",
        "torch.fake_quantize_per_channel_affine",
        "torch.fake_quantize_per_tensor_affine",
        "torch.fbgemm_linear_fp16_weight_fp32_activation",
        "torch.fbgemm_linear_fp16_weight",
        "torch.fbgemm_linear_int8_weight_fp32_activation",
        "torch.fbgemm_linear_int8_weight",
        "torch.fbgemm_linear_quantize_weight",
        "torch.fbgemm_pack_gemm_matrix_fp16",
        "torch.fbgemm_pack_quantized_matrix",
        "torch.feature_alpha_dropout_",
        "torch.feature_alpha_dropout",
        "torch.feature_dropout_",
        "torch.feature_dropout",
        "torch.fill_",
        "torch.fill",
        "torch.fix_",
        "torch.fix",
        "torch.flatten",
        "torch.flip",
        "torch.fliplr",
        "torch.flipud",
        "torch.float_power",
        "torch.floor_",
        "torch.floor_divide",
        "torch.floor",
        "torch.fmax",
        "torch.fmin",
        "torch.fmod",
        "torch.frac_",
        "torch.frac",
        "torch.frexp",
        "torch.frobenius_norm",
        "torch.from_file",
        "torch.from_numpy",
        "torch.frombuffer",
        "torch.full_like",
        "torch.full",
        "torch.fused_moving_avg_obs_fake_quant",
        "torch.gather",
        "torch.gcd_",
        "torch.gcd",
        "torch.ge",
        "torch.geqrf",
        "torch.ger",
        "torch.get_device",
        "torch.gradient",
        "torch.greater_equal",
        "torch.greater",
        "torch.grid_sampler_2d",
        "torch.grid_sampler_3d",
        "torch.grid_sampler",
        "torch.group_norm",
        "torch.gru_cell",
        "torch.gru",
        "torch.gt",
        "torch.hamming_window",
        "torch.hann_window",
        "torch.hardshrink",
        "torch.heaviside",
        "torch.hinge_embedding_loss",
        "torch.histc",
        "torch.histogram",
        "torch.histogramdd",
        "torch.hsmm",
        "torch.hsplit",
        "torch.hspmm",
        "torch.hstack",
        "torch.hypot",
        "torch.i0_",
        "torch.i0",
        "torch.igamma",
        "torch.igammac",
        "torch.imag",
        "torch.index_add",
        "torch.index_copy",
        "torch.index_fill",
        "torch.index_put_",
        "torch.index_put",
        "torch.index_reduce",
        "torch.index_select",
        "torch.indices_copy",
        "torch.inner",
        "torch.instance_norm",
        "torch.int_repr",
        "torch.inverse",
        "torch.is_complex",
        "torch.is_conj",
        "torch.is_distributed",
        "torch.is_floating_point",
        "torch.is_inference",
        "torch.is_neg",
        "torch.is_nonzero",
        "torch.is_same_size",
        "torch.is_signed",
        "torch.is_vulkan_available",
        "torch.isclose",
        "torch.isfinite",
        "torch.isin",
        "torch.isinf",
        "torch.isnan",
        "torch.isneginf",
        "torch.isposinf",
        "torch.isreal",
        "torch.istft",
        "torch.kaiser_window",
        "torch.kl_div",
        "torch.kron",
        "torch.kthvalue",
        "torch.layer_norm",
        "torch.lcm_",
        "torch.lcm",
        "torch.ldexp_",
        "torch.ldexp",
        "torch.le",
        "torch.lerp",
        "torch.less_equal",
        "torch.less",
        "torch.lgamma",
        "torch.linspace",
        "torch.log_",
        "torch.log_softmax",
        "torch.log",
        "torch.log10_",
        "torch.log10",
        "torch.log1p_",
        "torch.log1p",
        "torch.log2_",
        "torch.log2",
        "torch.logaddexp",
        "torch.logaddexp2",
        "torch.logcumsumexp",
        "torch.logdet",
        "torch.logical_and",
        "torch.logical_not",
        "torch.logical_or",
        "torch.logical_xor",
        "torch.logit_",
        "torch.logit",
        "torch.logspace",
        "torch.logsumexp",
        "torch.lstm_cell",
        "torch.lstm",
        "torch.lt",
        "torch.lu_solve",
        "torch.lu_unpack",
        "torch.margin_ranking_loss",
        "torch.masked_fill",
        "torch.masked_scatter",
        "torch.masked_select",
        "torch.matmul",
        "torch.matrix_exp",
        "torch.matrix_power",
        "torch.max_pool1d_with_indices",
        "torch.max_pool1d",
        "torch.max_pool2d",
        "torch.max_pool3d",
        "torch.max",
        "torch.maximum",
        "torch.mean",
        "torch.median",
        "torch.min",
        "torch.minimum",
        "torch.miopen_batch_norm",
        "torch.miopen_convolution_add_relu",
        "torch.miopen_convolution_relu",
        "torch.miopen_convolution_transpose",
        "torch.miopen_convolution",
        "torch.miopen_depthwise_convolution",
        "torch.miopen_rnn",
        "torch.mkldnn_adaptive_avg_pool2d",
        "torch.mkldnn_convolution",
        "torch.mkldnn_linear_backward_weights",
        "torch.mkldnn_max_pool2d",
        "torch.mkldnn_max_pool3d",
        "torch.mkldnn_rnn_layer",
        "torch.mm",
        "torch.mode",
        "torch.moveaxis",
        "torch.movedim",
        "torch.msort",
        "torch.mul",
        "torch.multinomial",
        "torch.multiply",
        "torch.mv",
        "torch.mvlgamma",
        "torch.nan_to_num_",
        "torch.nan_to_num",
        "torch.nanmean",
        "torch.nanmedian",
        "torch.nanquantile",
        "torch.nansum",
        "torch.narrow_copy",
        "torch.narrow",
        "torch.native_batch_norm",
        "torch.native_channel_shuffle",
        "torch.native_dropout",
        "torch.native_group_norm",
        "torch.native_layer_norm",
        "torch.native_norm",
        "torch.ne",
        "torch.neg_",
        "torch.neg",
        "torch.negative_",
        "torch.negative",
        "torch.nextafter",
        "torch.nonzero_static",
        "torch.nonzero",
        "torch.norm_except_dim",
        "torch.normal",
        "torch.not_equal",
        "torch.nuclear_norm",
        "torch.numel",
        "torch.ones_like",
        "torch.ones",
        "torch.orgqr",
        "torch.ormqr",
        "torch.outer",
        "torch.pairwise_distance",
        "torch.pdist",
        "torch.permute_copy",
        "torch.permute",
        "torch.pinverse",
        "torch.pixel_shuffle",
        "torch.pixel_unshuffle",
        "torch.poisson_nll_loss",
        "torch.poisson",
        "torch.polar",
        "torch.polygamma",
        "torch.positive",
        "torch.pow",
        "torch.prelu",
        "torch._print",
        "torch.prod",
        "torch.promote_types",
        "torch.put",
        "torch.q_per_channel_axis",
        "torch.q_per_channel_scales",
        "torch.q_per_channel_zero_points",
        "torch.q_scale",
        "torch.q_zero_point",
        "torch.qr",
        "torch.quantile",
        "torch.quantize_per_channel",
        "torch.quantize_per_tensor_dynamic",
        "torch.quantize_per_tensor",
        "torch.quantized_batch_norm",
        "torch.quantized_gru_cell",
        "torch.quantized_lstm_cell",
        "torch.quantized_max_pool1d",
        "torch.quantized_max_pool2d",
        "torch.quantized_max_pool3d",
        "torch.quantized_rnn_relu_cell",
        "torch.quantized_rnn_tanh_cell",
        "torch.rad2deg_",
        "torch.rad2deg",
        "torch.rand_like",
        "torch.rand",
        "torch.randint_like",
        "torch.randint",
        "torch.randn_like",
        "torch.randn",
        "torch.randperm",
        "torch.range",
        "torch.ravel",
        "torch.real",
        "torch.reciprocal_",
        "torch.reciprocal",
        "torch.relu_",
        "torch.relu",
        "torch.remainder",
        "torch.renorm",
        "torch.repeat_interleave",
        "torch.reshape",
        "torch.resolve_conj",
        "torch.resolve_neg",
        "torch.result_type",
        "torch.rms_norm",
        "torch.rnn_relu_cell",
        "torch.rnn_relu",
        "torch.rnn_tanh_cell",
        "torch.rnn_tanh",
        "torch.roll",
        "torch.rot90",
        "torch.round_",
        "torch.round",
        "torch.row_indices_copy",
        "torch.row_stack",
        "torch.rrelu_",
        "torch.rrelu",
        "torch.rsqrt_",
        "torch.rsqrt",
        "torch.rsub",
        "torch.saddmm",
        "torch.scalar_tensor",
        "torch.scatter_add",
        "torch.scatter_reduce",
        "torch.scatter",
        "torch.searchsorted",
        "torch.segment_reduce",
        "torch.select_copy",
        "torch.select_scatter",
        "torch.select",
        "torch.selu_",
        "torch.selu",
        "torch.sgn",
        "torch.sigmoid_",
        "torch.sigmoid",
        "torch.sign",
        "torch.signal.windows.windows.sqrt",
        "torch.signbit",
        "torch.sin_",
        "torch.sin",
        "torch.sinc_",
        "torch.sinc",
        "torch.sinh_",
        "torch.sinh",
        "torch.slice_copy",
        "torch.slice_scatter",
        "torch.slogdet",
        "torch.smm",
        "torch.softmax",
        "torch.sort",
        "torch.split_copy",
        "torch.split_with_sizes_copy",
        "torch.split_with_sizes",
        "torch.spmm",
        "torch.sqrt_",
        "torch.sqrt",
        "torch.square_",
        "torch.square",
        "torch.squeeze_copy",
        "torch.squeeze",
        "torch.sspaddmm",
        "torch.stack",
        "torch.std_mean",
        "torch.std",
        "torch.sub",
        "torch.subtract",
        "torch.sum",
        "torch.svd",
        "torch.swapaxes",
        "torch.swapdims",
        "torch.sym_constrain_range_for_size",
        "torch.sym_constrain_range",
        "torch.t_copy",
        "torch.t",
        "torch.take_along_dim",
        "torch.take",
        "torch.tan_",
        "torch.tan",
        "torch.tanh_",
        "torch.tanh",
        "torch.tensor_split",
        "torch.tensor",
        "torch.threshold_",
        "torch.threshold",
        "torch.tile",
        "torch.topk",
        "torch.trace",
        "torch.transpose_copy",
        "torch.transpose",
        "torch.trapezoid",
        "torch.trapz",
        "torch.triangular_solve",
        "torch.tril_indices",
        "torch.tril",
        "torch.triplet_margin_loss",
        "torch.triu_indices",
        "torch.triu",
        "torch.true_divide",
        "torch.trunc_",
        "torch.trunc",
        "torch.unbind_copy",
        "torch.unbind",
        "torch.unflatten",
        "torch.unfold_copy",
        "torch.unsafe_chunk",
        "torch.unsafe_split_with_sizes",
        "torch.unsafe_split",
        "torch.unsqueeze_copy",
        "torch.unsqueeze",
        "torch.values_copy",
        "torch.vander",
        "torch.var_mean",
        "torch.var",
        "torch.vdot",
        "torch.view_as_complex_copy",
        "torch.view_as_complex",
        "torch.view_as_real_copy",
        "torch.view_as_real",
        "torch.view_copy",
        "torch.vsplit",
        "torch.vstack",
        "torch.where",
        "torch.xlogy_",
        "torch.xlogy",
        "torch.zero_",
        "torch.zeros",
        "torch.zeros_like",
        "torch._fused_sgd_",
        "torch.slice_inverse",
        "torch._assert_scalar",
        "torch._functional_assert_scalar",
    ],
    TorchInGraphFunctionVariable,
)


if sys.version_info >= (3, 9):
    torch_c_binding_in_graph_functions["math.lcm"] = TorchInGraphFunctionVariable
if sys.version_info >= (3, 11):
    torch_c_binding_in_graph_functions["math.exp2"] = TorchInGraphFunctionVariable
    torch_c_binding_in_graph_functions["math.cbrt"] = TorchInGraphFunctionVariable


# In graph functions (including constant folding) that are not C bindings
torch_non_c_binding_in_graph_functions = dict.fromkeys(
    [
        "torch.__future__.get_overwrite_module_params_on_conversion",
        "torch.__future__.set_overwrite_module_params_on_conversion",
        "torch.__getattr__",
        "torch._assert",
        "torch._check_index",
        "torch._check_is_size",
        "torch._check_not_implemented",
        "torch._check_tensor_all_with",
        "torch._check_tensor_all",
        "torch._check_type",
        "torch._check_value",
        "torch._check_with",
        "torch._check",
        "torch._compile._disable_dynamo",
        "torch._functorch.apis.chunk_vmap",
        "torch._functorch.autograd_function.custom_function_call_functionalize",
        "torch._functorch.autograd_function.custom_function_call_grad",
        "torch._functorch.autograd_function.custom_function_call_vmap_generate_rule",
        "torch._functorch.autograd_function.custom_function_call_vmap",
        "torch._functorch.autograd_function.generate_single_level_function",
        "torch._functorch.autograd_function.get_tangents_in_dims",
        "torch._functorch.autograd_function.has_overriden_vmap_rule",
        "torch._functorch.autograd_function.reductify_leaf",
        "torch._functorch.autograd_function.reductify",
        "torch._functorch.autograd_function.validate_vmap_returns_tuple_of_two_elements",
        "torch._functorch.autograd_function.vmapify_autograd_function",
        "torch._functorch.autograd_function.wrap_outputs_maintaining_identity",
        "torch._functorch.batch_norm_replacement.batch_norm_without_running_stats",
        "torch._functorch.batch_norm_replacement.replace_all_batch_norm_modules_",
        "torch._functorch.deprecated.combine_state_for_ensemble",
        "torch._functorch.deprecated.functionalize",
        "torch._functorch.deprecated.get_warning",
        "torch._functorch.deprecated.make_functional_with_buffers",
        "torch._functorch.deprecated.make_functional",
        "torch._functorch.deprecated.setup_docs",
        "torch._functorch.deprecated.warn_deprecated",
        "torch._functorch.eager_transforms._any_differentiable",
        "torch._functorch.eager_transforms._autograd_grad",
        "torch._functorch.eager_transforms._vjp_treespec_compare",
        "torch._functorch.eager_transforms._set_tensor_requires_grad",
        "torch._functorch.eager_transforms._jvp_treespec_compare",
        "torch._functorch.eager_transforms._linearize_treespec_compare",
        "torch._functorch.eager_transforms._is_differentiable",
        "torch._functorch.eager_transforms._maybe_unwrap_functional_tensor",
        "torch._functorch.eager_transforms._maybe_wrap_functional_tensor",
        "torch._functorch.eager_transforms._unwrap_all_tensors_from_functional",
        "torch._functorch.eager_transforms._wrap_all_tensors_to_functional",
        "torch._functorch.eager_transforms.assert_flat_tuple_of_tensors",
        "torch._functorch.eager_transforms.functionalize",
        "torch._functorch.eager_transforms.lazy_dynamo_disable",
        "torch._functorch.eager_transforms.noop",
        "torch._functorch.pyfunctorch.coerce_cinterpreter",
        "torch._functorch.pyfunctorch.dispatch_functorch",
        "torch._functorch.pyfunctorch.nested",
        "torch._functorch.pyfunctorch.retrieve_current_functorch_interpreter",
        "torch._functorch.pyfunctorch.temporarily_pop_interpreter_stack",
        "torch._functorch.utils.enable_single_level_autograd_function",
        "torch._functorch.utils.exposed_in",
        "torch._functorch.utils.unwrap_dead_wrappers",
        "torch._functorch.vmap.lazy_load_decompositions",
        "torch._guards.compile_context",
        "torch._guards.detect_fake_mode",
        "torch._guards.tracing",
        "torch._higher_order_ops.map._has_potential_branch_input_alias",
        "torch._higher_order_ops.map._has_potential_branch_input_mutation",
        "torch._higher_order_ops.map._stack_pytree",
        "torch._higher_order_ops.map._unstack_pytree",
        "torch._higher_order_ops.map.create_fw_bw_graph",
        "torch._higher_order_ops.map.map_autograd",
        "torch._higher_order_ops.map.map_dense",
        "torch._higher_order_ops.map.map_fake_tensor_mode",
        "torch._higher_order_ops.map.map_functionalize",
        "torch._higher_order_ops.map.map_proxy_torch_dispatch_mode",
        "torch._higher_order_ops.map.map_wrapper",
        "torch._higher_order_ops.map.trace_map",
        "torch._higher_order_ops.out_dtype.elementwise_dtypes",
        "torch._higher_order_ops.out_dtype.is_int_mm",
        "torch._higher_order_ops.out_dtype.out_dtype_dense",
        "torch._higher_order_ops.out_dtype.out_dtype_fake_tensor_mode",
        "torch._higher_order_ops.out_dtype.out_dtype_fallback",
        "torch._higher_order_ops.out_dtype.out_dtype_func",
        "torch._higher_order_ops.out_dtype.out_dtype_proxy",
        "torch._higher_order_ops.out_dtype.trace_out_dtype",
        "torch._higher_order_ops.utils.autograd_not_implemented_inner",
        "torch._higher_order_ops.utils.autograd_not_implemented",
        "torch._linalg_utils._symeig",
        "torch._linalg_utils.basis",
        "torch._linalg_utils.bform",
        "torch._linalg_utils.eig",
        "torch._linalg_utils.get_floating_dtype",
        "torch._linalg_utils.is_sparse",
        "torch._linalg_utils.lstsq",
        "torch._linalg_utils.matmul",
        "torch._linalg_utils.matrix_rank",
        "torch._linalg_utils.qform",
        "torch._linalg_utils.solve",
        "torch._linalg_utils.symeig",
        "torch._load_global_deps",
        "torch._lowrank._svd_lowrank",
        "torch._lowrank.get_approximate_basis",
        "torch._lowrank.pca_lowrank",
        "torch._lowrank.svd_lowrank",
        "torch._ops._compute_keyset",
        "torch._ops._get_tensors",
        "torch._ops._to_flat_tuple",
        "torch._ops.add_cached_op",
        "torch._ops.dl_open_guard",
        "torch._ops.get_cached_ops",
        "torch._ops.key_extractor",
        "torch._ops.reset_cached_ops",
        "torch._ops.resolve_key",
        "torch._preload_cuda_deps",
        "torch._register_device_module",
        "torch._running_with_deploy",
        "torch._utils._dummy_type",
        "torch._weights_only_unpickler._get_allowed_globals",
        "torch._weights_only_unpickler.load",
        "torch.align_tensors",
        "torch.amp.autocast_mode._enter_autocast",
        "torch.amp.autocast_mode._exit_autocast",
        "torch.amp.autocast_mode.autocast_decorator",
        "torch.amp.autocast_mode.custom_bwd",
        "torch.amp.autocast_mode.custom_fwd",
        "torch.are_deterministic_algorithms_enabled",
        "torch.atleast_1d",
        "torch.atleast_2d",
        "torch.atleast_3d",
        "torch.autograd._calculate_shape",
        "torch.autograd._is_checkpoint_valid",
        "torch.autograd._make_grads",
        "torch.autograd._register_py_tensor_class_for_device",
        "torch.autograd._tensor_or_tensors_to_tuple",
        "torch.autograd.forward_ad._maybe_load_decompositions",
        "torch.autograd.function._iter_filter",
        "torch.autograd.function._iter_jit_values",
        "torch.autograd.function._iter_None_tensors",
        "torch.autograd.function._iter_tensors_permissive",
        "torch.autograd.function._iter_tensors",
        "torch.autograd.function._jit_unwrap_structured",
        "torch.autograd.function._map_tensor_data",
        "torch.autograd.function._nested_map",
        "torch.autograd.function._unflatten",
        "torch.autograd.function.once_differentiable",
        "torch.autograd.function.traceable",
        "torch.autograd.functional._as_tuple_nocheck",
        "torch.autograd.functional._as_tuple",
        "torch.autograd.functional._autograd_grad",
        "torch.autograd.functional._check_requires_grad",
        "torch.autograd.functional._construct_standard_basis_for",
        "torch.autograd.functional._fill_in_zeros",
        "torch.autograd.functional._grad_postprocess",
        "torch.autograd.functional._grad_preprocess",
        "torch.autograd.functional._jacfwd",
        "torch.autograd.functional._tuple_postprocess",
        "torch.autograd.functional._validate_v",
        "torch.autograd.functional.hessian",
        "torch.autograd.functional.hvp",
        "torch.autograd.functional.jacobian",
        "torch.autograd.functional.jvp",
        "torch.autograd.functional.vhp",
        "torch.autograd.functional.vjp",
        "torch.autograd.grad_mode._enter_inference_mode",
        "torch.autograd.grad_mode._exit_inference_mode",
        "torch.autograd.graph._get_sid",
        "torch.autograd.graph._get_tid",
        "torch.autograd.graph.allow_mutation_on_saved_tensors",
        "torch.autograd.graph.get_gradient_edge",
        "torch.autograd.graph.increment_version",
        "torch.autograd.graph.register_multi_grad_hook",
        "torch.autograd.variable",
        "torch.backends.__allow_nonbracketed_mutation",
        "torch.backends.cpu.get_cpu_capability",
        "torch.backends.cuda.can_use_efficient_attention",
        "torch.backends.cuda.can_use_flash_attention",
        "torch.backends.cuda.can_use_cudnn_attention",
        "torch.backends.cuda.enable_flash_sdp",
        "torch.backends.cuda.enable_math_sdp",
        "torch.backends.cuda.enable_mem_efficient_sdp",
        "torch.backends.cuda.flash_sdp_enabled",
        "torch.backends.cuda.is_built",
        "torch.backends.cuda.math_sdp_enabled",
        "torch.backends.cuda.mem_efficient_sdp_enabled",
        "torch.backends.cuda.cudnn_sdp_enabled",
        "torch.backends.cuda.enable_cudnn_sdp",
        "torch.backends.cuda.preferred_blas_library",
        "torch.backends.cuda.preferred_linalg_library",
        "torch.backends.cuda.sdp_kernel",
        "torch.backends.cudnn._init",
        "torch.backends.cudnn.flags",
        "torch.backends.cudnn.is_acceptable",
        "torch.backends.cudnn.is_available",
        "torch.backends.cudnn.set_flags",
        "torch.backends.cudnn.version",
        "torch.backends.disable_global_flags",
        "torch.backends.flags_frozen",
        "torch.backends.mkl.is_available",
        "torch.backends.mkldnn.flags",
        "torch.backends.mkldnn.is_available",
        "torch.backends.mkldnn.set_flags",
        "torch.backends.mps._init",
        "torch.backends.mps.is_available",
        "torch.backends.mps.is_built",
        "torch.backends.mps.is_macos13_or_newer",
        "torch.backends.openmp.is_available",
        "torch.backends.quantized._get_qengine_id",
        "torch.backends.quantized._get_qengine_str",
        "torch.block_diag",
        "torch.broadcast_tensors",
        "torch.cartesian_prod",
        "torch.cdist",
        "torch.chain_matmul",
        "torch.compile",
        "torch.compiled_with_cxx11_abi",
        "torch.cpu._is_cpu_support_avx2",
        "torch.cpu._is_cpu_support_avx512",
        "torch.cpu._is_cpu_support_avx512_vnni",
        "torch.cpu._is_cpu_support_amx_tile",
        "torch.cpu._init_amx",
        "torch.cpu.current_device",
        "torch.cpu.current_stream",
        "torch.cpu.device_count",
        "torch.cpu.is_available",
        "torch.cpu.set_device",
        "torch.cpu.stream",
        "torch.cpu.synchronize",
        "torch.cuda._check_capability",
        "torch.cuda._check_cubins",
        "torch.cuda._device_count_amdsmi",
        "torch.cuda._device_count_nvml",
        "torch.cuda._get_amdsmi_handler",
        "torch.cuda._get_amdsmi_device_index",
        "torch.cuda._get_device",
        "torch.cuda._get_generator",
        "torch.cuda._get_nvml_device_index",
        "torch.cuda._get_pynvml_handler",
        "torch.cuda._get_rng_state_offset",
        "torch.cuda._is_compiled",
        "torch.cuda._lazy_call",
        "torch.cuda._lazy_init",
        "torch.cuda._memory_viz._block_extra_legacy",
        "torch.cuda._memory_viz._block_extra",
        "torch.cuda._memory_viz._format_size",
        "torch.cuda._memory_viz._format_viz",
        "torch.cuda._memory_viz._frame_filter",
        "torch.cuda._memory_viz._frame_fmt",
        "torch.cuda._memory_viz._frames_fmt",
        "torch.cuda._memory_viz._profile_to_snapshot",
        "torch.cuda._memory_viz._report_free",
        "torch.cuda._memory_viz._write_blocks",
        "torch.cuda._memory_viz.calc_active",
        "torch.cuda._memory_viz.compare",
        "torch.cuda._memory_viz.format_flamegraph",
        "torch.cuda._memory_viz.memory",
        "torch.cuda._memory_viz.profile_plot",
        "torch.cuda._memory_viz.segment_plot",
        "torch.cuda._memory_viz.segments",
        "torch.cuda._memory_viz.segsum",
        "torch.cuda._memory_viz.trace_plot",
        "torch.cuda._memory_viz.trace",
        "torch.cuda._nvml_based_avail",
        "torch.cuda._parse_visible_devices",
        "torch.cuda._raw_device_count_amdsmi",
        "torch.cuda._raw_device_count_nvml",
        "torch.cuda._raw_device_uuid_amdsmi",
        "torch.cuda._raw_device_uuid_nvml",
        "torch.cuda._register_triton_kernels",
        "torch.cuda._set_rng_state_offset",
        "torch.cuda._set_stream_by_id",
        "torch.cuda._sleep",
        "torch.cuda._transform_uuid_to_ordinals",
        "torch.cuda._utils._get_device_index",
        "torch.cuda.amp.autocast_mode._cast",
        "torch.cuda.amp.autocast_mode.custom_bwd",
        "torch.cuda.amp.autocast_mode.custom_fwd",
        "torch.cuda.amp.common.amp_definitely_not_available",
        "torch.amp.grad_scaler._refresh_per_optimizer_state",
        "torch.cuda.can_device_access_peer",
        "torch.cuda.check_error",
        "torch.cuda.clock_rate",
        "torch.cuda.cudart",
        "torch.cuda.current_blas_handle",
        "torch.cuda.current_stream",
        "torch.cuda.default_stream",
        "torch.cuda.device_count",
        "torch.cuda.get_arch_list",
        "torch.cuda.get_device_capability",
        "torch.cuda.get_device_name",
        "torch.cuda.get_device_properties",
        "torch.cuda.get_gencode_flags",
        "torch.cuda.get_sync_debug_mode",
        "torch.cuda.graphs.graph_pool_handle",
        "torch.cuda.graphs.is_current_stream_capturing",
        "torch.cuda.graphs.make_graphed_callables",
        "torch.cuda.init",
        "torch.cuda.ipc_collect",
        "torch.cuda.is_available",
        "torch.cuda.is_bf16_supported",
        "torch.cuda.is_initialized",
        "torch.cuda.jiterator._create_jit_fn",
        "torch.cuda.jiterator._create_multi_output_jit_fn",
        "torch.cuda.memory_usage",
        "torch.cuda.memory._dump_snapshot",
        "torch.cuda.memory._free_mutex",
        "torch.cuda.memory._get_current_allocator",
        "torch.cuda.memory._host_allocator",
        "torch.cuda.memory._record_memory_history_impl",
        "torch.cuda.memory._record_memory_history_legacy",
        "torch.cuda.memory._record_memory_history",
        "torch.cuda.memory._save_memory_usage",
        "torch.cuda.memory._save_segment_usage",
        "torch.cuda.memory._set_allocator_settings",
        "torch.cuda.memory._snapshot",
        "torch.cuda.memory.caching_allocator_alloc",
        "torch.cuda.memory.caching_allocator_delete",
        "torch.cuda.memory.change_current_allocator",
        "torch.cuda.memory.empty_cache",
        "torch.cuda.memory.get_allocator_backend",
        "torch.cuda.memory.list_gpu_processes",
        "torch.cuda.memory.max_memory_allocated",
        "torch.cuda.memory.max_memory_cached",
        "torch.cuda.memory.max_memory_reserved",
        "torch.cuda.memory.mem_get_info",
        "torch.cuda.memory.memory_allocated",
        "torch.cuda.memory.memory_cached",
        "torch.cuda.memory.memory_reserved",
        "torch.cuda.memory.memory_snapshot",
        "torch.cuda.memory.memory_stats_as_nested_dict",
        "torch.cuda.memory.memory_stats",
        "torch.cuda.memory.memory_summary",
        "torch.cuda.memory.reset_accumulated_memory_stats",
        "torch.cuda.memory.reset_max_memory_allocated",
        "torch.cuda.memory.reset_max_memory_cached",
        "torch.cuda.memory.reset_peak_memory_stats",
        "torch.cuda.memory.set_per_process_memory_fraction",
        "torch.cuda.nccl._check_sequence_type",
        "torch.cuda.nccl.all_gather",
        "torch.cuda.nccl.all_reduce",
        "torch.cuda.nccl.broadcast",
        "torch.cuda.nccl.init_rank",
        "torch.cuda.nccl.is_available",
        "torch.cuda.nccl.reduce_scatter",
        "torch.cuda.nccl.reduce",
        "torch.cuda.nccl.unique_id",
        "torch.cuda.nccl.version",
        "torch.cuda.nvtx.mark",
        "torch.cuda.nvtx.range_end",
        "torch.cuda.nvtx.range_pop",
        "torch.cuda.nvtx.range_push",
        "torch.cuda.nvtx.range_start",
        "torch.cuda.nvtx.range",
        "torch.cuda.power_draw",
        "torch.cuda.profiler.init",
        "torch.cuda.profiler.profile",
        "torch.cuda.profiler.start",
        "torch.cuda.profiler.stop",
        "torch.cuda.random.get_rng_state_all",
        "torch.cuda.random.initial_seed",
        "torch.cuda.random.manual_seed_all",
        "torch.cuda.random.manual_seed",
        "torch.cuda.random.seed_all",
        "torch.cuda.random.seed",
        "torch.cuda.random.set_rng_state_all",
        "torch.cuda.set_stream",
        "torch.cuda.set_sync_debug_mode",
        "torch.cuda.stream",
        "torch.cuda.synchronize",
        "torch.cuda.temperature",
        "torch.cuda.utilization",
        "torch.einsum",
        "torch.functional._check_list_size",
        "torch.functional._consecutive_return_counts",
        "torch.functional._consecutive_return_inverse_false",
        "torch.functional._consecutive_return_inverse_true",
        "torch.functional._consecutive_return_inverse",
        "torch.functional._consecutive_return_output",
        "torch.functional._lu_impl",
        "torch.functional._lu_no_infos",
        "torch.functional._lu_with_infos",
        "torch.functional._meshgrid",
        "torch.functional._return_counts",
        "torch.functional._return_inverse_false",
        "torch.functional._return_inverse_true",
        "torch.functional._return_inverse",
        "torch.functional._return_output",
        "torch.functional._unique_consecutive_impl",
        "torch.functional._unique_impl",
        "torch.functional._unravel_index",
        "torch.functional.broadcast_shapes",
        "torch.functional.lu",
        "torch.functional.unique",
        "torch.functional.unravel_index",
        "torch.futures.collect_all",
        "torch.futures.wait_all",
        "torch.fx.experimental.const_fold.split_const_subgraphs",
        "torch.fx.experimental.proxy_tensor.make_fx",
        "torch.get_deterministic_debug_mode",
        "torch.get_float32_matmul_precision",
        "torch.is_deterministic_algorithms_warn_only_enabled",
        "torch.is_storage",
        "torch.is_tensor",
        "torch.is_warn_always_enabled",
        "torch.masked._ops._any",
        "torch.masked._ops._apply_docstring_templates",
        "torch.masked._ops._canonical_dim",
        "torch.masked._ops._combine_input_and_mask",
        "torch.masked._ops._generate_docstring",
        "torch.masked._ops._input_mask",
        "torch.masked._ops._output_mask",
        "torch.masked._ops._reduction_identity",
        "torch.masked._ops._sparse_coo_flatten_indices",
        "torch.masked._ops._sparse_coo_scatter_reduction_helper",
        "torch.masked._ops._sparse_coo_where",
        "torch.masked._ops._sparse_csr_segment_reduction_helper",
        "torch.masked._ops._sparse_csr_where",
        "torch.masked._ops._std_var",
        "torch.masked._ops._where",
        "torch.masked._ops.amax",
        "torch.masked._ops.amin",
        "torch.masked._ops.argmax",
        "torch.masked._ops.argmin",
        "torch.masked._ops.corresponding_real_dtype",
        "torch.masked._ops.cumprod",
        "torch.masked._ops.cumsum",
        "torch.masked._ops.log_softmax",
        "torch.masked._ops.logaddexp",
        "torch.masked._ops.logsumexp",
        "torch.masked._ops.mean",
        "torch.masked._ops.median",
        "torch.masked._ops.norm",
        "torch.masked._ops.normalize",
        "torch.masked._ops.prod",
        "torch.masked._ops.softmax",
        "torch.masked._ops.softmin",
        "torch.masked._ops.std",
        "torch.masked._ops.sum",
        "torch.masked._ops.var",
        "torch.meshgrid",
        "torch.mps._get_default_mps_generator",
        "torch.mps.current_allocated_memory",
        "torch.mps.driver_allocated_memory",
        "torch.mps.empty_cache",
        "torch.mps.get_rng_state",
        "torch.mps.manual_seed",
        "torch.mps.profiler.profile",
        "torch.mps.profiler.start",
        "torch.mps.profiler.stop",
        "torch.mps.seed",
        "torch.mps.set_per_process_memory_fraction",
        "torch.mps.set_rng_state",
        "torch.mps.synchronize",
        "torch.nested._internal.nested_tensor.buffer_from_jagged",
        "torch.nested._internal.nested_tensor.get_tensor_symint",
        "torch.nested._internal.nested_tensor.is_expandable_to",
        "torch.nested._internal.nested_tensor.jagged_from_list",
        "torch.nested._internal.nested_tensor.jagged_from_tensor_and_lengths",
        "torch.nested._internal.nested_tensor.nested_view_from_values_offsets",
        "torch.nested._internal.nested_tensor.nested_view_from_values_offsets_lengths",
        "torch.nested.as_nested_tensor",
        "torch.nested.narrow",
        "torch.nested.nested_tensor",
        "torch.nn._reduction.get_enum",
        "torch.nn._reduction.legacy_get_enum",
        "torch.nn._reduction.legacy_get_string",
        "torch.nn.factory_kwargs",
        "torch.nn.functional.adaptive_avg_pool2d",
        "torch.nn.functional.adaptive_avg_pool3d",
        "torch.nn.functional.adaptive_max_pool1d_with_indices",
        "torch.nn.functional.adaptive_max_pool1d",
        "torch.nn.functional.adaptive_max_pool2d_with_indices",
        "torch.nn.functional.adaptive_max_pool2d",
        "torch.nn.functional.adaptive_max_pool3d_with_indices",
        "torch.nn.functional.adaptive_max_pool3d",
        "torch.nn.functional.affine_grid",
        "torch.nn.functional.alpha_dropout",
        "torch.nn.functional.assert_int_or_pair",
        "torch.nn.functional.batch_norm",
        "torch.nn.functional.binary_cross_entropy_with_logits",
        "torch.nn.functional.binary_cross_entropy",
        "torch.nn.functional.celu",
        "torch.nn.functional.cosine_embedding_loss",
        "torch.nn.functional.cross_entropy",
        "torch.nn.functional.ctc_loss",
        "torch.nn.functional.dropout",
        "torch.nn.functional.dropout1d",
        "torch.nn.functional.dropout2d",
        "torch.nn.functional.dropout3d",
        "torch.nn.functional.elu",
        "torch.nn.functional.embedding_bag",
        "torch.nn.functional.embedding",
        "torch.nn.functional.feature_alpha_dropout",
        "torch.nn.functional.fold",
        "torch.nn.functional.fractional_max_pool2d_with_indices",
        "torch.nn.functional.fractional_max_pool2d",
        "torch.nn.functional.fractional_max_pool3d_with_indices",
        "torch.nn.functional.fractional_max_pool3d",
        "torch.nn.functional.gaussian_nll_loss",
        "torch.nn.functional.glu",
        "torch.nn.functional.grid_sample",
        "torch.nn.functional.group_norm",
        "torch.nn.functional.gumbel_softmax",
        "torch.nn.functional.hardsigmoid",
        "torch.nn.functional.hardswish",
        "torch.nn.functional.hardtanh",
        "torch.nn.functional.hinge_embedding_loss",
        "torch.nn.functional.huber_loss",
        "torch.nn.functional.instance_norm",
        "torch.nn.functional.interpolate",
        "torch.nn.functional.kl_div",
        "torch.nn.functional.l1_loss",
        "torch.nn.functional.layer_norm",
        "torch.nn.functional.leaky_relu",
        "torch.nn.functional.local_response_norm",
        "torch.nn.functional.log_softmax",
        "torch.nn.functional.lp_pool1d",
        "torch.nn.functional.lp_pool2d",
        "torch.nn.functional.margin_ranking_loss",
        "torch.nn.functional.max_pool1d_with_indices",
        "torch.nn.functional.max_pool1d",
        "torch.nn.functional.max_pool2d_with_indices",
        "torch.nn.functional.max_pool2d",
        "torch.nn.functional.max_pool3d_with_indices",
        "torch.nn.functional.max_pool3d",
        "torch.nn.functional.max_unpool1d",
        "torch.nn.functional.max_unpool2d",
        "torch.nn.functional.max_unpool3d",
        "torch.nn.functional.mish",
        "torch.nn.functional.mse_loss",
        "torch.nn.functional.multi_head_attention_forward",
        "torch.nn.functional.multi_margin_loss",
        "torch.nn.functional.multilabel_margin_loss",
        "torch.nn.functional.multilabel_soft_margin_loss",
        "torch.nn.functional.nll_loss",
        "torch.nn.functional.normalize",
        "torch.nn.functional.poisson_nll_loss",
        "torch.nn.functional.relu",
        "torch.nn.functional.relu6",
        "torch.nn.functional.rrelu",
        "torch.nn.functional.selu",
        "torch.nn.functional.sigmoid",
        "torch.nn.functional.silu",
        "torch.nn.functional.smooth_l1_loss",
        "torch.nn.functional.soft_margin_loss",
        "torch.nn.functional.softmax",
        "torch.nn.functional.softmin",
        "torch.nn.functional.softsign",
        "torch.nn.functional.tanh",
        "torch.nn.functional.tanhshrink",
        "torch.nn.functional.triplet_margin_loss",
        "torch.nn.functional.unfold",
        "torch.nn.functional.upsample_bilinear",
        "torch.nn.functional.upsample_nearest",
        "torch.nn.functional.upsample",
        "torch.nn.grad._pair",
        "torch.nn.grad._single",
        "torch.nn.grad._triple",
        "torch.nn.grad.conv1d_input",
        "torch.nn.grad.conv1d_weight",
        "torch.nn.grad.conv2d_input",
        "torch.nn.grad.conv2d_weight",
        "torch.nn.grad.conv3d_input",
        "torch.nn.grad.conv3d_weight",
        "torch.nn.modules.activation._is_make_fx_tracing",
        "torch.nn.modules.utils._list_with_default",
        "torch.nn.modules.utils._ntuple",
        "torch.nn.modules.utils._quadruple",
        "torch.nn.modules.utils._reverse_repeat_tuple",
        "torch.nn.modules.utils.consume_prefix_in_state_dict_if_present",
        "torch.nn.parameter.is_lazy",
        "torch.norm",
        "torch.quantization.default_eval_fn",
        "torch.random._seed_custom_device",
        "torch.random.fork_rng",
        "torch.random.initial_seed",
        "torch.random.seed",
        "torch.return_types.pytree_register_structseq",
        "torch.set_default_device",
        "torch.set_default_dtype",
        "torch.set_default_tensor_type",
        "torch.set_deterministic_debug_mode",
        "torch.set_float32_matmul_precision",
        "torch.set_warn_always",
        "torch.signal.windows.windows._add_docstr",
        "torch.signal.windows.windows._window_function_checks",
        "torch.signal.windows.windows.bartlett",
        "torch.signal.windows.windows.blackman",
        "torch.signal.windows.windows.cosine",
        "torch.signal.windows.windows.exponential",
        "torch.signal.windows.windows.gaussian",
        "torch.signal.windows.windows.general_cosine",
        "torch.signal.windows.windows.general_hamming",
        "torch.signal.windows.windows.hamming",
        "torch.signal.windows.windows.hann",
        "torch.signal.windows.windows.kaiser",
        "torch.signal.windows.windows.merge_dicts",
        "torch.signal.windows.windows.nuttall",
        "torch.signal.windows.windows.parse_kwargs",
        "torch.sparse.semi_structured.to_sparse_semi_structured",
        "torch.sparse.sum",
        "torch.split",
        "torch.stft",
        "torch.sym_float",
        "torch.sym_int",
        "torch.sym_ite",
        "torch.sym_max",
        "torch.sym_min",
        "torch.sym_not",
        "torch.tensordot",
        "torch.typename",
        "torch.unique_consecutive",
        "torch.use_deterministic_algorithms",
    ],
    TorchInGraphFunctionVariable,
)


torch_name_rule_map = [
    manual_torch_name_rule_map,
    torch_c_binding_in_graph_functions,
    torch_non_c_binding_in_graph_functions,
]


"""
Generate the torch object - Dynamo tracing rule (the wrapping variable) map.
"""


@functools.lru_cache(None)
def get_torch_obj_rule_map():
    d: Dict[Any, VariableTracker] = {}
    for m in torch_name_rule_map:
        for k, v in m.items():  # type: ignore[attr-defined]
            if ".py#" not in k:
                obj = load_object(k)
            else:
                obj = _module_dir(torch) + k[len("torch/") :]
            if obj is not None:
                if obj in d and d[obj] != v:
                    raise AssertionError(
                        f"Duplicate torch object {obj} with different rules: {v}, {d[obj]}"
                    )
                else:
                    d[obj] = v
    return d


def _load_obj_from_str(fully_qualified_name):
    module, obj_name = fully_qualified_name.rsplit(".", maxsplit=1)
    return getattr(importlib.import_module(module), obj_name)


"""
Load string represented torch objects.
"""


def load_object(name):
    try:
        x = name.split("#")
        if len(x) == 2:
            obj = _load_obj_from_str(x[0])
            val = getattr(obj, x[1])
        else:
            assert len(x) == 1, f"Invalid obj name {name}"
            val = _load_obj_from_str(x[0])
        val = unwrap_if_wrapper(val)
    except (AttributeError, ImportError):
        val = None
    return val


"""
Get all torch.Tensor methods which are allowed to be in graph functions.
"""


@functools.lru_cache(None)
def get_tensor_method():
    s = set()
    for name in dir(torch.Tensor):
        method = getattr(torch.Tensor, name)
        if isinstance(
            method, (types.MethodDescriptorType, types.WrapperDescriptorType)
        ):
            s.add(method)
    return frozenset(s)


"""
Return if a torch object is ATen op or torch.Tensor method.
"""


def is_aten_op_or_tensor_method(obj):
    return obj in get_tensor_method() or isinstance(
        obj,
        (torch._ops.OpOverloadPacket, torch._ops.OpOverload),
    )


class FunctionIdSet:
    """
    Track a set of `id()`s of objects which are either allowed or not
    allowed to go into the generated FX graph.  Use to test for torch.*,
    numpy.*, builtins.*, etc.

    Support user modification to permit customization of what can be
    added to the graph and what will cause a graph break.
    """

    function_ids: Optional[Set[int]] = None
    function_names: Optional[Dict[int, str]] = None

    def __init__(self, lazy_initializer: Callable[[], Union[Dict[int, str], Set[int]]]):
        self.lazy_initializer = lazy_initializer

    def __call__(self):
        if self.function_ids is None:
            value = self.lazy_initializer()
            if isinstance(value, dict):
                self.function_ids = set(value.keys())
                self.function_names = value
            else:
                assert isinstance(value, set)
                self.function_ids = value
        return self.function_ids

    def get_name(self, idx: int, default: str):
        self()  # lazy init
        assert self.function_names is not None
        return self.function_names.get(idx, default)

    def add(self, idx: int):
        function_ids = self()  # lazy init
        function_ids.add(idx)

    def remove(self, idx: int):
        function_ids = self()
        if idx in function_ids:
            function_ids.remove(idx)

    def __contains__(self, idx: int):
        return idx in self()


@FunctionIdSet
def _allowed_callable_ids() -> Dict[int, str]:
    rv: Dict[int, str] = {}
    return rv


@FunctionIdSet
def _disallowed_callable_ids() -> Dict[int, str]:
    rv: Dict[int, str] = {}
    return rv


@FunctionIdSet
def _builtin_function_ids() -> Dict[int, str]:
    rv = {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and callable(v)
    }
    rv.update(
        {
            id(v): f"operator.{k}"
            for k, v in operator.__dict__.items()
            if not k.startswith("_") and callable(v)
        }
    )
    rv.update(
        {id(v): f"functools.{v.__name__}" for v in (itertools.chain, itertools.islice)}
    )
    rv.update(
        {
            id(cast): "typing.cast",
            id(functools.reduce): "functools.reduce",
            id(copy.deepcopy): "copy.deepcopy",
        }
    )
    return rv


@FunctionIdSet
def _numpy_function_ids() -> Dict[int, str]:
    rv = {}
    for mod in NP_SUPPORTED_MODULES:
        rv.update(
            {
                id(v): f"{mod.__name__}.{k}"
                for k, v in mod.__dict__.items()
                if callable(v)
                and (getattr(v, "__module__", None) or mod.__name__) == mod.__name__
            }
        )
    return rv


@FunctionIdSet
def _builtin_constant_ids() -> Dict[int, str]:
    """
    Collects constant builtins by eliminating callable items.
    """
    rv = {
        id(v): f"builtins.{k}"
        for k, v in builtins.__dict__.items()
        if not k.startswith("_") and not callable(v)
    }
    return rv


_lazy_module_init: Dict[str, List[Callable[[], None]]] = defaultdict(list)


def add_module_init_func(name: str, init_func: Callable[[], None]) -> None:
    """Register a module without eagerly importing it"""
    # If the module is already imported, eagerly run init
    assert "." not in name, f"Expected a root module name, but got {name}"
    assert name not in _lazy_module_init
    _lazy_module_init[name].append(init_func)


def _maybe_init_lazy_module(obj: object) -> None:
    module = getattr(obj, "__module__", None)
    if module is None:
        return

    base_module = module.split(".")[0]
    init_funcs = _lazy_module_init.pop(base_module, None)
    if init_funcs is not None:
        for fn in init_funcs:
            fn()


def is_callable_allowed(obj) -> bool:
    _maybe_init_lazy_module(obj)
    return id(obj) in _allowed_callable_ids


def is_callable_disallowed(obj) -> bool:
    _maybe_init_lazy_module(obj)
    return id(obj) in _disallowed_callable_ids


def is_forbidden(obj) -> bool:
    _maybe_init_lazy_module(obj)
    return inspect.getattr_static(obj, "_dynamo_forbidden", False)


def is_builtin_callable(obj) -> bool:
    return id(obj) in _builtin_function_ids


def is_builtin_constant(obj) -> bool:
    return id(obj) in _builtin_constant_ids


def is_numpy(obj) -> bool:
    if np is None:
        return False
    return isinstance(obj, (np.ndarray, np.generic)) or id(obj) in _numpy_function_ids


def is_numpy_dtype(obj) -> bool:
    if np is None:
        return False
    return isinstance(obj, np.dtype)


def is_numpy_type_info(obj) -> bool:
    if np is None:
        return False
    return isinstance(obj, (np.finfo, np.iinfo))


BUILTIN_SKIPLIST = (
    abc,
    collections,
    contextlib,
    copy,
    copyreg,
    dataclasses,
    enum,
    functools,
    importlib,
    inspect,
    linecache,
    logging,
    multiprocessing,
    operator,
    os,
    posixpath,
    random,
    re,
    selectors,
    signal,
    tempfile,
    threading,
    tokenize,
    torch,  # torch/* is skipped by default unless specified in FUNC_INLINELIST or MOD_INLINELIST
    traceback,
    types,
    typing,
    unittest,
    weakref,
    _collections_abc,
    _weakrefset,
)

# third party libraries skiplist is defined by str, because users may not use these libraries.
# we should use lazy import & skip in the future.
THIRDPARTY_SKIPLIST = (
    "fx2trt_oss",
    "hypothesis",
    "networkx",
    "numpy",
    "omegaconf",
    "onnx",
    "onnxruntime",
    "onnx_tf",
    "pandas",
    "sklearn",
    "tabulate",
    "tensorflow",
    "tensorrt",
    "torch2trt",
    "tqdm",
    "tree",
    "tvm",
    "xarray",
)


def _strip_init_py(s):
    # TODO: Once we require py3.9 use removesuffix instead.
    suffix = "__init__.py"
    if s.endswith(suffix):
        return s[: -len(suffix)]
    else:
        return s


def _module_dir(m: types.ModuleType):
    # Protect against a module not exporting __file__ - this can happen for
    # frozen modules, for example.
    file = getattr(m, "__file__", None)
    return file and _strip_init_py(file)


# These are legacy workarounds, don't add new modules to this list.
# Please use the MOD_INLINELIST instead to force inline functions under particular modules.
LEGACY_MOD_INLINELIST = {
    "torch._dynamo.external_utils",
    "torch._export.db.examples",
    "torch._export.wrappers",
    "torch._functorch.apis",
    "torch._functorch.deprecated",
    "torch._higher_order_ops.cond",
    "torch.ao.quantization.pt2e.export_utils",
    "torch.ao.quantization.pt2e.qat_utils",
    "torch.ao.quantization.pt2e.representation.rewrite",
    "torch.ao.quantization.pt2e.utils",
    "torch.ao.quantization.quantizer.xnnpack_quantizer",
    "torch.optim",
}

if torch.distributed.is_available():
    LEGACY_MOD_INLINELIST |= {
        "torch.distributed._tensor.api",
        "torch.distributed._tensor.device_mesh",
        "torch.distributed.device_mesh",
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
        "torch.distributed.tensor.parallel._data_parallel_utils",
        "torch.distributed.tensor.parallel._utils",
        "torch.distributed.tensor.parallel.style",
        # we have to add replicate to LEGACY_MOD_INLINELIST to ensure
        # the forward_hook won't be ignored.
        "torch.distributed._composable.replicate",
    }


# Force inline functions under these modules, even they are in *_SKIPLIST.
# We are using python module name instead of file or directory object to avoid circular dependency.
# Please keep this sorted alphabetically.
MOD_INLINELIST = {
    "torch.utils._python_dispatch",
    "torch._refs",
    "torch._prims",
    "torch._decomp",
    "torch._dynamo._trace_wrapped_higher_order_op",
    "torch._dynamo.comptime",
    "torch._dynamo.polyfill",
    "torch._functorch.vmap",
    "torch._functorch.autograd_function",
    "torch._library.custom_ops",
    "torch._functorch.eager_transforms",
    "torch._inductor.test_operators",
    "torch.amp.autocast_mode",
    "torch.ao.nn",
    "torch.autograd.function",
    "torch.backends.cuda",
    "torch.cuda.amp.autocast_mode",
    "torch.distributions",
    "torch.fx._pytree",
    "torch.fx.passes.shape_prop",
    "torch.nn",
    "torch.overrides",
    "torch.random",
    "torch.sparse",
    "torch.testing",
    "torch.testing._internal.hypothesis_utils",
    "torch.utils._content_store",
    "torch.utils._contextlib",
    "torch.utils._foreach_utils",
    "torch.utils._pytree",
    "torch.utils.hooks",
    "torch._tensor",
    "torch._higher_order_ops.strict_mode",
    "torch._higher_order_ops.while_loop",
    "torch._higher_order_ops.associative_scan",
    "torch._functorch.functional_call",
}


if torch.distributed.is_available():
    MOD_INLINELIST.add("torch.distributed")
    MOD_INLINELIST.add("torch.distributed._functional_collectives")
    MOD_INLINELIST.add("torch.distributed._composable.replicate")


@functools.lru_cache(None)
def get_legacy_mod_inlinelist():
    inlinelist = {
        _module_dir(torch) + m[len("torch.") :].replace(".", "/")
        for m in LEGACY_MOD_INLINELIST
    }
    return inlinelist


@functools.lru_cache(None)
def get_mod_inlinelist():
    inlinelist = {
        _module_dir(torch) + m[len("torch.") :].replace(".", "/")
        for m in MOD_INLINELIST
    }
    return inlinelist


# skip some standard python builtin libs
SKIP_DIRS = [
    "<frozen importlib",
    "<__array_function__ internals>",
    _config_module.__file__,
    "triton/backends",
]
SKIP_DIRS.extend(filter(None, (_module_dir(m) for m in BUILTIN_SKIPLIST)))

SKIP_DIRS_RE = re.compile(r"match nothing^")

is_fbcode = importlib.import_module("torch._inductor.config").is_fbcode()
# Skip fbcode paths(including torch.package paths) containing
# one of the following strings.
FBCODE_SKIP_DIRS = set()
# Remove this after fbcode is fully migrated to tracing through torchrec.
if torch._dynamo.config.skip_torchrec:
    FBCODE_SKIP_DIRS.add("torchrec/distributed")
    FBCODE_SKIP_DIRS.add("torchrec/fb/distributed")
    FBCODE_SKIP_DIRS.add("caffe2/torch/fb/sparsenn/pooled_embeddings_modules.py")
FBCODE_SKIP_DIRS_RE = re.compile(f".*({'|'.join(map(re.escape, FBCODE_SKIP_DIRS))})")


# TODO(yanboliang, anijain2305) - There are a few concerns that we should
# resolve
# 1) Audit if torchrec/distributed is even required in FBCODE_SKIPS_DIR
# 2) To inline just one file but skip others in a directory, we could use
# manual_torch_name_rule_map but this one is hard because FBCODE can add unusual
# names like torch_package.
# So, this is a stop gap solution till then.
FBCODE_INLINE_FILES_IN_SKIPPED_DIRS = {
    "torchrec/distributed/types.py",
}
FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE = re.compile(
    f".*({'|'.join(map(re.escape, FBCODE_INLINE_FILES_IN_SKIPPED_DIRS))})"
)

# torch.optim is a special case,
# we usually want to inline it, but the directory
# structure does not match the module structure
# and we want to skip the functions in optim/lr_scheduler.py
# this has precedence over all other rules in check_file
FORCE_SKIP_FILES = {f"{_module_dir(torch)}optim/lr_scheduler.py"}


def _recompile_re():
    global SKIP_DIRS_RE
    SKIP_DIRS_RE = re.compile(rf"^[^\s<]*({'|'.join(map(re.escape, SKIP_DIRS))})")


def add(import_name: str):
    if isinstance(import_name, types.ModuleType):
        return add(import_name.__name__)
    assert isinstance(import_name, str)
    from importlib.util import find_spec

    module_spec = find_spec(import_name)
    if not module_spec:
        return
    origin = module_spec.origin
    if origin is None:
        return
    SKIP_DIRS.append(_strip_init_py(origin))
    _recompile_re()


@dataclasses.dataclass
class SkipResult:
    skipped: bool
    reason: Optional[str]


def check_file(filename, is_inlined_call=False):
    """Should skip this file?"""
    if filename is None:
        return SkipResult(True, "filename is None")
    if filename in FORCE_SKIP_FILES:
        return SkipResult(True, "FORCE_SKIP_FILES")
    if any(filename.startswith(d) for d in get_legacy_mod_inlinelist()):
        return SkipResult(
            False,
            "LEGACY_MOD_INLINELIST",
        )
    if is_inlined_call and is_torch_inline_allowed(filename):
        return SkipResult(
            False,
            "MOD_INLINELIST",
        )
    if (
        is_fbcode
        and bool(FBCODE_SKIP_DIRS_RE.match(filename))
        and not bool(FBCODE_INLINE_FILES_IN_SKIPPED_DIRS_RE.match(filename))
    ):
        return SkipResult(
            True,
            "FBCODE_SKIP_DIRS",
        )
    if bool(SKIP_DIRS_RE.match(filename)):
        return SkipResult(True, "SKIP_DIRS")
    else:
        return SkipResult(False, "inlined by default")


@dataclasses.dataclass
class FunctionInfo:
    py_obj: Optional[object]
    name: Optional[str]
    filename: str
    code: Optional[types.CodeType]


"""
This is the main entry point to determine whether an object (function) should be inlined or skipped.
Let's illustrate the logic with an example:
    @torch.compile
    def f1(x, y):
        ......
        f2(x, y)
        ......

    def f2(x, y):
        ......
        f3(x, y)
        ......

    def f3(x, y):
        ......

There are mainly three call sites of check/check_verbose:
* The compile region entrance (like function f1), the correspoinding code is located at eval_frame.py.
* When tracing the recursively called functions (like function f2 and f3).
    * Dynamo decides inline/skip everytime it encounters a new recursively function call, and the call site
      is in InliningInstructionTranslator.check_inlineable of symbolic_convert.py.
    * If f2 is skipped by Dynamo, when evaluating the frame of f3, Dynamo need the inline/skip check again
      and the call site is in catch_errors_wrapper.catch_errors of convert_frame.py.
* For global variables and function arguments, Dynamo needs to decide if they are wrapped as SkipFunctionVariable in builder.py.

`is_inlined_call` is used to indicate if the current function call is inlined (f2 is inlined call if it passes check)
or not (f3 is not inlined call if f2 is skipped). Inside of the `check_verbose` function, there are more rules
to be checked if this `is_inlined_call`.
The reason to have this flag is that if the upper level function call (e.g, f2) is skipped,
we don't want to inline the lower level function call (e.g, f3) by default.
"""


def check_verbose(obj, is_inlined_call=False):
    if isinstance(
        obj, (UserFunctionVariable, UserMethodVariable, NestedUserFunctionVariable)
    ):
        try:
            py_obj = obj.get_function()
        except NotImplementedError:
            py_obj = None
        fi = FunctionInfo(py_obj, obj.get_name(), obj.get_filename(), obj.get_code())
    elif isinstance(obj, types.CodeType):
        fi = FunctionInfo(None, obj.co_name, obj.co_filename, obj)
    elif isinstance(obj, (types.FunctionType, types.MethodType)):
        fi = FunctionInfo(
            obj, obj.__name__, getfile(obj), obj.__code__  # type: ignore[union-attr] # FIXME Add MethodType.__code__ to typeshed
        )
    else:
        fi = FunctionInfo(obj, None, getfile(obj), None)

    # Consulte the central trace rules defined in torch._dynamo.trace_rules.
    reasons: Set[str] = set()
    rule = torch._dynamo.trace_rules.lookup_inner(
        fi.py_obj, fi.name, fi.filename, is_inlined_call, reasons
    )
    if issubclass(rule, UserFunctionVariable):
        return SkipResult(
            False,
            f"inlined according trace_rules.lookup {reasons.pop()}",
        )
    else:
        assert rule == SkipFunctionVariable, rule
        return SkipResult(
            True,
            f"skipped according trace_rules.lookup {reasons.pop()}",
        )


def check(obj, is_inlined_call=False):
    return check_verbose(obj, is_inlined_call).skipped


# skip common third party libs
for _name in THIRDPARTY_SKIPLIST:
    add(_name)

_recompile_re()


def is_torch_inline_allowed(filename):
    return any(filename.startswith(d) for d in get_mod_inlinelist())


@functools.lru_cache(None)
def dynamo_dir():
    import torch._dynamo

    return _module_dir(torch._dynamo)


def is_torch(filename):
    if filename.startswith(dynamo_dir()):
        return False
    return filename.startswith(_module_dir(torch))


"""
Main entry point for looking up the trace rule (the Dynamo variable) for a given callable object.
"""


def lookup_callable(obj):
    if not hashable(obj):
        return None
    # Custom allow/disallow in graph takes precedence over the general lookup.
    if is_callable_disallowed(obj):
        return SkipFunctionVariable
    if is_callable_allowed(obj):
        return TorchInGraphFunctionVariable
    if is_builtin_callable(obj):
        return BuiltinVariable


"""
Main entry point for looking up the trace rule (the Dynamo variable) for a given function object.
E.g, the lookup result of `torch.sin` is `TorchInGraphFunctionVariable`.
"""


def lookup(obj):
    return lookup_inner(obj)


def lookup_inner(
    obj,
    name=None,
    filename=None,
    is_direct_call=True,
    reasons: Union[None, Set[str]] = None,
):
    # Step 1: lookup obj's tracing rule in `torch_name_rule_map`.
    # The rules defined in `torch_name_rule_map` mainly includes two parts:
    # - Manually defined rules for any functions.
    # - The list of torch in graph functions.
    try:
        can_hash = hashable(obj)
    except Exception:
        can_hash = False
    if not can_hash:
        if reasons is not None:
            reasons.add("obj is not hashable")
        return None
    if obj is not None:
        if is_aten_op_or_tensor_method(obj):
            return TorchInGraphFunctionVariable
        rule = get_torch_obj_rule_map().get(obj, None)
        if rule is not None:
            if reasons is not None:
                reasons.add("get_torch_obj_rule_map")
            return rule
    elif name is not None and filename is not None and not is_direct_call:
        if name.startswith(TORCH_DYNAMO_RESUME_IN_PREFIX):
            rule = get_torch_obj_rule_map().get(
                filename + "#" + TORCH_DYNAMO_RESUME_IN_PREFIX, None
            )
        else:
            rule = get_torch_obj_rule_map().get(filename + "#" + name, None)
        if rule is not None:
            if reasons is not None:
                reasons.add("get_torch_obj_rule_map")
            return rule

    # Step 2: lookup obj's tracing rule by function name.
    if is_direct_call:
        if name == "patched_init":
            if reasons is not None:
                reasons.add("func name is patched_init")
            return SkipFunctionVariable
        elif name == "__torch_function__":
            if reasons is not None:
                reasons.add("func name is __torch_function__")
            return UserFunctionVariable

    if not is_direct_call:
        if name == "__getattr__":
            # is_direct_call = False indicates that this is the top-level frame
            # being traced (i.e., it is not inlined and not called from
            # InliningInstructionTranslator).  Tracing __getattr__ at the top
            # level is unlikely because we inline it for
            # UserDefinedObjectVariable. This scenario occurs only for
            # UnspecializedNNModuleVariable, where Dynamo directly calls
            # __getattr__ during trace time, generating LOAD_ATTR bytecode
            # without going through the underlying __getattr__ data structures.
            # When this optimized bytecode is executed, Dynamo is triggered
            # again on the __getattr__ call. Therefore, we skip Dynamo tracing
            # in this case.
            if reasons is not None:
                reasons.add(
                    "Tracing __getattr__ as the top level frame, unsuitable for tracing."
                )
            return SkipFunctionVariable

    # Step 3: lookup obj's tracing rule by filename.
    if filename is None:
        filename = getfile(obj)

    skip_result = check_file(filename, is_direct_call)
    if reasons is not None:
        reasons.add(skip_result.reason)
    if skip_result.skipped:
        return SkipFunctionVariable
    else:
        return UserFunctionVariable


def clear_lru_cache():
    torch._dynamo.trace_rules.get_torch_obj_rule_map.cache_clear()
    torch._dynamo.trace_rules.get_tensor_method.cache_clear()
    torch._dynamo.trace_rules.get_legacy_mod_inlinelist.cache_clear()
    torch._dynamo.trace_rules.get_mod_inlinelist.cache_clear()
    torch._dynamo.trace_rules.dynamo_dir.cache_clear()
