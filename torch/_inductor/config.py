import os
import sys
from collections.abc import Callable
from typing import Any, Literal, Optional, TYPE_CHECKING, Union

import torch
import torch._inductor.custom_graph_pass
from torch._environment import is_fbcode
from torch.utils._config_module import Config, get_tristate_env, install_config_module


if TYPE_CHECKING:
    from torch._inductor.choices import InductorChoices

inplace_padding = os.environ.get("TORCHINDUCTOR_INPLACE_PADDING", "1") == "1"
can_inplace_pad_graph_input = False  # ease testing


def fx_graph_remote_cache_default() -> Optional[bool]:
    return get_tristate_env("TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE")


def vec_isa_ok_default() -> Optional[bool]:
    if os.environ.get("TORCHINDUCTOR_VEC_ISA_OK") == "1":
        return True
    if os.environ.get("TORCHINDUCTOR_VEC_ISA_OK") == "0":
        return False
    return None


def autotune_remote_cache_default() -> Optional[bool]:
    return get_tristate_env("TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE")


def bundled_autotune_remote_cache_default() -> Optional[bool]:
    return get_tristate_env("TORCHINDUCTOR_BUNDLED_AUTOTUNE_REMOTE_CACHE")


def bundle_triton_into_fx_graph_cache_default() -> Optional[bool]:
    return get_tristate_env(
        "TORCHINDUCTOR_BUNDLE_TRITON_INTO_FX_GRAPH_CACHE",
        True if not is_fbcode() else None,
    )


def static_cuda_launcher_default() -> bool:
    STATIC_CUDA_LAUNCHER_VERSION = 2

    if "TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER" in os.environ:
        return os.environ.get("TORCHINDUCTOR_USE_STATIC_CUDA_LAUNCHER") == "1"
    elif is_fbcode():
        version = torch._utils_internal.justknobs_getval_int(
            "pytorch/inductor:static_cuda_launcher_version"
        )
        return version <= STATIC_CUDA_LAUNCHER_VERSION
    else:
        # Default true in OSS
        return True


def prologue_fusion_enabled() -> bool:
    ENABLE_PROLOGUE_FUSION_VERSION = 0

    if "TORCHINDUCTOR_PROLOGUE_FUSION" in os.environ:
        return os.environ.get("TORCHINDUCTOR_PROLOGUE_FUSION") == "1"
    elif is_fbcode():
        jk_name = "pytorch/inductor:prologue_fusion_version"
        version = torch._utils_internal.justknobs_getval_int(jk_name)
        return version <= ENABLE_PROLOGUE_FUSION_VERSION
    else:
        return True


# Enable auto_functionalized_v2 (enabled by default)
enable_auto_functionalized_v2 = (
    os.environ.get("TORCHDYNAMO_AUTO_FUNCTIONALIZED_V2", "1") == "1"
)

# add some debug printouts
debug = False

# Whether to disable a progress bar for autotuning
disable_progress = True

# Whether to enable printing the source code for each future
verbose_progress = False

# Configurable compile worker logging path for subproc_pool
worker_log_path = (
    "/logs/dedicated_log_torch_compile_worker_rank" if is_fbcode() else None
)

# precompilation timeout
precompilation_timeout_seconds: int = 60 * 60

# use fx aot graph codegen cache
fx_graph_cache: bool = Config(
    justknob="pytorch/remote_cache:enable_local_fx_graph_cache",
    env_name_default="TORCHINDUCTOR_FX_GRAPH_CACHE_DEFAULT",
    env_name_force="TORCHINDUCTOR_FX_GRAPH_CACHE",
    default=True,
)

remote_gemm_autotune_cache: bool = False

# use remote fx aot graph codegen cache
# False: Disables the cache
# True: Enables the cache
# None: Not set -- Off for OSS, JustKnobs based for internal
fx_graph_remote_cache: Optional[bool] = fx_graph_remote_cache_default()

# should we bundle triton caching into fx graph cache
bundle_triton_into_fx_graph_cache: Optional[bool] = (
    bundle_triton_into_fx_graph_cache_default()
)

non_blocking_remote_cache_write: bool = Config(
    justknob="pytorch/remote_cache:enable_non_blocking_remote_cache_write_v2",
    env_name_force="TORCHINDUCTOR_NON_BLOCKING_REMOTE_CACHE_WRITE",
    default=True,
)

# Enable autotune local cache.
#
# See bundled_autotune_remote_cache for the effect this flag has on the bundled
# remote cache.
autotune_local_cache: bool = True

# Enable autotune remote cache.
#
# Enables/disables the autotune remote cache regardless of the state of
# autotune_local_cache. If both local and remote are enabled then on write both
# are written and on read local is checked first and only on a cache miss is
# remote read.
#
# False: Disables the cache
# True: Enables the cache
# None: Not set -- Off for OSS, JustKnobs based for internal
autotune_remote_cache: Optional[bool] = autotune_remote_cache_default()

# Enable bundled autotune cache.
#
# Enables/disables the bundled autotune cache regardless of the state of
# autotune_remote_cache. However it does depend on the local cache for local
# state management - as a result if the local cache is disabled this will also
# disable the bundled autotune cache.
#
# False: Disables the cache
# True: Enables the cache (requires autotune_local_cache)
# None: Not set -- Off for OSS, JustKnobs based for internal
bundled_autotune_remote_cache: Optional[bool] = bundled_autotune_remote_cache_default()

# See torch.compiler.config.force_disable_caches
force_disable_caches: bool = Config(alias="torch.compiler.config.force_disable_caches")

# Unsafe way to skip dynamic shape guards to get faster cache load
unsafe_skip_cache_dynamic_shape_guards: bool = False

# Unsafe way to mark non torch functions as safe to cache
# dictionary is from function name -> cache key
# Any function name in the dictionary will be allowed to be cacheable
# by AOTAutogradCache and FxGraphCache.
# changing the cache key value will change the resulting
# FXGraphCache key.
# Example usage:
# torch._inductor.config.unsafe_marked_cacheable_functions = {
# 'torch.ops.my_function' : torch.__version__
# }
# The above example causes the custom op torch.ops.my_function to be cacheable,
# and for cache keys to be keyed by the current torch version
unsafe_marked_cacheable_functions: dict[str, str] = {}

# sleep in inductor for testing
sleep_sec_TESTING_ONLY: Optional[int] = None

# The default layout constraint for user-defined triton kernels.
# See "The default layout constraint for custom operators" for options.
triton_kernel_default_layout_constraint: Literal[
    "needs_fixed_stride_order", "flexible_layout"
] = "needs_fixed_stride_order"

# use cpp wrapper instead of python wrapper
# incompatible with disable_cpp_codegen
cpp_wrapper: bool = os.environ.get("TORCHINDUCTOR_CPP_WRAPPER", "0") == "1"

# controls whether to compile entry and kernel separately for cpp_wrapper mode.
# turn on this option to compile entry and kernel separately and minimize compile time of the entry part.
# see https://github.com/pytorch/pytorch/pull/148773
# Note: compiling entry and kernel separately may have a non-negligible impact on the performance.
# see https://github.com/pytorch/pytorch/issues/156037
cpp_wrapper_build_separate: bool = (
    os.environ.get("TORCHINDUCTOR_CPP_WRAPPER_BUILD_SEPARATE", "0") == "1"
)

fx_wrapper: bool = os.environ.get("TORCHINDUCTOR_FX_WRAPPER", "0") == "1"

# Controls automatic precompiling of common include files for codecache.CppCodeCache
# (i.e. for cpp_wrapper mode and for cpp kernels on CPU).  AOTI header precompiling is
# controlled by a separate flag.
cpp_cache_precompile_headers: bool = not is_fbcode()

online_softmax = os.environ.get("TORCHINDUCTOR_ONLINE_SOFTMAX", "1") == "1"

# dead code elimination
dce = False

# assume weight tensors are fixed size
static_weight_shapes = True

# put correctness assertions in generated code
size_asserts = os.environ.get("TORCHINDUCTOR_SIZE_ASSERTS", "1") == "1"
nan_asserts = os.environ.get("TORCHINDUCTOR_NAN_ASSERTS") == "1"
runtime_triton_nan_asserts = (
    os.environ.get("TORCHINDUCTOR_RUNTIME_TRITON_NAN_ASSERTS") == "1"
)
scalar_asserts = os.environ.get("TORCHINDUCTOR_SCALAR_ASSERTS", "1") == "1"

# Disable by default in fbcode
alignment_asserts = (
    os.environ.get("TORCHINDUCTOR_ALIGNMENT_ASSERTS", "0" if is_fbcode() else "1")
    == "1"
)

# enable loop reordering based on input orders
pick_loop_orders = True

# reuse a kernel input as the output
inplace_buffers = True

# reuse a buffer for an unrelated purpose
allow_buffer_reuse = True

# Enable pooled allocations for non-output tensors
memory_planning = os.environ.get("TORCHINDUCTOR_MEMORY_PLANNING", "0") == "1"

# Enable to allow using ftz variant of exponenet instruction in triton codegen.
use_fast_math = os.environ.get("TORCHINDUCTOR_USE_FAST_MATH") == "1"

# Enable bfloat16 atomic adds (fbcode only until upstreamed to triton)
bfloat16_atomic_adds_enabled = True

# How to organize memory under memory_planning=True:
# - "none": do not try to pool storage, just reuse
# - "intermediates": all non-outputs share storage, outputs each get unique storage
# - "outputs": two pools, one for intermediates (freed on return) and one for outputs
# - "combined": a single pool for both intermediates and outputs
memory_pool: Literal["none", "intermediates", "outputs", "combined"] = os.environ.get(
    "TORCHINDUCTOR_MEMORY_POOL", "intermediates"
)  # type: ignore[assignment]

# codegen benchmark harness
benchmark_harness = True

# fuse pointwise into templates epilogues
epilogue_fusion = True

# fuse pointwise into template prologues
prologue_fusion = prologue_fusion_enabled()

# do epilogue fusions before other fusions
epilogue_fusion_first = False

# enable pattern match+replace optimizations
pattern_matcher = True

# set to True to enable the back-to-back GEMM pass
b2b_gemm_pass = False

# register custom graph optimization pass hook. so far, pre/post passes are
# only applied before/after pattern_matcher in post_grad_passes.
#
# Implement CustomGraphPass to allow Inductor to graph compiled artifacts
# to which your custom passes have been applied:
post_grad_custom_pre_pass: torch._inductor.custom_graph_pass.CustomGraphPassType = None
post_grad_custom_post_pass: torch._inductor.custom_graph_pass.CustomGraphPassType = None

# Allow users to pass in custom partition function
custom_partitioner_fn: torch._inductor.custom_graph_pass.CustomPartitionerFnType = None

# Registers a custom joint graph pass.
joint_custom_pre_pass: torch._inductor.custom_graph_pass.CustomGraphPassType = None
joint_custom_post_pass: torch._inductor.custom_graph_pass.CustomGraphPassType = None

# Registers a custom pregrad pass. Note that the pre-grad IR is 1.
# non-functional, 2. non-normalized, and 3. prone to change. Ideally we should
# use post-grad passes.
pre_grad_custom_pass: Optional[Callable[[torch.fx.graph.Graph], None]] = None

# Registers a custom pass to be run right before fusion in Inductor scheduler.
# WARNING: Inductor scheduler IR is at prototype stage and subject to change,
# hence custom IR passes built on top of it might break in the future.
_pre_fusion_custom_pass: Optional[
    Callable[
        [list["torch._inductor.scheduler.BaseSchedulerNode"]],
        list["torch._inductor.scheduler.BaseSchedulerNode"],
    ]
] = None

# Registers a custom pass to be run right after fusion in Inductor scheduler.
# WARNING: Inductor scheduler IR is at prototype stage and subject to change,
# hence custom IR passes built on top of it might break in the future.
_post_fusion_custom_pass: Optional[
    Callable[
        [list["torch._inductor.scheduler.BaseSchedulerNode"]],
        list["torch._inductor.scheduler.BaseSchedulerNode"],
    ]
] = None

# Deprecated
split_cat_fx_passes = True

# Optimize conv-batchnorm if batchnorm is in eval mode. Slightly reduces numerical stability.
efficient_conv_bn_eval_fx_passes = False

# Enable predispatch aten IR for export
is_predispatch = False

# Deprecated
group_fusion = False

# Deprecated
batch_fusion = True

# Pre grad fusion and options in order, set to empty dict to disable fusion.
# Call `torch._inductor.fx_passes.group_batch_fusion.list_group_batch_fusions()` to see available fusions.
# batch fusion options:
# batch_linear
# batch_linear_lhs
# batch_layernorm
# batch_tanh
# batch_relu
# batch_sigmoid

# split cat fusion options:
# normalization_pass
# remove_split_with_size_one_pass
# merge_getitem_cat_pass
# merge_stack_tahn_unbind
# merge_splits_pass
# mutate_cat_pass
# split_cat_pass
pre_grad_fusion_options: dict[str, dict[str, Any]] = {}

# Post grad fusion and options, set to empty dict to disable fusion.
# Call `torch._inductor.fx_passes.group_batch_fusion.list_group_batch_fusions(False)` to see available fusions.
post_grad_fusion_options: dict[str, dict[str, Any]] = {}

# enable reordering pass for improving memory locality
reorder_for_locality = True

# Scale down Rn_BLOCK for better occupancy
dynamic_scale_rblock = os.environ.get("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK", "1") == "1"

# this forces fusion for int_mm with mul. Needed when you want to avoid realizing the int32
# but the mul gets fused with other pointwise ops instead.
force_fuse_int_mm_with_mul = False

# DEPRECATED. This setting is ignored.
use_mixed_mm = True

# enable runtime numeric check for pre/post grad fx passes
# floating point provides limited accuracy (about 7 decimal digits for single precision
# floating point numbers,about 16 decimal digits for double precision floating point numbers)
# according to PyTorch documentation.
# https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations
fx_passes_numeric_check: dict[str, Any] = {
    "pre_grad": False,
    "precision": 1e-4,
    "num_iterations": 1,
    "requires_optimizer": True,
}

# DEPRECATED. This setting is ignored.
mixed_mm_choice: Literal["default", "triton", "aten", "heuristic"] = "heuristic"

# enable reordering pass for increasing overlap between compute and communication
reorder_for_compute_comm_overlap = False

# passes (in execution order) for increasing overlap between compute and communication
# for built-in passes, use string name; for user-defined passes, pass in the function handle
# WARNING: Inductor scheduler IR is at prototype stage and subject to change,
# hence custom IR passes built on top of it might break in the future.
#
# See aten_distributed_optimizations, it is recommended way for distributed optimizations.
#
# Recommended configuration for reorder_for_compute_comm_overlap_passes:
# [
#     "reorder_communication_preserving_peak_memory",
#     "sink_waits_iterative",
#     "reorder_communication_preserving_peak_memory",
# ]
reorder_for_compute_comm_overlap_passes: list[
    Union[
        str,
        Callable[
            [list["torch._inductor.scheduler.BaseSchedulerNode"]],
            list["torch._inductor.scheduler.BaseSchedulerNode"],
        ],
    ]
] = []

# Maximum number of positions to advance a given collective, unlimited by default
reorder_prefetch_limit: Optional[int] = None

# enable operator reordering for peak memory optimization
reorder_for_peak_memory = True
reorder_for_peak_memory_debug = False

# In some cases, when all the nodes that can be scheduled are quite large,
# it is beneficial to switch the scheduling strategy. So instead of using
# size as the criterion, we choose a node that can unlock more nodes to
# become schedulable by analyzing their successor nodes. The default value
# is zero, which turns off this optimization.
size_threshold_for_succ_based_strategy: int = 0


bucket_all_gathers_fx: Literal["none", "all", "only_fsdp"] = "none"
# By default torch._inductor.fx_passes.bucketing.bucket_size_determinator is used
bucket_all_gathers_fx_bucket_size_determinator: Optional[Callable[[int], int]] = None

bucket_reduce_scatters_fx: Literal["none", "all"] = "none"
# By default torch._inductor.fx_passes.bucketing.bucket_size_determinator is used
bucket_reduce_scatters_fx_bucket_size_determinator: Optional[Callable[[int], int]] = (
    None
)

# runtime estimation function for ops
# for built-in estimation function, pass in "default"; for user-defined estimation function, pass in the function handle
estimate_op_runtime = "default"

runtime_estimations_mms_benchmark: bool = False

# unit: GB/s, uni-directional P2P bandwidth per card
# default value is NVLink
intra_node_bw = 300

# unit: GB/s, uni-directional P2P bandwidth per node
# default value is InfiniBand
inter_node_bw = 25

# use Inductor's experimental benchmarker (runtime/benchmarking.py)
# to benchmark kernels during autotuning, otherwise fall back to
# Triton's `do_bench`. the experimental benchmarker may produce
# results that are not consistent with `do_bench`'s results
use_experimental_benchmarker: bool = Config(
    default=True,
    env_name_force="TORCHINDUCTOR_USE_EXPERIMENTAL_BENCHMARKER",
    justknob="pytorch/inductor:use_experimental_benchmarker",
)

# Enable distributed autotuning. When this is enabled we will distribute the
# autotuning across distributed ranks in the same program group - so instead of
# each rank autotuning every kernel they only autotune 1/world size kernels and
# then share the results.
distributed_max_autotune_gemm = (
    os.environ.get("TORCHINDUCTOR_DISTRIBUTED_MAX_AUTOTUNE_GEMM") == "1"
)

# enable slow autotuning passes to select algorithms
max_autotune = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"

# enable slow autotuning passes to select pointwise/reductions algorithms
max_autotune_pointwise = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE") == "1"

# enable slow autotuning passes to select gemm algorithms
max_autotune_gemm = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM") == "1"

# Modifies the number of autotuning choices displayed, set to None for all
autotune_num_choices_displayed: Optional[int] = 10

# Report the autotune choices and their benchmark results. Default is True.
max_autotune_report_choices_stats = (
    os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_REPORT_CHOICES_STATS", "1") == "1"
)

# Prune configs that require more shared memory than the hardware limit
max_autotune_prune_choices_based_on_shared_mem = (
    os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM", "1")
    == "1"
)

# Disable triton from trying to initialize and detect devices on the host
triton_disable_device_detection = (
    os.environ.get("TORCHINDUCTOR_TRITON_DISABLE_DEVICE_DETECTION", "0") == "1"
)

# enable inductor graph partition to allow multiple inductor graphs for the same dynamo graph
graph_partition: bool = (
    os.environ.get("TORCHINDUCTOR_GRAPH_PARTITION", "1" if not is_fbcode() else "0")
    == "1"
)

# register ops upon which inductor should partition the graph. name format should be
# "namespace::kernel_name" (e.g., aten::mm) for op overload packet, or
# "namespace::kernel_name.overload" (e.g., aten::mm.default).
custom_should_partition_ops: list[str] = []

# whether template autotuning should allow flexible layouts if possible (e.g. only extern choices)
max_autotune_allow_flexible_layouts: bool = False

# force cublas and triton to use the same precision; cublas supports TF32 for matmul operations
# when m, n, k are multiples of 16, 16, 8, whereas triton supports TF32 for matmul operations
# for any combinations of m, n, k, regardless of their alignment. setting this flag will ensure
# that triton does not use TF32 wherever cublas would not use TF32
# DEPRECATED. cuBLAS no longer has the above alignment requirements. will remove in the future.
force_same_precision: bool = Config(
    justknob="pytorch/compiler:force_same_precision",
    env_name_force="TORCHINDUCTOR_FORCE_SAME_PRECISION",
    default=False,
)

# Size hints for multi-kernel dispatch.
# A reasonable default value of this config would be [64, 256, 4096]
# TODO: @bobrenjc93 to roll this out to a few internal models to ensure this works
# as expected before turning it on for everyone.
multi_kernel_hints: list[int] = []

# Specify candidate backends for gemm autotune.
# Possible choices are combinations of: ATen, Triton, CUTLASS, CK, CKTILE, CPP.
# ATen: default Pytorch ATen kernels.
# Triton: Triton templates defined in torch inductor (AMD and NVidia GPUs).
# CUTLASS: Cutlass templates and kernels (NVidia GPUs only).
# CK: Composable Kernel templates and kernels (AMD Instinct GPUs only).
# CKTILE: Composable Kernel templates and kernels, new API (AMD Instinct GPUs only).
# CPP: CPP templates and kernels for CPU.
max_autotune_gemm_backends = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON,CPP"
).upper()


# As above, specify candidate backends for conv autotune.
# NB: in some cases for 1x1 convs we emit as matmul,
# which will use the backends of `max_autotune_gemm_backends`
max_autotune_conv_backends = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_CONV_BACKENDS", "ATEN,TRITON"
).upper()


# Specify the size of the search space for GEMM autotuning.
# DEFAULT     - balance between compile time overhead and performance
# EXHAUSTIVE  - maximize performance
max_autotune_gemm_search_space: Literal["DEFAULT", "EXHAUSTIVE"] = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE", "DEFAULT"
).upper()  # type: ignore[assignment]

# Specify the size of the search space for flex attention autotuning.
# DEFAULT     - balance between compile time overhead and performance
# EXHAUSTIVE  - maximize performance
max_autotune_flex_search_space: Literal["DEFAULT", "EXHAUSTIVE"] = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_FLEX_SEARCH_SPACE", "DEFAULT"
).upper()  # type: ignore[assignment]

cutedsl_enable_autotuning: bool = (
    os.environ.get("CUTEDSL_ENABLE_AUTOTUNING", "0") == "1"
)

# DEPRECATED. This setting is ignored.
autotune_fallback_to_aten = False

# the value used as a fallback for the unbacked SymInts
# that can appear in the input shapes (e.g., in autotuning)
unbacked_symint_fallback = 8192

# DEPRECATED. This setting is ignored.
search_autotune_cache = False

save_args = os.environ.get("TORCHINDUCTOR_SAVE_ARGS") == "1"

# We will disable creating subprocess for autotuning if this is False
autotune_in_subproc = os.environ.get("TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC") == "1"

# The following three timeouts are applicable if autotune_in_subproc is True:

# Max time that a valid benchmark result may take during autotuning
max_autotune_subproc_result_timeout_seconds = 60.0
# DEPRECATED. This setting is ignored.
max_autotune_subproc_graceful_timeout_seconds = 0.0
# DEPRECATED. This setting is ignored.
max_autotune_subproc_terminate_timeout_seconds = 0.0

# If autotuning in subprocess, whether to use multiple devices
autotune_multi_device = os.environ.get("TORCHINDUCTOR_AUTOTUNE_MULTI_DEVICE") == "1"

coordinate_descent_tuning = (
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING") == "1"
)
coordinate_descent_check_all_directions = (
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS") == "1"
)
coordinate_descent_search_radius = int(
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS", "1")
)

# AutoHeuristic is a framework that allows one to collect data from autotuning, use the data to learn a heuristic, and
# generate the learned heuristic to code which is shipped with the compiler
# Specify a list of comma separated optimizations to collect data for
autoheuristic_collect = os.environ.get("TORCHINDUCTOR_AUTOHEURISTIC_COLLECT", "")
# Specify a list of comma separated optimizations to use learned heuristics for
autoheuristic_use = os.environ.get("TORCHINDUCTOR_AUTOHEURISTIC_USE", "mixed_mm")

# If set to 1, will run a JIT post compile hook if one is set.
run_jit_post_compile_hook = (
    os.environ.get("TORCHINDUCTOR_RUN_JIT_POST_COMPILE_HOOK", "0") == "1"
)


def run_autoheuristic(name: str) -> bool:
    return collect_autoheuristic(name) or use_autoheuristic(name)


def collect_autoheuristic(name: str) -> bool:
    return name in torch._inductor.config.autoheuristic_collect.split(",")


def use_autoheuristic(name: str) -> bool:
    return name in torch._inductor.config.autoheuristic_use.split(",")


# If set to "DEFAULT", this will use the default log path specified in autoheuristic.py.
# If set to another path, autoheuristic will instead log results to the given path.
autoheuristic_log_path = os.environ.get(
    "TORCHINDUCTOR_AUTOHEURISTIC_LOG_PATH", "DEFAULT"
)

# Disabled by default on ROCm, opt-in if model utilises NHWC convolutions
layout_opt_default = "1" if not torch.version.hip else "0"
layout_optimization = (
    os.environ.get("TORCHINDUCTOR_LAYOUT_OPTIMIZATION", layout_opt_default) == "1"
)

force_layout_optimization = os.environ.get("TORCHINDUCTOR_FORCE_LAYOUT_OPT", "0") == "1"


# Whether to keep the output strides the same as eager after layout optimization.
keep_output_stride = os.environ.get("TORCHINDUCTOR_KEEP_OUTPUT_STRIDE", "1") == "1"

# Enabling this will let compiler print warning messages if a generated triton
# kernel has inputs with mixed layouts.  This is helpful for perf debugging
# since kernel with mixed layout inputs may run much slower then one whose inputs
# have uniform layouts.
warn_mix_layout = os.environ.get("TORCHINDUCTOR_WARN_MIX_LAYOUT") == "1"

# control store vs recompute heuristic
# For fanouts, rematerialization can lead to exponential blowup. So, have
# smaller threshold
realize_reads_threshold = 4
realize_opcount_threshold = 30

# Threshold to prevent excessive accumulation of ops in one buffer during lowering
realize_acc_reads_threshold = 8
realize_acc_reads_size_threshold: Optional[int] = (
    None  # TODO(xuanzh): harden this to make it non optional
)

# fallback to eager for random/dropout, this is slow but useful for debugging
fallback_random = False

# fallback embedding_bag_byte_unpack to eager
fallback_embedding_bag_byte_unpack = False

# automatically create fallbacks when encountering an unhandled op
implicit_fallbacks = True
assume_unaligned_fallback_output = (
    os.environ.get("TORCHINDUCTOR_ASSUME_UNALIGNED_FALLBACK_OUTPUT") == "1"
)

# Custom InductorChoices callable to use (can be a class or functools.partial with kwargs)
inductor_choices_class: Optional[Callable[[], "InductorChoices"]] = None

# fuse even in cases without common reads
aggressive_fusion = False

# For each fused kernel in the wrapper, comment with the nodes that get fused.
# Useful for debugging fusion.
debug_fusion: bool = os.environ.get("TORCHINDUCTOR_DEBUG_FUSION") == "1"
benchmark_fusion: bool = os.environ.get("TORCHINDUCTOR_BENCHMARK_FUSION") == "1"
enabled_metric_tables = os.environ.get("TORCHINDUCTOR_ENABLED_METRIC_TABLES", "")
loop_ordering_after_fusion: bool = (
    os.environ.get(
        "TORCHINDUCTOR_LOOP_ORDERING_AFTER_FUSION", "0" if is_fbcode() else "1"
    )
    == "1"
)


# When trying to fuse two nodes, one with:
# a[contiguous_writes] = fn(...)
# and another node:
# b[contiguous_writes] = a[discontiguous_reads]
# If b is unary, and we can figure out an inverse formula for
# discontiguous writes, invert b as :
# b[inverse(discontiguous_writes)] = a[contiguous_reads]
# so that the nodes can fuse. for more details: https://gist.github.com/eellison/6f9f4a7ec10a860150b15b719f9285a9
loop_index_inversion_in_fusion: bool = True

# If fusing two nodes only save less then score_fusion_memory_threshold memory,
# we should not bother fusing the nodes.
#
# This is especially helpful to resolve https://github.com/pytorch/pytorch/issues/133242
# Previously we fuse two nodes because of common read of a scalar tensor.
# If we skip it, the loop ordering after fusion mechanism kicks in and can
# brings more savings.
#
# For the cases loop ordering after fusion does not help, we don't lose much.
score_fusion_memory_threshold = 10

# For Triton Templates, select fastest of best template + epilogue vs best template + separate epilogue kernel
benchmark_epilogue_fusion = (
    os.environ.get("TORCHINDUCTOR_BENCHMARK_EPILOGUE_FUSION", "1") == "1"
)

# Take how many of the top triton kernels to benchmark epilogue
max_epilogue_benchmarked_choices = 1

# how many nodes to allow into a single fusion
max_fusion_size = 64

# how many nodes to attempt pairwise fusion with in a buffer group
max_fusion_buffer_group_pairwise_attempts = 64

# maximum number of unique input/output buffers allowed in fused kernels.
# The check is disabled if set to None.
max_fusion_unique_io_buffers: Optional[int] = None

# max number of inputs to generate cat as a pointwise op with masked loads
max_pointwise_cat_inputs = 8

# force concat to be generated as a pointwise op with masked loads
force_pointwise_cat = False

# replace small reductions with pointwise, disable with `= 1`
unroll_reductions_threshold = 8

# Add extra comments to output code (causes compile cache misses)
comment_origin = False

# Convert 1x1 convs into matmuls
conv_1x1_as_mm = False

# For reductions with a small output size (usually 1, e.g. x.sum()) there is not enough
# parallelism to saturate the GPU.  We have two ways of handling this, either `split_reductions`
# or `triton.cooperative_reductions` which are mutually exclusive.
#   split_reductions: uses multiple kernels to gain more parallelism
#   triton.cooperative_reductions: uses cross thread-block synchronization to gain more parallelism
# enabling both of these will implicitly disable split_reductions
split_reductions = os.getenv("TORCHINDUCTOR_SPLIT_REDUCTIONS", "1") == "1"

# A deterministic mode that skips any on device benchmarking in Inductor
# if we know they affect numerics.  WARNING: Expect perf hit in this mode.
deterministic = os.getenv("TORCHINDUCTOR_DETERMINISTIC") == "1"

# When we do split reduction, this number control the minimum value for
# num_split. Too small num_split make the split reduction less efficient.
# It's a much bigger problem when we compile a dynamic shape kernel with
# non-representative inputs.
min_num_split = int(os.environ.get("TORCHINDUCTOR_MIN_NUM_SPLIT", 0))

benchmark_kernel = os.environ.get("TORCHINDUCTOR_BENCHMARK_KERNEL", "0") == "1"

# Enable constant and index_expr folding
constant_and_index_propagation = True

# we always add constants into graph.constants without
# performing any constant-inlining optimization
always_keep_tensor_constants = False

# assert that indirect indexing does not read / write out of bounds
assert_indirect_indexing = True

# compute CSE bounds on variables that do not appear in the FX graph
compute_all_bounds = False

# enable the combo kernel that combines data-independent kernels (additional
# to foreach kernels) into a single one (Experimental)
combo_kernels = False
# benchmark combo kernels and only allow ones with perf gains
benchmark_combo_kernel = False
# combo_kernel autotuning options: 0 - disable, 1 - enable except for foreach,
# 2 - enable for all
combo_kernels_autotune = 1
# Enable masking for combining kernels of mixed sizes: 0 - disable, 1 - enable
# for all except for foreach, 2 - enable for all
combo_kernel_allow_mixed_sizes = 1
# Enable dynamic shapes for foreach kernels
combo_kernel_foreach_dynamic_shapes = True
# Maximum number of arguments (read/write buffers) allowed in a combo kernel
combo_kernel_max_num_args = 250

# constant folding on the joint graph
joint_graph_constant_folding = True

# Enable indirect_indexing asserts for decompositions and lowerings
debug_index_asserts = False

# Mode to emulate PyTorch eager numerics when doing lower precision compute
# (fp16, bf16).  PyTorch eager computes bf16/fp16 by upcasting inputs to fp32
# and downcasting after.  When two low precision operators are fused together,
# Inductor will elide the downcast-upcast pairs (effectively a precision
# truncation) that would occur between these two operators.  Typically,
# Inductor's behavior should be closer to fp64 ref numerics.  However, with
# this knob you can ensure the downcast-upcast are preserved so that you can
# emulate the eager numerics.
emulate_precision_casts = (
    os.environ.get("TORCHINDUCTOR_EMULATE_PRECISION_CASTS", "0") == "1"
)

# x / y in Triton is lowered to div.full which is approx
# PyTorch eager uses the equivalent of Triton's div_rn, which can
# come at a performance penalty
emulate_divison_rounding = (
    os.environ.get("TORCHINDUCTOR_EMULATE_DIVISION_ROUNDING", "0") == "1"
)

# warnings intended for PyTorch developers, disable for point releases
is_nightly_or_source = "dev" in torch.__version__ or "git" in torch.__version__
developer_warnings = is_fbcode() or is_nightly_or_source

# This pattern matches a special usage of scatter
# 1. It's applied to a constant tensor
# 2. The index tensor has size 1 in the scatter dimension
# Such pattern generates a sparse matrix when the const tensor is all-zero.
# We can lower this pattern to a pointwise kernel for more fusion opportunities
# and saving memory footprint.
optimize_scatter_upon_const_tensor = (
    os.environ.get("TORCHINDUCTOR_OPTIMIZE_SCATTER_UPON_CONST_TENSOR", "1") == "1"
)

# options in caffe2/torch/_inductor/fx_passes/pre_grad.py
add_pre_grad_passes: Optional[str] = None
remove_pre_grad_passes: Optional[str] = None


# The multiprocessing start method to use for inductor workers in the codecache.
def decide_worker_start_method() -> str:
    if "TORCHINDUCTOR_WORKER_START" in os.environ:
        start_method = os.environ["TORCHINDUCTOR_WORKER_START"]
    else:
        start_method = "subprocess"
    assert start_method in (
        "subprocess",
        "fork",
        "spawn",
    ), f"Invalid start method: {start_method}"
    return start_method


worker_start_method: str = decide_worker_start_method()

# Threshold to decide if a kernel has small memory access in bytes
# Default value is 16 MB which is arbitrarily selected.
small_memory_access_threshold: int = 16777216

# Whether to log from subprocess workers that are launched.
worker_suppress_logging: bool = Config(
    justknob="pytorch/compiler:worker_suppress_logging",
    env_name_force="TORCHINDUCTOR_WORKER_SUPPRESS_LOGGING",
    default=True,
)

# Log per-operation runtime estimates for TLParse analysis.
log_tlparse: bool = Config(
    env_name_force="LOG_TLPARSE",
    default=False,
)

# Flags to turn on all_reduce fusion. These 2 flags should be automatically turned
# on by DDP and should not be set by the users.
_fuse_ddp_communication = False
_fuse_ddp_bucket_size = 25

# Flag to control which fusion passes to apply. Functions in the list will
# be applied in order. There are two different different fusion passes
# --"fuse_ddp_with_concat_op" and "fuse_ddp_with_coalesced_op". The default
# one is "fuse_ddp_with_concat_op". Users can also change this to a customized
# fusion function.
#
# The fusion currently does not support multiple DDP with different PG or
# data type. This feature will be added in the future PRs.
#
# "schedule_comm_wait" is used to delay the wait ops to maximize comm/comp
# overlapping. At this moment, this pass performs better than
# reorder_for_compute_comm_overlap_passes but we will add the logic of
# "schedule_comm_wait" in the future and remove the one here.
_fuse_ddp_communication_passes: list[Union[Callable[..., None], str]] = [
    "fuse_ddp_with_concat_op",
    "schedule_comm_wait",
]

_micro_pipeline_tp: bool = False


class _collective:
    auto_select: bool = False
    one_shot_all_reduce_threshold_bytes: int = 128 * 1024


class aten_distributed_optimizations:
    """Configuration for distributed optimization passes on ATen FX graphs."""

    # Enable overlap scheduling pass
    enable_overlap_scheduling: bool = False

    # Enable overlap-preserving collective bucketing
    collective_bucketing: Optional[bool] = None

    # Insert ordering dependencies to preserve overlap relationships. This should only be used if
    # compiling with inductor, or for subsequent passes before removing the ops prior to execution
    insert_overlap_deps: Optional[bool] = None

    # Maximum compute node prefetch distance for overlap scheduling
    max_compute_pre_fetch: Optional[int] = None

    # Custom runtime estimation function for ops
    # For user-defined estimation function, pass in the function handle
    # None means use default estimations
    # TODO - need estimated and profile based version
    custom_runtime_estimation: Optional[Callable[[torch.fx.Node], Optional[float]]] = (
        None
    )

    # Method for estimating collective runtime
    # "analytical": Use bandwidth formulas (default)
    # "benchmark": Use CUDA events with power-of-2 rounding and interpolation
    collective_estimator: Literal["analytical", "benchmark"] = "analytical"


def parallel_compile_enabled_internally() -> bool:
    """
    TODO: Remove when parallel compiled is fully enabled internally. For rollout, use a
    knob to enable / disable. The justknob should not be performed at import, however.
    So for fbcode, we assign compile_threads to 'None' below and initialize lazily in
    async_compile.py.
    """
    ENABLE_PARALLEL_COMPILE_VERSION = 1

    jk_name = "pytorch/inductor:enable_parallel_compile_version"
    version = torch._utils_internal.justknobs_getval_int(jk_name)
    return ENABLE_PARALLEL_COMPILE_VERSION >= version


def decide_compile_threads() -> int:
    """
    Here are the precedence to decide compile_threads
    1. User can override it by TORCHINDUCTOR_COMPILE_THREADS.  One may want to disable async compiling by
       setting this to 1 to make pdb happy.
    2. Set to 1 if it's win32 platform
    3. decide by the number of CPU cores
    """
    import logging

    # Defined locally so install_config_module doesn't try to parse
    # as a config option.
    log = logging.getLogger(__name__)

    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        compile_threads = int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
        log.info("compile_threads set to %d via env", compile_threads)
    elif sys.platform == "win32":
        compile_threads = 1
        log.info("compile_threads set to 1 for win32")
    elif is_fbcode() and not parallel_compile_enabled_internally():
        compile_threads = 1
        log.info("compile_threads set to 1 in fbcode")
    else:
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        )
        assert cpu_count
        compile_threads = min(32, cpu_count)
        log.info("compile_threads set to %d", compile_threads)

    return compile_threads


# TODO: Set directly after internal rollout.
compile_threads: Optional[int] = None if is_fbcode() else decide_compile_threads()

# Whether to quiesce the Triton-compile subprocess pool at the end of each compilation.
quiesce_async_compile_pool: bool = Config(
    justknob="pytorch/inductor:quiesce_async_compile_pool",
    env_name_force="TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL",
    default=False,
)

# Time in seconds to wait before quiescing
quiesce_async_compile_time: int = Config(
    default=60,
)

# Whether or not to enable statically launching CUDA kernels
# compiled by triton (instead of using triton's own launcher)
use_static_cuda_launcher: bool = static_cuda_launcher_default()

# Attempt to statically launch user defined triton kernels
# Requires use_static_cuda_launcher
static_launch_user_defined_triton_kernels: bool = Config(
    justknob="pytorch/inductor:static_launch_user_defined_triton_kernels",
    env_name_force="TORCHINDUCTOR_STATIC_LAUNCH_USER_DEFINED_TRITON_KERNELS",
    default=False,
)

# Raise error if we bypass the launcher
strict_static_cuda_launcher: bool = (
    os.environ.get("TORCHINDUCTOR_STRICT_STATIC_CUDA_LAUNCHER", "0") == "1"
)

# gemm autotuning global cache dir
global_cache_dir: Optional[str]
if is_fbcode():
    try:
        from libfb.py import parutil

        if __package__:
            global_cache_dir = parutil.get_dir_path(
                os.path.join(__package__.replace(".", os.sep), "fb/cache")
            )
        else:
            global_cache_dir = parutil.get_dir_path("fb/cache")
    except (ValueError, ImportError):
        global_cache_dir = None

else:
    global_cache_dir = None

# If kernel is fused, the name is generated from the origin node op names
# for larger kernels limit this
kernel_name_max_ops = 10

# Pad input tensors of matmul/bmm/addmm to leverage Tensor Cores in NVIDIA GPUs
shape_padding = os.environ.get("TORCHINDUCTOR_SHAPE_PADDING", "1") == "1"

# Control if we will do padding for pointwise/reductions
comprehensive_padding = (
    os.environ.get("TORCHINDUCTOR_COMPREHENSIVE_PADDING", "1") == "1"
)
pad_channels_last = False

# Control if we will do padding on dynamic shapes
pad_dynamic_shapes = False

# Disable comprehensive padding on the CPU
disable_padding_cpu = True

# Control if we will expand the dimension of pointwise nodes to fuse
expand_dimension_for_pointwise_nodes = False

# The width of comprehensive padding, in bytes.
# CUDA max memory transaction size is 128 bytes for a warp.
padding_alignment_bytes = 128

# Threshold on the minimum stride that will be padded.
#
# Don't align a too small stride since that causes too much memory increase.
# Pad too small stride may also cause perf loss. We may result in many tiny data blocks
# with gaps in between. That causes less coalesced GPU memory access!
#
# Initially we pick 320 as the threshold since for alignment=16,
# that results in at most 5% memory cost.
#
# But later on we raise the threshold to 1024 to avoid interfere with persistent reduction.
# Let's say an inner reduction has a row size 513. Inductor will generate
# persistent reduction code.
# If we do padding, the strides are not contiguous any more. Inductor
# uses a much smaller threshold for persistent reduction in this case and
# generates potentially worse non-persistent reduction code.
#
# This change turns HF AllenaiLongformerBase amp training from a loss of 1.09x to a win of 1.05x.
# (baseline: 71.09ms, padding w/o this change: 77.38ms, padding with this change: 67.77ms)
padding_stride_threshold = 1024

# Enable padding outputs, even if they would not be padded in eager mode.
# By default, we use the same strides as eager mode.
pad_outputs = False

# Whether to treat output of the backward graph as user visible.
# For user visible outputs, inductor will make sure the stride matches with eager.
bw_outputs_user_visible = True

# Whether to always use shape padding if it is enabled and possible
force_shape_pad: bool = False

# Fx-based linear/matmul/bmm + permute/transpose vertical fusion
permute_fusion = os.environ.get("TORCHINDUCTOR_PERMUTE_FUSION", "0") == "1"

# Mark the wrapper call in PyTorch profiler
profiler_mark_wrapper_call = False

# Generate hook calls to torch._inductor.hooks.run_intermediate_hooks for
# every intermediate for which we can correlate it with an intermediate
# from the original FX graph
generate_intermediate_hooks = False

# Populate traceback field on IRNode; good for debugging why origin_node is
# not populated, or finding out where an IRNode was constructed
debug_ir_traceback = False

# used for debugging to make sure config is properly set
_raise_error_for_testing = False

_profile_var = os.environ.get("TORCHINDUCTOR_PROFILE", "")
profile_bandwidth = _profile_var != ""
profile_bandwidth_regex = "" if _profile_var == "1" else _profile_var
# Specify a file where we print out the profiling results.
# None means we do not dump results to a file.
profile_bandwidth_output: Optional[str] = os.environ.get(
    "TORCHINDUCTOR_PROFILE_OUTPUT", None
)
# Switch to do_bench_using_profiling to exclude the CPU overheads
profile_bandwidth_with_do_bench_using_profiling = (
    os.environ.get("TORCHINDUCTOR_PROFILE_WITH_DO_BENCH_USING_PROFILING") == "1"
)


# TODO: remove later
# incompatible with cpp_wrapper
disable_cpp_codegen = False


# Freezing will attempt to inline weights as constants in optimization
# and run constant folding and other optimizations on them. After freezing, weights
# can no longer be updated.
freezing: bool = os.environ.get("TORCHINDUCTOR_FREEZING", "0") == "1"

# Make freezing invalidate the eager Parameters of nn modules, to avoid memory overhead
# of potentially keeping multiple copies of weights.
freezing_discard_parameters: bool = False

# decompose some memory bound matmul/bmm to mul
decompose_mem_bound_mm: bool = False

# assume_aligned_inputs means that we assume that inputs will be aligned; we generate
# code using this assumption, and clone tensors before use if they aren't aligned.
# In the common case, most inputs will be aligned.
assume_aligned_inputs: bool = False

# For the user-written Triton kernels compiled with the model, ignore the unsupported
# arguments passed to the @triton.autotune in the user's code; this is unsafe, as
# ignoring the unsupported args may lead to unexpected autotuning behavior: don't
# set unless you know what you're doing.
unsafe_ignore_unsupported_triton_autotune_args: bool = False

# When True, we will check in scheduler.py _codegen that there are no "loops"
# in the call stack; that is to say, the same frame multiple times.  This
# ensures that a cProfile trace to this frame will be a straight line without
# any cycles. Incompatible with cpp_wrapper.
check_stack_no_cycles_TESTING_ONLY: bool = False

# When True, complex_memory_overlap always reports True
always_complex_memory_overlap_TESTING_ONLY: bool = False

# enable linear binary folding
enable_linear_binary_folding = (
    os.environ.get("TORCHINDUCTOR_ENABLE_LINEAR_BINARY_FOLDING", "0") == "1"
)


# Adds NVTX annotations around training phases
annotate_training: bool = os.environ.get("TORCHINDUCTOR_ANNOTATE_TRAINING", "0") == "1"

# Enable caching codegen of triton templates.
enable_caching_generated_triton_templates: bool = True

# Lookup table for overriding autotune configs based on hash of Triton source code
autotune_lookup_table: dict[str, dict[str, Any]] = {}

file_lock_timeout: int = int(os.environ.get("TORCHINDUCTOR_FILE_LOCK_TIMEOUT", "600"))


def get_worker_log_path() -> Optional[str]:
    log_loc = None
    if is_fbcode():
        mast_job_name = os.environ.get("MAST_HPC_JOB_NAME", None)
        global_rank = os.environ.get("ROLE_RANK", "0")

        if mast_job_name is not None:
            log_loc = f"/logs/dedicated_log_torch_compile_worker_rank{global_rank}"

    return log_loc


torchinductor_worker_logpath: str = Config(
    env_name_force="TORCHINDUCTOR_WORKER_LOGPATH",
    default="",
)


# config specific to codegen/cpp.py
class cpp:
    """
    Settings for cpp backend.
    This class provides a centralized location for managing cpp backend settings.
    """

    # set to torch.get_num_threads()
    threads = -1

    # Do not generate loops when the condition doesn't hold, like:
    # for(long i0=4096; i0<4096; i0+=1)
    no_redundant_loops = (
        os.environ.get("TORCHINDUCTOR_CPP_NO_REDUNDANT_LOOPS", "1") == "1"
    )

    # Assume number of threads is dynamic, don't specialize thread number.
    # Kernels don't recompile on thread number changes with this flag on.
    # For single-threaded workload, turning it on would incur a slight
    # performance degradation.
    dynamic_threads = os.environ.get("TORCHINDUCTOR_CPP_DYNAMIC_THREADS", "0") == "1"

    simdlen: Optional[int] = None
    min_chunk_size = int(os.environ.get("TORCHINDUCTOR_CPP_MIN_CHUNK_SIZE", "512"))

    cxx: tuple[None, str] = (
        None,  # download gcc12 from conda-forge if conda is installed
        os.environ.get("CXX", "clang++" if sys.platform == "darwin" else "g++"),
    )  # type: ignore[assignment]

    # Allow kernel performance profiling via PyTorch profiler
    enable_kernel_profile = (
        os.environ.get("TORCHINDUCTOR_CPP_ENABLE_KERNEL_PROFILE", "0") == "1"
    )

    # enable weight prepacking to get a better performance; may lead to large memory footprint
    weight_prepack = os.environ.get("TORCHINDUCTOR_CPP_WEIGHT_PREPACK", "1") == "1"

    # Inject a bug into our relu implementation; useful for testing our repro
    # extraction and minification functionality.
    # Valid values: "compile_error", "runtime_error", "accuracy"
    inject_relu_bug_TESTING_ONLY: Optional[str] = None
    inject_log1p_bug_TESTING_ONLY: Optional[str] = None

    # If None, autodetect whether or not AVX512/AVX2 can be used.  Otherwise,
    # force usage as specified, without testing. Default None.
    vec_isa_ok: Optional[bool] = get_tristate_env("TORCHINDUCTOR_VEC_ISA_OK")

    # similar to config.triton.descriptive_names
    descriptive_names: Literal["torch", "original_aten", "inductor_node"] = (
        "original_aten"
    )

    # how many nodes to allow into a single horizontal fusion
    max_horizontal_fusion_size = int(
        os.environ.get("TORCHINDUCTOR_CPP_MAX_HORIZONTAL_FUSION_SIZE", "16")
    )

    # Make scatter_reduce fallback when reduce is sum to avoid performance regression
    # using atomic_add.
    fallback_scatter_reduce_sum = (
        os.environ.get("TORCHINDUCTOR_CPP_FALLBACK_SCATTER_REDUCE_SUM", "1") == "1"
    )

    # Use funsafe-math-optimizations when compiling
    enable_unsafe_math_opt_flag = (
        os.environ.get("TORCHINDUCTOR_CPP_ENABLE_UNSAFE_MATH_OPT_FLAG", "0") == "1"
    )

    # Use ffp-contract when compiling
    # Options: "off" (default), "on", "fast"
    # Per https://godbolt.org/z/bf4bvfc9r , clang/gcc has different behavior for "fast"
    enable_floating_point_contract_flag = os.environ.get(
        "TORCHINDUCTOR_CPP_ENABLE_FLOATING_POINT_CONTRACT_FLAG", "off"
    )

    # Disable the tiling select heuristic
    enable_tiling_heuristics = (
        os.environ.get("TORCHINDUCTOR_CPP_ENABLE_TILING_HEURISTIC", "1") == "1"
    )

    # Enable the Grouped GEMM Fusion
    enable_grouped_gemm_template = False

    # Maximal allowed number of slices on K-dim for a GEMM kernel. This controls
    # the maximal parallelism of K-slicing. Since K-slicing requires extra thread
    # synchronization and buffers,  the maximal number of slices is limited to
    # mitigate the sync overhead and memory usage.
    # When set to 0, the number of slices is unlimited.
    gemm_max_k_slices = int(os.environ.get("TORCHINDUCTOR_CPP_GEMM_MAX_K_SLICES", "1"))

    # For perf tuning and debugging purpose, configure the pre-defined cache blocking for
    # MxNxK dims respectively. The blockings are separated by comma and the unit is
    # the number of register blocks.
    # For example, "4,1,10" means 4 register blocks on M, 1 on N and 10 on K respectively.
    gemm_cache_blocking = os.environ.get("TORCHINDUCTOR_CPP_GEMM_CACHE_BLOCKING", None)

    # For perf tuning and debugging purpose, configure the pre-defined thread blocking factors for
    # MxNxK dims respectively. The factors are separated by comma and their product
    # should be the same as the total number of threads.
    # For example, if the total number of threads is 56, "7,4,2" means the work is
    # decomposed into 7x4x2 thread blocks along MxNxK of a GEMM.
    gemm_thread_factors = os.environ.get("TORCHINDUCTOR_CPP_GEMM_THREAD_FACTORS", None)

    # Whether to enable masked vectorization for the tail_loop.
    enable_loop_tail_vec = True

    # Whether to enable concat linear for cpu device
    # Currently concat linear on CPU not always have benefit, depends on linear'shape or
    # computing resource. We set this default to False to avoid regressions. User and
    # enable this feature by their need.
    enable_concat_linear = False

    # Whether to use decomposed tanh for cpu device
    # Disable by default due to https://github.com/pytorch/pytorch/issues/148241
    use_decompose_tanh = (
        os.environ.get("TORCHINDUCTOR_CPP_USE_DECOMPOSE_TANH", "0") == "1"
    )

    # Use a small dequant buffer for wgt of woq int4 size as: [q_group_size, Nr]
    use_small_dequant_buffer = False

    force_inline_kernel = (
        os.environ.get("TORCHINDUCTOR_CPP_FORCE_INLINE_KERNEL", "0") == "1"
    )

    # Use static constexpr or static const for int array
    use_constexpr_for_int_array = (
        os.environ.get("TORCHINDUCTOR_CPP_USE_CONSTEXPR_FOR_INT_ARRAY", "1") == "1"
    )


class triton:
    """
    Config specific to codegen/triton.py
    """

    # Use cudagraphs on output code
    cudagraphs = os.environ.get("TORCHINDUCTOR_CUDAGRAPHS") == "1"

    # Use cudagraph trees for memory pooling if `cudagraphs` is True
    cudagraph_trees = True

    # Should we skip cudagraphing graphs with dynamic shape inputs
    # If False, we will re-record a graph for each unique set of shape inputs
    cudagraph_skip_dynamic_graphs = False

    # Specify dynamic shapes to capture cudagraphs and skip cudagraph for other shapes.
    # Default to None, which means we capture cudagraphs for all shapes.
    cudagraph_capture_sizes: Optional[tuple[Union[int, tuple[int, ...]]]] = None

    # assertions not on the fast path, steady state
    slow_path_cudagraph_asserts = True

    # TODO - need to debug why this prevents cleanup
    cudagraph_trees_history_recording = False

    # Enable cudagraph support for mutated inputs from prior cudagraph pool
    cudagraph_support_input_mutation = not is_fbcode()

    # Maximal number of allowed cudagraph re-record for a function and
    # a cudagraph node due to static input tensor address changes or
    # cudagraph managed tensor data pointer changed.
    # i.e., allow num_recording <= cudagraph_unexpected_rerecord_limit
    # note: we are conservative here and choose a large limit.
    cudagraph_unexpected_rerecord_limit = 128

    # Warn loudly when the number of cudagraphs due to dynamic shape
    # exceeds this limit
    cudagraph_dynamic_shape_warn_limit: Optional[int] = 8

    # synchronize after cudagraph invocation
    force_cudagraph_sync = False

    # always run cudagraphs in the eager warmup stage
    # instead of recording and executing cudagraphs
    force_cudagraphs_warmup = False

    # If False (default), torch.compile skips cudagraph for a graph if it
    # contains cudagraph-unsafe ops. If True, we require that all cuda ops
    # be captured into cudagraph. If this is not possible, this will raise
    # an error.
    cudagraph_or_error: bool = Config(
        env_name_force="TORCHINDUCTOR_CUDAGRAPH_OR_ERROR",
        default=False,
    )

    # assertions on the fast path
    fast_path_cudagraph_asserts = False

    # skip warmup for cudagraph trees
    skip_cudagraph_warmup = False

    # Synchronize before and after every compiled graph.
    debug_sync_graph = False

    # Synchronize after every kernel launch, to help pinpoint bugs
    debug_sync_kernel = False

    # Always load full blocks (rather than broadcasting inside the block)
    dense_indexing = False

    # TODO - enable by default
    coalesce_tiling_analysis: bool = (
        os.environ.get(
            "TORCHINDUCTOR_COALESCE_TILING_ANALYSIS", "1" if not is_fbcode() else "0"
        )
        == "1"
    )

    # limit tiling dimensions
    #   - max_tiles=1 disables tiling
    #   - max_tiles=2
    #   - max_tiles=3 is experimental and may have bugs
    # higher values are unsupported

    #  We use a max of 3 if coalesce_tiling_analysis is True, and 2 otherwise.
    #  Note - coalesce_tiling_analysis does not yet apply to dynamic shapes.
    max_tiles: Optional[int] = None

    # Prefer higher dimensional tilings. This simplifies indexing expressions, making
    # it easier to identify block pointers.
    prefer_nd_tiling: bool = False

    # use triton.autotune for pointwise ops with complex layouts
    # this should only be disabled for debugging/testing
    autotune_pointwise = True

    # max autotune gemm with cublasLt
    autotune_cublasLt = True

    # Tune the generated Triton kernels at compile time instead of first time they run
    # Setting to None means uninitialized
    autotune_at_compile_time: Optional[bool] = None

    # We use random tensors for autotune by default. Setting this as true will let us
    # use inputs from sample inputs to autotune user defined triton kernels.
    # Side effect for this option is increased memory footprint during first pass compilation.
    autotune_with_sample_inputs: bool = False

    # Allows tiling reductions into multiple dimensions.
    # For best results, this should be used with prefer_nd_tiling.
    tile_reductions: bool = False

    # Codegen matmul natively with tl.dot without using a template.
    # This option makes Inductor generate matrix multiplication from scratch,
    # instead of calling predefined Triton templates (mm, bmm, mm_plus_mm).
    # Compile time may be longer because native matmul benchmarks more Triton configs
    # than regular pointwise or reduction kernels.
    # Native matmul often aggressively fuses operations around the matrix multiply,
    # which can make it faster or slower depending on your program.
    #
    # This option takes priority over other GEMM implementations. If Inductor determines
    # that a matmul can be generated, it will always generate it with native_matmul.
    # That means optimized kernels such as decompose_k or persistent_tma_matmul will
    # not be called when this option is enabled.
    #
    # Note: Native matmul does not currently support block pointers or TMA matmul.
    # If both native_matmul and (use_block_ptr or enable_persistent_tma_matmul) are enabled,
    # an error will be thrown.
    native_matmul: bool = False

    # should we stop a fusion to allow better tiling?
    tiling_prevents_pointwise_fusion = True
    tiling_prevents_reduction_fusion = True

    # should we give different names to kernels
    # Note: This is orthogonal to descriptive_names - this is deciding whether
    # our triton kernel names should all be `triton_` (to maximize caching) or
    # whether they should be unique.
    unique_kernel_names = (
        os.environ.get("TORCHINDUCTOR_UNIQUE_KERNEL_NAMES", "1") == "1"
    )

    # similar to the option above, but this is specific to user defined kernels,
    # while unique_kernel_name is for kernels generated by inductor.
    # We have this option because sometimes we reuse user's kernel code with different
    # configs which would result in the same name.
    # Note: This MODIFIES the user's kernel function name within inductor phase.
    unique_user_kernel_names = (
        os.environ.get("TORCHINDUCTOR_UNIQUE_USER_KERNEL_NAMES", "0") == "1"
    )

    # should we put op names in kernel names
    # "torch": Maps to the fx op in the Dynamo graph (module name, method name, etc.)
    # "original_aten": Maps to the highest-level aten op (i.e. pre-decompositions)
    # "inductor_node": Maps to the node name in the FX graph passed to Inductor
    descriptive_names: Literal["torch", "original_aten", "inductor_node"] = (
        "original_aten"
    )

    # use alternate codegen for smaller reductions
    persistent_reductions = (
        os.environ.get("TORCHINDUCTOR_PERSISTENT_REDUCTIONS", "1") == "1"
    )

    # For small output size reductions uses cross thread-block synchronization to gain more parallelism
    cooperative_reductions = (
        os.environ.get("TORCHINDUCTOR_COOPERATIVE_REDUCTIONS", "0") == "1"
    )

    # used for debugging cooperative reduction codegen, always generate cooperative_reductions
    force_cooperative_reductions = False

    # 0: disable
    # 1/True: enable, use tuning to pick between different subkernels
    # 2: enable, force using persistent reduction (for debugging)
    # 3: enable, force using non-persistent reduction (for debugging)
    multi_kernel: Literal[0, 1, 2, 3] = int(
        os.environ.get("TORCHINDUCTOR_MULTI_KERNEL", "0")
    )  # type: ignore[assignment]

    # hint to Triton when arguments are divisible by 16
    divisible_by_16 = os.environ.get("TORCHINDUCTOR_DIVISIBLE_BY_16", "1") == "1"

    # Minimum R0_BLOCK to be used for a TritonSplitScanKernel
    # NOTE: This also indirectly controls the size of workspace buffer required
    min_split_scan_rblock = 256

    # Store the generated cubin files for cpp wrapper code to load
    store_cubin = False

    # the max number of spills we allow for the configs we benchmark.
    # Setting this to 0 means we skip a config if it spills even a single
    # register.
    # Setting it to a larger value allows a config spilling a small amount
    # of registers being benchmarked.
    #
    # NOTE: triton will always report >0 register spills for kernels using sin/cos.
    # (check this issue https://github.com/triton-lang/triton/issues/1756 )
    # So far we see a fixed 8 spilled registers for kernels using sin/cos.
    # Raise the threshold to 16 to be safe.
    # We should revisit this once we understand more of the source of register spills.
    spill_threshold: int = 32 if torch.version.hip else 16

    # Generate code containing the newer tl.make_block_ptr() API for loads/store
    use_block_ptr = False

    # (Experimental)
    # Generate code using the tl.make_tensor_descriptor() API for loads/store
    # [Note: TMA API Restrictions] Currently the TMA API requires the following:
    # - For Nvidia GPUs, the compute capability should be >= 9.0
    # - The innermost stride of a descriptor should be 1
    # - The size of the block shape in the innermost dimension should load / store
    #   at least 16 bytes.
    # - Tensors are 16 byte aligned. Enabling this option therefore requires
    #   assume_aligned_inputs to also be enabled
    # TMA descriptors are only going to be generated if the above conditions
    # can be satisfied, along with any existing requirements for index expressions
    use_tensor_descriptor = False

    # Inject a bug into our relu implementation; useful for testing our repro
    # extraction and minification functionality.
    # Valid values: "compile_error", "runtime_error", "accuracy"
    inject_relu_bug_TESTING_ONLY: Optional[str] = None

    # Whether to upcast float16 / bfloat16 to float32 in triton codegen (Experimental)
    codegen_upcast_to_fp32 = True

    # Whether persistent matmul kernels should be enabled this flag only has effect when on h100
    # with a version of triton new enough to support TMA
    enable_persistent_tma_matmul = (
        os.environ.get("ENABLE_PERSISTENT_TMA_MATMUL", "0") == "1"
    )
    # Should TMA store be enable from templates. TODO: Remove once we
    # can autotune over the result.
    enable_template_tma_store = os.environ.get("ENABLE_TEMPLATE_TMA_STORE", "0") == "1"
    # Use epilogue subtiling. We allow disabling it due to limited B200 testing.
    enable_epilogue_subtiling = os.environ.get("ENABLE_EPILOGUE_SUBTILING", "1") == "1"
    # Skip L1 cache for buffers that are used only once.  Disabled by default
    skip_l1_cache = os.environ.get("TORCHINDUCTOR_SKIP_L1", "0") == "1"

    # During autotuning, if one of the kernels/configs fails for some reason,
    # Inductor will usually skip it (and assign its latency to inf).
    # For testing it's helpful to be able to assert that none of the configs fail.
    # Note: it may also need to be used with config.compile_threads = 1
    disallow_failing_autotune_kernels_TESTING_ONLY = False

    # specify number of splits to autotune on for decompose_k. 0 disables decompose_k
    num_decompose_k_splits = int(
        os.environ.get("TORCHINDUCTOR_NUM_DECOMPOSE_K_SPLITS", "10")
    )

    # specify minimum ratio of K to M AND N in order to autotune on decompose_k. 0 enables
    # it as an autotuning choice for all matmuls
    decompose_k_threshold = int(
        os.environ.get("TORCHINDUCTOR_DECOMPOSE_K_THRESHOLD", "32")
    )

    # Programmatic Dependent Launch improves launch latency on Nvidia Hopper+ devices
    # If set to true, will generate PDL code on devices that support it.
    # If set to false, will never generate PDL code.
    enable_pdl = False

    mix_order_reduction = (
        os.environ.get("TORCHINDUCTOR_MIX_ORDER_REDUCTION", "0" if is_fbcode() else "1")
        == "1"
    )
    mix_order_reduction_initial_xblock = 1

    mix_order_reduction_split_size: Optional[int] = None
    mix_order_reduction_autotune_split_size = (
        os.environ.get("TORCHINDUCTOR_MIX_ORDER_REDUCTION_AUTOTUNE_SPLIT_SIZE", "0")
        == "1"
    )


class aot_inductor:
    """
    Settings for Ahead-Of-Time Inductor Compilation
    """

    # AOTInductor output path
    # If an absolute path is specified, the generated lib files will be stored under the directory;
    # If a relative path is specified, it will be used as a subdirectory under the default caching path;
    # If not specified, a temp directory will be created under the default caching path.
    # If the specified path contains something like "model.so", the sub-string will be used
    # to name the generated library.
    output_path = ""

    debug_compile = os.environ.get("AOT_INDUCTOR_DEBUG_COMPILE", "0") == "1"
    debug_symbols = os.environ.get("AOT_INDUCTOR_DEBUG_SYMBOLS", "0") == "1"

    # Annotate generated main wrapper function, i.e. AOTInductorModel::run_impl,
    # to use which cpp compiler optimization level, default to O1
    compile_wrapper_opt_level = os.environ.get(
        "AOT_INDUCTOR_COMPILE_WRAPPER_OPT_LEVEL", "O1"
    )

    # option for debug printing/saving for intermediate tensor values for aot inductor
    # 0: disable debug dumping
    # 1: enable saving intermediate tensor values
    # 2: enable printing intermediate tensor values
    # 3: enable printing kernel names only (useful for pinpointing troublesome kernels)
    debug_intermediate_value_printer: Literal["0", "1", "2", "3"] = os.environ.get(
        "AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER", "0"
    )  # type: ignore[assignment]

    # filtered nodes to be printed for debug values. Specify this option when debug_intermediate_value_printer is set to 2
    filtered_kernel_names = os.environ.get(
        "AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT", None
    )

    # Serialized tree spec for flattening inputs
    # TODO: Move this into metadata
    serialized_in_spec = ""

    # Serialized tree spec for flattening outputs
    # TODO: Move this into metadata
    serialized_out_spec = ""

    # flag to decide whether to create a submodule for constant graph.
    use_runtime_constant_folding: bool = False

    # flag to force weight to be appended to the shared library and mapped by the runtime
    # rather than embedded into the data section. Needed to support 1B+ parameter models
    force_mmap_weights: bool = False

    # Default value of use_consts_asm_build is True, it will build by assembly language.
    # When the value is False, it will build by c++ language.
    use_consts_asm_build = True

    package: bool = False
    package_cpp_only: Optional[bool] = None

    # If package_cpp_only is True, whether cpp files will be compiled to a
    # dynamically linked library or static linked library
    dynamic_linkage: bool = True

    # Dictionary of metadata users might want to save to pass to the runtime.
    # TODO: Move this somewhere else, since it's no longer really a config
    metadata: dict[str, str] = {}

    # fbcode only. Whether to raise error if C++ codegen is too big to optimize
    raise_error_on_ignored_optimization: bool = (
        os.environ.get("AOTINDUCTOR_RAISE_ERROR_ON_IGNORED_OPTIMIZATION", "1") == "1"
    )

    # dump an aoti minifier if program errors
    dump_aoti_minifier: bool = os.environ.get("DUMP_AOTI_MINIFIER", "0") == "1"

    # Compiler compilation debug info
    # 1: Dumps the original graph out to repro.py if compilation fails
    # 2: Dumps a minifier_launcher.py if aoti fails.
    # 3: Always dumps a minifier_launcher.py. Good for segfaults.
    # 4: Dumps a minifier_launcher.py if the accuracy fails.
    repro_level: int = int(os.environ.get("AOTINDUCTOR_REPRO_LEVEL", 2))

    # Dictionary of presets that can be passed in
    presets: dict[str, Any] = {}

    # Kill switch for allowing temporary tensors to be allocated as stack arrays. Tests
    # should be run with this flag both on and off to make sure we have coverage.
    allow_stack_allocation: bool = False

    # Enables an alternate DSO interface (the "minimal ArrayRef interface") intended
    # to maximize performance for use cases that it can accommodate at the expense of
    # generality. In brief:
    # - inputs and outputs are ArrayRefTensor<T> (note that strides are required, but the
    #   tensor must be contiguous)
    # - constant handling is unchanged because it is not a per-inference-iteration bottleneck
    #
    # When the DSO is generated in this mode, the usual interface will also be supported,
    # but performance for that interface may be degraded.
    use_minimal_arrayref_interface: bool = False

    # Set to True if we want to use Pytorch's CUDACachingAllocator for weight management
    weight_use_caching_allocator: bool = (
        os.environ.get("AOT_INDUCTOR_WEIGHT_USE_CACHING_ALLOCATOR", "0") == "1"
    )

    # Experimental. Flag to control whether to include weight in .so
    # Not supported for cross_target_platform="windows".
    package_constants_in_so: bool = True

    # Experimental. Flag to control whether to package weight separately on disk and which
    # format to package it in.
    # Options:
    # None:
    #       Do not package weight separately on disk.
    # "pickle_weights":
    #       Each weight is pickled and stored separately in data/weights. We also store the
    #       FQN names of each weight in a weights_config.json in each model's data/aot_inductor/model folder.
    #       Can only be load back from python using torch._inductor.aoti_load_package API now.
    # "binary_blob":
    #       Stores all weights in a single binary blob in data/aot_inductor/model folder for each model.
    #       This option and config.aot_inductor.force_mmap_weights cannot both be True
    package_constants_on_disk_format: Optional[str] = None

    # Experimental.  Controls automatic precompiling of common AOTI include files.
    precompile_headers: bool = not is_fbcode()

    # Embed generated kernel binary files into model.so
    embed_kernel_binary: Optional[bool] = None

    # Generate kernel files that support multiple archs
    # For CUDA, this means generating fatbin files for kernels, and the fatbin files
    # contains PTX and SASS for the current architecture.
    emit_multi_arch_kernel: Optional[bool] = None

    # If not None, the generated files with use this name in file stem.
    # If None, we will use a hash to name files.
    #
    # If package_cpp_only, this name is also used for the target name in CMakelists.txt
    # The default target name is "aoti_model"
    #
    # If compile_standalone, the aoti model class name is f"AOTInductorModel{name}"
    #
    # This name can only contain letters, numbers, and underscores.
    model_name_for_generated_files: Optional[str] = None

    # Custom ops that have implemented C shim wrappers, defined as an op to C shim declaration dict
    custom_ops_to_c_shims: dict[torch._ops.OpOverload, list[str]] = {}
    # custom op libs that have implemented C shim wrappers
    custom_op_libs: Optional[list[str]] = None

    # Whether to enable link-time-optimization
    enable_lto = os.environ.get("AOT_INDUCTOR_ENABLE_LTO", "0") == "1"

    # Whether the compiled .so should link to libtorch
    link_libtorch: bool = True

    # Currently the only valid option is "windows".
    # We'll use x86_64-w64-mingw32-gcc to cross-compile a .dll file
    # If using cuda, you also need to set WINDOWS_CUDA_HOME env var
    # to point to windows CUDA toolkit.
    # Example: WINDOWS_CUDA_HOME=cuda-windows-base/cuda_cudart/cudart/
    # The path should contain lib cuda and lib cudart
    cross_target_platform: Optional[str] = None

    # If link_libtorch is False and cross_target_platform is windows,
    # a library needs to be provided to provide the shim implementations.
    aoti_shim_library: Optional[str | list[str]] = None
    aoti_shim_library_path: Optional[str] = None


# a convenient class that automatically sets a group of the configs in aot_inductor
# it should only control the flags in aot_inductor.
# it should not do anything else.
class aot_inductor_mode:
    # dynamic_linkage=False
    # link_libtorch=False
    # package_cpp_only=True
    # embed_kernel_binary=True
    # emit_multi_arch_kernel=True
    compile_standalone: bool = False


class cuda:
    """Settings for cuda backend, today this consists of cutlass"""

    # CUDA arch to use for CUDA template kernel compilation.
    # e.g. "70", "75", "80", "90", etc.
    # When arch is None, Inductor uses torch.cuda.get_device_capability(0).
    arch: Optional[str] = None

    # CUDA version to use for CUDA template kernel compilation.
    # e.g. "11.4", "12.1", etc.
    # When version is None, Inductor uses torch.version.cuda.
    version: Optional[str] = None

    # Optimization level for the host compiler.
    compile_opt_level: Literal["-O0", "-O1", "-O2", "-O3", "-OS"] = "-O1"

    # Whether to enable device LTO (link-time-optimization).
    enable_cuda_lto = False

    # Whether to keep intermediate files dring compilation.
    enable_ptxas_info = False

    # Whether to enable debug info, e.g. line number, cutlass debug info.
    enable_debug_info = False

    # Whether to use fast math.
    use_fast_math = False

    # Path to the CUTLASS repo root directory.
    # The default path only works under PyTorch local development environment.
    cutlass_dir = os.path.realpath(
        os.environ.get(
            "TORCHINDUCTOR_CUTLASS_DIR",
            os.path.join(os.path.dirname(torch.__file__), "../third_party/cutlass/"),
        )
    )

    # Configures the maximum number of CUTLASS configs to profile in max_autotune.
    # By default it's None, so that all CUTLASS configs are tuned.
    # This is mainly used to reduce test time in CI.
    cutlass_max_profiling_configs: Optional[int] = None

    # The L2 swizzle values to consider when profiling CUTLASS configs in max_autotune.
    cutlass_max_profiling_swizzle_options: list[int] = [1, 2, 4, 8]

    # Whether to use CUTLASS EVT for epilogue fusion
    cutlass_epilogue_fusion_enabled = (
        os.environ.get("CUTLASS_EPILOGUE_FUSION", "0") == "1"
    )

    # Whether to only use TMA-compatible kernels in CUTLASS
    cutlass_tma_only = False

    # Path to CUDA NVCC.
    # NVCC search order:
    # 1) cuda_cxx set in this config
    # 2) CUDACXX environment variable
    # 3) CUDA_HOME environment variable
    # 4) default system search PATH.
    cuda_cxx: Optional[str] = None

    # Minimum value of M*N*K to consider the CUTLASS backend for GEMM ops.
    cutlass_backend_min_gemm_size: int = 1

    # enable generation of inline standalone runner in CUDA CPP generated code
    # which allows to compile the generated code into a standalone executable.
    generate_test_runner: bool = (
        os.environ.get("INDUCTOR_CUDA_BACKEND_GENERATE_TEST_RUNNER_CODE", "0") == "1"
    )

    # Keep only Cutlass op configs which contain this regular expression pattern
    # Set this to "warpspecialized_cooperative_epi_tma" to enable only SM90 TMA Cutlass Kernels for large GEMMs
    cutlass_op_allowlist_regex: Optional[str] = os.environ.get(
        "TORCHINDUCTOR_CUTLASS_ALLOWLIST"
    )

    # Note: Names of Cutlass ops names can be obtained by calling
    # op.configuration_name() on a Cutlass op instance, for example those
    # returned from cutlass_utils.gen_ops() or the op argument passed to
    # CUTLASSGemmTemplate.render(...)

    # Filter Cutlass configs which contain this regular expression pattern
    # Set this to "pingpong" to avoid numerical issues
    # caused by the op ordering of the "pingpong" memory access
    # pattern used by some Cutlass Kernels.
    cutlass_op_denylist_regex: Optional[str] = os.environ.get(
        "TORCHINDUCTOR_CUTLASS_DENYLIST"
    )

    # Non-negative integer which determines how many kernels are instantiated.
    # 0 = 0000 generates the fewest kernels, 9999 generates all possible combinations.
    # increasing first digit reduces schedule / mixed type pruning,
    # increasing second digit generates more cluster sizes,
    # increasing third digit generates more MMA multipliers,
    # increasing fourth digit generates more instruction shapes.
    cutlass_instantiation_level: str = os.environ.get(
        "TORCHINDUCTOR_CUTLASS_INSTANTIATION_LEVEL", "0"
    )

    # use compile command to create kernel .cu and .so name
    cutlass_hash_with_compile_cmd: bool = (
        os.environ.get("TORCHINDUCTOR_CUTLASS_HASH_WITH_COMPILE_CMD", "0") == "1"
    )

    # Experimental. Prescreen top x configs before tuning on swizzle.
    cutlass_prescreening: bool = (
        os.environ.get("TORCHINDUCTOR_CUTLASS_PRESCREENING", "1") == "1"
    )

    # Specify which operations should use CUTLASS backend
    # Comma-separated list like "mm,addmm,bmm", "all" for all operations, and "" for none.
    # Acceptable operations: mm, int_mm, addmm, sparse_semi_structured_mm, bmm, scaled_mm
    cutlass_enabled_ops: str = os.environ.get(
        "TORCHINDUCTOR_CUTLASS_ENABLED_OPS", "all"
    )

    # Whether to consult the binary remote cache
    use_binary_remote_cache: bool = True

    # Whether to upload compiled kernels to remote cache
    upload_to_binary_remote_cache: bool = False

    # Whether to force upload if the key already exists
    # Use this to overwrite and handle cache pollution
    binary_remote_cache_force_write: bool = False

    # Enable caching codegen of cuda templates.
    enable_caching_codegen: bool = True


class rocm:
    # Offload arch list for device code compilation, e.g. ["gfx90a", "gfx942"].
    # If empty, the `native` arch is used
    arch: list[str] = []

    # Enable the CK backend for CDNA2 and CDNA3 only (for now)
    # Processor name reference: https://llvm.org/docs/AMDGPUUsage.html#processors
    ck_supported_arch: list[Literal["gfx90a", "gfx942", "gfx950"]] = [
        "gfx90a",
        "gfx942",
        "gfx950",
    ]

    # Optimization level, use to balance compilation speed and runtime performance.
    # The type will not necessarily be comprehensive and won't be enforced at runtime.
    compile_opt_level: Literal[
        "-O0", "-O1", "-O2", "-O3", "-Os", "-Oz", "-Omin", "-Ofast", "-Omax"
    ] = "-O2"

    # Flag to keep debug information in compiled objects
    is_debug = False

    # Flag to keep intermediate files (assembly listings, preprocessed sources, etc.)
    save_temps = False

    # Flag to add `-ffast-math`` to compile flags
    use_fast_math = True

    # Flag to add `-fgpu-flush-denormals-to-zero` to compile flags
    flush_denormals = True

    # Flag to print register and LDS usage during compilation
    print_kernel_resource_usage = False

    # Path to ROCm installation, if None, use env variable ROCM_HOME.
    # In fbcode see triton/fb/TARGETS for how ROCM_HOME gets set.
    rocm_home: Optional[str] = None

    # Path to Composable Kernel library.
    # Install with `pip install git+https://github.com/rocm/composable_kernel@develop`.
    ck_dir = os.environ.get("TORCHINDUCTOR_CK_DIR")

    # generate standalone executables for instances generated with the CK backend
    generate_test_runner: bool = (
        os.environ.get("INDUCTOR_CK_BACKEND_GENERATE_TEST_RUNNER_CODE", "0") == "1"
    )

    # Deprecated, use CK and/or CK-tile specific settings
    n_max_profiling_configs: Optional[int] = None

    # Number of op instance choices to trade off between runtime perf and compilation time
    # For CK Kernels
    ck_max_profiling_configs: Optional[int] = None

    # Number of op instance choices to trade off between runtime perf and compilation time
    # For CK-Tile Kernels
    ck_tile_max_profiling_configs: Optional[int] = None

    # Flag to use a short list of CK instances which perform well across a variety of shapes.
    # Currently RCR and F16 only
    use_preselected_instances: bool = False

    # List to determine kBatch parameters to sweep over. By default, we calculate one in splitK
    # scenarios, and run on kBatch=1 in non-splitK scenarios
    kBatch_sweep: Optional[list[int]] = None

    # The threshold at which we trigger a splitK config - K // max(M,N) has to be greater than this
    split_k_threshold: int = 16

    # The threshold at which we trigger a contiguous subgraph transformation
    contiguous_threshold: int = 16


# Backend to use for CPU codegen either "cpp" or "triton" (experimental) or "halide" (experimental) or "pallas" (experimental)
cpu_backend: Literal["cpp", "triton", "halide", "pallas"] = "cpp"

# Backend to use for CUDA codegen either
# "triton", "halide" (experimental) or "pallas" (experimental)
cuda_backend: Literal["triton", "halide", "pallas"] = "triton"

# Backend to use for XPU codegen either "triton"
xpu_backend: Literal["triton"] = "triton"


class halide:
    # Base halide target to use for CPU devices
    cpu_target = "host"

    # Base halide target to use for CUDA devices
    gpu_target = "host-cuda"

    # Halide autoscheduler to use, choices are:
    # "Anderson2021" (gpu-only), "Li2018", "Adams2019" (cpu-only), or "Mullapudi2016" (cpu-only)
    scheduler_cuda: Literal["Anderson2021", "Li2018", "Adams2019", "Mullapudi2016"] = (
        "Anderson2021"
    )
    scheduler_cpu: Literal["Anderson2021", "Li2018", "Adams2019", "Mullapudi2016"] = (
        "Adams2019"
    )

    # Controls `no_asserts` flag passed to Halide target (warning: can false positive)
    asserts = False

    # Controls `debug` flag passed to Halide target
    debug = False

    # Enable (or fallback on) scan kernels such as cumsum
    # Halide autoschedulers struggle with these kernels
    scan_kernels = False


# create a directory containing lots of debug information
class trace:
    # master switch for all debugging flags below
    enabled = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    # save real tensors
    save_real_tensors = os.environ.get("TORCH_COMPILE_DEBUG_SAVE_REAL", "0") == "1"

    # Save debug information to a temporary directory
    # If not specified, a temp directory will be created by system
    debug_dir: Optional[str] = None

    # Save python logger call >=logging.DEBUG
    debug_log = False

    # Save python logger call >=logging.INFO
    info_log = False

    # Save input FX graph (post decomps, pre optimization)
    fx_graph = True

    # Save FX graph after transformations
    fx_graph_transformed = True

    # Save TorchInductor IR before fusion pass
    ir_pre_fusion = True

    # Save TorchInductor IR after fusion pass
    ir_post_fusion = True

    # Copy generated code to trace dir
    output_code = True

    # SVG figure showing post-fusion graph
    graph_diagram = os.environ.get("INDUCTOR_POST_FUSION_SVG", "0") == "1"

    # SVG figure showing fx with fusion
    draw_orig_fx_graph = os.environ.get("INDUCTOR_ORIG_FX_SVG", "0") == "1"

    # We draw our fx graphs with the "record" shape attribute by default.
    # Sometimes, when the graph is very complex, we may hit dot errors like below:
    #   "flat edge between adjacent nodes one of which has a record shape -
    #    replace records with HTML-like labels"
    # and thus fail to generate a graph. So, let's give the user an option
    # to specify the shape attribute for the dot graph. For example, passing
    # INDUCTOR_DOT_GRAPH_SHAPE_SVG = "none" would let us generate HTML-like labels
    # to workaround the above failure.
    dot_graph_shape = os.environ.get("INDUCTOR_DOT_GRAPH_SHAPE_SVG", None)

    # If not None, this is the URL that saves the SVG files of the input/output
    # graph of each pass that changed the graph
    # The nodes that are being transformed in each pass will be colored in yellow
    # URL only supports local directory for now
    log_url_for_graph_xform = os.environ.get("INDUCTOR_LOG_URL_FOR_GRAPH_XFORM", None)

    # Store cProfile (see snakeviz to view)
    compile_profile = False

    # Upload the .tar.gz file
    # Needs to be overridden based on specific environment needs
    upload_tar: Optional[Callable[[str], None]] = None

    log_autotuning_results = os.environ.get("LOG_AUTOTUNE_RESULTS", "0") == "1"

    # Save mapping info from inductor generated kernel to post_grad/pre_grad fx nodes
    # Levels:
    #   0 - disabled (default)
    #   1 - normal
    #   2 - basic
    # Backward compatibility:
    #   If TORCH_COMPILE_DEBUG=1, level is set to at least 1.
    #   If INDUCTOR_PROVENANCE is set, use its integer value.
    provenance_tracking_level: int = int(
        os.environ.get(
            "INDUCTOR_PROVENANCE", os.environ.get("TORCH_COMPILE_DEBUG", "0")
        )
    )


_save_config_ignore: list[str] = [
    # workaround: "Can't pickle <function ...>"
    "trace.upload_tar",
    "joint_custom_pre_pass",
    "joint_custom_post_pass",
    "pre_grad_custom_pass",
    "aot_inductor.repro_level",
    "aot_inductor.dump_aoti_minifier",
    "post_grad_custom_pre_pass",
    "post_grad_custom_post_pass",
    "_fuse_ddp_communication_passes",
    "_pre_fusion_custom_pass",
]

_cache_config_ignore_prefix: list[str] = [
    # trace functions are not relevant to config caching
    "trace",
    # uses absolute path
    "cuda.cutlass_dir",
    # not relevant
    "worker_start_method",
    "compile_threads",
    # see CustomGraphPass; these are handled specially
    "post_grad_custom_post_pass",
    "post_grad_custom_pre_pass",
    "joint_custom_pre_pass",
    "joint_custom_post_pass",
    "_fuse_ddp_communication_passes",
    "_pre_fusion_custom_pass",
    # tests assume that changes here don't invalidate cache
    "always_complex_memory_overlap_TESTING_ONLY",
    # cache related options are not relevant to cache results
    "fx_graph_cache",
    "fx_graph_remote_cache",
    "autotune_local_cache",
    "autotune_remote_cache",
]

# External callable for matmul tuning candidates
external_matmul: list[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]] = []

write_are_deterministic_algorithms_enabled = (
    os.getenv("TORCHINDUCTOR_WRITE_ARE_DETERMINISTIC_ALGORITHMS_ENABLED", "1") == "1"
)


class lookup_table:
    # Lookup table for template config overrides
    table: Optional[dict[str, list[dict[str, Any]]]] = None

    # Enable template src_hash checking in lookup table to prevent using stale configs.
    # If True, configs with 'template_hash' field will be compared against the template's
    # src_hash at runtime and filtered out if they don't match. If False, no
    # hash checking is performed.
    check_src_hash: bool = True


class test_configs:
    force_extern_kernel_in_multi_template: bool = False

    max_mm_configs: Optional[int] = None

    runtime_triton_dtype_assert = False
    runtime_triton_shape_assert = False
    static_cpp_dtype_assert = False

    # regex to control the set of considered autotuning
    # choices (aka configs) by name and / or description
    autotune_choice_name_regex: Optional[str] = None
    autotune_choice_desc_regex: Optional[str] = None

    graphsafe_rng_func_ignores_fallback_random = False

    track_memory_lifecycle: Optional[Literal["assert", "log"]] = None

    # If set to True, AOTI-generated CMakelists.txt will still use libtorch
    # for unit testing
    use_libtorch = False

    # Assume bucketing reduces latency (mostly for testing)
    assume_bucketing_reduces_latency: bool = True

    # A test config to ease the test for perf of reduction config filtering
    force_filter_reduction_configs = (
        os.getenv("TORCHINDUCTOR_FORCE_FILTER_REDUCTION_CONFIGS") == "1"
    )

    # a testing config to distort benchmarking result
    # - empty string to disable
    # - "inverse" to inverse the numbers
    # - "random" return a random value
    distort_benchmarking_result = os.getenv(
        "TORCHINDUCTOR_DISTORT_BENCHMARKING_RESULT", ""
    )

    bisect_pre_grad_graph = False
    bisect_keep_custom_backend_for_inductor = False


if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403


# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
