import os  # noqa: C101
import sys
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Union

import torch


def is_fbcode() -> bool:
    return not hasattr(torch.version, "git_version")


def fx_graph_remote_cache_default() -> Optional[bool]:
    if os.environ.get("TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE") == "1":
        return True
    if os.environ.get("TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE") == "0":
        return False
    return None


def autotune_remote_cache_default() -> Optional[bool]:
    if os.environ.get("TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE") == "1":
        return True
    if os.environ.get("TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE") == "0":
        return False
    return None


# add some debug printouts
debug = False

# Whether to disable a progress bar for autotuning
disable_progress = True

# Whether to enable printing the source code for each future
verbose_progress = False

# use fx aot graph codegen cache
fx_graph_cache = (
    os.environ.get("TORCHINDUCTOR_FX_GRAPH_CACHE", "0" if is_fbcode() else "1") == "1"
)

# use remote fx aot graph codegen cache
# False: Disables the cache
# True: Enables the cache
# None: Not set -- Off for OSS, JustKnobs based for internal
fx_graph_remote_cache: Optional[bool] = fx_graph_remote_cache_default()

# enable autotune local cache
autotune_local_cache = True

# enable autotune remote cache
# False: Disables the cache
# True: Enables the cache
# None: Not set -- Off for OSS, JustKnobs based for internal
autotune_remote_cache: Optional[bool] = autotune_remote_cache_default()

# Force disabled all inductor level caching -- This will override any other caching flag
force_disable_caches = os.environ.get("TORCHINDUCTOR_FORCE_DISABLE_CACHES") == "1"

# sleep in inductor for testing
sleep_sec_TESTING_ONLY: Optional[int] = None

# use cpp wrapper instead of python wrapper
cpp_wrapper = os.environ.get("TORCHINDUCTOR_CPP_WRAPPER", "0") == "1"

# codegen cpp wrapper code in an ABI compatible mode
abi_compatible = (
    os.environ.get("TORCHINDUCTOR_ABI_COMPATIBLE", "1" if is_fbcode() else "0") == "1"
)

c_shim_version = os.environ.get("TORCHINDUCTOR_C_SHIM_VERSION", "2")

# dead code elimination
dce = False

# assume weight tensors are fixed size
static_weight_shapes = True

# put correctness assertions in generated code
size_asserts = os.environ.get("TORCHINDUCTOR_SIZE_ASSERTS", "1") == "1"
nan_asserts = os.environ.get("TORCHINDUCTOR_NAN_ASSERTS") == "1"

# enable loop reordering based on input orders
pick_loop_orders = True

# reuse a kernel input as the output
inplace_buffers = True

# reuse a buffer for an unrelated purpose
allow_buffer_reuse = True

# Enable pooled allocations for non-output tensors
memory_planning = os.environ.get("TORCHINDUCTOR_MEMORY_PLANNING", "0") == "1"

# How to organize memory under memory_planning=True:
# - "none": do not try to pool storage, just reuse
# - "intermediates": all non-outputs share storage, outputs each get unique storage
# - "outputs": two pools, one for intermediates (freed on return) and one for outputs
# - "combined": a single pool for both intermediates and outputs
memory_pool = os.environ.get("TORCHINDUCTOR_MEMORY_POOL", "intermediates")

# codegen benchmark harness
benchmark_harness = True

# fuse pointwise into templates
epilogue_fusion = True

# do epilogue fusions before other fusions
epilogue_fusion_first = False

# enable pattern match+replace optimizations
pattern_matcher = True

# set to True to enable the back-to-back GEMM pass
b2b_gemm_pass = False

# register custom graph optimization pass hook. so far, pre/post passes are
# only applied before/after pattern_matcher in post_grad_passes.
#
# def my_custom_pre_pass(graph: torch.fx.graph.Graph):
#     # my custom graph optimization pass
#     ...
#
# def my_custom_post_pass(graph: torch.fx.graph.Graph):
#     # my custom graph optimization pass
#     ...
#
# torch._inductor.config.post_grad_custom_pre_pass = my_custom_pre_pass
# torch._inductor.config.post_grad_custom_post_pass = my_custom_post_pass
post_grad_custom_pre_pass: Optional[Callable[[torch.fx.graph.Graph], None]] = None
post_grad_custom_post_pass: Optional[Callable[[torch.fx.graph.Graph], None]] = None

# Registers a custom joint graph pass.
joint_custom_pre_pass: Optional[Callable[[torch.fx.Graph], None]] = None
joint_custom_post_pass: Optional[Callable[[torch.fx.Graph], None]] = None

# Registers a custom pregrad pass. Note that the pre-grad IR is 1.
# non-functional, 2. non-normalized, and 3. prone to change. Ideally we should
# use post-grad passes.
pre_grad_custom_pass: Optional[Callable[[torch.fx.graph.Graph], None]] = None

# Registers a custom pass to be run right before fusion in Inductor scheduler.
# WARNING: Inductor scheduler IR is at prototype stage and subject to change,
# hence custom IR passes built on top of it might break in the future.
_pre_fusion_custom_pass: Optional[
    Callable[
        [List["torch._inductor.scheduler.BaseSchedulerNode"]],
        List["torch._inductor.scheduler.BaseSchedulerNode"],
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
pre_grad_fusion_options: Dict[str, Dict[str, Any]] = {
    "batch_linear": {},
    "batch_linear_lhs": {},
    "batch_layernorm": {},
    "batch_tanh": {},
    "batch_relu": {},
    "batch_sigmoid": {},
}

# Post grad fusion and options, set to empty dict to disable fusion.
# Call `torch._inductor.fx_passes.group_batch_fusion.list_group_batch_fusions(False)` to see available fusions.
post_grad_fusion_options: Dict[str, Dict[str, Any]] = {}

# enable reordering pass for improving memory locality
reorder_for_locality = True

# Scale down RBLOCK for better occupancy
dynamic_scale_rblock = os.environ.get("TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK", "1") == "1"

# this forces fusion for int_mm with mul. Needed when you want to avoid realizing the int32
# but the mul gets fused with other pointwise ops instead.
force_fuse_int_mm_with_mul = False

# for pattern torch.mm(a, b.to(dtype)) with cuda tensors,
# enable torch._inductor.kernel.mm.tuned_mixed_mm fused kernel.
# Autotune will compare perf with normal cast->then->mm option
use_mixed_mm = True

# enable runtime numeric check for pre/post grad fx passes
# floating point provides limited accuracy (about 7 decimal digits for single precision
# floating point numbers,about 16 decimal digits for double precision floating point numbers)
# according to PyTorch documentation.
# https://pytorch.org/docs/stable/notes/numerical_accuracy.html#batched-computations-or-slice-computations
fx_passes_numeric_check: Dict[str, Any] = {
    "pre_grad": False,
    "precision": 1e-4,
    "num_iterations": 1,
    "requires_optimizer": True,
}

# mixed_mm_choice can be used to control the behaviour for pattern torch.mm(a, b.to(dtype)) with cuda tensors.
# The fallback aten implementation is normal cast->then->mm option.
# If mixed_mm_choice is "default": this flag will be ignored.
# If mixed_mm_choice is "triton":
# - Always use torch._inductor.kernel.mm.tuned_mixed_mm's fused kernel.
# - Autotune will not compare with fallback.
# If mixed_mm_choice is "aten": always use the fallback aten implementation.
# If mixed_mm_choice is "heuristic":
# - Enables the heuristic.
# - If the heuristic decides to add a config, it will add the config as the first choice.
# - If autotune is disabled, this config will always be chosen.
# - If autotune is enabled, it will also compare with fallback aten implementation and fused kernel.
# The use_mixed_mm flag will be ignored if mixed_mm_choice != "default".
mixed_mm_choice = "heuristic"

# enable reordering pass for increasing overlap between compute and communication
reorder_for_compute_comm_overlap = False

# passes (in execution order) for increasing overlap between compute and communication
# for built-in passes, use string name; for user-defined passes, pass in the function handle
# WARNING: Inductor scheduler IR is at prototype stage and subject to change,
# hence custom IR passes built on top of it might break in the future.
reorder_for_compute_comm_overlap_passes = [
    "reorder_compute_for_overlap",
    "sink_waits",
    "raise_comms",
]

# runtime estimation function for ops
# for built-in estimation function, pass in "default"; for user-defined estimation function, pass in the function handle
estimate_op_runtime = "default"

# unit: GB/s, uni-directional P2P bandwidth per card
# default value is NVLink
intra_node_bw = 300

# unit: GB/s, uni-directional P2P bandwidth per node
# default value is InfiniBand
inter_node_bw = 25

# enable slow autotuning passes to select algorithms
max_autotune = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"

# enable slow autotuning passes to select pointwise/reductions algorithms
max_autotune_pointwise = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE") == "1"

# enable slow autotuning passes to select gemm algorithms
max_autotune_gemm = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM") == "1"

# force cublas and triton to use the same precision; cublas supports TF32 for matmul operations
# when m, n, k are multiples of 16, 16, 8, whereas triton supports TF32 for matmul operations
# for any combinations of m, n, k, regardless of their alignment. setting this flag will ensure
# that triton does not use TF32 wherever cublas would not use TF32
force_same_precision = (
    True if is_fbcode() else os.environ.get("TORCHINDUCTOR_FORCE_SAME_PRECISION") == "1"
)

# Specify candidate backends for gemm autotune.
# Possible choices are combinations of: ATen, Triton, CUTLASS, CK, CPP.
# ATen: default Pytorch ATen kernels.
# Triton: Triton templates defined in torch inductor (AMD and NVidia GPUs).
# CUTLASS: Cutlass templates and kernels (NVidia GPUs only).
# CK: Composable Kernel templates and kernels (AMD Instinct GPUs only).
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
max_autotune_gemm_search_space = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE", "DEFAULT"
).upper()

# Whether we fall back to ATen or hard error when no matches are found during autotuning
autotune_fallback_to_aten = (
    os.environ.get("TORCHINDUCTOR_AUTOTUNE_FALLBACK_TO_ATEN", "1") == "1"
)

# the value used as a fallback for the unbacked SymInts
# that can appear in the input shapes (e.g., in autotuning)
unbacked_symint_fallback = 8192

# enable searching global and local cache regardless of `max_autotune`
search_autotune_cache = os.environ.get("TORCHINDUCTOR_SEARCH_AUTOTUNE_CACHE") == "1"

save_args = os.environ.get("TORCHINDUCTOR_SAVE_ARGS") == "1"

# We will disable creating subprocess for autotuning if this is False
autotune_in_subproc = os.environ.get("TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC") == "1"

# The following three timeouts are applicable if autotune_in_subproc is True:

# Max time that a a valid benchmark result may take during autotuning
max_autotune_subproc_result_timeout_seconds = 60.0
# Additional time we allow subprocesses to terminate gracefully after the timeout until we send a SIGTERM
max_autotune_subproc_graceful_timeout_seconds = 1.0
# Additional time that we grant after a SIGTERM until we do a hard SIGKILL of subprocesses
max_autotune_subproc_terminate_timeout_seconds = 2.0

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
# generate the learned heursitic to code which is shipped with the compiler
# Specify a list of comma separated optimizations to collect data for
autoheuristic_collect = os.environ.get("TORCHINDUCTOR_AUTOHEURISTIC_COLLECT", "")
# Specify a list of comma separated optimizations to use learned heuristics for
autoheuristic_use = os.environ.get("TORCHINDUCTOR_AUTOHEURISTIC_USE", "mixed_mm")


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

# fallback to eager for random/dropout, this is slow but useful for debugging
fallback_random = False

# automatically create fallbacks when encountering an unhandled op
implicit_fallbacks = True

# fuse even in cases without common reads
aggressive_fusion = False

# For each fused kernel in the wrapper, comment with the nodes that get fused.
# Useful for debugging fusion.
debug_fusion = os.environ.get("TORCHINDUCTOR_DEBUG_FUSION") == "1"
benchmark_fusion = os.environ.get("TORCHINDUCTOR_BENCHMARK_FUSION") == "1"
enabled_metric_tables = os.environ.get("TORCHINDUCTOR_ENABLED_METRIC_TABLES", "")

# For Triton Templates, select fastest of best template + epilogue vs best template + separate epilogue kernel
benchmark_epilogue_fusion = (
    os.environ.get("TORCHINDUCTOR_BENCHMARK_EPILOGUE_FUSION", "1") == "1"
)

# Take how many of the top triton kernels to benchmark epilogue
max_epilogue_benchmarked_choices = 1

# how many nodes to allow into a single fusion
max_fusion_size = 64

# max number of inputs to generate cat as a pointwise op with masked laods
max_pointwise_cat_inputs = 8

# replace small reductions with pointwise, disable with `= 1`
unroll_reductions_threshold = 8

# Add extra comments to output code (causes compile cache misses)
comment_origin = False

# Convert 1x1 convs into matmuls
conv_1x1_as_mm = False

# Enable split reductions for better utilization when the dimension
# being reduced over is large (by splitting it)
split_reductions = True

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

# constant folding on the joint graph
joint_graph_constant_folding = True

# Enable indirect_indexing asserts for decompositions and lowerings
debug_index_asserts = False

# Mode to emulate pytorch eager numerics for lower precision (fp16, bf16)
# Pytorch eager computes bf16/fp16 by upcasting inputs to fp32 and downcasting after
# For multiple, fused pointwise nodes, inductor will elide the intermediary upcasts and downcasts
# Typically this should be closer to fp64 ref numerics. However, it can be useful for debugging
# to emulate the eager numerics.
emulate_precision_casts = False

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


# The multiprocessing start method to use for inductor workers in the codecache.
# Can be "subprocess" or "fork".
def decide_worker_start_method() -> str:
    start_method = os.environ.get(
        "TORCHINDUCTOR_WORKER_START", "fork" if is_fbcode() else "subprocess"
    )
    assert start_method in (
        "subprocess",
        "fork",
    ), f"Invalid start method: {start_method}"
    return start_method


worker_start_method = decide_worker_start_method()

# Flags to turn on all_reduce fusion. These 2 flags should be automaticaly turned
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
_fuse_ddp_communication_passes: List[Union[Callable[..., None], str]] = [
    "fuse_ddp_with_concat_op",
    "schedule_comm_wait",
]

_micro_pipeline_tp: bool = False


def decide_compile_threads() -> int:
    """
    Here are the precedence to decide compile_threads
    1. User can override it by TORCHINDUCTOR_COMPILE_THREADS.  One may want to disable async compiling by
       setting this to 1 to make pdb happy.
    2. Set to 1 if it's win32 platform
    3. decide by the number of CPU cores
    """
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        return int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
    elif sys.platform == "win32":
        return 1
    elif is_fbcode():
        return 1
    else:
        cpu_count = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count()
        )
        assert cpu_count
        return min(32, cpu_count)


compile_threads = decide_compile_threads()

# gemm autotuning global cache dir
if is_fbcode():
    from libfb.py import parutil

    try:
        if __package__:
            global_cache_dir = parutil.get_dir_path(
                os.path.join(__package__.replace(".", os.sep), "fb/cache")
            )
        else:
            global_cache_dir = parutil.get_dir_path("fb/cache")
    except ValueError:
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
profile_bandwidth_output = os.environ.get("TORCHINDUCTOR_PROFILE_OUTPUT", None)

# TODO: remove later
disable_cpp_codegen = False


# Freezing will attempt to inline weights as constants in optimization
# and run constant folding and other optimizations on them. After freezing, weights
# can no longer be updated.
freezing: bool = os.environ.get("TORCHINDUCTOR_FREEZING", "0") == "1"

# Make freezing invalidate the eager Parameters of nn modules, to avoid memory overhead
# of potentially keeping multiple copies of weights.
freezing_discard_parameters: bool = False

# Kill switch for allowing temporary tensors to be allocated as stack arrays. Tests
# should be run with this flag both on and off to make sure we have coverage.
allow_stack_allocation: bool = (
    os.environ.get("TORCHINDUCTOR_STACK_ALLOCATION", "1" if is_fbcode() else "0") == "1"
)

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


# config specific to codegen/cpp.py
class cpp:
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
    min_chunk_size = int(os.environ.get("TORCHINDUCTOR_CPP_MIN_CHUNK_SIZE", "4096"))
    cxx = (
        None,  # download gcc12 from conda-forge if conda is installed
        # "g++-12",
        # "g++-11",
        # "g++-10",
        # "clang++",
        os.environ.get("CXX", "clang++" if sys.platform == "darwin" else "g++"),
        # "g++.par",
    )
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
    # force usage as specified, without testing.
    vec_isa_ok: Optional[bool] = None

    # similar to config.triton.descriptive_names
    descriptive_names = "original_aten"

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
    enable_floating_point_contract_flag = (
        os.environ.get("TORCHINDUCTOR_CPP_ENABLE_FLOATING_POINT_CONTRACT_FLAG", "0")
        == "1"
    )

    # Maximal allowed number of slices on K-dim for a GEMM kernel. This controls
    # the maximal parallelism of K-slicing. Since K-slicing requires extra thread
    # synchronization and buffers,  the maximal number of slices is limited to
    # mitigate the sync overhead and memory usage.
    # When set to 0, the number of slices is unlimited.
    gemm_max_k_slices = int(os.environ.get("TORCHINDUCTOR_CPP_GEMM_MAX_K_SLICES", "1"))


# config specific to codegen/triton.py
class triton:
    # Use cudagraphs on output code
    cudagraphs = os.environ.get("TORCHINDUCTOR_CUDAGRAPHS") == "1"

    # Use cudagraph trees for memory pooling if `cudagraphs` is True
    cudagraph_trees = True

    # Should we skip cudagraphing graphs with dynamic shape inputs
    # If False, we will re-record a graph for each unique set of shape inputs
    cudagraph_skip_dynamic_graphs = False

    # assertions not on the fast path, steady state
    slow_path_cudagraph_asserts = True

    # TODO - need to debug why this prevents cleanup
    cudagraph_trees_history_recording = False

    # Enable cudagraph support for mutated inputs from prior cudagraph pool
    cudagraph_support_input_mutation = False if is_fbcode() else True

    # Maximal number of allowed cudagraph re-record for a function and
    # a cudagraph node due to static input tensor address changes or
    # cudagraph managed tensor data pointer changed.
    # i.e., allow num_recording <= cudagraph_unexpected_rerecord_limit
    # note: we are conservative here and choose a large limit.
    cudagraph_unexpected_rerecord_limit = 128

    # Warn loudly when the number of cudagraphs due to dynamic shape
    # exceeds this limit
    cudagraph_dynamic_shape_warn_limit: Optional[int] = 50

    # synchronize after cudagraph invocation
    force_cudagraph_sync = False

    # always run cudagraphs in the eager warmup stage
    # instead of recording and executing cudagraphs
    force_cudagraphs_warmup = False

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

    # limit tiling dimensions
    max_tiles = 2

    # Prefer higher dimensional tilings. This simplifies indexing expressions, making
    # it easier to identify block pointers.
    prefer_nd_tiling: bool = False

    # use triton.autotune for pointwise ops with complex layouts
    # this should only be disabled for debugging/testing
    autotune_pointwise = True

    # max autotune gemm with cublasLt
    autotune_cublasLt = True

    # Tune the generated Triton kernels at compile time instead of first time they run
    autotune_at_compile_time = False

    # should we stop a fusion to allow better tiling?
    tiling_prevents_pointwise_fusion = True
    tiling_prevents_reduction_fusion = True

    # should we give different names to kernels
    # Note: This is orthogonal to descriptive_names - this is deciding whether
    # our triton kernel names should all be `triton_` (to maximize caching) or
    # whether they should be unique.
    unique_kernel_names = os.environ.get("TORCHINDUCTOR_UNIQUE_KERNEL_NAMES") == "1"

    # should we put op names in kernel names
    # False: No special names (just triton__1, triton__2, etc.)
    # "torch": Maps to the fx op in the Dynamo graph (module name, method name, etc.)
    # "original_aten": Maps to the highest-level aten op (i.e. pre-decompositions)
    # "inductor_node": Maps to the node name in the FX graph passed to Inductor
    descriptive_names = "original_aten"

    # use alternate codegen for smaller reductions
    persistent_reductions = (
        os.environ.get("TORCHINDUCTOR_PERSISTENT_REDUCTIONS", "1") == "1"
    )

    # 0/False: disable
    # 1/True: enable, use tuning to pick between different subkernels
    # 2: enable, force using persistent reduction (for debugging)
    # 3: enable, force using non-persistent reduction (for debugging)
    multi_kernel = int(os.environ.get("TORCHINDUCTOR_MULTI_KERNEL", "0"))

    # hint to Triton when arguments are divisible by 16
    divisible_by_16 = True

    # Minimum RBLOCK to be used for a TritonSplitScanKernel
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
    # (check this issue https://github.com/openai/triton/issues/1756 )
    # So far we see a fixed 8 spilled registers for kernels using sin/cos.
    # Raise the threshold to 16 to be safe.
    # We should revisit this once we understand more of the source of register spills.
    spill_threshold: int = 16

    # Generate code containing the newer tl.make_block_ptr() API for loads/store
    use_block_ptr = False

    # Inject a bug into our relu implementation; useful for testing our repro
    # extraction and minification functionality.
    # Valid values: "compile_error", "runtime_error", "accuracy"
    inject_relu_bug_TESTING_ONLY: Optional[str] = None


class aot_inductor:
    # AOTInductor output path
    # If an absolute path is specified, the generated lib files will be stored under the directory;
    # If a relative path is specified, it will be used as a subdirectory under the default caching path;
    # If not specified, a temp directory will be created under the default caching path.
    # If the specified path contains something like "model.so", the sub-string will be used
    # to name the generated library.
    output_path = ""

    debug_compile = os.environ.get("AOT_INDUCTOR_DEBUG_COMPILE", "0") == "1"

    debug_dump_consts_bin: bool = (
        os.environ.get("AOT_INDUCTOR_DEBUG_DUMP_CONSTS_BIN", "0") == "1"
    )

    # enable debug mode for aot inductor and it will print out more information including the intermediate tensor values, etc
    # for debugging purpose
    debug_intermediate_value_printer = (
        os.environ.get("AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER", "0") == "1"
    )

    # filtered nodes to be printed for debug values. If not set, it will dump all debug tensor value info by default
    filtered_kernel_names = os.environ.get("AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT", "")

    # Serialized tree spec for flattening inputs
    serialized_in_spec = ""

    # Serialized tree spec for flattening outputs
    serialized_out_spec = ""

    # flag to decide whether to create a submodule for constant graph.
    use_runtime_constant_folding: bool = False

    # flag to force weight to be appened to the shared library and mmaped  by the runtime
    # rather than embedded into the data section. Needed to support 1B+ parameter models
    force_mmap_weights: bool = False

    package: bool = False


class cuda:
    # CUDA arch to use for CUDA template kernel compilation.
    # e.g. "70", "75", "80", "90", etc.
    # When arch is None, Inductor uses torch.cuda.get_device_capability(0).
    arch: Optional[str] = None

    # CUDA version to use for CUDA template kernel compilation.
    # e.g. "11.4", "12.1", etc.
    # When version is None, Inductor uses torch.version.cuda.
    version: Optional[str] = None

    # Optimization level for the host compiler.
    compile_opt_level = "-O1"

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
    cutlass_dir = os.environ.get(
        "TORCHINDUCTOR_CUTLASS_DIR",
        os.path.abspath(
            os.path.join(os.path.dirname(torch.__file__), "../third_party/cutlass/")
        ),
    )

    # Configures the maximum number of CUTLASS configs to profile in max_autotune.
    # By default it's None, so that all CUTLASS configs are tuned.
    # This is mainly used to reduce test time in CI.
    cutlass_max_profiling_configs: Optional[int] = None

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
        os.environ.get("INDUCTOR_CUDA_BACKEND_GENERATE_TEST_RUNNER_CODE", "1") == "1"
    )

    # Keep only Cutlass op configs which contain this regular expression pattern
    # Set this to "warpspecialized_cooperative_epi_tma" to enable only SM90 TMA Cutlass Kernels for large GEMMs
    cutlass_op_allowlist_regex: Optional[str] = None

    # Note: Names of Cutlass ops names can be obtained by calling
    # op.configuration_name() on a Cutlass op instance, for example those
    # returned from cutlass_utils.gen_ops() or the op argument passed to
    # CUTLASSGemmTemplate.render(...)

    # Filter Cutlass configs which contain this regular expression pattern
    # Set this to "pingpong" to avoid numerical issues
    # caused by the op ordering of the "pingpong" memory access
    # pattern used by some Cutlass Kernels.
    cutlass_op_denylist_regex: Optional[str] = "pingpong"


class rocm:
    # Offload arch list for device code compilation, e.g. ["gfx941", "gfx942"].
    # If empty, the `native` arch is used
    arch: List[str] = []

    # Enable the CK backend for CDNA2 and CDNA3 only (for now)
    # Processor name reference: https://llvm.org/docs/AMDGPUUsage.html#processors
    ck_supported_arch: List[str] = ["gfx90a", "gfx940", "gfx941", "gfx942"]

    # Optimization level, use to balance compilation speed and runtime performance
    compile_opt_level = "-O2"

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

    # Path to ROCm installation, if None, use env variable ROCM_HOME
    rocm_home: Optional[str] = None

    # Path to Composable Kernel library.
    # Install with `pip install git+https://github.com/rocm/composable_kernel@develop`.
    ck_dir = os.environ.get("TORCHINDUCTOR_CK_DIR")

    # Number of op instance choices to trade off between runtime perf and compilation time
    n_max_profiling_configs: Optional[int] = None

    # Flag to use a short list of CK instances which perform well across a variety of shapes.
    # Currently RCR and F16 only
    use_preselected_instances: bool = False


# Backend to use for CPU codegen either "cpp" or "halide" (experimental)
cpu_backend = "cpp"

# Backend to use for CUDA codegen either "triton" or "halide" (experimental)
cuda_backend = "triton"


class halide:
    # Base halide target to use for CPU devices
    cpu_target = "host"

    # Base halide target to use for CUDA devices
    gpu_target = "host-cuda"

    # Halide autoscheduler to use, choices are:
    # "Anderson2021" (gpu-only), "Li2018", "Adams2019" (cpu-only), or "Mullapudi2016" (cpu-only)
    scheduler_cuda = "Anderson2021"
    scheduler_cpu = "Adams2019"

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
    # INDUCTOR_DOT_GRAPH_SHAPE_SVG = "none" would let us generate HTML-like lables
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
    # Needs to be overriden based on specific environment needs
    upload_tar: Optional[Callable[[str], None]] = None

    log_autotuning_results: bool = False


_save_config_ignore = [
    # workaround: "Can't pickle <function ...>"
    "trace.upload_tar",
    "post_grad_custom_post_pass",
    "post_grad_custom_pre_pass",
    "joint_custom_pre_pass",
    "joint_custom_post_pass",
    "pre_grad_custom_pass",
]

_cache_config_ignore_prefix = [
    # trace functions are not relevant to config caching
    "trace",
    # uses absolute path
    "cuda.cutlass_dir",
    # not relevant
    "compile_threads",
]

if TYPE_CHECKING:
    from torch.utils._config_typing import *  # noqa: F401, F403

from torch.utils._config_module import install_config_module


# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
