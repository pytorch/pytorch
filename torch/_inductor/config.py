import os
import sys

import torch

# add some debug printouts
debug = False

# Whether to disable a progress bar for autotuning
disable_progress = True

# Whether to enable printing the source code for each future
verbose_progress = False

# use cpp wrapper instead of python wrapper
cpp_wrapper = False

# dead code elimination
dce = False

# assume weight tensors are fixed size
static_weight_shapes = True

# put correctness assertions in generated code
size_asserts = os.environ.get("TORCHINDUCTOR_SIZE_ASSERTS", "1") == "1"

# enable loop reordering based on input orders
pick_loop_orders = True

# reuse a kernel input as the output
inplace_buffers = True

# reuse a buffer for an unrelated purpose
allow_buffer_reuse = True

# codegen benchmark harness
benchmark_harness = True

# fuse pointwise into templates
epilogue_fusion = True

# do epilogue fusions before other fusions
epilogue_fusion_first = False

# enable pattern match+replace optimizations
pattern_matcher = True

# register custom graph optimizatin pass hook. so far, pre/post passes are
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
post_grad_custom_pre_pass = None
post_grad_custom_post_pass = None

# Optimize away split cat patterns (Experimental)
split_cat_fx_passes = True

# enable pattern match with group fusion (using fbgemm)
group_fusion = False

# enable pattern match with batch fusion (using torch op)
batch_fusion = True

# enable reordering pass
reordering = True

# for pattern torch.mm(a, b.to(dtype)) with cuda tensors,
# enable torch._inductor.kernel.mm.tuned_mixed_mm fused kernel.
# Autotune will compare perf with normal cast->then->mm option
use_mixed_mm = False

# for pattern torch.mm(a, b.to(dtype)) with cuda tensors, always use
# torch._inductor.kernel.mm.tuned_mixed_mm's fused kernel.
# Autotune will not compare with normal cast->then->mm option.
# (if force_mixed_mm is true, the use_mixed_mm flag will be ignored)
force_mixed_mm = False

# TODO: capture whether the graph is from export
from_export = False

# enable slow autotuning passes to select algorithms
max_autotune = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"

# enable slow autotuning passes to select pointwise/reductions algorithms
max_autotune_pointwise = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE") == "1"

# enable slow autotuning passes to select gemm algorithms
max_autotune_gemm = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM") == "1"

# Specify candidate backends for gemm autotune.
# Possible choices are combinations of: ATen, Triton, CUTLASS.
# ATen: default Pytorch ATen kernels.
# Triton: Triton templates defined in torch inductor.
# CUTLASS: Cutlass templates and kernels.
max_autotune_gemm_backends = os.environ.get(
    "TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS", "ATEN,TRITON"
).upper()

# enable searching global and local cache regardless of `max_autotune`
search_autotune_cache = os.environ.get("TORCHINDUCTOR_SEARCH_AUTOTUNE_CACHE") == "1"

save_args = os.environ.get("TORCHINDUCTOR_SAVE_ARGS") == "1"

# We will disable creating subprocess for autotuning if this is False
autotune_in_subproc = os.environ.get("TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC") == "1"

coordinate_descent_tuning = (
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING") == "1"
)
coordinate_descent_check_all_directions = (
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS") == "1"
)
coordinate_descent_search_radius = int(
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS", "1")
)

layout_optimization = os.environ.get("TORCHINDUCTOR_LAYOUT_OPTIMIZATION", "1") == "1"

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
realize_bytes_threshold = 2000

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

# how many nodes to allow into a single fusion
max_fusion_size = 64

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


def is_fbcode():
    return not hasattr(torch.version, "git_version")


# constant folding on the joint graph
# Turn off constant folding due to issue #108388
joint_graph_constant_folding = not is_fbcode()

# Enable indirect_indexing asserts for decompositions and lowerings
debug_index_asserts = False

# warnings intended for PyTorch developers, disable for point releases
is_nightly_or_source = "dev" in torch.__version__ or "git" in torch.__version__
developer_warnings = is_fbcode() or is_nightly_or_source


def decide_compile_threads():
    """
    Here are the precedence to decide compile_threads
    1. User can override it by TORCHINDUCTOR_COMPILE_THREADS.  One may want to disable async compiling by
       setting this to 1 to make pdb happy.
    2. Set to 1 if it's win32 platform or it's a fbcode build
    3. decide by the number of CPU cores
    """
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        return int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
    elif sys.platform == "win32" or is_fbcode():
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
    from libfb.py import parutil  # type: ignore[import]

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

# TODO: remove later
disable_cpp_codegen = False


# Freezing will attempt to inline weights as constants in optimization
# and run constant folding and other optimizations on them. After freezing, weights
# can no longer be updated.
freezing: bool = os.environ.get("TORCHINDUCTOR_FREEZING", "0") == "1"

# Make freezing invalidate the eager Parameters of nn modules, to avoid memory overhead
# of potentially keeping multiple copies of weights.
freezing_discard_parameters: bool = False


# config specific to codegen/cpp.py
class cpp:
    # set to torch.get_num_threads()
    threads = -1

    # Do not generate loops when the condition doesn't hold, like:
    # for(long i0=4096; i0<4096; i0+=1)
    no_redundant_loops = True

    # Assume number of threads is dynamic, don't specialize thread number.
    # Kernels don't recompile on thread number changes with this flag on.
    # For single-threaded workload, turning it on would incur a slight
    # performance degradation.
    dynamic_threads = False

    simdlen = None
    min_chunk_size = 4096
    cxx = (
        None,  # download gcc12 from conda-forge if conda is installed
        # "g++-12",
        # "g++-11",
        # "g++-10",
        # "clang++",
        os.environ.get("CXX", "g++"),
        # "g++.par",
    )
    # Allow kernel performance profiling via PyTorch profiler
    enable_kernel_profile = False

    # enable weight prepacking to get a better performance; may lead to large memory footprint
    weight_prepack = True

    # Inject a bug into our relu implementation; useful for testing our repro
    # extraction and minification functionality.
    # Valid values: "compile_error", "runtime_error", "accuracy"
    inject_relu_bug_TESTING_ONLY = None
    inject_log1p_bug_TESTING_ONLY = None

    # If None, autodetect whether or not AVX512/AVX2 can be used.  Otherwise,
    # force usage as specified, without testing.
    vec_isa_ok = None

    # similar to config.triton.descriptive_names
    descriptive_names = "original_aten"

    # how many nodes to allow into a single horizontal fusion
    max_horizontal_fusion_size = 16

    # Make scatter_reduce fallback when reduce is sum to avoid performance regression
    # using atomic_add.
    fallback_scatter_reduce_sum = True


# config specific to codegen/triton.py
class triton:
    # Use cudagraphs on output code
    cudagraphs = False

    # Use cudagraph trees for memory pooling if `cudagraphs` is True
    cudagraph_trees = True

    # assertions not on the fast path, steady state
    slow_path_cudagraph_asserts = True

    # TODO - need to debug why this prevents cleanup
    cudagraph_trees_history_recording = False

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

    # use triton.autotune for pointwise ops with complex layouts
    # this should only be disabled for debugging/testing
    autotune_pointwise = True

    # max autotune gemm with cublasLt
    autotune_cublasLt = True

    # should we stop a fusion to allow better tiling?
    tiling_prevents_pointwise_fusion = True
    tiling_prevents_reduction_fusion = True

    # assert that indirect indexing does not read / write out of bounds
    assert_indirect_indexing = True

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

    # hint to Triton when arguments are divisible by 16
    divisible_by_16 = True

    # theses are not enforced, but they are used by asserts in triton_heuristics.py
    # NOTE: mobilevit_s in timm_models required X to be set to the higher value 2048
    max_block = {"X": 2048, "Y": 1024, "Z": 1024}

    # Store the generated cubin files for cpp wrapper code to load
    store_cubin = False

    # the max number of spills we allow for the configs we benchmark.
    # Setting this to 0 means we skip a config if it spills even a single
    # register.
    # Settting it to a larger value allows a config spilling a small amount
    # of registers being benchmarked.
    #
    # NOTE: triton will always report >0 register spills for kernels using sin/cos.
    # (check this issue https://github.com/openai/triton/issues/1756 )
    # So far we see a fixed 8 spilled registers for kernels using sin/cos.
    # Raise the threshold to 16 to be safe.
    # We should revisit this once we understand more of the source of register spills.
    spill_threshold: int = 16

    # Inject a bug into our relu implementation; useful for testing our repro
    # extraction and minification functionality.
    # Valid values: "compile_error", "runtime_error", "accuracy"
    inject_relu_bug_TESTING_ONLY = None


class aot_inductor:
    # AOTInductor output path
    # If an absolute path is specified, the generated lib files will be stored under the directory;
    # If a relative path is specified, it will be used as a subdirectory under the default caching path;
    # If not specified, a temp directory will be created under the default caching path
    output_path = ""


class cuda:
    # CUDA arch to use for CUDA template kernel compilation.
    # e.g. "70", "75", "80", "90", etc.
    # When arch is None, Inductor uses torch.cuda.get_device_capability(0).
    arch = None

    # CUDA version to use for CUDA template kernel compilation.
    # e.g. "11.4", "12.1", etc.
    # When version is None, Inductor uses torch.version.cuda.
    version = None

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
    cutlass_max_profiling_configs = None

    # Path to CUDA NVCC.
    # NVCC search order:
    # 1) cuda_cxx set in this config
    # 2）CUDACXX environment variable
    # 3）CUDA_HOME environment variable
    # 4) default system search PATH.
    cuda_cxx = None


# create a directory containing lots of debug information
class trace:
    # master switch for all debugging flags below
    enabled = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

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

    # Store cProfile (see snakeviz to view)
    compile_profile = False

    # Upload the .tar.gz file
    # Needs to be overriden based on specific environment needs
    upload_tar = None


_save_config_ignore = {
    # workaround: "Can't pickle <function ...>"
    "trace.upload_tar",
}


from .._dynamo.config_utils import install_config_module

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
