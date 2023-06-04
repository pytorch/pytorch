import dataclasses
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
from torch._config_utils import ConfigMixin

# add some debug printouts
debug: bool = False

# Whether to disable a progress bar for autotuning
disable_progress: bool = True

# Whether to enable printing the source code for each future
verbose_progress: bool = False

# use cpp wrapper instead of python wrapper
cpp_wrapper: bool = False

# dead code elimination
dce: bool = False

# assume weight tensors are fixed size
static_weight_shapes: bool = True

# put correctness assertions in generated code
size_asserts: bool = os.environ.get("TORCHINDUCTOR_SIZE_ASSERTS", "1") == "1"

# enable loop reordering based on input orders
pick_loop_orders: bool = True

# generate inplace computations
inplace_buffers: bool = True

# allow reusing buffers for more efficient memory use
allow_buffer_reuse: bool = True

# codegen benchmark harness
benchmark_harness: bool = True

# fuse pointwise into templates
epilogue_fusion: bool = True

# do epilogue fusions before other fusions
epilogue_fusion_first: bool = False

# enable pattern match+replace optimizations
pattern_matcher: bool = True

# Optimize away split cat patterns (Experimental)
split_cat_fx_passes: bool = True

# enable reordering pass
reordering: bool = True

# enable slow autotuning passes to select algorithms
max_autotune: bool = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"

# enable slow autotuning passes to select pointwise/reductions algorithms
max_autotune_pointwise: bool = (
    os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE") == "1"
)

# enable slow autotuning passes to select gemm algorithms
max_autotune_gemm: bool = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM") == "1"

# enable searching global and local cache regardless of `max_autotune`
search_autotune_cache: bool = (
    os.environ.get("TORCHINDUCTOR_SEARCH_AUTOTUNE_CACHE") == "1"
)

# We will disable creating subprocess for autotuning if this is False
autotune_in_subproc: bool = os.environ.get("TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC") == "1"

coordinate_descent_tuning: bool = (
    os.environ.get("TORCHINDUCTOR_COORDINATE_DESCENT_TUNING") == "1"
)

layout_optimization: bool = (
    os.environ.get("TORCHINDUCTOR_LAYOUT_OPTIMIZATION", "1") == "1"
)

# Whether to keep the output strides the same as eager after layout optimization.
keep_output_stride: bool = (
    os.environ.get("TORCHINDUCTOR_KEEP_OUTPUT_STRIDE", "1") == "1"
)

# Enabling this will let compiler print warning messages if a generated triton
# kernel has inputs with mixed layouts.  This is helpful for perf debugging
# since kernel with mixed layout inputs may run much slower then one whose inputs
# have uniform layouts.
warn_mix_layout: bool = os.environ.get("TORCHINDUCTOR_WARN_MIX_LAYOUT") == "1"

# control store vs recompute heuristic
# For fanouts, rematerialization can lead to exponential blowup. So, have
# smaller threshold
realize_reads_threshold: int = 4
realize_bytes_threshold: int = 2000

# Threshold to prevent excessive accumulation of ops in one buffer during lowering
realize_acc_reads_threshold: int = 8

# fallback to eager for random/dropout, this is slow but useful for debugging
fallback_random: bool = False

# automatically create fallbacks when encountering an unhandled op
implicit_fallbacks: bool = True

# fuse even in cases without common reads
aggressive_fusion: bool = False

# how many nodes to allow into a single fusion
max_fusion_size: int = 64

# replace small reductions with pointwise, disable with `= 1`
unroll_reductions_threshold: int = 8

# Add extra comments to output code (causes compile cache misses)
comment_origin: bool = False

# Convert 1x1 convs into matmuls
conv_1x1_as_mm: bool = False

# Enable split reductions for better utilization when the dimension
# being reduced over is large (by splitting it)
split_reductions: bool = True

benchmark_kernel: bool = os.environ.get("TORCHINDUCTOR_BENCHMARK_KERNEL", "0") == "1"

# Enable constant and index_expr folding
constant_and_index_propagation: bool = True

# Enable indirect_indexing asserts for decompositions and lowerings
debug_index_asserts: bool = False


# warnings intended for PyTorch developers, disable for point releases
is_nightly_or_source: bool = "dev" in torch.__version__ or "git" in torch.__version__
developer_warnings: bool = ConfigMixin.is_fbcode() or is_nightly_or_source


def decide_compile_threads():
    """
    Here are the precedence to decide compile_threads
    1. User can override it by TORCHINDUCTOR_COMPILE_THREADS.
       One may want to disable async compiling by
       setting this to 1 to make pdb happy.
    2. Set to 1 if it's win32 platform or it's a fbcode build
    3. decide by the number of CPU cores
    """
    if "TORCHINDUCTOR_COMPILE_THREADS" in os.environ:
        return int(os.environ["TORCHINDUCTOR_COMPILE_THREADS"])
    elif sys.platform == "win32" or ConfigMixin.is_fbcode():
        return 1
    else:
        return min(
            32,
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else os.cpu_count(),
        )


compile_threads: int = decide_compile_threads()

# gemm autotuning global cache dir

global_cache_dir: Optional[str] = "fb/cache" if ConfigMixin.is_fbcode() else None

# If kernel is fused, the name is generated from the origin node op names
# for larger kernels limit this
kernel_name_max_ops: int = 10

# Pad input tensors of matmul/bmm/addmm to leverage Tensor Cores in NVIDIA GPUs
shape_padding: bool = os.environ.get("TORCHINDUCTOR_SHAPE_PADDING", "1") == "1"

# Fx-based linear/matmul/bmm + permute/transpose vertical fusion
permute_fusion: bool = os.environ.get("TORCHINDUCTOR_PERMUTE_FUSION", "0") == "1"

# Mark the wrapper call in PyTorch profiler
profiler_mark_wrapper_call: bool = False

# Generate hook calls to torch._inductor.hooks.run_intermediate_hooks for
# every intermediate for which we can correlate it with an intermediate
# from the original FX graph
generate_intermediate_hooks: bool = False

# Populate traceback field on IRNode; good for debugging why origin_node is
# not populated, or finding out where an IRNode was constructed
debug_ir_traceback: bool = False

# used for debugging to make sure config is properly set
_raise_error_for_testing: bool = False

_profile_var: str = os.environ.get("TORCHINDUCTOR_PROFILE", "")
profile_bandwidth: bool = _profile_var != ""
profile_bandwidth_regex: str = "" if _profile_var == "1" else _profile_var

disable_cpp_codegen: bool = ConfigMixin.is_fbcode()


# config specific to codegen/cpp.pp
@dataclasses.dataclass
class CppConfig(ConfigMixin):
    # set to torch.get_num_threads()
    threads: int = -1

    # Do not generate loops when the condition doesn't hold, like:
    # for(long i0=4096; i0<4096; i0+=1)
    no_redundant_loops: bool = True

    # Assume number of threads is dynamic, don't specialize thread number.
    # Kernels don't recompile on thread number changes with this flag on.
    # For single-threaded workload, turning it on would incur a slight
    # performance degradation.
    dynamic_threads: bool = False

    simdlen: Optional[int] = None
    min_chunk_size: int = 4096
    cxx: Tuple[Optional[str]] = (
        None,  # download gcc12 from conda-forge if conda is installed
        # "g++-12",
        # "g++-11",
        # "g++-10",
        # "clang++",
        os.environ.get("CXX", "g++"),
        # "g++.par",
    )
    # Allow kernel performance profiling via PyTorch profiler
    enable_kernel_profile: bool = False

    # enable weight prepacking to get a better performance; may lead to large memory footprint
    weight_prepack: bool = True

    # Inject a bug into our relu implementation; useful for testing our repro
    # extraction and minification functionality.
    # Valid values: "compile_error", "runtime_error", "accuracy"
    inject_relu_bug_TESTING_ONLY: Optional[str] = None
    inject_log1p_bug_TESTING_ONLY: Optional[str] = None

    # If None, autodetect whether or not AVX512/AVX2 can be used.  Otherwise,
    # force usage as specified, without testing.
    vec_isa_ok: Optional[str] = None

    # similar to config.triton.descriptive_names
    descriptive_names: str = "original_aten"

    # how many nodes to allow into a single horizontal fusion
    max_horizontal_fusion_size: int = 16


# config specific to codegen/triton.py
@dataclasses.dataclass
class TritonConfig(ConfigMixin):
    # Use cudagraphs on output code
    cudagraphs: bool = False

    # Use cudagraph trees for memory pooling if `cudagraphs` is True
    cudagraph_trees: bool = not ConfigMixin.is_fbcode()

    # assertions not on the fast path, steady state
    slow_path_cudagraph_asserts: bool = True

    # TODO - need to debug why this prevents cleanup
    cudagraph_trees_history_recording: bool = False

    # assertions on the fast path
    fast_path_cudagraph_asserts: bool = False

    # skip warmup for cudagraph trees
    skip_cudagraph_warmup: bool = False

    # Synchronize before and after every compiled graph.
    debug_sync_graph: bool = False

    # Synchronize after every kernel launch, to help pinpoint bugs
    debug_sync_kernel: bool = False

    # Always load full blocks (rather than broadcasting inside the block)
    dense_indexing: bool = False

    # limit tiling dimensions
    max_tiles: int = 2

    # use triton.autotune for pointwise ops with complex layouts
    # this should only be disabled for debugging/testing
    autotune_pointwise: bool = True

    # should we stop a fusion to allow better tiling?
    tiling_prevents_pointwise_fusion: bool = True
    tiling_prevents_reduction_fusion: bool = True

    # assert that indirect indexing does not read / write out of bounds
    assert_indirect_indexing: bool = True

    # should we give different names to kernels
    # Note: This is orthogonal to descriptive_names - this is deciding whether
    # our triton kernel names should all be `triton_` (to maximize caching) or
    # whether they should be unique.
    unique_kernel_names: bool = (
        os.environ.get("TORCHINDUCTOR_UNIQUE_KERNEL_NAMES") == "1"
    )

    # should we put op names in kernel names
    # False: No special names (just triton__1, triton__2, etc.)
    # "torch": Maps to the fx op in the Dynamo graph (module name, method name, etc.)
    # "original_aten": Maps to the highest-level aten op (i.e. pre-decompositions)
    # "inductor_node": Maps to the node name in the FX graph passed to Inductor
    descriptive_names: str = "original_aten"

    # use alternate codegen for smaller reductions
    persistent_reductions: bool = (
        os.environ.get("TORCHINDUCTOR_PERSISTENT_REDUCTIONS", "1") == "1"
    )

    # hint to Triton when arguments are divisible by 16
    divisible_by_16: bool = True

    # theses are not enforced, but they are used by asserts in triton_heuristics.py
    # NOTE: mobilevit_s in timm_models required X to be set to the higher value 2048
    max_block: Dict[str, int] = dataclasses.field(
        default_factory=lambda: {"X": 2048, "Y": 1024, "Z": 1024}
    )

    # Store the generated cubin files for cpp wrapper code to load
    store_cubin: bool = False

    # the max number of spills we allow for the configs we benchmark.
    # Setting this to 0 means we skip a config if it spills even a single
    # register.
    # Settting it to a larger value allows a config spilling a small amount
    # of registers being benchmarked.
    spill_threshold: int = 0

    # Inject a bug into our relu implementation; useful for testing our repro
    # extraction and minification functionality.
    # Valid values: "compile_error", "runtime_error", "accuracy"
    inject_relu_bug_TESTING_ONLY: Optional[str] = None


# create a directory containing lots of debug information
@dataclasses.dataclass
class TraceConfig(ConfigMixin):
    # master switch for all debugging flags below
    enabled: bool = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    # Save python logger call >=logging.DEBUG
    debug_log: bool = False

    # Save python logger call >=logging.INFO
    info_log: bool = False

    # Save input FX graph (post decomps, pre optimization)
    fx_graph: bool = True

    # Save FX graph after transformations
    fx_graph_transformed: bool = True

    # Save TorchInductor IR before fusion pass
    ir_pre_fusion: bool = True

    # Save TorchInductor IR after fusion pass
    ir_post_fusion: bool = True

    # Copy generated code to trace dir
    output_code: bool = True

    # SVG figure showing post-fusion graph
    graph_diagram: bool = False

    # Store cProfile (see snakeviz to view)
    compile_profile: bool = False

    # Upload the .tar.gz file
    # Needs to be overriden based on specific environment needs
    # skip_pickle will make this field skipped by pickle
    upload_tar: Any = dataclasses.field(default=None, metadata={"skip_pickle": True})


cpp: CppConfig = CppConfig()
triton: TritonConfig = TritonConfig()
trace: TraceConfig = TraceConfig()

from torch._config_utils import install_config_module

install_config_module("InductorConfig", sys.modules[__name__])
