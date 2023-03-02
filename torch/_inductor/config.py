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
size_asserts = True

# enable loop reordering based on input orders
pick_loop_orders = True

# generate inplace computations
inplace_buffers = True

# codegen benchmark harness
benchmark_harness = True

# fuse pointwise into templates
epilogue_fusion = False

# do epilogue fusions before other fusions
epilogue_fusion_first = False

# enable pattern match+replace optimizations
pattern_matcher = True

# enable reordering pass
reordering = False

# enable slow autotuning passes to select algorithms
max_autotune = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"

# control store vs recompute heuristic
# For fanouts, rematearialization can lead to exponential blowup. So, have
# smaller threshold
realize_reads_threshold = 4
realize_bytes_threshold = 2000

# Threshold to prevent excessive accumulation of ops in one buffer during lowering
realize_acc_reads_threshold = 8

# fallback to eager for random/dropout, this is slow but useful for debugging
fallback_random = False

# automatically create fallbacks when encountering an unhandled op
implicit_fallbacks = True

# do bench to decide best layout, currently only for aten.conv
tune_layout = False

# fuse even in cases without common reads
aggressive_fusion = False

# how many nodes to allow into a single fusion
max_fusion_size = 64

# replace small reductions with pointwise, disable with `= 1`
unroll_reductions_threshold = 8

comment_origin = False


def is_fbcode():
    return not hasattr(torch.version, "git_version")


# warnings intended for PyTorch developers, disable for point releases
developer_warnings = is_fbcode() or "+" in torch.__version__

compile_threads = (
    1
    if sys.platform == "win32" or is_fbcode()
    else min(
        32,
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count(),
    )
)

# If kernel is fused, the name is generated from the origin node op names
# for larger kernels limit this
kernel_name_max_ops = 10

# Pad input tensors of matmul/bmm/addmm to leverage Tensor Cores in NVIDIA GPUs
shape_padding = os.environ.get("TORCHINDUCTOR_SHAPE_PADDING", "0") == "1"

# Fx-based linear/matmul/bmm + permute/transpose vertical fusion
permute_fusion = os.environ.get("TORCHINDUCTOR_PERMUTE_FUSION", "0") == "1"

# Mark the wrapper call in PyTorch profiler
profiler_mark_wrapper_call = False

# used for debugging to make sure config is properly set
_raise_error_for_testing = False

# config specific to codegen/cpp.pp
class cpp:
    # set to torch.get_num_threads()
    threads = -1

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
        "g++",
        # "g++.par",
    )
    # Allow kernel performance profiling via PyTorch profiler
    enable_kernel_profile = False

    # enable weight prepacking to get a better performance; may lead to large memory footprint
    weight_prepack = True


# config specific to codegen/triton.py
class triton:

    # Use cudagraphs on output code
    cudagraphs = False

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

    # should we stop a fusion to allow better tiling?
    tiling_prevents_pointwise_fusion = True
    tiling_prevents_reduction_fusion = True

    # should we give different names to kernels
    ordered_kernel_names = False

    # should we put op names in kernel names
    descriptive_kernel_names = False

    # use alternate codegen for smaller reductions
    persistent_reductions = False


# create a directory containing lots of debug information
class trace:
    # master switch for all debugging flags below
    enabled = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    # Save python logger call >=logging.DEBUG
    debug_log = True

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
    graph_diagram = False

    # Store cProfile (see snakeviz to view)
    compile_profile = False

    # Upload the .tar.gz file
    # Needs to be overriden based on specific environment needs
    upload_tar = None


from .._dynamo.config_utils import install_config_module

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
