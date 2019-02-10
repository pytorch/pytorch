# In the open-source build, these are generated into
# torch/csrc/{autgrad,jit}/generated. In fbcode, this distinction is
# not currently relevant so they are combined into one list.
from __future__ import absolute_import, division, print_function, unicode_literals
load("@bazel_skylib//lib:new_sets.bzl", "sets")


GENERATED_CPP = [
    "Functions.cpp",
    "THCUNN.cpp",
    "THNN.cpp",
    "VariableType_0.cpp",
    "VariableType_1.cpp",
    "VariableType_2.cpp",
    "VariableType_3.cpp",
    "VariableType_4.cpp",
    "register_aten_ops_0.cpp",
    "register_aten_ops_1.cpp",
    "register_aten_ops_2.cpp",
    "python_functions.cpp",
    "python_nn_functions.cpp",
    "python_torch_functions.cpp",
    "python_variable_methods.cpp",
]

# copied from https://github.com/pytorch/pytorch/blob/master/tools/cpp_build/torch/CMakeLists.txt
libtorch_sources = [
    ":generate-code=Functions.cpp",
    ":generate-code=register_aten_ops_0.cpp",
    ":generate-code=register_aten_ops_1.cpp",
    ":generate-code=register_aten_ops_2.cpp",
    ":generate-code=VariableType_0.cpp",
    ":generate-code=VariableType_1.cpp",
    ":generate-code=VariableType_2.cpp",
    ":generate-code=VariableType_3.cpp",
    ":generate-code=VariableType_4.cpp",
    "torch/csrc/autograd/VariableTypeManual.cpp",
    "torch/csrc/autograd/anomaly_mode.cpp",
    "torch/csrc/autograd/engine.cpp",
    "torch/csrc/autograd/function.cpp",
    "torch/csrc/autograd/functions/accumulate_grad.cpp",
    "torch/csrc/autograd/functions/basic_ops.cpp",
    "torch/csrc/autograd/functions/tensor.cpp",
    "torch/csrc/autograd/functions/utils.cpp",
    "torch/csrc/autograd/grad_mode.cpp",
    "torch/csrc/autograd/input_buffer.cpp",
    "torch/csrc/autograd/profiler.cpp",
    "torch/csrc/autograd/saved_variable.cpp",
    "torch/csrc/autograd/variable.cpp",
    "torch/csrc/Exceptions.cpp",
    "torch/csrc/jit/autodiff.cpp",
    "torch/csrc/jit/attributes.cpp",
    "torch/csrc/jit/constants.cpp",
    "torch/csrc/jit/node_hashing.cpp",
    "torch/csrc/jit/export.cpp",
    "torch/csrc/jit/graph_executor.cpp",
    "torch/csrc/jit/import.cpp",
    "torch/csrc/jit/interpreter.cpp",
    "torch/csrc/jit/ir.cpp",
    "torch/csrc/jit/caffe2_operator.cpp",
    "torch/csrc/jit/register_caffe2_ops.cpp",
    "torch/csrc/jit/register_c10_ops.cpp",
    "torch/csrc/jit/symbolic_script.cpp",
    "torch/csrc/jit/operator.cpp",
    "torch/csrc/jit/passes/alias_analysis.cpp",
    "torch/csrc/jit/passes/batch_mm.cpp",
    "torch/csrc/jit/passes/canonicalize_ops.cpp",
    "torch/csrc/jit/passes/canonicalize.cpp",
    "torch/csrc/jit/passes/common_subexpression_elimination.cpp",
    "torch/csrc/jit/passes/constant_propagation.cpp",
    "torch/csrc/jit/passes/constant_pooling.cpp",
    "torch/csrc/jit/passes/create_autodiff_subgraphs.cpp",
    "torch/csrc/jit/passes/dead_code_elimination.cpp",
    "torch/csrc/jit/passes/erase_number_types.cpp",
    "torch/csrc/jit/passes/graph_fuser.cpp",
    "torch/csrc/jit/passes/inline_autodiff_subgraphs.cpp",
    "torch/csrc/jit/passes/inplace_check.cpp",
    "torch/csrc/jit/passes/loop_unrolling.cpp",
    "torch/csrc/jit/passes/lower_grad_of.cpp",
    "torch/csrc/jit/passes/lower_tuples.cpp",
    "torch/csrc/jit/passes/peephole.cpp",
    "torch/csrc/jit/passes/python_print.cpp",
    "torch/csrc/jit/passes/remove_expands.cpp",
    "torch/csrc/jit/passes/requires_grad_analysis.cpp",
    "torch/csrc/jit/passes/shape_analysis.cpp",
    "torch/csrc/jit/passes/specialize_undef.cpp",
    "torch/csrc/jit/passes/utils/subgraph_utils.cpp",
    "torch/csrc/jit/passes/utils/alias_tracker.cpp",
    "torch/csrc/jit/register_prim_ops.cpp",
    "torch/csrc/jit/register_special_ops.cpp",
    "torch/csrc/jit/scope.cpp",
    "torch/csrc/jit/script/compiler.cpp",
    "torch/csrc/jit/script/edit_distance.cpp",
    "torch/csrc/jit/script/final_returns.cpp",
    "torch/csrc/jit/script/type_parser.cpp",
    "torch/csrc/jit/script/sugared_value.cpp",
    "torch/csrc/jit/script/schema_matching.cpp",
    "torch/csrc/jit/script/parser.cpp",
    "torch/csrc/jit/import_method.cpp",
    "torch/csrc/jit/hooks_for_testing.cpp",
    "torch/csrc/jit/script/builtin_functions.cpp",
    "torch/csrc/jit/script/lexer.cpp",
    "torch/csrc/jit/script/module.cpp",
    "torch/csrc/jit/tracer.cpp",
    "torch/csrc/utils/tensor_flatten.cpp",
    "torch/csrc/utils/variadic.cpp",
    "torch/csrc/jit/fuser/kernel_cache.cpp",
    "torch/csrc/jit/fuser/compiler.cpp",
    "torch/csrc/jit/fuser/executor.cpp",
    "torch/csrc/jit/fuser/codegen.cpp",
    "torch/csrc/jit/fuser/fallback.cpp",
    "torch/csrc/jit/fuser/cpu/fused_kernel.cpp",
    "torch/csrc/jit/fuser/cpu/dynamic_library_unix.cpp",
    "torch/csrc/jit/fuser/interface.cpp",
]

libtorch_cuda_sources = [
    "torch/csrc/cuda/comm.cpp",
    "torch/csrc/cuda/nccl.cpp",
    "torch/csrc/jit/fuser/cuda/fused_kernel.cpp",
    "torch/csrc/autograd/profiler_cuda.cpp",
    "torch/csrc/autograd/functions/comm.cpp"
]


def add_torch_libs():
    r = {}
    # We start torch_python_sources with all cpp files, and exclude some
    # including the files already contained in the torch and cuda bindings
    globbed_sources = (native.glob(
        ["torch/csrc/**/*.cpp"],
        exclude=[
            # remove anything that has "generic" in it"s path
            "torch/csrc/**/generic/**/*.cpp",
            # distributed only uses Module.cpp
            # so remove all other files and just include that
            "torch/csrc/distributed/**/*.cpp",
        ],
    ) + [
        "torch/csrc/distributed/Module.cpp",
        "torch/csrc/distributed/c10d/init.cpp",
        "torch/csrc/distributed/c10d/ddp.cpp",
    ] + [":generate-code=" + x for x in GENERATED_CPP])
    libtorch_python_sources = sets.to_list(sets.difference(
        sets.make(globbed_sources),
        sets.make(libtorch_sources + libtorch_cuda_sources),
    ))

    common_flags = {
        "compiler_flags": [
            "-D_THP_CORE",
            "-DUSE_C10D",
            "-DUSE_CUDNN",
            "-DUSE_DISTRIBUTED",
            "-DUSE_NCCL",
            "-DUSE_NUMPY",
            "-DUSE_SCALARS",
            "-DTH_INDEX_BASE=0",
            "-DNO_CUDNN_DESTROY_HANDLE",
            "-DPYTORCH_ONNX_CAFFE2_BUNDLE",
            "-Wno-write-strings",
            "-Wno-format",
            "-Wno-strict-aliasing",
            "-Wno-non-virtual-dtor",
            "-Wno-shadow-compatible-local",
            "-Wno-empty-body",
        ],
        "compiler_specific_flags": {
            "clang": [
                "-Wno-absolute-value",
                "-Wno-expansion-to-defined",
                "-Wno-pessimizing-move",
                "-Wno-return-type-c-linkage",
                "-Wno-unknown-pragmas",
            ]
        },
        "headers": native.glob(["torch/csrc/**/*.h", "torch/csrc/generic/*.cpp"]),
        "preprocessor_flags": [
            "-Icaffe2",
            "-Icaffe2/torch/csrc/api/include",
            "-Icaffe2/torch/csrc",
            "-Icaffe2/torch/csrc/nn",
            "-Icaffe2/torch/lib",
        ],
    }

    cpp_library(
        name="libtorch",
        srcs=libtorch_sources,
        link_whole=True,
        deps=[
            ":generated-autograd-headers",
            ":generated-autograd-headers-bare",
            ":generated-jit-headers",
            "//caffe2/aten:ATen-cpu",
            "//caffe2/caffe2:caffe2_cpu",
            "//caffe2/torch/lib/libshm:libshm",
        ],
        external_deps=[
            ("nanopb", None, "protobuf-nanopb"),
            ("protobuf", None),
        ],
        **common_flags
    )

    cpp_library(
        name="libtorch_cuda",
        srcs=libtorch_cuda_sources,
        link_whole=True,
        propagated_pp_flags=[
            "-DUSE_CUDA",
        ],
        deps=[
            ":generated-autograd-headers",
            ":generated-autograd-headers-bare",
            ":generated-jit-headers",
            ":libtorch",
            "//caffe2/aten:ATen",
            "//caffe2/aten:generated-aten-headers-cuda",
            "//caffe2/caffe2:caffe2_cpu",
            "//caffe2/torch/lib/libshm:libshm",
        ],
        external_deps=[
            ("cudnn", "7.1.2", "cudnn-lazy"),
            ("nccl", "2.1.15", "nccl-lazy"),
            ("cuda", None, "nvToolsExt-lazy"),
            ("cuda", None, "nvrtc-lazy"),
            ("cuda", None, "nvrtc-builtins-lazy"),
        ],
        **common_flags
    )

    cpp_python_extension(
        name="_C",
        srcs=libtorch_python_sources,
        base_module="torch",
        deps=[
            ":libtorch_cuda",
            ":thnn",
            ":torch-lib-headers",
            "//caffe2/torch/lib/THD:THD",
            "//caffe2/torch/lib/c10d:c10d",
            "//caffe2/torch/lib/libshm:libshm",
        ],
        external_deps=[
            ("numpy", None, "cpp"),
            ("pybind11", None),
        ],
        **common_flags
    )

    return r
