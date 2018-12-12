# In the open-source build, these are generated into
# torch/csrc/{autgrad,jit}/generated. In fbcode, this distinction is
# not currently relevant so they are combined into one list.
from __future__ import absolute_import, division, print_function, unicode_literals
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

# copied from https://github.com/pytorch/pytorch/blob/master/tools/cpp_build/libtorch/CMakeLists.txt
torch_sources_no_python_default = [
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
    "torch/csrc/autograd/functions/comm.cpp",
    "torch/csrc/autograd/functions/tensor.cpp",
    "torch/csrc/autograd/functions/utils.cpp",
    "torch/csrc/autograd/grad_mode.cpp",
    "torch/csrc/autograd/input_buffer.cpp",
    "torch/csrc/autograd/profiler.cpp",
    "torch/csrc/autograd/saved_variable.cpp",
    "torch/csrc/autograd/variable.cpp",
    "torch/csrc/Exceptions.cpp",
    "torch/csrc/jit/autodiff.cpp",
    "torch/csrc/jit/constants.cpp",
    "torch/csrc/jit/node_hashing.cpp",
    "torch/csrc/jit/export.cpp",
    "torch/csrc/jit/graph_executor.cpp",
    "torch/csrc/jit/import.cpp",
    "torch/csrc/jit/interpreter.cpp",
    "torch/csrc/jit/ir.cpp",
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
    "torch/csrc/jit/register_prim_ops.cpp",
    "torch/csrc/jit/register_special_ops.cpp",
    "torch/csrc/jit/scope.cpp",
    "torch/csrc/jit/script/compiler.cpp",
    "torch/csrc/jit/import_method.cpp",
    "torch/csrc/jit/hooks_for_testing.cpp",
    "torch/csrc/jit/script/builtin_functions.cpp",
    "torch/csrc/jit/script/lexer.cpp",
    "torch/csrc/jit/script/module.cpp",
    "torch/csrc/jit/tracer.cpp",
    "torch/csrc/utils/tensor_flatten.cpp",
    "torch/csrc/utils/variadic.cpp",
]


def torch_vars():
    r = {}
    # We start torch_sources with all cpp files, and exclude some.
    # This is a much better approach than listing all of them manually because
    # the number of excluded files is small and doesn"t change very frequently
    r["torch_sources"] = native.glob(
        ["torch/csrc/**/*.cpp"],
        exclude = [
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
    ] + [":generate-code=" + x for x in GENERATED_CPP]

    r["torch_sources_no_python"] = torch_sources_no_python_default + [
        "torch/csrc/cuda/comm.cpp",
        "torch/csrc/cuda/nccl.cpp",
    ] + native.glob(
        [
            "torch/csrc/jit/fuser/**/*.cpp",
        ],
    )

    r["torch_sources_no_python_cpu"] = torch_sources_no_python_default + native.glob(
        [
            "torch/csrc/jit/fuser/**/*.cpp",
        ],
        exclude = ["torch/csrc/jit/fuser/cuda/*.cpp"],
    )

    r["torch_csrc_flags"] = {
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
            ],
        },
        "headers": native.glob(
            [
                "torch/csrc/**/*.h",
                "torch/csrc/generic/*.cpp",
            ],
        ),
        "preprocessor_flags": [
            "-Icaffe2",
            "-Icaffe2/torch/csrc/api/include",
            "-Icaffe2/torch/csrc",
            "-Icaffe2/torch/csrc/nn",
            "-Icaffe2/torch/lib",
            "-DUSE_CPU_FUSER_FBCODE=1",
            "-DUSE_CUDA_FUSER_FBCODE=1",
        ],
    }

    r["torch_csrc_flags_cpu"] = dict(r['torch_csrc_flags'])

    r["torch_csrc_flags_cpu"]["preprocessor_flags"] = [
        "-Icaffe2",
        "-Icaffe2/torch/csrc/api/include",
        "-Icaffe2/torch/csrc",
        "-Icaffe2/torch/csrc/nn",
        "-Icaffe2/torch/lib",
        "-DUSE_CPU_FUSER_FBCODE=1",
        "-DUSE_CUDA_FUSER_FBCODE=0",
    ]
    return r
