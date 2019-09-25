# In the open-source build, these are generated into
# torch/csrc/{autgrad,jit}/generated. In fbcode, this distinction is
# not currently relevant so they are combined into one list.
from __future__ import absolute_import, division, print_function, unicode_literals
load("@bazel_skylib//lib:new_sets.bzl", "sets")
load("//caffe2/caffe2/fb:defs_gpu.bzl", "gpu_library_selector")

GENERATED_CPP = [
    "Functions.cpp",
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
    "torch/csrc/autograd/autograd.cpp",
    "torch/csrc/autograd/custom_function.cpp",
    "torch/csrc/autograd/cpp_hook.cpp",
    "torch/csrc/autograd/engine.cpp",
    "torch/csrc/autograd/function.cpp",
    "torch/csrc/autograd/function_hook.cpp",
    "torch/csrc/autograd/functions/accumulate_grad.cpp",
    "torch/csrc/autograd/functions/basic_ops.cpp",
    "torch/csrc/autograd/functions/tensor.cpp",
    "torch/csrc/autograd/functions/utils.cpp",
    "torch/csrc/autograd/input_buffer.cpp",
    "torch/csrc/autograd/profiler.cpp",
    "torch/csrc/autograd/record_function.cpp",
    "torch/csrc/autograd/saved_variable.cpp",
    "torch/csrc/autograd/variable.cpp",
    "torch/csrc/distributed/autograd/utils.cpp",
    "torch/csrc/distributed/autograd/context/dist_autograd_container.cpp",
    "torch/csrc/distributed/autograd/context/dist_autograd_context.cpp",
    "torch/csrc/distributed/autograd/functions/sendrpc_backward.cpp",
    "torch/csrc/distributed/rpc/future_message.cpp",
    "torch/csrc/distributed/rpc/message.cpp",
    "torch/csrc/distributed/rpc/script_call.cpp",
    "torch/csrc/distributed/rpc/script_remote_call.cpp",
    "torch/csrc/distributed/rpc/script_rref_proto.cpp",
    "torch/csrc/distributed/rpc/script_ret.cpp",
    "torch/csrc/Exceptions.cpp",
    "torch/csrc/jit/autodiff.cpp",
    "torch/csrc/jit/attributes.cpp",
    "torch/csrc/jit/argument_spec.cpp",
    "torch/csrc/jit/constants.cpp",
    "torch/csrc/jit/node_hashing.cpp",
    "torch/csrc/jit/export.cpp",
    "torch/csrc/jit/pass_manager.cpp",
    "torch/csrc/jit/pickler.cpp",
    "torch/csrc/jit/unpickler.cpp",
    "torch/csrc/jit/graph_executor.cpp",
    "torch/csrc/jit/import.cpp",
    "torch/csrc/jit/import_legacy.cpp",
    "torch/csrc/jit/pickle.cpp",
    "torch/csrc/jit/import_export_helpers.cpp",
    "torch/csrc/jit/instruction.cpp",
    "torch/csrc/jit/interpreter.cpp",
    "torch/csrc/jit/ir.cpp",
    "torch/csrc/jit/irparser.cpp",
    "torch/csrc/jit/jit_log.cpp",
    "torch/csrc/jit/netdef_converter.cpp",
    "torch/csrc/jit/register_c10_ops.cpp",
    "torch/csrc/jit/subgraph_matcher.cpp",
    "torch/csrc/jit/symbolic_script.cpp",
    "torch/csrc/jit/profiling_graph_executor_impl.cpp",
    "torch/csrc/jit/profiling_record.cpp",
    "torch/csrc/jit/operator.cpp",
    "torch/csrc/jit/passes/alias_analysis.cpp",
    "torch/csrc/jit/passes/batch_mm.cpp",
    "torch/csrc/jit/passes/bailout_graph.cpp",
    "torch/csrc/jit/passes/canonicalize_ops.cpp",
    "torch/csrc/jit/passes/decompose_ops.cpp",
    "torch/csrc/jit/passes/canonicalize.cpp",
    "torch/csrc/jit/passes/common_subexpression_elimination.cpp",
    "torch/csrc/jit/passes/constant_propagation.cpp",
    "torch/csrc/jit/passes/constant_pooling.cpp",
    "torch/csrc/jit/passes/create_autodiff_subgraphs.cpp",
    "torch/csrc/jit/passes/dead_code_elimination.cpp",
    "torch/csrc/jit/passes/erase_number_types.cpp",
    "torch/csrc/jit/passes/graph_fuser.cpp",
    "torch/csrc/jit/passes/guard_elimination.cpp",
    "torch/csrc/jit/passes/inline_autodiff_subgraphs.cpp",
    "torch/csrc/jit/passes/inliner.cpp",
    "torch/csrc/jit/passes/lift_closures.cpp",
    "torch/csrc/jit/passes/inline_forked_closures.cpp",
    "torch/csrc/jit/passes/inplace_check.cpp",
    "torch/csrc/jit/passes/insert_guards.cpp",
    "torch/csrc/jit/passes/liveness.cpp",
    "torch/csrc/jit/passes/loop_unrolling.cpp",
    "torch/csrc/jit/passes/lower_grad_of.cpp",
    "torch/csrc/jit/passes/lower_tuples.cpp",
    "torch/csrc/jit/passes/peephole.cpp",
    "torch/csrc/jit/passes/python_print.cpp",
    "torch/csrc/jit/passes/quantization.cpp",
    "torch/csrc/jit/passes/fuse_linear.cpp",
    "torch/csrc/jit/passes/remove_expands.cpp",
    "torch/csrc/jit/passes/requires_grad_analysis.cpp",
    "torch/csrc/jit/passes/shape_analysis.cpp",
    "torch/csrc/jit/passes/specialize_autogradzero.cpp",
    "torch/csrc/jit/passes/subgraph_rewrite.cpp",
    "torch/csrc/jit/passes/utils/subgraph_utils.cpp",
    "torch/csrc/jit/passes/utils/memory_dag.cpp",
    "torch/csrc/jit/print_handler.cpp",
    "torch/csrc/jit/register_prim_ops.cpp",
    "torch/csrc/jit/register_string_ops.cpp",
    "torch/csrc/jit/register_special_ops.cpp",
    "torch/csrc/jit/scope.cpp",
    "torch/csrc/jit/script/compiler.cpp",
    "torch/csrc/jit/script/edit_distance.cpp",
    "torch/csrc/jit/script/logging.cpp",
    "torch/csrc/jit/script/convert_to_ssa.cpp",
    "torch/csrc/jit/script/exit_transforms.cpp",
    "torch/csrc/jit/script/inline_loop_condition.cpp",
    "torch/csrc/jit/script/canonicalize_modified_loop.cpp",
    "torch/csrc/jit/script/script_type_parser.cpp",
    "torch/csrc/jit/script/sugared_value.cpp",
    "torch/csrc/jit/script/schema_matching.cpp",
    "torch/csrc/jit/script/class_type.cpp",
    "torch/csrc/jit/script/parser.cpp",
    "torch/csrc/jit/script/jit_exception.cpp",
    "torch/csrc/jit/source_range_serialization.cpp",
    "torch/csrc/jit/testing/file_check.cpp",
    "torch/csrc/jit/import_source.cpp",
    "torch/csrc/jit/hooks_for_testing.cpp",
    "torch/csrc/jit/script/builtin_functions.cpp",
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
    "torch/csrc/jit/fuser/interface.cpp",
    "torch/csrc/jit/function.cpp",
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

    torch_cpp_headers = {
        header[len("torch/csrc/api/include/torch/"):]: header
        for header in glob(["torch/csrc/api/include/**/*.h"])
    }

    torch_cpp_headers["script.h"] = "torch/script.h"

    torch_cpp_srcs = [
        "torch/csrc/api/src/cuda.cpp",  # this just forwards stuff, no real CUDA
        "torch/csrc/api/src/data/datasets/mnist.cpp",
        "torch/csrc/api/src/data/samplers/distributed.cpp",
        "torch/csrc/api/src/data/samplers/random.cpp",
        "torch/csrc/api/src/data/samplers/sequential.cpp",
        "torch/csrc/api/src/data/samplers/stream.cpp",
        "torch/csrc/api/src/enum.cpp",
        "torch/csrc/api/src/jit.cpp",
        "torch/csrc/api/src/serialize.cpp",
        "torch/csrc/api/src/nn/init.cpp",
        "torch/csrc/api/src/nn/module.cpp",
        "torch/csrc/api/src/nn/modules/batchnorm.cpp",
        "torch/csrc/api/src/nn/modules/conv.cpp",
        "torch/csrc/api/src/nn/modules/dropout.cpp",
        "torch/csrc/api/src/nn/modules/distance.cpp",
        "torch/csrc/api/src/nn/modules/embedding.cpp",
        "torch/csrc/api/src/nn/modules/fold.cpp",
        "torch/csrc/api/src/nn/modules/linear.cpp",
        "torch/csrc/api/src/nn/modules/loss.cpp",
        "torch/csrc/api/src/nn/modules/pooling.cpp",
        "torch/csrc/api/src/nn/modules/rnn.cpp",
        "torch/csrc/api/src/nn/modules/container/functional.cpp",
        "torch/csrc/api/src/nn/modules/container/named_any.cpp",
        "torch/csrc/api/src/nn/options/batchnorm.cpp",
        "torch/csrc/api/src/nn/options/conv.cpp",
        "torch/csrc/api/src/nn/options/dropout.cpp",
        "torch/csrc/api/src/nn/options/linear.cpp",
        "torch/csrc/api/src/nn/options/pooling.cpp",
        "torch/csrc/api/src/nn/options/rnn.cpp",
        "torch/csrc/api/src/optim/adagrad.cpp",
        "torch/csrc/api/src/optim/adam.cpp",
        "torch/csrc/api/src/optim/lbfgs.cpp",
        "torch/csrc/api/src/optim/optimizer.cpp",
        "torch/csrc/api/src/optim/rmsprop.cpp",
        "torch/csrc/api/src/optim/serialize.cpp",
        "torch/csrc/api/src/optim/sgd.cpp",
        "torch/csrc/api/src/serialize/input-archive.cpp",
        "torch/csrc/api/src/serialize/output-archive.cpp",
    ]

    libtorch_python_sources = [
        ":generate-code=python_functions.cpp",
        ":generate-code=python_nn_functions.cpp",
        ":generate-code=python_torch_functions.cpp",
        ":generate-code=python_variable_methods.cpp",
        "torch/csrc/CudaIPCTypes.cpp",
        "torch/csrc/DataLoader.cpp",
        "torch/csrc/Device.cpp",
        "torch/csrc/Dtype.cpp",
        "torch/csrc/DynamicTypes.cpp",
        "torch/csrc/Generator.cpp",
        "torch/csrc/Layout.cpp",
        "torch/csrc/MemoryFormat.cpp",
        "torch/csrc/QScheme.cpp",
        "torch/csrc/Module.cpp",
        "torch/csrc/PtrWrapper.cpp",
        "torch/csrc/python_dimname.cpp",
        "torch/csrc/Size.cpp",
        "torch/csrc/Storage.cpp",
        "torch/csrc/TypeInfo.cpp",
        "torch/csrc/api/src/python/init.cpp",
        "torch/csrc/autograd/functions/init.cpp",
        "torch/csrc/autograd/init.cpp",
        "torch/csrc/autograd/python_anomaly_mode.cpp",
        "torch/csrc/autograd/python_cpp_function.cpp",
        "torch/csrc/autograd/python_engine.cpp",
        "torch/csrc/autograd/python_function.cpp",
        "torch/csrc/autograd/python_hook.cpp",
        "torch/csrc/autograd/python_legacy_variable.cpp",
        "torch/csrc/autograd/python_variable.cpp",
        "torch/csrc/autograd/python_variable_indexing.cpp",
        "torch/csrc/byte_order.cpp",
        "torch/csrc/distributed/autograd/init.cpp",
        "torch/csrc/distributed/c10d/comm.cpp",
        "torch/csrc/distributed/c10d/init.cpp",
        "torch/csrc/distributed/c10d/reducer.cpp",
        "torch/csrc/distributed/autograd/init.cpp",
        "torch/csrc/distributed/rpc/functions.cpp",
        "torch/csrc/distributed/rpc/init.cpp",
        "torch/csrc/distributed/rpc/process_group_agent.cpp",
        "torch/csrc/distributed/rpc/python_functions.cpp",
        "torch/csrc/distributed/rpc/python_rpc_handler.cpp",
        "torch/csrc/distributed/rpc/rpc_agent.cpp",
        "torch/csrc/distributed/rpc/rref.cpp",
        "torch/csrc/distributed/rpc/rref_context.cpp",
        "torch/csrc/distributed/rpc/types.cpp",
        "torch/csrc/jit/init.cpp",
        "torch/csrc/jit/passes/inline_fork_wait.cpp",
        "torch/csrc/jit/passes/onnx.cpp",
        "torch/csrc/jit/passes/onnx/cast_all_constant_to_floating.cpp",
        "torch/csrc/jit/passes/onnx/constant_fold.cpp",
        "torch/csrc/jit/passes/onnx/fixup_onnx_loop.cpp",
        "torch/csrc/jit/passes/onnx/peephole.cpp",
        "torch/csrc/jit/passes/onnx/prepare_division_for_onnx.cpp",
        "torch/csrc/jit/passes/onnx/scalar_type_analysis.cpp",
        "torch/csrc/jit/passes/remove_inplace_ops.cpp",
        "torch/csrc/jit/passes/utils/check_alias_annotation.cpp",
        "torch/csrc/jit/python_arg_flatten.cpp",
        "torch/csrc/jit/python_interpreter.cpp",
        "torch/csrc/jit/python_ir.cpp",
        "torch/csrc/jit/python_tracer.cpp",
        "torch/csrc/jit/script/init.cpp",
        "torch/csrc/jit/script/python_sugared_value.cpp",
        "torch/csrc/jit/script/python_tree_views.cpp",
        "torch/csrc/multiprocessing/init.cpp",
        "torch/csrc/onnx/init.cpp",
        "torch/csrc/serialization.cpp",
        "torch/csrc/tensor/python_tensor.cpp",
        "torch/csrc/utils/init.cpp",
        "torch/csrc/utils/throughput_benchmark.cpp",
        "torch/csrc/utils.cpp",
        "torch/csrc/utils/cuda_lazy_init.cpp",
        "torch/csrc/utils/invalid_arguments.cpp",
        "torch/csrc/utils/object_ptr.cpp",
        "torch/csrc/utils/python_arg_parser.cpp",
        "torch/csrc/utils/structseq.cpp",
        "torch/csrc/utils/tensor_apply.cpp",
        "torch/csrc/utils/tensor_dtypes.cpp",
        "torch/csrc/utils/tensor_layouts.cpp",
        "torch/csrc/utils/tensor_memoryformats.cpp",
        "torch/csrc/utils/tensor_qschemes.cpp",
        "torch/csrc/utils/tensor_list.cpp",
        "torch/csrc/utils/tensor_new.cpp",
        "torch/csrc/utils/tensor_numpy.cpp",
        "torch/csrc/utils/tensor_types.cpp",
        "torch/csrc/utils/tuple_parser.cpp",
        "test/cpp/jit/torch_python_test.cpp",
    ]

    libtorch_python_sources.extend(glob(["test/cpp/jit/test_*.cpp"]))

    libtorch_python_cuda_sources = [
        "torch/csrc/cuda/Event.cpp",
        "torch/csrc/cuda/Module.cpp",
        "torch/csrc/cuda/Storage.cpp",
        "torch/csrc/cuda/Stream.cpp",
        "torch/csrc/cuda/Tensor.cpp",
        "torch/csrc/cuda/python_comm.cpp",
        "torch/csrc/cuda/python_nccl.cpp",
        "torch/csrc/cuda/serialization.cpp",
        "torch/csrc/cuda/utils.cpp",
        "torch/csrc/distributed/c10d/ddp.cpp",
    ]

    compiler_flags_cpu = [
        "-D_THP_CORE",
        "-DUSE_C10D",
        "-DUSE_DISTRIBUTED",
        "-DUSE_NUMPY",
        "-DUSE_SCALARS",
        "-DNO_CUDNN_DESTROY_HANDLE",
        "-DPYTORCH_ONNX_CAFFE2_BUNDLE",
        "-Wno-write-strings",
        "-Wno-format",
        "-Wno-strict-aliasing",
        "-Wno-non-virtual-dtor",
        "-Wno-shadow-compatible-local",
        "-Wno-empty-body",
    ]
    compiler_flags_cuda = [
        "-DUSE_CUDNN",
        "-DUSE_NCCL",
    ]
    common_flags = {
        "compiler_specific_flags": {
            "clang": [
                "-Wno-absolute-value",
                "-Wno-expansion-to-defined",
                "-Wno-pessimizing-move",
                "-Wno-return-type-c-linkage",
                "-Wno-unknown-pragmas",
            ]
        },
        "headers": native.glob(["torch/csrc/**/*.h", "torch/csrc/generic/*.cpp", "test/cpp/jit/*.h"]),
    }
    propagated_pp_flags = [
        "-Icaffe2",
        "-Icaffe2/torch/csrc/api/include",
        "-Icaffe2/torch/csrc",
        "-Icaffe2/torch/csrc/nn",
        "-Icaffe2/torch/lib",
    ]

    cpp_library(
        name="libtorch",
        srcs=libtorch_sources,
        link_whole=True,
        propagated_pp_flags=propagated_pp_flags,
        deps=[
            ":generated-autograd-headers",
            ":generated-autograd-headers-bare",
            ":generated-jit-headers",
            "//caffe2/aten:ATen-cpu",
            "//caffe2/caffe2:caffe2_cpu",
            "//caffe2/torch/lib/libshm:libshm",
            "//caffe2/caffe2/quantization/server:dnnlowp_ops",
        ],
        external_deps=[
            ("nanopb", None, "protobuf-nanopb"),
            ("protobuf", None),
        ],
        compiler_flags=compiler_flags_cpu,
        **common_flags
    )

    cpp_library(
        name="libtorch_cuda",
        srcs=libtorch_cuda_sources,
        link_whole=True,
        # TODO: putting USE_CUDA in propagated_pp_flags is error-prone
        propagated_pp_flags=propagated_pp_flags + [
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
            "//caffe2/caffe2:caffe2_gpu",
            "//caffe2/torch/lib/libshm:libshm",
        ],
        external_deps=[
            ("cudnn", None, "cudnn-lazy"),
            ("nccl", None, "nccl-lazy"),
            ("cuda", None, "nvToolsExt-lazy"),
            ("cuda", None, "nvrtc-lazy"),
            ("cuda", None, "nvrtc-builtins-lazy"),
        ],
        compiler_flags=compiler_flags_cpu + compiler_flags_cuda,
        **common_flags
    )

    # torch-cpp is still conditionally compiled based on USE_CUDA. Ideally we'd
    # separate it out as an additive library instead.
    gpu_library_selector(
        name="torch-cpp",
        deps_cpu=[":torch-cpp-cpu"],
        deps_cuda=[":torch-cpp-cuda"],
        merge_cpu_deps=False,
    )

    # USE_CUDA flag is propagated through propagated_pp_flags on libtorch
    cpp_library(
        name="torch-cpp-cuda",
        srcs=torch_cpp_srcs,
        headers=torch_cpp_headers,
        header_namespace="torch",
        deps=[
            ":libtorch_cuda",
            "//caffe2/torch/fb/init:init",
        ],
        external_deps=[
            ("cuda", None, "cuda-lazy"),
            ("cudnn", None, "cudnn-lazy"),
        ],
    )

    cpp_library(
        name="torch-cpp-cpu",
        srcs=torch_cpp_srcs,
        headers=torch_cpp_headers,
        header_namespace="torch",
        deps=[
            ":libtorch",
            "//caffe2/torch/fb/init:init",
        ],
    )

    # _C_impl is still conditionally compiled based on USE_CUDA. Ideally we'd
    # separate it out as an additive library instead.
    # TODO: split it into cpp and cuda parts similarly to libtorch
    gpu_library_selector(
        name="_C_impl",
        deps_cpu=[":_C_impl_cpu"],
        deps_cuda=[":_C_impl_cuda"],
        merge_cpu_deps=False,
    )

    cpp_library(
        name="_C_impl_cpu",
        srcs=libtorch_python_sources,
        link_whole=True,
        deps=[
            ":torch-cpp-cpu",
            "//caffe2/torch/fb/init:init",
            "//caffe2/torch/lib/c10d:c10d_cpu",
            "//caffe2/torch/lib/libshm:libshm",
        ],
        external_deps=[
            ("numpy", None, "cpp"),
            ("pybind11", None),
            ("python", None),
        ],
        compiler_flags=compiler_flags_cpu,
        **common_flags
    )

    cpp_library(
        name="_C_impl_cuda",
        srcs=libtorch_python_sources + libtorch_python_cuda_sources,
        link_whole=True,
        deps=[
            ":torch-cpp-cuda",
            "//caffe2/torch/fb/init:init",
            "//caffe2/torch/lib/c10d:c10d",
            "//caffe2/torch/lib/libshm:libshm",
        ],
        external_deps=[
            ("numpy", None, "cpp"),
            ("pybind11", None),
            ("python", None),
        ],
        compiler_flags=compiler_flags_cpu + compiler_flags_cuda,
        **common_flags
    )

    cpp_python_extension(
        name="_C",
        srcs=[
            "torch/csrc/stub.cpp",
        ],
        base_module="torch",
        deps=[":_C_impl"],
    )

    return r
