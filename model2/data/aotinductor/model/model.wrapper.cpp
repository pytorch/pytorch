
#include <torch/csrc/inductor/aoti_include/cuda.h>
// Definition of AOTI runtime interface functions

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>

#include <iostream>
#include <vector>


#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)      \
  try {                                           \
    __VA_ARGS__                                   \
  } catch (const std::exception& e) {             \
    std::cerr << "Error: " << e.what() << '\n';   \
    return AOTI_RUNTIME_FAILURE;                  \
  } catch (...) {                                 \
    std::cerr << "Unknown exception occurred.\n"; \
    return AOTI_RUNTIME_FAILURE;                  \
  }                                               \
  return AOTI_RUNTIME_SUCCESS;

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't supports
// arguments that requires gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(false);
  }
  AOTINoGradGuard(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard(AOTINoGradGuard&&) noexcept = delete;
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  AOTINoGradGuard& operator=(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard& operator=(AOTINoGradGuard&&) noexcept = delete;
  bool prev_mode{aoti_torch_grad_mode_is_enabled()};
};

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
}

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {
  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0\n";
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}

AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerRunSingleThreaded(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_single_threaded(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *num_constants = container->num_constants(); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *original_fqn = container->constant_original_fqn(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantType(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* type) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *type = container->constant_type(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDataSize(
  AOTInductorModelContainerHandle container_handle,
  size_t idx,
  size_t* data_size) {
  auto* container =
    reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
        container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *data_size = container->constant_data_size(idx); })
}

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMap(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto constants_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { const auto ret = container->extract_constants_map(use_inactive);
      for (const auto& pair: ret) {
        constants_map->emplace(pair.first, pair.second);
      }
    })
}

AOTIRuntimeError AOTInductorModelContainerUpdateUserManagedConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update, /* user_managed = */ true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(container_handle,
          constant_map_handle,
          /*use_inactive*/ true,
          /*validate_full_update*/ true);
}

AOTIRuntimeError AOTInductorModelContainerFreeInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->free_inactive_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
      auto constant_array = std::make_shared<std::vector<torch::aot_inductor::ConstantHandle>>();
      auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);

      auto model = new torch::aot_inductor::AOTInductorModel(
          constant_map,
          constant_array,
          "cpu", // device_str is hardcoded, as AOTInductorModelCreate is only use for CPU models
          ""
      );

      if (input_map) {
        for (auto const& kv : *input_map) {
          constant_map->emplace(kv.first, kv.second);
        }
      } else {
        model->load_constants();
      }

      *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
    })}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto input_map =
        reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);

    for (auto const& kv : *input_map) {
      constant_map->emplace(kv.first, kv.second);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

} // extern "C"


#define CUDA_DRIVER_CHECK(EXPR)                    \
do {                                               \
    CUresult code = EXPR;                          \
    const char *msg;                               \
    CUresult code_get_error = cuGetErrorString(code, &msg); \
    if (code_get_error != CUDA_SUCCESS) {          \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string("invalid error code!"));   \
    }                                              \
    if (code != CUDA_SUCCESS) {                    \
        throw std::runtime_error(                  \
            std::string("CUDA driver error: ") +   \
            std::string(msg));                     \
    }                                              \
} while (0);

static inline CUfunction loadKernel(
        std::string filePath,
        const std::string &funcName,
        uint32_t sharedMemBytes,
        const std::optional<std::string> &cubinDir = std::nullopt) {
    if (cubinDir) {
        std::filesystem::path p1{*cubinDir};
        std::filesystem::path p2{filePath};
        filePath = (p1 / p2.filename()).string();
    }

    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoad(&mod, filePath.c_str()));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline CUfunction loadKernel(const void* start, const std::string &funcName, uint32_t sharedMemBytes) {
    CUmodule mod;
    CUfunction func;
    CUDA_DRIVER_CHECK(cuModuleLoadData(&mod, start));
    CUDA_DRIVER_CHECK(cuModuleGetFunction(&func, mod, funcName.c_str()));
    if (sharedMemBytes > 0) {
        CUDA_DRIVER_CHECK(cuFuncSetAttribute(
            func,
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            sharedMemBytes
        ))
    }
    return func;
}

static inline void launchKernel(
        CUfunction func,
        uint32_t gridX,
        uint32_t gridY,
        uint32_t gridZ,
        uint32_t numWarps,
        uint32_t sharedMemBytes,
        void* args[],
        cudaStream_t stream) {
    CUDA_DRIVER_CHECK(cuLaunchKernel(
        func, gridX, gridY, gridZ, 32*numWarps, 1, 1, sharedMemBytes, stream, args, nullptr
    ));
}
CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DEVICE(cuda);
CACHE_TORCH_LAYOUT(strided);
namespace torch::aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
    CUfunction model_triton_tem_fused_addmm_relu_sigmoid_t_1{nullptr};
    CUfunction model_triton_tem_fused_addmm_relu_t_0{nullptr};
};
}  // namespace



AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(1,
                           1,
                           3,
                           device_str,
                           std::move(cubin_dir),
                           true) {
    inputs_info_[0].name = "arg4_1";
    constants_info_[0].name = "fc1_weight";
    constants_info_[0].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 640;
    constants_info_[0].from_folded = false;
    constants_info_[0].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[0].shape = {16, 10};
    constants_info_[0].stride = {10, 1};
    constants_info_[0].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[0].original_fqn = "fc1.weight";
    constants_info_[1].name = "fc1_bias";
    constants_info_[1].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[1].offset = 0;
    constants_info_[1].data_size = 64;
    constants_info_[1].from_folded = false;
    constants_info_[1].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[1].shape = {16};
    constants_info_[1].stride = {1};
    constants_info_[1].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[1].original_fqn = "fc1.bias";
    constants_info_[2].name = "fc2_weight";
    constants_info_[2].dtype = static_cast<int32_t>(cached_torch_dtype_float32);
    constants_info_[2].offset = 0;
    constants_info_[2].data_size = 64;
    constants_info_[2].from_folded = false;
    constants_info_[2].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Parameter);
    constants_info_[2].shape = {1, 16};
    constants_info_[2].stride = {16, 1};
    constants_info_[2].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[2].original_fqn = "fc2.weight";
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = R"([1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.dict", "context": "[]", "children_spec": []}]}])";
    out_spec_ = R"([1, {"type": null, "context": null, "children_spec": []}])";
    outputs_info_[0].name = "output0";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization
) {

    if (!initialization) {
        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                  << "aot_inductor.use_runtime_constant_folding=False\n";
    }
    return {};
}
} // namespace torch::aot_inductor
using namespace torch::aot_inductor;

template <typename arg_A_type_, typename arg_B_type_, typename in_ptr2_type_, typename out_ptr1_type_, typename kernels_type_>
static inline void call_model_triton_tem_fused_addmm_relu_t_0(
    const arg_A_type_& arg_A,
    const arg_B_type_& arg_B,
    const in_ptr2_type_& in_ptr2,
    const out_ptr1_type_& out_ptr1,
    int64_t ks0,
    int64_t _grid_0,
    int64_t _grid_1,
    int64_t _grid_2,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('model_triton_tem_fused_addmm_relu_t_0', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

    @triton_heuristics.template(

    num_stages=1,
    num_warps=1,
    triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'in_ptr2': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=76, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'kernel_name': 'model_triton_tem_fused_addmm_relu_t_0', 'backend_hash': '7606AC1BD735D3E3F140115999815ACFE642967D9047962703B386A670225BF4', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': False, 'ALLOW_TF32': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16, 'GROUP_M': 8}},

    )
    @triton.jit
    def model_triton_tem_fused_addmm_relu_t_0(arg_A, arg_B, in_ptr2, out_ptr1, ks0):
        EVEN_K : tl.constexpr = False
        ALLOW_TF32 : tl.constexpr = False
        USE_FAST_ACCUM : tl.constexpr = False
        ACC_TYPE : tl.constexpr = tl.float32
        BLOCK_M : tl.constexpr = 16
        BLOCK_N : tl.constexpr = 16
        BLOCK_K : tl.constexpr = 16
        GROUP_M : tl.constexpr = 8
        INDEX_DTYPE : tl.constexpr = tl.int32
        A = arg_A
        B = arg_B

        M = ks0
        N = 16
        K = 10
        if M * N == 0:
            # early exit due to zero-size input(s)
            return
        stride_am = 10
        stride_ak = 1
        stride_bk = 1
        stride_bn = 10

        # based on triton.ops.matmul
        pid = tl.program_id(0)
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N

        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
            offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        else:
            offs_a_m = rm % M
        if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
            offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        else:
            offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

            a_mask = offs_k[None, :] < (K - k_idx * BLOCK_K)
            b_mask = offs_k[:, None] < (K - k_idx * BLOCK_K)

            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

            idx_m = offs_a_m[:, None]
            idx_n = a_k_idx_vals
            xindex = idx_n + 10*idx_m
            a = tl.load(A + (xindex), mask=a_mask, other=0.0)

            idx_m = b_k_idx_vals
            idx_n = offs_b_n[None, :]
            xindex = idx_n + 16*idx_m
            b = tl.load(B + ((tl.broadcast_to(idx_m + 10*idx_n, xindex.shape)).broadcast_to(xindex.shape)), mask=b_mask, other=0.0)


            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # inductor generates a suffix
        xindex = idx_n + 16*idx_m
        tmp0 = tl.load(in_ptr2 + (tl.broadcast_to(idx_n, acc.shape)), mask, eviction_policy='evict_last')
        tmp1 = acc + tmp0
        tmp2 = tl.full([1], 0, tl.int32)
        tmp3 = triton_helpers.maximum(tmp2, tmp1)
        tl.store(out_ptr1 + (tl.broadcast_to(idx_n + 16*idx_m, acc.shape)), tmp3, mask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = _grid_0;
    uint32_t grid_1 = _grid_1;
    uint32_t grid_2 = _grid_2;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.model_triton_tem_fused_addmm_relu_t_0 == nullptr) {
        kernels_.model_triton_tem_fused_addmm_relu_t_0 = loadKernel("/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model_triton_tem_fused_addmm_relu_t_0.cubin", "model_triton_tem_fused_addmm_relu_t_0", 2048, cubin_dir_); 
    }
    CUdeviceptr var_0 = reinterpret_cast<CUdeviceptr>(arg_A.data_ptr());
    CUdeviceptr var_1 = reinterpret_cast<CUdeviceptr>(arg_B.data_ptr());
    CUdeviceptr var_2 = reinterpret_cast<CUdeviceptr>(in_ptr2.data_ptr());
    CUdeviceptr var_3 = reinterpret_cast<CUdeviceptr>(out_ptr1.data_ptr());
    int32_t var_4 = ks0;
    CUdeviceptr global_scratch_scratch_5 = 0;
    void* kernel_args_[] = {&var_0, &var_1, &var_2, &var_3, &var_4, &global_scratch_scratch_5};
    launchKernel(kernels_.model_triton_tem_fused_addmm_relu_t_0, grid_0, grid_1, grid_2, 1, 2048, kernel_args_, stream_);
}

template <typename arg_A_type_, typename arg_B_type_, typename out_ptr1_type_, typename kernels_type_>
static inline void call_model_triton_tem_fused_addmm_relu_sigmoid_t_1(
    const arg_A_type_& arg_A,
    const arg_B_type_& arg_B,
    const out_ptr1_type_& out_ptr1,
    int64_t ks0,
    int64_t _grid_0,
    int64_t _grid_1,
    int64_t _grid_2,
    int32_t device_idx_,
    cudaStream_t stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('model_triton_tem_fused_addmm_relu_sigmoid_t_1', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

    @triton_heuristics.template(

    num_stages=1,
    num_warps=1,
    triton_meta={'signature': {'arg_A': '*fp32', 'arg_B': '*fp32', 'out_ptr1': '*fp32', 'ks0': 'i32'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=76, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'kernel_name': 'model_triton_tem_fused_addmm_relu_sigmoid_t_1', 'backend_hash': '7606AC1BD735D3E3F140115999815ACFE642967D9047962703B386A670225BF4', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': True, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'grid_type': 'FixedGrid', 'fixed_grid': ['_grid_0', '_grid_1', '_grid_2'], 'extra_launcher_args': ['_grid_0', '_grid_1', '_grid_2'], 'config_args': {'EVEN_K': True, 'ALLOW_TF32': False, 'USE_FAST_ACCUM': False, 'ACC_TYPE': 'tl.float32', 'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16, 'GROUP_M': 8}},

    )
    @triton.jit
    def model_triton_tem_fused_addmm_relu_sigmoid_t_1(arg_A, arg_B, out_ptr1, ks0):
        EVEN_K : tl.constexpr = True
        ALLOW_TF32 : tl.constexpr = False
        USE_FAST_ACCUM : tl.constexpr = False
        ACC_TYPE : tl.constexpr = tl.float32
        BLOCK_M : tl.constexpr = 16
        BLOCK_N : tl.constexpr = 16
        BLOCK_K : tl.constexpr = 16
        GROUP_M : tl.constexpr = 8
        INDEX_DTYPE : tl.constexpr = tl.int32
        A = arg_A
        B = arg_B

        M = ks0
        N = 1
        K = 16
        if M * N == 0:
            # early exit due to zero-size input(s)
            return
        stride_am = 16
        stride_ak = 1
        stride_bk = 1
        stride_bn = 16

        # based on triton.ops.matmul
        pid = tl.program_id(0)
        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N

        # re-order program ID for better L2 performance
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // (group_size)
        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        if ((stride_am == 1 and stride_ak == M) or (stride_am == K and stride_ak == 1)) and (M >= BLOCK_M and K > 1):
            offs_a_m = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
        else:
            offs_a_m = rm % M
        if ((stride_bk == 1 and stride_bn == K) or (stride_bk == N and stride_bn == 1)) and (N >= BLOCK_N and K > 1):
            offs_b_n = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
        else:
            offs_b_n = rn % N
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

        for k_idx in range(0, tl.cdiv(K, BLOCK_K)):

            a_k_idx_vals = offs_k[None, :] + (k_idx * BLOCK_K)
            b_k_idx_vals = offs_k[:, None] + (k_idx * BLOCK_K)

            idx_m = offs_a_m[:, None]
            idx_n = a_k_idx_vals
            xindex = idx_n + 16*idx_m
            a = tl.load(A + (xindex))

            idx_m = b_k_idx_vals
            idx_n = offs_b_n[None, :]
            xindex = idx_m + idx_n
            b = tl.load(B + ((tl.broadcast_to(idx_m, xindex.shape)).broadcast_to(xindex.shape)))


            acc += tl.dot(a, b, allow_tf32=ALLOW_TF32, out_dtype=ACC_TYPE)


        # rematerialize rm and rn to save registers
        rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        idx_m = rm[:, None]
        idx_n = rn[None, :]
        mask = (idx_m < M) & (idx_n < N)

        # inductor generates a suffix
        xindex = idx_m + idx_n
        tmp0 = -0.13563597202301025
        tmp1 = acc + tmp0
        tmp2 = tl.sigmoid(tmp1)
        tl.store(out_ptr1 + (tl.broadcast_to(xindex, acc.shape)), tmp2, mask)
    ''', device_str='cuda')
    */
    uint32_t grid_0 = _grid_0;
    uint32_t grid_1 = _grid_1;
    uint32_t grid_2 = _grid_2;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.model_triton_tem_fused_addmm_relu_sigmoid_t_1 == nullptr) {
        kernels_.model_triton_tem_fused_addmm_relu_sigmoid_t_1 = loadKernel("/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model_triton_tem_fused_addmm_relu_sigmoid_t_1.cubin", "model_triton_tem_fused_addmm_relu_sigmoid_t_1", 2048, cubin_dir_); 
    }
    CUdeviceptr var_6 = reinterpret_cast<CUdeviceptr>(arg_A.data_ptr());
    CUdeviceptr var_7 = reinterpret_cast<CUdeviceptr>(arg_B.data_ptr());
    CUdeviceptr var_8 = reinterpret_cast<CUdeviceptr>(out_ptr1.data_ptr());
    int32_t var_9 = ks0;
    CUdeviceptr global_scratch_scratch_10 = 0;
    void* kernel_args_[] = {&var_6, &var_7, &var_8, &var_9, &global_scratch_scratch_10};
    launchKernel(kernels_.model_triton_tem_fused_addmm_relu_sigmoid_t_1, grid_0, grid_1, grid_2, 1, 2048, kernel_args_, stream_);
}

namespace torch::aot_inductor {

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {}

AOTI_NOINLINE static void check_input_0(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg4_1 = ConstantHandle(input_handles[0]);
    int32_t arg4_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg4_1, &arg4_1_dtype));

    int32_t arg4_1_expected_dtype = aoti_torch_dtype_float32();
    if (arg4_1_expected_dtype != arg4_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dtype, "
           << "expected: " << arg4_1_expected_dtype << "(at::kFloat), "
           << "but got: " << arg4_1_dtype << "\n";
        throw std::runtime_error(ss.str());
    }
    auto arg4_1_size = arg4_1.sizes();

    if (arg4_1_size[0] < 2) {
        std::stringstream ss;
        ss << "input_handles[0]: dim value is too small at 0, "
           << "expected it to be >= 2, " << "but got: "
           << arg4_1_size[0] << "\n";
        throw std::runtime_error(ss.str());
    }

    if (arg4_1_size[0] > 1024) {
        std::stringstream ss;
        ss << "input_handles[0]: dim value is too large at 0, "
           << "expected to be <= 1024, " << "but got: "
           << arg4_1_size[0] << "\n";
        throw std::runtime_error(ss.str());
    }

    if (10 != arg4_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 1, "
           << "expected: 10, " << "but got: " << arg4_1_size[1]
           << "\n";
        throw std::runtime_error(ss.str());
    }
    auto arg4_1_stride = arg4_1.strides();

    if (10 != arg4_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 0, "
           << "expected: 10, " << "but got: " << arg4_1_stride[0]
           << "\n";
        throw std::runtime_error(ss.str());
    }

    if (1 != arg4_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 1, "
           << "expected: 1, " << "but got: " << arg4_1_stride[1]
           << "\n";
        throw std::runtime_error(ss.str());
    }
    int32_t arg4_1_device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(arg4_1, &arg4_1_device_type));

    int32_t arg4_1_expected_device_type = 1;
    if (arg4_1_expected_device_type != arg4_1_device_type) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched device type, "
        << "expected: " << arg4_1_expected_device_type << "1(cuda), "
        << "but got: " << arg4_1_device_type << "\n";
        throw std::runtime_error(ss.str());
    }
}

static bool _check_aoti_runtime_check_inputs_env() {
    const static char* env_var_value = getenv("AOTI_RUNTIME_CHECK_INPUTS");
    const static bool result = env_var_value != nullptr && env_var_value[0] != '0';
    return result;
}

AOTI_NOINLINE static void __check_inputs_outputs(
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
    if (!_check_aoti_runtime_check_inputs_env()){
        return;
    }
    check_input_0(input_handles);
}

void AOTInductorModel::run_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {
    __check_inputs_outputs(input_handles, output_handles);

    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 1);
    auto arg4_1 = std::move(inputs[0]);
    [[maybe_unused]] auto& fc1_weight = constants_->at(0);
    [[maybe_unused]] auto& fc1_bias = constants_->at(1);
    [[maybe_unused]] auto& fc2_weight = constants_->at(2);

    if ((long(arg4_1.data_ptr()) & (16 -1)) != 0) {
        AOTI_TORCH_WARN("Input 0 was compiled as 16-bytes aligned, but it is not aligned at run time. Copying to an aligned tensor to guarantee correctness, but expect a performance hit.");
        AtenTensorHandle arg4_1_aligned;
        aoti_torch_clone_preserve_strides(arg4_1, &arg4_1_aligned);
        arg4_1 = std::move(RAIIAtenTensorHandle(arg4_1_aligned));
    }
    auto arg4_1_size = arg4_1.sizes();
    int64_t s77 = arg4_1_size[0];
    inputs.clear();
    [[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());
    int64_t _xnumel = 16L*s77;

    AOTICudaStreamGuard stream_guard(stream, this->device_idx_);
    const int64_t int_array_0[] = {s77, 16L};
    static constexpr int64_t int_array_1[] = {16L, 1L};
    AtenTensorHandle buf1_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf1_handle));
    RAIIAtenTensorHandle buf1(buf1_handle);
    // Topologically Sorted Source Nodes: [linear, relu], Original ATen: [aten.t, aten.addmm, aten.relu]
    call_model_triton_tem_fused_addmm_relu_t_0(arg4_1, fc1_weight, fc1_bias, buf1, s77, c10::div_floor_integer(static_cast<int64_t>(15L + s77), static_cast<int64_t>(16L)), 1, 1, this->device_idx_, stream, kernels, this->cubin_dir_);
    arg4_1.reset();
    const int64_t int_array_2[] = {s77, 1L};
    static constexpr int64_t int_array_3[] = {1L, 1L};
    AtenTensorHandle buf3_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_2, int_array_3, cached_torch_dtype_float32, cached_torch_device_type_cuda, this->device_idx_, &buf3_handle));
    RAIIAtenTensorHandle buf3(buf3_handle);
    // Topologically Sorted Source Nodes: [linear, relu, linear_1, sigmoid], Original ATen: [aten.addmm, aten.relu, aten.t, aten.sigmoid]
    call_model_triton_tem_fused_addmm_relu_sigmoid_t_1(buf1, fc2_weight, buf3, s77, c10::div_floor_integer(static_cast<int64_t>(15L + s77), static_cast<int64_t>(16L)), 1, 1, this->device_idx_, stream, kernels, this->cubin_dir_);
    buf1.reset();
    output_handles[0] = buf3.release();
} // AOTInductorModel::run_impl
} // namespace torch::aot_inductor





// Compile cmd
// /home/shangdiy/miniconda3/envs/pytorch-3.10/bin/x86_64-conda-linux-gnu-c++ /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/include/python3.10" /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/Include" /I "/home/shangdiy/pytorch/torch/include" /I "/home/shangdiy/pytorch/torch/include/torch/csrc/api/include" /I "/usr/local/cuda-12/include"  /D NOMINMAX /D TORCH_INDUCTOR_CPP_WRAPPER /D STANDALONE_TORCH_HEADER /D  C10_USING_CUSTOM_GENERATED_MACROS /D  USE_CUDA  /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental  /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.cpp   /c /Fo/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.obj
// Link cmd
// /home/shangdiy/miniconda3/envs/pytorch-3.10/bin/x86_64-conda-linux-gnu-c++ /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/include/python3.10" /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/Include" /I "/home/shangdiy/pytorch/torch/include" /I "/home/shangdiy/pytorch/torch/include/torch/csrc/api/include" /I "/usr/local/cuda-12/include"  /D NOMINMAX /D TORCH_INDUCTOR_CPP_WRAPPER /D STANDALONE_TORCH_HEADER /D  C10_USING_CUSTOM_GENERATED_MACROS /D  USE_CUDA  /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental  /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.obj /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.kernel.obj /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model/model_consts.weights.obj   /Fe/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.pyd /LD /link /LIBPATH:"/home/shangdiy/miniconda3/envs/pytorch-3.10/libs" /LIBPATH:"/home/shangdiy/pytorch/torch/lib" /LIBPATH:"/usr/local/cuda-12.8/targets/x86_64-linux/lib" /LIBPATH:"/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs"  "torch.lib" "torch_cpu.lib" "sleef.lib" "c10.lib" "c10_cuda.lib" "cuda.lib" "torch_cuda.lib"  

// Compile cmd
// /home/shangdiy/miniconda3/envs/pytorch-3.10/bin/x86_64-conda-linux-gnu-c++ /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/include/python3.10" /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/Include" /I "/home/shangdiy/pytorch/torch/include" /I "/home/shangdiy/pytorch/torch/include/torch/csrc/api/include" /I "/usr/local/cuda-12/include"  /D NOMINMAX /D TORCH_INDUCTOR_CPP_WRAPPER /D STANDALONE_TORCH_HEADER /D  C10_USING_CUSTOM_GENERATED_MACROS /D  USE_CUDA  /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental  /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.cpp   /c /Fo/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.obj
// Link cmd
// /home/shangdiy/miniconda3/envs/pytorch-3.10/bin/x86_64-conda-linux-gnu-c++ /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/include/python3.10" /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/Include" /I "/home/shangdiy/pytorch/torch/include" /I "/home/shangdiy/pytorch/torch/include/torch/csrc/api/include" /I "/usr/local/cuda-12/include"  /D NOMINMAX /D TORCH_INDUCTOR_CPP_WRAPPER /D STANDALONE_TORCH_HEADER /D  C10_USING_CUSTOM_GENERATED_MACROS /D  USE_CUDA  /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental  /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.obj /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.kernel.obj /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model/model_consts.weights.obj   /Fe/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.pyd /LD /link /LIBPATH:"/home/shangdiy/miniconda3/envs/pytorch-3.10/libs" /LIBPATH:"/home/shangdiy/pytorch/torch/lib" /LIBPATH:"/usr/local/cuda-12.8/targets/x86_64-linux/lib" /LIBPATH:"/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs"  "torch.lib" "torch_cpu.lib" "sleef.lib" "c10.lib" "c10_cuda.lib" "cuda.lib" "torch_cuda.lib"  

// Compile cmd
// /home/shangdiy/miniconda3/envs/pytorch-3.10/bin/x86_64-conda-linux-gnu-c++ /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/include/python3.10" /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/Include" /I "/home/shangdiy/pytorch/torch/include" /I "/home/shangdiy/pytorch/torch/include/torch/csrc/api/include" /I "/usr/local/cuda-12/include"  /D NOMINMAX /D TORCH_INDUCTOR_CPP_WRAPPER /D STANDALONE_TORCH_HEADER /D  C10_USING_CUSTOM_GENERATED_MACROS /D  USE_CUDA  /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental  /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.cpp   /c /Fo/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.obj
// Link cmd
// /home/shangdiy/miniconda3/envs/pytorch-3.10/bin/x86_64-conda-linux-gnu-c++ /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/include/python3.10" /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/Include" /I "/home/shangdiy/pytorch/torch/include" /I "/home/shangdiy/pytorch/torch/include/torch/csrc/api/include" /I "/usr/local/cuda-12/include"  /D NOMINMAX /D TORCH_INDUCTOR_CPP_WRAPPER /D STANDALONE_TORCH_HEADER /D  C10_USING_CUSTOM_GENERATED_MACROS /D  USE_CUDA  /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental  /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.obj /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.kernel.obj /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model/model_consts.weights.obj   /Fe/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.pyd /LD /link /LIBPATH:"/home/shangdiy/miniconda3/envs/pytorch-3.10/libs" /LIBPATH:"/home/shangdiy/pytorch/torch/lib" /LIBPATH:"/usr/local/cuda-12.8/targets/x86_64-linux/lib" /LIBPATH:"/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs"  "torch.lib" "torch_cpu.lib" "sleef.lib" "c10.lib" "c10_cuda.lib" "cuda.lib" "torch_cuda.lib"  

// Compile cmd
// /home/shangdiy/miniconda3/envs/pytorch-3.10/bin/x86_64-conda-linux-gnu-c++ /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/include/python3.10" /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/Include" /I "/home/shangdiy/pytorch/torch/include" /I "/home/shangdiy/pytorch/torch/include/torch/csrc/api/include" /I "/usr/local/cuda-12/include"  /D NOMINMAX /D TORCH_INDUCTOR_CPP_WRAPPER /D STANDALONE_TORCH_HEADER /D  C10_USING_CUSTOM_GENERATED_MACROS /D CPU_CAPABILITY_AVX512 /D  USE_CUDA  /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental  /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.cpp  -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mavx512bf16  /c /Fo/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.obj
// Link cmd
// /home/shangdiy/miniconda3/envs/pytorch-3.10/bin/x86_64-conda-linux-gnu-c++ /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/include/python3.10" /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/Include" /I "/home/shangdiy/pytorch/torch/include" /I "/home/shangdiy/pytorch/torch/include/torch/csrc/api/include" /I "/usr/local/cuda-12/include"  /D NOMINMAX /D TORCH_INDUCTOR_CPP_WRAPPER /D STANDALONE_TORCH_HEADER /D  C10_USING_CUSTOM_GENERATED_MACROS /D CPU_CAPABILITY_AVX512 /D  USE_CUDA  /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental  /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.obj /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.kernel.obj /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model/model_consts.weights.obj  -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mavx512bf16  /Fe/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.pyd /LD /link /LIBPATH:"/home/shangdiy/miniconda3/envs/pytorch-3.10/libs" /LIBPATH:"/home/shangdiy/pytorch/torch/lib" /LIBPATH:"/usr/local/cuda-12.8/targets/x86_64-linux/lib" /LIBPATH:"/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs"  "torch.lib" "torch_cpu.lib" "sleef.lib" "c10.lib" "c10_cuda.lib" "cuda.lib" "torch_cuda.lib"  

// Compile cmd
// /home/shangdiy/miniconda3/envs/pytorch-3.10/bin/x86_64-conda-linux-gnu-c++ /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/include/python3.10" /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/Include" /I "/home/shangdiy/pytorch/torch/include" /I "/home/shangdiy/pytorch/torch/include/torch/csrc/api/include" /I "/usr/local/cuda-12/include"  /D NOMINMAX /D TORCH_INDUCTOR_CPP_WRAPPER /D STANDALONE_TORCH_HEADER /D  C10_USING_CUSTOM_GENERATED_MACROS /D CPU_CAPABILITY_AVX512 /D  USE_CUDA  /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental  /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.cpp  -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mavx512bf16  /c /Fo/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.obj
// Link cmd
// /home/shangdiy/miniconda3/envs/pytorch-3.10/bin/x86_64-conda-linux-gnu-c++ /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/include/python3.10" /I "/home/shangdiy/miniconda3/envs/pytorch-3.10/Include" /I "/home/shangdiy/pytorch/torch/include" /I "/home/shangdiy/pytorch/torch/include/torch/csrc/api/include" /I "/usr/local/cuda-12/include"  /D NOMINMAX /D TORCH_INDUCTOR_CPP_WRAPPER /D STANDALONE_TORCH_HEADER /D  C10_USING_CUSTOM_GENERATED_MACROS /D CPU_CAPABILITY_AVX512 /D  USE_CUDA  /O2 /DLL /MD /std:c++20 /wd4819 /wd4251 /wd4244 /wd4267 /wd4275 /wd4018 /wd4190 /wd4624 /wd4067 /wd4068 /EHsc /Zc:__cplusplus /permissive- /openmp /openmp:experimental  /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.obj /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.kernel.obj /tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model/model_consts.weights.obj  -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mavx512bf16  /Fe/tmp/torchinductor_shangdiy/cx7jxbnff2tlwdz2gpv4yy5zoxvd7b6o2t5zekqulqe6zo5ld5vs/model.wrapper.pyd /LD /link /LIBPATH:"/home/shangdiy/miniconda3/envs/pytorch-3.10/libs" /LIBPATH:"/home/shangdiy/pytorch/torch/lib" /LIBPATH:"/usr/local/cuda-12.8/targets/x86_64-linux/lib" /LIBPATH:"/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs"  "torch.lib" "torch_cpu.lib" "sleef.lib" "c10.lib" "c10_cuda.lib" "cuda.lib" "torch_cuda.lib"  
