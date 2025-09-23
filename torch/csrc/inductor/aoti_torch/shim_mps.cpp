#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <ATen/mps/MPSStream.h>
#include <unordered_map>

using namespace torch::aot_inductor;

// Global storage to keep shared_ptr alive while raw pointers are used
static std::unordered_map<at::native::mps::MetalKernelFunction*, std::shared_ptr<at::native::mps::MetalKernelFunction>> function_storage;

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle handle,
    unsigned idx,
    AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto t = tensor_handle_to_tensor_pointer(tensor);
    if (t == nullptr) {
      throw std::runtime_error("Tensor is null.");
    }
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->setArg(idx, *t);
  });
}

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle handle,
    unsigned idx,
    int64_t val) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->setArg(idx, val);
  });
}

// MetalShaderLibrary functions (pure C++)
AOTITorchError aoti_torch_mps_create_shader_library(
    const char* metal_shader_source,
    AOTIMetalShaderLibraryHandle* library_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* library = new at::native::mps::DynamicMetalShaderLibrary(std::string(metal_shader_source));
    *library_handle = reinterpret_cast<AOTIMetalShaderLibraryHandle>(library);
  });
}

AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle library_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* library = reinterpret_cast<at::native::mps::DynamicMetalShaderLibrary*>(library_handle);
    delete library;
  });
}

AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle library_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* function_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* library = reinterpret_cast<at::native::mps::DynamicMetalShaderLibrary*>(library_handle);
    auto function_shared_ptr = library->getKernelFunction(std::string(kernel_name));
    auto* raw_function = function_shared_ptr.get();

    // Store the shared_ptr to keep the object alive
    function_storage[raw_function] = function_shared_ptr;

    // Return raw pointer to match existing API
    *function_handle = reinterpret_cast<AOTIMetalKernelFunctionHandle>(raw_function);
  });
}

// MetalKernelFunction functions (pure C++)
AOTITorchError aoti_torch_mps_run_command_block(
    AOTIMetalKernelFunctionHandle func,
    std::function<void(AOTIMetalKernelFunctionHandle)> command_block) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr = reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    function_ptr->runCommandBlock([&command_block, func]() {
      command_block(func);
    });
  });
}

AOTITorchError aoti_torch_mps_start_encoding(
    AOTIMetalKernelFunctionHandle func) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr = reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    function_ptr->startEncoding();
  });
}

// Pure C dispatch functions - single value versions
AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr = reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    function_ptr->dispatch(length);
  });
}

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length,
    uint64_t group_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr = reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    function_ptr->dispatch(length, group_size);
  });
}

// Pure C dispatch functions - array versions
AOTITorchError aoti_torch_mps_dispatch_array(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr = reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    c10::ArrayRef<uint64_t> length_ref(length, length_size);
    function_ptr->dispatch(length_ref);
  });
}

AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size,
    const uint64_t* group_size,
    size_t group_size_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr = reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    c10::ArrayRef<uint64_t> length_ref(length, length_size);
    c10::ArrayRef<uint64_t> group_size_ref(group_size, group_size_size);
    function_ptr->dispatch(length_ref, group_size_ref);
  });
}

AOTITorchError aoti_torch_mps_synchronize_stream() {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* stream = at::mps::getCurrentMPSStream();
    stream->synchronize(at::mps::SyncType::COMMIT_AND_WAIT);
  });
}
