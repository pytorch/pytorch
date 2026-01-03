#ifndef AOTI_TORCH_SHIM_MPS
#define AOTI_TORCH_SHIM_MPS

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

struct AOTIMetalKernelFunctionOpaque;
using AOTIMetalKernelFunctionHandle = AOTIMetalKernelFunctionOpaque*;

struct AOTIMetalShaderLibraryOpaque;
using AOTIMetalShaderLibraryHandle = AOTIMetalShaderLibraryOpaque*;

#ifdef __cplusplus
extern "C" {
#endif

// MetalShaderLibrary functions
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_create_shader_library(
    const char* metal_shader_source,
    AOTIMetalShaderLibraryHandle* library_handle);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle library_handle);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle library_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* function_handle);

// MetalKernelFunction functions
AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_mps_start_encoding(AOTIMetalKernelFunctionHandle func);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AtenTensorHandle tensor);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    int64_t val);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length,
    uint64_t group_size);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_dispatch_array(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size,
    const uint64_t* group_size,
    size_t group_size_size);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_mps_malloc(void** buffer, size_t num_bytes);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_free(void* ptr);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_memcpy(
    void* buffer,
    size_t constant_offset,
    size_t bytes_read,
    size_t data_size,
    uint8_t* constants_start);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_copy_buffer(
    void* src_buffer,
    void* dst_buffer,
    size_t data_size,
    size_t src_offset,
    size_t dst_offset);

// C callback function type for command block execution
typedef void (*aoti_torch_mps_command_block_callback_t)(
    AOTIMetalKernelFunctionHandle func,
    void* user_data);

// Shared callback function for std::function trampoline
AOTI_TORCH_EXPORT void aoti_torch_mps_shared_callback(
    AOTIMetalKernelFunctionHandle func,
    void* user_data);

// Pure C version using function pointer and user data for trampoline pattern
AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_run_command_block(
    AOTIMetalKernelFunctionHandle func,
    aoti_torch_mps_command_block_callback_t callback,
    void* user_data);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOTI_TORCH_SHIM_MPS
