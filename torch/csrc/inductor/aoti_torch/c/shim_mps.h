#ifndef AOTI_TORCH_SHIM_MPS
#define AOTI_TORCH_SHIM_MPS

#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AOTIMetalKernelFunctionOpaque;
using AOTIMetalKernelFunctionHandle = AOTIMetalKernelFunctionOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    AtenTensorHandle tensor);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle func,
    unsigned idx,
    int64_t val);

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

#ifdef __cplusplus
} // extern "C"
#endif

#endif // AOTI_TORCH_SHIM_MPS
