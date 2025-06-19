#ifndef AOTI_TORCH_SHIM_XPU
#define AOTI_TORCH_SHIM_XPU

#include <ATen/Config.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef USE_XPU
#ifdef __cplusplus
extern "C" {
#endif

struct XPUGuardOpaque;
using XPUGuardHandle = XPUGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_xpu_guard(
    int32_t device_index,
    XPUGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_xpu_guard(XPUGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_xpu_guard_set_index(XPUGuardHandle guard, int32_t device_index);

struct XPUStreamGuardOpaque;
using XPUStreamGuardHandle = XPUStreamGuardOpaque*;

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_create_xpu_stream_guard(
    void* stream,
    int32_t device_index,
    XPUStreamGuardHandle* ret_guard // returns new reference
);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_delete_xpu_stream_guard(XPUStreamGuardHandle guard);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_xpu_stream(int32_t device_index, void** ret_stream);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_get_current_xpu_device(int32_t* device_index);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_set_current_xpu_device(const int32_t& device_index);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_get_current_sycl_queue(void** ret);

#if AT_MKLDNN_ENABLED()

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_xpu_mkldnn__convolution_pointwise_binary(
    AtenTensorHandle X,
    AtenTensorHandle other,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    const char* binary_attr,
    double* alpha,
    const char** unary_attr,
    const double** unary_scalars,
    int64_t unary_scalars_len_,
    const char** unary_algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_xpu_mkldnn__convolution_pointwise(
    AtenTensorHandle X,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    const char* attr,
    const double** scalars,
    int64_t scalars_len_,
    const char** algorithm,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError
aoti_torch_xpu_mkldnn__convolution_pointwise_binary_(
    AtenTensorHandle other,
    AtenTensorHandle X,
    AtenTensorHandle W,
    AtenTensorHandle* B,
    const int64_t* padding,
    int64_t padding_len_,
    const int64_t* stride,
    int64_t stride_len_,
    const int64_t* dilation,
    int64_t dilation_len_,
    int64_t groups,
    const char* binary_attr,
    double* alpha,
    const char** unary_attr,
    const double** unary_scalars,
    int64_t unary_scalars_len_,
    const char** unary_algorithm,
    AtenTensorHandle* ret0);

#endif // AT_MKLDNN_ENABLED()
#ifdef __cplusplus
} // extern "C"
#endif

#endif // USE_XPU
#endif // AOTI_TORCH_SHIM_XPU
