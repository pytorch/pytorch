
#include <torch/csrc/inductor/aoti_torch/c/shim_xpu.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#include <c10/core/DeviceGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/core/StreamGuard.h>
#include <c10/xpu/XPUStream.h>

using namespace torch::aot_inductor;

AOTITorchError aoti_torch_create_xpu_guard(
    int32_t device_index,
    XPUGuardHandle* ret_guard // returns new reference
) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    at::DeviceGuard* guard =
        new at::DeviceGuard(at::Device(at::DeviceType::XPU, device_index));
    *ret_guard = reinterpret_cast<XPUGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_xpu_guard(XPUGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::DeviceGuard*>(guard); });
}

AOTITorchError aoti_torch_xpu_guard_set_index(
    XPUGuardHandle guard,
    int32_t device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { reinterpret_cast<at::DeviceGuard*>(guard)->set_index(device_index); });
}

AOTITorchError aoti_torch_create_xpu_stream_guard(
    void* stream,
    int32_t device_index,
    XPUStreamGuardHandle* ret_guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    assert(stream);
    at::StreamGuard* guard =
        new at::StreamGuard(at::xpu::getStreamFromExternal(
                                static_cast<sycl::queue*>(stream), device_index)
                                .unwrap());
    *ret_guard = reinterpret_cast<XPUStreamGuardHandle>(guard);
  });
}

AOTITorchError aoti_torch_delete_xpu_stream_guard(XPUStreamGuardHandle guard) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { delete reinterpret_cast<at::StreamGuard*>(guard); });
}

AOTITorchError aoti_torch_get_current_xpu_stream(
    int32_t device_index,
    void** ret_stream) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_stream = &(at::xpu::getCurrentXPUStream(device_index).queue()); });
}

AOTITorchError aoti_torch_get_current_xpu_device(int32_t* device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    *device_index =
        static_cast<int32_t>(static_cast<uint16_t>(c10::xpu::current_device()));
  });
}

AOTITorchError aoti_torch_set_current_xpu_device(const int32_t& device_index) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE(
      { c10::xpu::set_device(static_cast<int8_t>(device_index)); });
}

AOTITorchError aoti_torch_get_current_sycl_queue(void** ret) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    int32_t device_index =
        static_cast<int32_t>(static_cast<uint16_t>(c10::xpu::current_device()));
    *ret = &(at::xpu::getCurrentXPUStream(device_index).queue());
  });
}

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/xpu/Conv.h>

AOTITorchError aoti_torch_xpu_mkldnn__convolution_pointwise_binary(
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
    AtenTensorHandle* ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::List<std::optional<c10::Scalar>> unary_scalars_list;
    unary_scalars_list.reserve(unary_scalars_len_);
    for (int64_t i = 0; i < unary_scalars_len_; i++) {
      unary_scalars_list.emplace_back(pointer_to_optional(unary_scalars[i]));
    }
    auto tmp_result = at::native::xpu::convolution_pointwise_binary(
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(other),
        *tensor_handle_to_tensor_pointer(W),
        pointer_to_optional<at::Tensor>(B),
        pointer_to_list<int64_t>(padding, padding_len_),
        pointer_to_list<int64_t>(stride, stride_len_),
        pointer_to_list<int64_t>(dilation, dilation_len_),
        groups,
        binary_attr,
        pointer_to_optional<c10::Scalar>(alpha),
        pointer_to_optional<std::string_view>(unary_attr),
        unary_scalars_list,
        pointer_to_optional<std::string_view>(unary_algorithm));
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_xpu_mkldnn__convolution_pointwise_binary_(
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
    AtenTensorHandle* ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::List<std::optional<c10::Scalar>> unary_scalars_list;
    unary_scalars_list.reserve(unary_scalars_len_);
    for (int64_t i = 0; i < unary_scalars_len_; i++) {
      unary_scalars_list.emplace_back(pointer_to_optional(unary_scalars[i]));
    }
    auto tmp_result = at::native::xpu::convolution_pointwise_binary_(
        *tensor_handle_to_tensor_pointer(other),
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        pointer_to_optional<at::Tensor>(B),
        pointer_to_list<int64_t>(padding, padding_len_),
        pointer_to_list<int64_t>(stride, stride_len_),
        pointer_to_list<int64_t>(dilation, dilation_len_),
        groups,
        binary_attr,
        pointer_to_optional<c10::Scalar>(alpha),
        pointer_to_optional<std::string_view>(unary_attr),
        unary_scalars_list,
        pointer_to_optional<std::string_view>(unary_algorithm));
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_xpu_mkldnn__convolution_pointwise(
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
    AtenTensorHandle* ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    c10::List<std::optional<c10::Scalar>> scalars_list;
    scalars_list.reserve(scalars_len_);
    for (int64_t i = 0; i < scalars_len_; i++) {
      scalars_list.emplace_back(pointer_to_optional(scalars[i]));
    }
    auto tmp_result = at::native::xpu::convolution_pointwise(
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(W),
        pointer_to_optional<at::Tensor>(B),
        pointer_to_list<int64_t>(padding, padding_len_),
        pointer_to_list<int64_t>(stride, stride_len_),
        pointer_to_list<int64_t>(dilation, dilation_len_),
        groups,
        attr,
        scalars_list,
        pointer_to_optional<std::string_view>(algorithm));
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

#endif // AT_MKLDNN_ENABLED()
