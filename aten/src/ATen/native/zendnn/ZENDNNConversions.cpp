#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/native/zendnn/ZENDNNCommon.h>

namespace at {
namespace native {

#if AT_ZENDNN_ENABLED()

Tensor zendnn_to_dense(
    const Tensor& zendnn_tensor,
    c10::optional<ScalarType> dtype) {
  TORCH_CHECK(
      zendnn_tensor.scalar_type() == ScalarType::Float,
      "zendnn_to_dense expects float tensor input");
  zendnn::tensor& stensor = itensor_from_zendnn(zendnn_tensor);
  auto dims = stensor.get_dims();
  auto data_type =
      dtype.has_value() ? dtype.value() : zendnn_tensor.scalar_type();

  TORCH_CHECK(
      data_type == ScalarType::Float,
      "zendnn tensor only can be converted to be a float cpu tensor")
  // NOTE: int32_t dims from zendnn::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::empty(
      std::vector<int64_t>(dims.begin(), dims.end()),
      zendnn_tensor.options().layout(c10::kStrided).dtype(data_type));
  if (stensor.is_empty())
    return cpu_tensor;
  auto pub_tensor = data_type == ScalarType::Float
      ? stensor.to_public(
            cpu_tensor.template data_ptr<float>(),
            zendnn::tensor::data_type::f32)
      : stensor.to_public(
            cpu_tensor.template data_ptr<BFloat16>(),
            zendnn::tensor::data_type::bf16);
  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  return cpu_tensor;
}

Tensor dense_to_zendnn(
    const Tensor& cpu_tensor,
    c10::optional<ScalarType> dtype) {
  TORCH_CHECK(
      cpu_tensor.device().is_cpu(), "dense_to_zendnn expects CPU tensor input");
  TORCH_CHECK(
      cpu_tensor.layout() == Layout::Strided,
      "dense_to_zendnn expects strided tensor input");
  TORCH_CHECK(
      cpu_tensor.scalar_type() == ScalarType::Float,
      "dense_to_zendnn expects float tensor input");
  TORCH_CHECK(
      cpu_tensor.dim() <= 5,
      "dense_to_zendnn: Can't convert cpu tensor with the number of dimensions > 5");
  // TODO: consider to convert non-contiguous tensor to `zendnn::tensor`
  // directly.
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  auto data_type = dtype.has_value() ? dtype.value() : cpu_tensor.scalar_type();
  TORCH_CHECK(
      data_type == ScalarType::Float,
      "dense_to_zendnn: cpu tensor only can be converted to be a float zendnn tensor")
  Tensor zendnn_tensor = empty_zendnn(
      cpu_tensor_cont.sizes(),
      data_type,
      cpu_tensor_cont.options().layout_opt(),
      cpu_tensor_cont.options().device_opt(),
      cpu_tensor_cont.options().pinned_memory_opt());
  zendnn::tensor& dtensor = itensor_from_zendnn(zendnn_tensor);
  if (cpu_tensor.scalar_type() == ScalarType::Float) {
    dtensor.feed_from(
        dtensor.get_dims(),
        zendnn::tensor::data_type::f32,
        (cpu_tensor_cont.template data_ptr<float>()));
  } else {
    dtensor.feed_from(
        dtensor.get_dims(),
        zendnn::tensor::data_type::bf16,
        cpu_tensor_cont.template data_ptr<BFloat16>());
  }
  return zendnn_tensor;
}

#else

Tensor zendnn_to_dense(
    const Tensor& zendnn_tensor,
    c10::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "zendnn_to_dense: ATen not compiled with ZENDNN support");
}

Tensor dense_to_zendnn(
    const Tensor& cpu_tensor,
    c10::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "dense_to_zendnn: ATen not compiled with ZENDNN support");
}

#endif // AT_ZENDNN_ENABLED()

} // namespace native
} // namespace at
