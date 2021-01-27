#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#include <ATen/native/utils/ParamUtils.h>

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(mkldnn_tensor.scalar_type() == ScalarType::Float ||
              mkldnn_tensor.scalar_type() == ScalarType::BFloat16,
              "mkldnn_to_dense expects float or bfloat16 tensor input");
  ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
  auto dims = stensor.get_dims();
  auto data_type = dtype.has_value() ? dtype.value() : mkldnn_tensor.scalar_type();
  TORCH_CHECK(data_type == ScalarType::Float || data_type == ScalarType::BFloat16,
              "mkldnn tensor only can be converted to be a float or bfloat16 cpu tensor")
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::empty(
    std::vector<int64_t>(dims.begin(), dims.end()),
    mkldnn_tensor.options().layout(c10::kStrided).dtype(data_type));
  if (stensor.is_empty()) return cpu_tensor;
  auto pub_tensor =
      data_type == ScalarType::Float
      ? stensor.to_public(cpu_tensor.template data_ptr<float>(),
                          ideep::tensor::data_type::f32)
      : stensor.to_public(cpu_tensor.template data_ptr<BFloat16>(),
                         ideep::tensor::data_type::bf16);
  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  return cpu_tensor;
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(cpu_tensor.device().type() == DeviceType::CPU,
             "dense_to_mkldnn expects CPU tensor input");
  TORCH_CHECK(cpu_tensor.layout() == Layout::Strided,
             "dense_to_mkldnn expects strided tensor input");
  TORCH_CHECK(cpu_tensor.scalar_type() == ScalarType::Float ||
              cpu_tensor.scalar_type() == ScalarType::BFloat16,
             "dense_to_mkldnn expects float or bfloat16 tensor input");
  TORCH_CHECK(cpu_tensor.dim() <= 5,
             "Can't convert cpu tensor with the number of dimensions > 5");
  // TODO: consider to convert non-contiguous tensor to `ideep::tensor` directly.
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  auto data_type = dtype.has_value() ? dtype.value() : cpu_tensor.scalar_type();
  TORCH_CHECK(data_type == ScalarType::Float || data_type == ScalarType::BFloat16,
              "cpu tensor only can be converted to be a float or bfloat16 mkldnn tensor")
  Tensor mkldnn_tensor = empty_mkldnn(cpu_tensor_cont.sizes(), data_type,
                                      cpu_tensor_cont.options().layout_opt(), cpu_tensor_cont.options().device_opt(),
                                      cpu_tensor_cont.options().pinned_memory_opt());
  ideep::tensor& dtensor = itensor_from_mkldnn(mkldnn_tensor);
  if (cpu_tensor.scalar_type() == ScalarType::Float) {
    dtensor.feed_from(dtensor.get_dims(),
                      ideep::tensor::data_type::f32,
                      (cpu_tensor_cont.template data_ptr<float>()));
  } else {
    dtensor.feed_from(dtensor.get_dims(),
                      ideep::tensor::data_type::bf16,
                      cpu_tensor_cont.template data_ptr<BFloat16>());
  }
  return mkldnn_tensor;
}

// Mkldnn tensor has special non-public format for conv2d weights
// (dense_to_mkldnn only converts dense tensor to mkldnn tensor with
// public format). Ideep conv kernel will do implicit reorder if the
// weight is not already in this optimized format. By the time I'm
// writing this note, we are seeing ~20% perf cost of doing the
// on-the-fly reorder.
Tensor mkldnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_reorder_conv2d_weight: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  auto w = itensor_from_mkldnn(self);

  // Legacy mkldnn conv2d jitted module may contain a 5-d weight with an extra
  // dimension when groups > 1, having dimension [g, o/g, i, h, w] instead of
  // [o, i, h, w]. Ideally we should reorder the weight back in serialization.
  // For backward compatibility, we squash the first two dims (g * o/g) back to
  // its original form.
  if (w.ndims() == 5) {
    auto wdims = w.get_dims();
    w.reshape({wdims[0] * wdims[1], wdims[2], wdims[3], wdims[4]});
  }

  auto desc =
      ideep::convolution_forward::expected_weights_desc(
          w.get_dims(),
          w.get_data_type(),
          {stride.begin(), stride.end()},
          {padding.begin(), padding.end()},
          {padding.begin(), padding.end()},
          {dilation.begin(), dilation.end()},
          groups,
          ideep::algorithm::convolution_direct);
  ideep::tensor result;
  result.init(desc);
  result.feed_from(w);

  return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()),
                                 self.options().device_opt());
}

Tensor mkldnn_reorder_conv3d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  if (self.scalar_type() == ScalarType::BFloat16) {
    TORCH_CHECK(mkldnn_bf16_device_check(),
        "mkldnn_reorder_conv3d_weight: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
  }

  auto w = itensor_from_mkldnn(self);

  auto desc =
      ideep::convolution_forward::expected_weights_desc(
          w.get_dims(),
          w.get_data_type(),
          {stride.begin(), stride.end()},
          {padding.begin(), padding.end()},
          {padding.begin(), padding.end()},
          {dilation.begin(), dilation.end()},
          groups,
          ideep::algorithm::convolution_direct);
  ideep::tensor result;
  result.init(desc);
  result.feed_from(w);

  return new_with_itensor_mkldnn(std::move(result), optTypeMetaToScalarType(self.options().dtype_opt()), self.options().device_opt());
}

#else

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor, c10::optional<ScalarType> dtype) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

Tensor mkldnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(false, "mkldnn_reorder_conv2d_weight: MKL-DNN build is disabled");
}

Tensor mkldnn_reorder_conv3d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  TORCH_CHECK(false, "mkldnn_reorder_conv3d_weight: MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

}}
