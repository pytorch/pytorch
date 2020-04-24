#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/utils/ParamUtils.h>

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor, c10::optional<ScalarType> dtype) {
  TORCH_INTERNAL_ASSERT(mkldnn_tensor.scalar_type() == ScalarType::Float ||
                        mkldnn_tensor.scalar_type() == ScalarType::BFloat16,
                        "mkldnn_to_dense expects float or bfloat16 tensor input");
  ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
  auto dims = stensor.get_dims();
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  auto data_type = dtype.has_value() ? dtype.value() : mkldnn_tensor.scalar_type();
  Tensor cpu_tensor = at::empty(
    std::vector<int64_t>(dims.begin(), dims.end()),
    mkldnn_tensor.options().layout(c10::kStrided).dtype(data_type));
  if (stensor.is_empty()) return cpu_tensor;
  auto pub_tensor =
      cpu_tensor.scalar_type() == ScalarType::Float
          ? stensor.to_public(cpu_tensor.template data_ptr<float>(),
                              get_mkldnn_dtype(data_type))
          : stensor.to_public(cpu_tensor.template data_ptr<BFloat16>(),
                              get_mkldnn_dtype(data_type));
  cpu_tensor.as_strided_(dims, pub_tensor.get_strides());
  return cpu_tensor;
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor, c10::optional<ScalarType> dtype) {
  TORCH_INTERNAL_ASSERT(cpu_tensor.device().type() == DeviceType::CPU,
                        "dense_to_mkldnn expects CPU tensor input");
  TORCH_INTERNAL_ASSERT(cpu_tensor.layout() == Layout::Strided,
                        "dense_to_mkldnn expects strided tensor input");
  TORCH_INTERNAL_ASSERT(cpu_tensor.scalar_type() == ScalarType::Float ||
                        cpu_tensor.scalar_type() == ScalarType::BFloat16,
                        "dense_to_mkldnn expects bfloat16 or float tensor input");
  TORCH_INTERNAL_ASSERT(cpu_tensor.dim() <= 5,
                        "Can't convert cpu tensor with the number of dimensions > 5");
  // TODO: consider to convert non-contiguous tensor to `ideep::tensor` directly.
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  auto data_type = dtype.has_value() ? dtype.value() : cpu_tensor.scalar_type();
  Tensor mkldnn_tensor = empty_mkldnn(
      cpu_tensor_cont.sizes(), cpu_tensor_cont.options().dtype(data_type));
  ideep::tensor& dtensor = itensor_from_mkldnn(mkldnn_tensor);
  if (cpu_tensor.scalar_type() == ScalarType::Float) {
    dtensor.feed_from(dtensor.get_dims(),
                      get_mkldnn_dtype(cpu_tensor_cont.scalar_type()),
                      cpu_tensor_cont.template data_ptr<float>());
  } else {
    dtensor.feed_from(dtensor.get_dims(),
                      get_mkldnn_dtype(cpu_tensor_cont.scalar_type()),
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

  auto stride_vec = expand_param_if_needed(stride, "stride", 2);
  auto padding_vec = expand_param_if_needed(padding, "padding", 2);
  auto dilation_vec = expand_param_if_needed(dilation, "dilation", 2);

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
          {stride_vec.cbegin(), stride_vec.cend()},
          {padding_vec.cbegin(), padding_vec.cend()},
          {padding_vec.cbegin(), padding_vec.cend()},
          {dilation_vec.cbegin(), dilation_vec.cend()},
          groups,
          ideep::algorithm::convolution_direct);
  ideep::tensor result;
  result.init(desc);
  result.feed_from(w);

  return new_with_itensor_mkldnn(std::move(result), self.options());
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

#endif // AT_MKLDNN_ENABLED()

}}
