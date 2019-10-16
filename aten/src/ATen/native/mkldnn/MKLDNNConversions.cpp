#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/utils/ParamUtils.h>

namespace at { namespace native {

#if AT_MKLDNN_ENABLED()

//
// Semantics of _dnnl_reorder, for both directions
//   1. If output is in nature format (nchw, oihw), return CPUTensor
//   2. otherwise, return Opaque Tensor
//   3. 'from' can be any to adaptly accept incoming Tensor
//
Tensor dnnl_reorder(
    const Tensor& input, int64_t from, int64_t to, int64_t groups) {
  AT_ASSERTM(input.scalar_type() == c10::ScalarType::Float,
             "dnnl_reorder: Expects float tensor input");
  AT_ASSERTM(input.dim() <= 5,
             "dnnl_reorder: Can't convert cpu tensor with dimensions > 5");

  // `get_mkldnn_tensor` accepts both aten and dnnl tensors
  auto src_itensor = get_mkldnn_tensor(input);
  if (static_cast<ideep::format>(from) != ideep::format::any)
    AT_ASSERTM(src_itensor.get_descriptor().get_internal_format() ==
        static_cast<ideep::format>(from)
        || src_itensor.as_weights().get_internal_format() ==
        static_cast<ideep::format>(from),
               "dnnl_reorder: Incompatible input format");

  // TODO:
  //  1. Optimize it when from == to
  //
  if (static_cast<ideep::format>(to) == ideep::format::nchw
      || static_cast<ideep::format>(to) == ideep::format::oihw) {
    // We go to CPUTensor realm
    auto dims = src_itensor.get_dims();
    // casts int32_t dims to int64_t
    auto sizes = std::vector<int64_t>(dims.begin(), dims.end());
    auto cpu_tensor = at::empty(sizes, input.options().layout(c10::kStrided));
    src_itensor.to_public(cpu_tensor.template data_ptr<float>());
    return cpu_tensor;
  } else {
    // Go to DNNL special realm
    auto& input_cont = input.is_mkldnn() ? input : input.contiguous();
    // pre-alloc a tensor, managed by ideep
    auto dst_tensor = empty_dnnl(input_cont.sizes(), input_cont.options(),
        static_cast<ideep::format>(to), groups);
    auto& dst_itensor = itensor_from_mkldnn(dst_tensor);
    // do reordering
    dst_itensor.feed_from(dst_itensor.get_dims(), ideep::tensor::data_type::f32,
                          input_cont.template data_ptr<float>());
    return dst_tensor;
  }
}

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor) {
  ideep::tensor& stensor = itensor_from_mkldnn(mkldnn_tensor);
  auto dims = stensor.get_dims();
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  Tensor cpu_tensor = at::empty(
    std::vector<int64_t>(dims.begin(), dims.end()),
    mkldnn_tensor.options().layout(c10::kStrided));
  stensor.to_public(cpu_tensor.template data_ptr<float>());
  return cpu_tensor;
}

// TODO: There is no way an Opaque Tensor share custody with a CPUTensor
// resulted the inevitable copy, should we enable OpaqueTensor to have
// Storage?
Tensor dense_to_mkldnn(const Tensor& cpu_tensor) {
  AT_ASSERTM(cpu_tensor.device().type() == DeviceType::CPU,
             "dense_to_mkldnn expects CPU tensor input");
  AT_ASSERTM(cpu_tensor.layout() == Layout::Strided,
             "dense_to_mkldnn expects strided tensor input");
  AT_ASSERTM(cpu_tensor.scalar_type() == ScalarType::Float,
             "dense_to_mkldnn expects float tensor input");
  AT_ASSERTM(cpu_tensor.dim() <= 5,
             "Can't convert cpu tensor with the number of dimensions > 5");
  // TODO: consider to convert non-contiguous tensor to `ideep::tensor` directly.
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  Tensor mkldnn_tensor = empty_mkldnn(cpu_tensor_cont.sizes(), cpu_tensor_cont.options());
  ideep::tensor& dtensor = itensor_from_mkldnn(mkldnn_tensor);
  dtensor.feed_from(dtensor.get_dims(),
                    ideep::tensor::data_type::f32,
                    (cpu_tensor_cont.template data_ptr<float>()));
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

  ideep::tensor w = itensor_from_mkldnn(self).as_weights();
  w.make_group(groups);
  ideep::tensor::descriptor desc =
      ideep::convolution_forward::expected_weights_descriptor(
          w.get_dims(),
          w.get_data_type(),
          {stride_vec.cbegin(), stride_vec.cend()},
          {padding_vec.cbegin(), padding_vec.cend()},
          {padding_vec.cbegin(), padding_vec.cend()},
          {dilation_vec.cbegin(), dilation_vec.cend()},
          groups,
          ideep::algorithm::convolution_direct);
  ideep::tensor result;
  result.init<AllocForMKLDNN>(desc);
  result.feed_from(w);

  return new_with_itensor_mkldnn(std::move(result), self.options());
}

#else

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor) {
  AT_ERROR("MKL-DNN build is disabled");
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor) {
  AT_ERROR("MKL-DNN build is disabled");
}

Tensor mkldnn_reorder_conv2d_weight(
    const Tensor& self,
    IntArrayRef padding,
    IntArrayRef stride,
    IntArrayRef dilation,
    int64_t groups) {
  AT_ERROR("mkldnn_reorder_conv2d_weight: MKL-DNN build is disabled");
}

Tensor _dnnl_reorder(
    const Tensor& input, int64_t from, int64_t to, int64_t groups) {
  AT_ERROR("_dnnl_reorder: MKL-DNN build is disabled");
}

#endif // AT_MKLDNN_ENABLED()

}}
