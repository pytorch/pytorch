#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/utils/ParamUtils.h>
#include <ATen/Parallel.h>

#if AT_MKLDNN_ENABLED()

namespace {
template <typename T>
void mkldnn_tensor_to_buffer(const at::Tensor& mkldnn_tensor, T* output_point) {
  ideep::tensor& input = at::native::itensor_from_mkldnn(mkldnn_tensor);
  // NHWC is a public format in MKL-DNN so we have to explicitly convert it to
  // NCHW tensor to align with PyTorch format.
  if (input.is_nhwc_format()) {
    ideep::tensor output;
    output.init({input.get_dims(), input.get_data_type()}, output_point);
    if (input.has_scale())
      output.set_scale(input.get_scale());
    output.feed_from(input);
  } else {
    input.to_public_format(output_point);
  }

  // MKLDNN support symatric quantized tensor only,so we need to convert asymatric
  // quantized tensor to symatric quantized tensor here.
  if (input.get_data_type() == ideep::tensor::data_type::s8 &&
      mkldnn_tensor.scalar_type() == at::kQUInt8) {
    auto data_src = reinterpret_cast<int8_t*>(output_point);
    auto data_dst = reinterpret_cast<uint8_t*>(output_point);
    auto n = mkldnn_tensor.numel();
    at::parallel_for(0, n, 2048, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; i++)
        data_dst[i] = static_cast<int>(data_src[i]) + 128;
    });
  }
}

template <typename T>
void dense_tensor_to_mkldnn(
    const at::Tensor& cpu_tensor,
    at::Tensor& mkldnn_tensor) {
  ideep::tensor& tensor_out = at::native::itensor_from_mkldnn(mkldnn_tensor);
  auto scalar_type = cpu_tensor.scalar_type();
  // MKLDNN support symatric quantized tensor only,so we need to convert asymatric
  // quantized tensor to symatric quantized tensor here.
  if (scalar_type == at::kQUInt8 &&
      tensor_out.get_data_type() == ideep::tensor::data_type::s8) {
    uint8_t* data_src =
        reinterpret_cast<uint8_t*>(cpu_tensor.template data<c10::quint8>());
    auto data_dst = reinterpret_cast<int8_t*>(
        at::native::itensor_from_mkldnn(mkldnn_tensor).get_data_handle());
    auto n = cpu_tensor.numel();
    at::parallel_for(0, n, 2048, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; i++)
        data_dst[i] = static_cast<int>(data_src[i]) - 128;
    });
  } else {
    ideep::tensor tensor_in;
    tensor_in.init(
        {tensor_out.get_dims(), at::native::get_mkldnn_dtype(scalar_type)},
        cpu_tensor.template data<T>());
    if (c10::isQIntType(scalar_type)) {
      auto scale = cpu_tensor.q_scale();
      tensor_in.set_scale(at::native::ConvertScales({scale}));
    }
    tensor_out.feed_from(tensor_in);
  }
}
} // namespace

namespace at {
namespace native {

Tensor mkldnn_to_dense(const Tensor& mkldnn_tensor) {
  ideep::tensor& tensor_src = itensor_from_mkldnn(mkldnn_tensor);
  // NOTE: int32_t dims from ideep::tensor but sizes needs int64_t
  auto dims = tensor_src.get_dims();

  Tensor cpu_tensor;

  if (tensor_src.is_empty())
    return cpu_tensor;

  auto scalar_type = mkldnn_tensor.scalar_type();
  if (isQIntType(scalar_type)) {
    ideep::scale_t scale_mkldnn {1};
    if (tensor_src.has_scale()) {
      scale_mkldnn = tensor_src.get_scale();
      AT_ASSERTM(scale_mkldnn.size() == 1, "Only support per tensor quantized tensor.");
    }
    auto scale = ConvertScales(scale_mkldnn);
    cpu_tensor = empty_affine_quantized_cpu(
        std::vector<int64_t>(dims.begin(), dims.end()),
        mkldnn_tensor.options().layout(c10::kStrided),
        scale[0],
        (tensor_src.get_data_type() == ideep::tensor::data_type::s8 && scalar_type == kQUInt8) ? 128
                                                                           : 0);
    AT_DISPATCH_QINT_TYPES(scalar_type, "mkldnn_to_dense", [&]() {
      mkldnn_tensor_to_buffer<scalar_t>(mkldnn_tensor, cpu_tensor.template data<scalar_t>());
    });
  } else {
    AT_ASSERTM(scalar_type == kFloat, "Only support float data type!");
    cpu_tensor = at::empty(
        std::vector<int64_t>(dims.begin(), dims.end()),
        mkldnn_tensor.options().layout(c10::kStrided));
    mkldnn_tensor_to_buffer<float>(mkldnn_tensor, cpu_tensor.template data<float>());
  }
  return cpu_tensor;
}

Tensor dense_to_mkldnn(const Tensor& cpu_tensor) {
  auto scalar_type = cpu_tensor.scalar_type();
  int64_t zero_point;

  AT_ASSERTM(
      cpu_tensor.type_id() == CPUTensorId() ||
          cpu_tensor.type_id() == QuantizedCPUTensorId(),
      "dense_to_mkldnn expects dense CPU or quantizedCPU tensor input");
  if (cpu_tensor.type_id() == QuantizedCPUTensorId()) {
    zero_point = cpu_tensor.q_zero_point();
    AT_ASSERTM(
        (zero_point == 0 ||zero_point == 128),
        "mkldnn only support 0 or 128 zero point in quantized tensor");
  } else
    AT_ASSERTM(
        scalar_type == kFloat,
        "dense_to_mkldnn expects float tensor input");
  AT_ASSERTM(
      cpu_tensor.dim() <= 5,
      "Can't convert cpu tensor to mkldnn tensor with the number of dimensions > 5");
  // TODO: consider to convert non-contiguous tensor to `ideep::tensor` directly.
  auto cpu_tensor_cont = cpu_tensor.contiguous();
  Tensor mkldnn_tensor;
  if (cpu_tensor.type_id() == QuantizedCPUTensorId()) {
    mkldnn_tensor = empty_affine_quantized_mkldnn(
        cpu_tensor_cont.sizes(), cpu_tensor_cont.options(), cpu_tensor.q_scale(), zero_point);
  } else
    mkldnn_tensor =
        empty_mkldnn(cpu_tensor_cont.sizes(), cpu_tensor_cont.options());

  switch (scalar_type) {
    case kFloat:
      dense_tensor_to_mkldnn<float>(cpu_tensor_cont, mkldnn_tensor);
      break;
    case kQUInt8:
    case kQInt8:
    case kQInt32:
      AT_DISPATCH_QINT_TYPES(scalar_type, "dense_to_mkldnn", [&]() {
        dense_tensor_to_mkldnn<scalar_t>(cpu_tensor_cont, mkldnn_tensor);
      });
      break;
    default:
      AT_ERROR("dense_to_mkldnn: unsupport data type!");
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

} // namespace native
} // namespace at
#else

namespace at {
namespace native {

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

} // namespace native
} // namespace at
#endif // AT_MKLDNN_ENABLED()
