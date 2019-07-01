#include <ATen/native/TensorFactories.h>
#include <ATen/native/mkldnn/MKLDNNCommon.h>

namespace at {
namespace native {

#if !AT_MKLDNN_ENABLED()

Tensor empty_affine_quantized_mkldnn(
    IntArrayRef sizes,
    const TensorOptions& options,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  AT_ERROR(
      "empty_affine_quantized_mkldnn: ATen not compiled with MKLDNN support");
}

Tensor quantize_linear_mkldnn(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  AT_ERROR("quantize_linear_mkldnn: ATen not compiled with MKLDNN support");
}

Tensor dequantize_quant_mkldnn(const Tensor& self) {
  AT_ERROR("dequantize_quant_mkldnn: ATen not compiled with MKLDNN support");
}

double q_scale_quant_mkldnn(const Tensor& self) {
  AT_ERROR("q_scale_quant_mkldnn: ATen not compiled with MKLDNN support");
}

int64_t q_zero_point_quant_mkldnn(const Tensor& self) {
  AT_ERROR("q_zero_point_quant_mkldnn: ATen not compiled with MKLDNN support");
}

#else // AT_MKLDNN_ENABLED()

Tensor empty_affine_quantized_mkldnn(
    IntArrayRef sizes,
    const TensorOptions& options,
    double scale,
    int64_t zero_point,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  TORCH_CHECK(
      options.has_dtype(),
      "Must provide data type for Tensor creation functions.");
  TORCH_CHECK(
      (options.dtype() == kQUInt8 && (zero_point == 0 || zero_point == 128)) ||
          ((options.dtype() == kQInt32 || options.dtype() == kQInt8) &&
           zero_point == 0),
      "Only support uint8 with 0, 128 zero point and int32,int8 with 0 zero point.");
  native::check_size_nonnegative(sizes);
  ideep::tensor::dims dst_dims(sizes.begin(), sizes.end());
  auto data_type =
      get_mkldnn_dtype(at::typeMetaToScalarType(options.dtype()), zero_point);

  ideep::tensor itensor;
  itensor.resize<AllocForMKLDNN>(dst_dims, data_type);
  ideep::scale_t Y_scales_ = ConvertScales({scale});
  itensor.set_scale(Y_scales_);
  return new_with_itensor_mkldnn(std::move(itensor), options);
}

Tensor quantize_linear_mkldnn(
    const Tensor& self,
    double scale,
    int64_t zero_point,
    ScalarType dtype) {
  TORCH_CHECK(
      self.scalar_type() == kFloat,
      "Mkldnn quantize only works on Float Tensor.");
  TORCH_CHECK(
      self.device() == kCPU,
      "Mkldnn quantize only works for CPU backend right now.");
  TORCH_CHECK(
      (dtype == kQUInt8 && (zero_point == 128 || zero_point == 0)) ||
          ((dtype == kQInt8 || dtype == kQInt32) && zero_point == 0),
      "Only support uint8 with 0, 128 zero point and int32,int8 with 0 zero point.");

  ideep::tensor& src_itensor = itensor_from_mkldnn(self);

  TensorOptions options = self.options().dtype(dtype);

  native::check_size_nonnegative(self.sizes());
  auto data_type = get_mkldnn_dtype(dtype, zero_point);

  ideep::tensor dst_itensor;
  dst_itensor.init<AllocForMKLDNN>(
      {src_itensor.get_dims(), data_type, src_itensor.get_public_format()});
  ideep::scale_t Y_scales_ = ConvertScales({scale});
  dst_itensor.set_scale(Y_scales_);
  dst_itensor.feed_from(src_itensor);

  Tensor qtensor = new_with_itensor_mkldnn(std::move(dst_itensor), options);

  return qtensor;
}

Tensor dequantize_quant_mkldnn(const Tensor& self) {
  Tensor tensor = empty_mkldnn(self.sizes(), self.options().dtype(kFloat));
  ideep::tensor& dst_itensor = itensor_from_mkldnn(tensor);
  ideep::tensor& src_itensor = itensor_from_mkldnn(self);
  dst_itensor.feed_from(src_itensor);
  return tensor;
}

double q_scale_quant_mkldnn(const Tensor& self) {
  ideep::tensor& itensor = itensor_from_mkldnn(self);
  TORCH_CHECK(itensor.has_scale(), "quantized tensor hasn't scale!");
  auto mkldnn_scales = itensor.get_scale();
  auto scales = ConvertScales(mkldnn_scales);
  return scales[0];
}

int64_t q_zero_point_quant_mkldnn(const Tensor& self) {
  ideep::tensor& itensor = itensor_from_mkldnn(self);
  switch (self.scalar_type()) {
    case kQInt32:
    case kQInt8:
      return 0;
    case kQUInt8:
      if (itensor.get_data_type() == ideep::tensor::data_type::u8)
        return 0;
      else if (itensor.get_data_type() == ideep::tensor::data_type::s8)
        return 128;
      else
        AT_ERROR("q_zero_point_quant_mkldnn: unsupport mkldnn data type!");
    default:
      AT_ERROR("q_zero_point_quant_mkldnn: unsupport data type!");
  }
}

#endif // AT_MKLDNN_ENABLED()

} // namespace native
} // namespace at
