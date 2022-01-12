#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <ATen/native/quantized/cpu/qnnpack_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>         // for _empty_affine_q...
#include <ATen/ops/mean.h>                            // for mean
#include <ATen/ops/mean_native.h>                     // for mean_out_quanti...
#include <ATen/ops/quantize_per_tensor.h>             // for quantize_per_te...
#include <ATen/ops/zeros_like_ops.h>
#endif

namespace at {
namespace native {
#ifdef USE_PYTORCH_QNNPACK
Tensor qnnpack_mean(const Tensor& input, IntArrayRef dim, bool keepdim) {
  Tensor output;
  TORCH_CHECK(
      input.ndimension() == 4,
      "qnnpack_global_average_pool: Expected input to be 4-dimensional: got ",
      input.ndimension());
  TORCH_CHECK(
      dim.size() == 2,
      "qnnpack_global_average_pool: dim size must be a tuple of two ints");
  TORCH_CHECK(
      dim[0] == 2 && dim[1] == 3,
      "qnnpack_global_average_pool: Reduction dimensions must match last 2 dimensions of input tensor")

  const int64_t batch_size = input.size(0);
  const int64_t inC = input.size(1);
  const int64_t inH = input.size(2);
  const int64_t inW = input.size(3);

  Tensor input_contig = input.contiguous(MemoryFormat::ChannelsLast);

  initQNNPACK();
  const auto scale = input_contig.q_scale();
  const auto zero_point = input_contig.q_zero_point();
  const auto outC = inC;

  output = at::_empty_affine_quantized(
      keepdim ? IntArrayRef{batch_size, outC, 1, 1}
              : IntArrayRef{batch_size, outC},
      at::device(kCPU).dtype(kQUInt8),
      scale,
      zero_point);

  pytorch_qnnp_operator_t qnnpack_operator{nullptr};
  const pytorch_qnnp_status createStatus =
      pytorch_qnnp_create_global_average_pooling_nwc_q8(
          inC,
          zero_point,
          scale,
          zero_point,
          scale,
          std::numeric_limits<uint8_t>::min() /* output min */,
          std::numeric_limits<uint8_t>::max() /* output max */,
          0,
          &qnnpack_operator);

  CAFFE_ENFORCE(
      createStatus == pytorch_qnnp_status_success,
      "failed to create QNNPACK Global Average Pooling operator");
  std::unique_ptr<pytorch_qnnp_operator, QnnpackOperatorDeleter>
      qnnpack_uniq_ptr(qnnpack_operator);

  const pytorch_qnnp_status setupStatus =
      pytorch_qnnp_setup_global_average_pooling_nwc_q8(
          qnnpack_operator,
          batch_size,
          inH * inW,
          (uint8_t*)input_contig.data_ptr<c10::quint8>() /* input data */,
          inC,
          (uint8_t*)output.data_ptr<c10::quint8>() /* output data */,
          outC);
  CAFFE_ENFORCE(
      setupStatus == pytorch_qnnp_status_success,
      "failed to setup QNNPACK Global Average Pooling operator");
  pthreadpool_t threadpool = caffe2::pthreadpool_();
  const pytorch_qnnp_status runStatus =
      pytorch_qnnp_run_operator(qnnpack_operator, threadpool);
  TORCH_INTERNAL_ASSERT(
      runStatus == pytorch_qnnp_status_success,
      "failed to run QNNPACK Global Average Pool operator");
  return output;
}
#endif
Tensor& mean_out_quantized_cpu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype,
    Tensor& result) {
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      self.scalar_type() == kQUInt8 &&
      // QNNPACK currently is only supported for NCHW + dim=(2, 3)
      // Remove these checks after generic version is implemented.
      self.ndimension() == 4 && dim.size() == 2 && dim[0] == 2 && dim[1] == 3) {
    result = qnnpack_mean(self, dim, keepdim);
    return result;
  }
#endif
  auto self_dequantized = self.dequantize();
  auto result_dequantized = at::mean(self_dequantized, dim, keepdim, opt_dtype);
  result = at::quantize_per_tensor(
      result_dequantized,
      self.q_scale(),
      self.q_zero_point(),
      opt_dtype.value_or(self.scalar_type()));
  return result;
}

Tensor mean_quantized_cpu(
    const Tensor& self,
    IntArrayRef dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  Tensor result;
  mean_out_quantized_cpu(self, dim, keepdim, dtype, result);
  return result;
}

Tensor mean_quantized_cpu(
    const Tensor& self,
    DimnameList dim,
    bool keepdim,
    optional<ScalarType> dtype) {
  return mean_quantized_cpu(
      self, dimnames_to_positions(self, dim), keepdim, dtype);
}

Tensor& mean_out_quantized_cpu(
    Tensor& result,
    const Tensor& self,
    DimnameList dim,
    bool keepdim,
    c10::optional<ScalarType> opt_dtype) {
  return mean_out_quantized_cpu(
      self, dimnames_to_positions(self, dim), keepdim, opt_dtype, result);
}

} // namespace native
} // namespace at
