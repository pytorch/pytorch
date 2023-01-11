#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Parallel.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/quantized/cpu/QnnpackUtils.h>
#include <ATen/native/quantized/cpu/QuantizedOps.h>
#include <ATen/native/quantized/cpu/init_qnnpack.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_empty_affine_quantized.h>
#endif

#include <algorithm>

namespace at {
namespace native {

DEFINE_DISPATCH(qhardswish_stub);

namespace {

template <typename DTYPE>
std::vector<DTYPE> create_hswish_lookup_table(
    int32_t input_zero_point,
    float input_scale,
    int32_t output_zero_point,
    float output_scale) {
  DTYPE dtype_min = std::numeric_limits<DTYPE>::min();
  DTYPE dtype_max = std::numeric_limits<DTYPE>::max();

  static_assert(
      std::is_same<DTYPE, int8_t>() || std::is_same<DTYPE, uint8_t>());

  std::vector<DTYPE> lookup_table(256, 0);
  const float scaled_min = (float)(int32_t)dtype_min;
  const float scaled_max = (float)(int32_t)dtype_max;
  const float inv_output_scale = 1.0f / output_scale;
  for (int32_t i = dtype_min, index = 0; i <= dtype_max; i++, index++) {
    float x = input_scale * (float)(i - input_zero_point);
    // hardswish, no min/max functions in C
    float x2 = x + 3.0f;
    x2 = x2 > 0.0f ? x2 : 0.0f;
    x2 = x2 < 6.0f ? x2 : 6.0f;
    x2 = x * x2 / 6.0f;
    float scaled_hardswish_x = inv_output_scale * x2 + output_zero_point;
    if (scaled_hardswish_x < scaled_min) {
      scaled_hardswish_x = scaled_min;
    }
    if (scaled_hardswish_x > scaled_max) {
      scaled_hardswish_x = scaled_max;
    }
    lookup_table[index] = static_cast<DTYPE>(lrintf(scaled_hardswish_x));
  }

  return lookup_table;
}

template <typename DTYPE>
void lookup_hardswish_table(
    const DTYPE* data,
    const int64_t num_elements,
    const std::vector<DTYPE> lookup_table,
    DTYPE* output_data) {
  int32_t offset = -std::numeric_limits<DTYPE>::min();
  at::parallel_for(0, num_elements, 0, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; i++) {
      const size_t index = (static_cast<int32_t>(data[i]) + offset);
      output_data[i] = lookup_table.at(index);
    }
  });
}

template <typename DTYPE>
void hardswish_int8(const Tensor& qx, Tensor& qy) {
  size_t num_elems = qx.numel();
  const auto i_zero_point = qx.q_zero_point();
  const auto i_scale = qx.q_scale();
  const auto o_zero_point = qy.q_zero_point();
  const auto o_scale = qy.q_scale();
  std::vector<typename DTYPE::underlying> lookup_table =
      create_hswish_lookup_table<typename DTYPE::underlying>(
          i_zero_point, i_scale, o_zero_point, o_scale);
  lookup_hardswish_table(
      (typename DTYPE::underlying*)qx.data_ptr<DTYPE>(), // input data
      num_elems, // input stride
      lookup_table,
      (typename DTYPE::underlying*)qy.data_ptr<DTYPE>()); // output data
}

Tensor quantized_hardswish(
    const Tensor& qx,
    double output_scale,
    int64_t output_zero_point) {
  Tensor qy = at::_empty_affine_quantized(
      qx.sizes(),
      at::device(kCPU).dtype(qx.scalar_type()),
      output_scale,
      output_zero_point,
      qx.suggest_memory_format());
#ifdef USE_PYTORCH_QNNPACK
  if (at::globalContext().qEngine() == at::QEngine::QNNPACK &&
      qx.scalar_type() == kQUInt8) {
    Tensor qx_contig = qx.contiguous(qx.suggest_memory_format());
    hardswish_int8<c10::quint8>(qx_contig, qy);
    return qy;
  }
#endif // USE_PYTORCH_QNNPACK
  if (qx.scalar_type() == kQInt8) {
    Tensor qx_contig = qx.contiguous(qx.suggest_memory_format());
    hardswish_int8<c10::qint8>(qx_contig, qy);
    return qy;
  }
  qhardswish_stub(qx.device().type(), qx, qy);
  return qy;
}

} // namespace

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::hardswish"),
      TORCH_FN(quantized_hardswish));
}

}
} // namespace at::native
