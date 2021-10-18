#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/mkldnn/quantization/Utils.h>
#include <ATen/native/quantized/cpu/quant_utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/library.h>

#include <torch/custom_class.h>

#include <c10/util/irange.h>

#include <algorithm>
#include <string>

torch::class_<LinearPackedParamsBase> register_linear_params();

#if AT_MKLDNN_ENABLED()
template <bool ReluFused>
at::Tensor PackedLinearWeightsMkldnn::apply_dynamic_impl(at::Tensor input, bool reduce_range) {
  using at::Tensor;
  // fp32 * int8 -> fp32 (with quantization on activation, and dequantization on the result).

  TORCH_CHECK(
      input.dim() >= 2,
      "The dimension of input tensor should be larger than or equal to 2");
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float,
      "qlinear_dynamic (MKLDNN): data type of input should be float.");

  // Input -> uint8
  auto input_contig = input.contiguous();
  const int64_t dim = input.dim();
  auto input_reshaped =
      dim == 2 ? input : input.reshape({-1, input.size(input.dim() - 1)});
  auto input_dims = input_reshaped.sizes().vec();
  auto input_data_type = dnnl::memory::data_type::f32;
  auto input_desc = ideep::tensor::desc(input_dims, input_data_type);
  ideep::attr_t op_attr = ReluFused ? ideep::attr_t::fuse_relu() : ideep::attr_t();
  ideep::tensor x;
  x.init(input_desc, input_contig.data_ptr());
  float x_max, x_min;
  const int precision = 8;
  find_min_max(input_reshaped.data_ptr<float>(), &x_min, &x_max, input.numel());
  auto q_params = quant_utils::ChooseQuantizationParams(
      /*min=*/x_min,
      /*max=*/x_max,
      /*qmin=*/0,
      /*qmax=*/(1 << precision) - 1,
      /*preserve_sparsity=*/false,
      /*force_scale_power_of_two=*/false,
      /*reduce_range=*/reduce_range);
  const std::vector<int32_t>& src_zero_point = std::vector<int32_t>(1, q_params.zero_point);
  // weights, dst
  auto w = *(weight_.get());
  auto dst_dims = {x.get_dim(0), w.get_dim(1)};
  const ideep::scale_t& src_scales = ideep::scale_t(1, 1.0/q_params.scale);
  const ideep::scale_t& weights_scales = w.get_scale();
  // Compute -> f32
  // Use ideep::matmul_forward instead of ideep::inner_product_forward, since the latter does not support asymmetric quantization
  // Allocate output Tensor
  at::Tensor output = at::empty(dst_dims, input.options().dtype(at::kFloat));
  if (output.numel() == 0) return output;
  ideep::tensor y({dst_dims, ideep::tensor::data_type::f32, {output.strides().cbegin(), output.strides().cend()}},
                  output.data_ptr());
  if (bias_.has_value()) {
    // Bias might be modified outside (e.g. by quantization bias correction).
    // If so, update the prepacked bias as well.
    if (bias_.value().get_data_handle() != orig_bias_.value().data_ptr()) {
      bias_.value().init(bias_.value().get_desc(), orig_bias_.value().data_ptr());
    }
    const ideep::tensor b = bias_.value();
    ideep::matmul_forward::compute_v2(x, w, b, y, 1.0f, 1.0f, src_scales, weights_scales, ideep::scale_t(),
                                      src_zero_point, ideep::zero_point_t(), op_attr);
  } else {
    ideep::matmul_forward::compute_v2(x, w, y, 1.0f, 1.0f, src_scales, weights_scales, ideep::scale_t(),
                                      src_zero_point, ideep::zero_point_t(), op_attr);
  }
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = w.get_dim(1);
  return output.reshape(out_sizes);
}

at::Tensor PackedLinearWeightsMkldnn::apply_dynamic(at::Tensor input, bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/false>(std::move(input), reduce_range);
}

at::Tensor PackedLinearWeightsMkldnn::apply_dynamic_relu(at::Tensor input, bool reduce_range) {
  return apply_dynamic_impl</*ReluFused=*/true>(std::move(input), reduce_range);
}

void PackedLinearWeightsMkldnn::find_min_max(const float* a, float* min, float* max, int len) {
  if (len <= 0) {
    *min = 0.0f;
    *max = 0.0f;
    return;
  }

  float temp_min = *a, temp_max = *a;
  int i = 0;

#ifdef __AVX__
  __m256 min_v = _mm256_set1_ps(*a), max_v = _mm256_set1_ps(*a);
  constexpr int VLEN = 8;
  if (len >= VLEN) {
    for (; i < len / VLEN * VLEN; i += VLEN) {
      min_v = _mm256_min_ps(min_v, _mm256_loadu_ps(a + i));
      max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(a + i));
    }

    float min_buf[VLEN], max_buf[VLEN];
    _mm256_storeu_ps(min_buf, min_v);
    _mm256_storeu_ps(max_buf, max_v);
    for (int j = 0; j < VLEN; ++j) {
      temp_min = std::min(temp_min, min_buf[j]);
      temp_max = std::max(temp_max, max_buf[j]);
    }
  }
#endif

  for (; i < len; i++) {
    temp_min = std::min(temp_min, a[i]);
    temp_max = std::max(temp_max, a[i]);
  }
  *min = temp_min;
  *max = temp_max;
}
#endif // #if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace {

template <bool ReluFused>
class QLinearDynamicInt8Mkldnn final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      bool reduce_range) {
    if (ReluFused) {
      return packed_weight->apply_dynamic_relu(std::move(input), reduce_range);
    } else {
      return packed_weight->apply_dynamic(std::move(input), reduce_range);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_dynamic_mkldnn"), TORCH_FN(QLinearDynamicInt8Mkldnn<false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_relu_dynamic_mkldnn"), TORCH_FN(QLinearDynamicInt8Mkldnn<true>::run));
}

TORCH_LIBRARY_IMPL(_quantized, CPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_dynamic_mkldnn"), TORCH_FN(QLinearDynamicInt8Mkldnn<false>::run));
}

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_dynamic_mkldnn"), TORCH_FN(QLinearDynamicInt8Mkldnn<false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_relu_dynamic_mkldnn"), TORCH_FN(QLinearDynamicInt8Mkldnn<true>::run));
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_dynamic_mkldnn"), TORCH_FN(QLinearDynamicInt8Mkldnn<false>::run));
}

} // namespace
} // namespace native
} // namespace at
