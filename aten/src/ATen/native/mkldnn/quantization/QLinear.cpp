#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/quantized/cpu/packed_params.h>
#include <ATen/native/mkldnn/quantization/Utils.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <torch/custom_class.h>
#include <torch/library.h>

#include <c10/util/irange.h>

#include <algorithm>
#include <string>

torch::class_<LinearPackedParamsBase> register_linear_params();

#if AT_MKLDNN_ENABLED()
template <bool ReluFused>
at::Tensor PackedLinearWeightsMkldnn::apply_impl(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  const int64_t dim = input.dim();
  TORCH_CHECK(
      input.dim() != 0,
      "mkldnn_linear: input needs to has dim at least 1, input dim ",
      input.dim());
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::QUInt8,
      "qlinear (MKLDNN): data type of input should be QUint8.");

  auto input_contig = input.expect_contiguous();
  auto& w = *(weight_.get());
  auto K = input.size(input.dim() - 1), M = input.numel() / K, N = w.get_dim(1);
  auto input_dims = {M, K};
  auto input_data_type = dnnl::memory::data_type::u8;
  auto input_desc = ideep::tensor::desc(input_dims, input_data_type);
  ideep::attr_t op_attr = ReluFused ? ideep::attr_t::fuse_relu() : ideep::attr_t();
  ideep::tensor x(input_desc, input_contig->data_ptr<c10::quint8>());
  auto dst_dims = {M, N};
  const ideep::scale_t& src_scales = ideep::scale_t(1, 1.0/input.q_scale());
  const ideep::scale_t& weights_scales = w.get_scale();
  const ideep::scale_t& dst_scales = ideep::scale_t(1, 1.0/output_scale); // Scales of MKLDNN and PyTorch are reciprocal
  const ideep::zero_point_t& src_zero_point = ideep::zero_point_t(1, input.q_zero_point());
  const ideep::zero_point_t& dst_zero_point = ideep::zero_point_t(1, output_zero_point);
  // Compute: Use ideep::matmul_forward to support asymmetric quantization
  // Allocate output Tensor
  at::Tensor output = at::_empty_affine_quantized(
      dst_dims,
      at::device(c10::kCPU).dtype(c10::kQUInt8),
      output_scale,
      output_zero_point);
  if (output.numel() == 0) {
    return output;
  }
  ideep::tensor y({dst_dims, ideep::tensor::data_type::u8, {output.strides().cbegin(), output.strides().cend()}},
                  output.data_ptr());
  if (bias_.has_value()) {
    // Bias might be modified outside (e.g. by quantization bias correction).
    // If so, update the prepacked bias as well.
    if (bias_.value().get_data_handle() != orig_bias_.value().data_ptr()) {
      bias_.value().init(bias_.value().get_desc(), orig_bias_.value().data_ptr());
    }
    const auto& b = bias_.value();
    TORCH_CHECK(b.get_dim(0) == 1, "bias should be a vector (1D Tensor), but got [", b.get_dims(), "]");
    TORCH_CHECK(
        b.get_dim(1) == N, "bias should have N elements: " + std::to_string(w.get_dim(1)), ", but got ", b.get_dim(1));
    ideep::matmul_forward::compute_v2(x, w, b, y, 1.0f, 1.0f, src_scales, weights_scales, dst_scales,
                                      src_zero_point, dst_zero_point, op_attr);
  } else {
    ideep::matmul_forward::compute_v2(x, w, y, 1.0f, 1.0f, src_scales, weights_scales, dst_scales,
                                      src_zero_point, dst_zero_point, op_attr);
  }
  auto out_sizes = input.sizes().vec();
  out_sizes.back() = N;
  if (output.sizes().vec() == out_sizes)
      return output;
  return output.reshape(out_sizes);
}

at::Tensor PackedLinearWeightsMkldnn::apply(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<false>(std::move(input), output_scale, output_zero_point);
}

at::Tensor PackedLinearWeightsMkldnn::apply_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<true>(std::move(input), output_scale, output_zero_point);
}

#endif // #if AT_MKLDNN_ENABLED()

namespace at {
namespace native {
namespace {

template <bool ReluFused>
class QLinearInt8Mkldnn final {
 public:
  static at::Tensor run(
      at::Tensor input,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    if (ReluFused) {
      return packed_weight->apply_relu(
          std::move(input), output_scale, output_zero_point);
    } else {
      return packed_weight->apply(
          std::move(input), output_scale, output_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_mkldnn"), TORCH_FN(QLinearInt8Mkldnn<false>::run));
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear_relu_mkldnn"), TORCH_FN(QLinearInt8Mkldnn<true>::run));
}

TORCH_LIBRARY_IMPL(_quantized, QuantizedCPU, m) {
  m.impl(TORCH_SELECTIVE_NAME("_quantized::linear_mkldnn"), TORCH_FN(QLinearInt8Mkldnn<false>::run));
}

} // namespace
} // namespace native
} // namespace at
