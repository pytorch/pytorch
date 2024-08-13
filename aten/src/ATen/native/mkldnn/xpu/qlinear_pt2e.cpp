#include <torch/library.h>

#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>

using namespace at::native::onednn;

namespace at{
namespace native{
namespace xpu{
    // Operators for pt2.0
Tensor q_linear_pointwise(
    Tensor act, // int8 cpu tensor
    double act_scale,
    int64_t act_zero_point,
    Tensor weight,
    Tensor weight_scales,
    Tensor weight_zero_points,
    c10::optional<Tensor> bias,
    double output_scale,
    int64_t output_zero_point,
    bool fp32_output,
    std::string post_op_name,
    torch::List<double> post_op_args,
    std::string post_op_algorithm) {
  TORCH_CHECK(
      !fp32_output,
      "ipex gpu currently does not support fp32 output int8 qlinear");
  Tensor b_raw = bias.has_value() ? bias.value() : at::Tensor();

  const int64_t dim = act.dim();
  int64_t K = act.size(dim - 1);
  int64_t M = act.numel() / K;
  // [M, K] x [K, N]
  int64_t N = weight.size(1);

  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> dst_dims = {M, N};
  Tensor qout = at::empty(dst_dims, device(c10::kXPU).dtype(c10::kByte));

  Attr attr = Attr();

  quantized_matmul_pt2(
      qout,
      act,
      weight,
      b_raw,
      /*m2_trans=*/false,
      act_scale,
      act_zero_point,
      weight_scales,
      weight_zero_points,
      output_scale,
      output_zero_point,
      attr);

  return qout;
}


TORCH_LIBRARY_IMPL(onednn, XPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("onednn::qlinear_pointwise"),
      TORCH_FN(q_linear_pointwise));
}


}
}
}
