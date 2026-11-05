#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>

#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace at::native::mkldnn {

// Ported from ideep::matmul_forward_dyn_quant_params.
// Parameters for dynamic quantization
struct matmul_forward_dyn_quant_params {
  ideep::tensor::desc src_desc; // to create src tensor
  dnnl::reorder::primitive src_reorder; // to reorder src

  matmul_forward_dyn_quant_params() {}

  matmul_forward_dyn_quant_params(
      ideep::tensor::desc&& src_desc,
      dnnl::reorder::primitive&& src_reorder)
      : src_desc(std::move(src_desc)), src_reorder(std::move(src_reorder)) {}
};

// Ported from ideep::matmul_forward_params.
// Common parameters for computation
struct matmul_forward_params {
  dnnl::matmul::primitive_desc pd;
  dnnl::matmul primitive;
  ideep::attr_t op_attr;
  ideep::attr_t src_attr;
  ideep::attr_t weights_attr;
  ideep::attr_t bias_attr; // contains requantization scales for bias
  std::shared_ptr<matmul_forward_dyn_quant_params> dq_param_ptr;
  // Keep {dnnl_arg, tensor} pairs of scales and zero points for primitive
  // execution
  std::shared_ptr<std::unordered_map<int, ideep::tensor>> all_scales = nullptr;
  std::shared_ptr<std::unordered_map<int, ideep::tensor>> all_zero_points =
      nullptr;

  matmul_forward_params() {}

  matmul_forward_params(
      dnnl::matmul::primitive_desc&& pd,
      ideep::attr_t&& op_attr,
      ideep::attr_t&& src_attr,
      ideep::attr_t&& weights_attr,
      ideep::attr_t&& bias_attr)
      : pd(std::move(pd)),
        op_attr(std::move(op_attr)),
        src_attr(std::move(src_attr)),
        weights_attr(std::move(weights_attr)),
        bias_attr(std::move(bias_attr)) {
    primitive = dnnl::matmul(pd);
  }
};

// Ported from ideep::matmul_forward::compute (int8 2-in-1, with bias).
// 2-in-1 compute for int8 op with bias. Bias is not used if it is empty.
// Set reorder flags to false if you are sure the memory layout aligns
// with primitive descriptor. Otherwise, checks are made and reorder
// may be needed.
template <bool reorder_src = true, bool reorder_weight = true>
void matmul_forward_compute(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_points,
    const std::vector<int32_t>& dst_zero_points,
    const float dst_coeff = 1.0f,
    const float sum_coeff = 1.0f,
    const ideep::attr_t& attr = ideep::attr_t(),
    const dnnl::memory::data_type dst_type = dnnl::memory::data_type::undef,
    const ideep::lowp_kind alowp_kind = ideep::u8s8);

// Ported from ideep::matmul_forward::prepare (int8, with bias).
// Prepare for int8 op with bias. Bias is not used if it is empty.
// Static: int8 * int8 -> int8. Dynamic: fp32 * int8 -> fp32
template <bool is_dynamic = false>
void matmul_forward_prepare(
    matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const std::vector<float>& src_scales,
    const std::vector<float>& weights_scales,
    const std::vector<float>& dst_scales,
    const std::vector<int32_t>& src_zero_points,
    const std::vector<int32_t>& dst_zero_points,
    const float dst_coeff = 1.0f,
    const float sum_coeff = 1.0f, // for post-op sum
    const ideep::attr_t& attr = ideep::attr_t(),
    const dnnl::memory::data_type dst_type = dnnl::memory::data_type::u8,
    const ideep::lowp_kind alowp_kind = ideep::u8s8);

// Ported from ideep::matmul_forward::compute (pre-prepared params, with bias).
// Compute for fp32 and static int8 (int8 * int8 -> int8)
// With bias. Bias is not used if it is empty.
// Set reorder flags to false if you are sure the memory layout aligns
// with primitive descriptor. Otherwise, checks are made and reorder
// may be needed.
template <bool reorder_src = true, bool reorder_weight = true>
void matmul_forward_compute(
    const matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst);

// Ported from ideep::matmul_forward::compute (dynamic int8, with bias).
// Compute for dynamic int8 (fp32 * int8 -> fp32)
// With bias. Bias is not used if it is empty.
// Set reorder flags to false if you are sure the memory layout aligns
// with primitive descriptor. Otherwise, checks are made and reorder
// may be needed.
template <bool reorder_weight = true>
void matmul_forward_compute(
    const matmul_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const std::vector<float>& src_scales,
    const std::vector<int32_t>& src_zero_points,
    const float dst_coeff = 1.0f);

// Ported from ideep::matmul_forward::compute (without bias).
// Deprecated 2-in-1 compute. Without bias.
void matmul_forward(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    ideep::tensor& dst,
    const float dst_coeff = 1.0f,
    const float sum_coeff = 1.0f,
    const ideep::attr_t& attr = ideep::attr_t());

// Ported from ideep::matmul_forward::expected_weights_desc (with y_dtype).
ideep::tensor::desc matmul_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims,
    const dnnl::memory::dims& src_dims,
    dnnl::memory::data_type dtype,
    dnnl::memory::data_type x_dtype,
    dnnl::memory::data_type y_dtype,
    const ideep::attr_t& attr = ideep::attr_t());

// Ported from ideep::matmul_forward::expected_weights_desc (deduced y_dtype).
ideep::tensor::desc matmul_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims,
    const dnnl::memory::dims& src_dims = dnnl::memory::dims(),
    dnnl::memory::data_type dtype = dnnl::memory::data_type::f32,
    dnnl::memory::data_type x_dtype = dnnl::memory::data_type::f32,
    const ideep::attr_t& attr = ideep::attr_t());

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
