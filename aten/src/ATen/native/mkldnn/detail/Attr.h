#pragma once

#include <ATen/Config.h>

#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/detail/PrimitiveCache.h>
// ideep headers use assert() without including <cassert>.
#include <cassert>
#include <ideep/tensor.hpp>

#include <cstdint>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace at::native::mkldnn {

using scale_mask_pair = std::pair<std::vector<float>, int>;
using zp_mask_pair = std::pair<std::vector<int32_t>, int>;
// Map DNNL arg to pair of scales and mask
using scale_map = std::unordered_map<int, scale_mask_pair>;
// Map DNNL arg to pair of zero points and mask
using zero_point_map = std::unordered_map<int, zp_mask_pair>;
inline const scale_map empty_scale_map;
inline const zero_point_map empty_zp_map;

// Ported from ideep::attr_t.
/// Attribute class for extra information into computations
struct Attr : public dnnl::primitive_attr {
  Attr() {}

  Attr(const Attr& other) : dnnl::primitive_attr(other) {
    *this = other;
  }

  Attr(const dnnl::primitive_attr& other) : dnnl::primitive_attr(other) {}

  /// @param arg Parameter argument index
  Attr(int arg, int mask, const std::vector<float>& scales) {
    set_scales_mask(arg, mask);
    if (!scales_) {
      scales_.reset(new scale_map());
    }
    (*scales_)[arg] = std::make_pair(scales, mask);
  }

  // Default arg = DNNL_ARG_DST
  Attr(int mask, const std::vector<float>& scales) {
    set_scales_mask(DNNL_ARG_DST, mask);
    if (!scales_) {
      scales_.reset(new scale_map());
    }
    (*scales_)[DNNL_ARG_DST] = std::make_pair(scales, mask);
  }

  Attr(dnnl_fpmath_mode_t fpmath_mode,
       dnnl::scratchpad_mode sp_mode = dnnl::scratchpad_mode::user) {
    set_fpmath_mode(fpmath_mode);
    set_scratchpad_mode(sp_mode);
  }

  Attr& set_fpmath_mode(dnnl_fpmath_mode_t mode) {
    dnnl::error::wrap_c_api(
        dnnl_primitive_attr_set_fpmath_mode(get(), mode),
        "could not set fpmath mode primitive attribute");
    return *this;
  }

  /// @param arg Parameter argument index, e.g. DNNL_ARG_SRC
  void set_scales(int arg, int mask, const std::vector<float>& scales) {
    set_scales_mask(arg, mask);
    if (!scales_) {
      scales_.reset(new scale_map());
    }
    (*scales_)[arg] = std::make_pair(scales, mask);
  }

  const scale_map& get_all_scales() const {
    return scales_ ? *scales_ : empty_scale_map;
  }

  bool has_scales() const {
    return (scales_ && !(*scales_).empty());
  }

  /// @param arg Parameter argument index, e.g. DNNL_ARG_SRC
  void set_zero_points(
      int arg,
      int mask,
      const std::vector<int32_t>& zero_points) {
    set_zero_points_mask(arg, mask);
    if (!zero_points_) {
      zero_points_.reset(new zero_point_map());
    }
    (*zero_points_)[arg] = std::make_pair(zero_points, mask);
  }

  const zero_point_map& get_all_zero_points() const {
    return zero_points_ ? *zero_points_ : empty_zp_map;
  }

  bool has_zero_points() const {
    return (zero_points_ && !(*zero_points_).empty());
  }

  // Helper factory
  static inline Attr fuse_eltwise(
      dnnl::algorithm alg,
      float alpha,
      float beta) {
    Attr attr;
    dnnl::post_ops po;
    po.append_eltwise(alg, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static Attr fuse_sum(float scale = 1.0, int32_t sum_zero_point = 0) {
    Attr attr;
    dnnl::post_ops po;
    po.append_sum(scale, sum_zero_point);
    attr.set_post_ops(po);
    return attr;
  }

  // Deprecated
  static Attr fuse_relu(
      float scale = 1.0f, // unused since onednn 3.0
      float alpha = 0.f,
      float beta = 0.f) {
    return fuse_eltwise(dnnl::algorithm::eltwise_relu, alpha, beta);
  }

  static Attr fuse_relu_v2(float alpha = 0.f, float beta = 0.f) {
    return fuse_eltwise(dnnl::algorithm::eltwise_relu, alpha, beta);
  }

  // Deprecated
  static Attr fuse_gelu(
      float scale, // unused since onednn 3.0
      float alpha = 0.f,
      float beta = 0.f,
      dnnl::algorithm gelu_type = dnnl::algorithm::eltwise_gelu_erf) {
    return fuse_eltwise(gelu_type, alpha, beta);
  }

  static Attr fuse_gelu_v2(
      float alpha = 0.f,
      float beta = 0.f,
      dnnl::algorithm gelu_type = dnnl::algorithm::eltwise_gelu_erf) {
    return fuse_eltwise(gelu_type, alpha, beta);
  }

  static Attr fuse_sigmoid(float alpha = 1.0, float beta = 0.f) {
    return fuse_eltwise(dnnl::algorithm::eltwise_logistic, alpha, beta);
  }

  static Attr fuse_swish(float alpha = 1.0, float beta = 0.f) {
    return fuse_eltwise(dnnl::algorithm::eltwise_swish, alpha, beta);
  }

  static Attr fuse_tanh(float alpha = 0.f, float beta = 0.f) {
    return fuse_eltwise(dnnl::algorithm::eltwise_tanh, alpha, beta);
  }

  static Attr residual(
      float sum_scale = 1.0,
      float alpha = 0.f,
      float beta = 0.f) {
    Attr attr;
    dnnl::post_ops po;
    po.append_sum(sum_scale);
    po.append_eltwise(dnnl::algorithm::eltwise_relu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static Attr residual_with_sum_zero_point(
      float sum_scale = 1.0,
      int32_t sum_zero_point = 0,
      float alpha = 0.f,
      float beta = 0.f) {
    Attr attr;
    dnnl::post_ops po;
    po.append_sum(sum_scale, sum_zero_point);
    po.append_eltwise(dnnl::algorithm::eltwise_relu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static Attr fuse_clamp(float lower_bound = -1.0, float upper_bound = 1.0) {
    return fuse_eltwise(
        dnnl::algorithm::eltwise_clip, lower_bound, upper_bound);
  }

  static Attr fuse_hardswish(float alpha = 1.0f / 6.0f, float beta = 0.5f) {
    return fuse_eltwise(dnnl::algorithm::eltwise_hardswish, alpha, beta);
  }

  static Attr fuse_binary(dnnl::algorithm alg, dnnl::memory::desc src_desc) {
    Attr attr;
    dnnl::post_ops po;
    po.append_binary(alg, src_desc);
    attr.set_post_ops(po);
    return attr;
  }

  static Attr attr_post_ops(dnnl::post_ops po) {
    Attr attr;
    attr.set_post_ops(po);
    return attr;
  }

  bool has_op_kind(dnnl::primitive::kind op_kind) const {
    auto po = get_post_ops();
    for (int i = 0; i < po.len(); i++)
      if (op_kind == po.kind(i))
        return true;
    return false;
  }

  std::tuple<
      dnnl::primitive::kind,
      float,
      float,
      float,
      dnnl::algorithm,
      int32_t,
      dnnl::memory::desc>
  get_params(int index) const {
    auto po = get_post_ops();
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        index < po.len() ? dnnl_success : dnnl_invalid_arguments,
        "post_ops index is out of range");
#endif

    dnnl::algorithm alg = dnnl::algorithm::undef;
    float scale = 1.0, alpha = 1.0, beta = 0.0;
    dnnl::memory::desc binary_src_desc;
    int32_t zero_point = 0;
    dnnl::memory::data_type post_op_sum_dtype;

    auto akind = po.kind(index);
    switch (akind) {
      case dnnl::primitive::kind::sum:
        po.get_params_sum(index, scale, zero_point, post_op_sum_dtype);
        break;
      case dnnl::primitive::kind::eltwise:
        po.get_params_eltwise(index, alg, alpha, beta);
        break;
      case dnnl::primitive::kind::binary:
        po.get_params_binary(index, alg, binary_src_desc);
        break;
      default:
        dnnl::error::wrap_c_api(dnnl_invalid_arguments, "could not get params");
        break;
    }

    return std::make_tuple(
        akind, scale, alpha, beta, alg, zero_point, binary_src_desc);
  }

  bool non_negitive_output() const {
    auto po = get_post_ops();
    auto last = po.len() - 1;
    if (last < 0) {
      return false;
    }

    auto params = get_params(last);
    if (std::get<0>(params) != dnnl::primitive::kind::eltwise ||
        std::get<1>(params) <= 0.f || std::get<2>(params) != 0.f ||
        std::get<3>(params) != 0.f ||
        std::get<4>(params) != dnnl::algorithm::eltwise_relu)
      return false;

    return true;
  }

  // deep data copy
  Attr& operator=(const Attr& rhs) {
    if (this == &rhs) {
      return *this;
    }
    dnnl_primitive_attr_t result;
    dnnl::error::wrap_c_api(
        dnnl_primitive_attr_clone(&result, rhs.get()),
        "could not clone primitive attributes");
    this->reset(result);
    if (rhs.has_scales()) {
      if (scales_) {
        *scales_ = rhs.get_all_scales();
      } else {
        scales_.reset(new scale_map(rhs.get_all_scales()));
      }
    }
    if (rhs.has_zero_points()) {
      if (zero_points_) {
        *zero_points_ = rhs.get_all_zero_points();
      } else {
        zero_points_.reset(new zero_point_map(rhs.get_all_zero_points()));
      }
    }
    return *this;
  }

  void to_bytes(utils::bytestring& bytes) const {
    // encode post ops
    auto num_ops = get_post_ops().len();
    for (int i = 0; i < num_ops; i++) {
      dnnl::primitive::kind akind;
      dnnl::algorithm alg = dnnl::algorithm::undef;
      float scale = 1.0, alpha = 1.0, beta = 0.0;
      int32_t zp = 0;
      dnnl::memory::desc binary_src_desc;
      std::tie(akind, scale, alpha, beta, alg, zp, binary_src_desc) =
          get_params(i);

      switch (akind) {
        case dnnl::primitive::kind::sum:
          utils::to_bytes(bytes, akind);
          bytes.append(1, '.');
          utils::to_bytes(bytes, scale);
          bytes.append(1, '.');
          utils::to_bytes(bytes, zp);
          break;
        case dnnl::primitive::kind::eltwise:
          utils::to_bytes(bytes, akind);
          bytes.append(1, '.');
          utils::to_bytes(bytes, scale);
          bytes.append(1, '.');
          utils::to_bytes(bytes, alpha);
          bytes.append(1, '.');
          utils::to_bytes(bytes, beta);
          bytes.append(1, '.');
          utils::to_bytes(bytes, alg);
          // ideep falls through here; kept so cache keys stay identical.
          [[fallthrough]];
        case dnnl::primitive::kind::binary:
          utils::to_bytes(bytes, akind);
          bytes.append(1, '.');
          utils::to_bytes(bytes, alg);
          bytes.append(1, '.');
          utils::to_bytes(bytes, binary_src_desc.get_dims());
          [[fallthrough]];
        default:
          break;
      }
    }

    // encode output scales
    if (has_scales()) {
      for (auto& scales : get_all_scales()) {
        utils::to_bytes(bytes, scales.first); // dnnl arg index
        utils::to_bytes(bytes, scales.second.first); // scale vector
        utils::to_bytes(bytes, scales.second.second); // mask
      }
    }

    // encode zero points
    if (has_zero_points()) {
      for (auto& zp : get_all_zero_points()) {
        utils::to_bytes(bytes, zp.first); // dnnl arg index
        utils::to_bytes(bytes, zp.second.first); // zero point vector
        utils::to_bytes(bytes, zp.second.second); // mask
      }
    }

    // encode computation modes
    utils::to_bytes(bytes, get_fpmath_mode()); // fpmath mode
    utils::to_bytes(bytes, get_scratchpad_mode()); // scratchpad mode
    utils::to_bytes(bytes, get_accumulation_mode()); // accumulation mode
    utils::to_bytes(bytes, get_deterministic()); // deterministic mode

    // Note: depthwise/binary post op, zero points, scales, rnn params are
    // not encoded so far. PD cache is supposed to use in convolution only
    // as a temporary workaround for gemm-based conv pd overhead
  }

  // ideep::tensor's reorders still take ideep::attr_t until the tensor port.
  operator ideep::attr_t() const {
    if (!has_scales() && !has_zero_points()) {
      return ideep::attr_t(static_cast<const dnnl::primitive_attr&>(*this));
    }
    dnnl_primitive_attr_t result;
    dnnl::error::wrap_c_api(
        dnnl_primitive_attr_clone(&result, get()),
        "could not clone primitive attributes");
    ideep::attr_t attr{dnnl::primitive_attr(result)};
    for (const auto& s : get_all_scales()) {
      attr.set_scales(s.first, s.second.second, s.second.first);
    }
    for (const auto& z : get_all_zero_points()) {
      attr.set_zero_points(z.first, z.second.second, z.second.first);
    }
    return attr;
  }

 private:
  std::shared_ptr<scale_map> scales_;
  std::shared_ptr<zero_point_map> zero_points_;
};

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
