#pragma once

#include "ATen/native/zendnn/AbstractTypes.hpp"
#include "ATen/native/zendnn/Utils.hpp"

namespace zendnn {

using post_ops = zendnn::post_ops;

/// Attribute class for extra information into computations
struct attr_t : public zendnn::primitive_attr {
  attr_t() {}

  attr_t(int mask, const scale_t& scales) {
    set_output_scales(mask, scales);
  }

  std::pair<scale_t, int> get_output_scales() const {
    zendnn_dim_t count;
    int c_mask;
    const float* c_scales;
    error::wrap_c_api(
        zendnn_primitive_attr_get_output_scales(
            get(), &count, &c_mask, &c_scales),
        "could not get int output scales");
    return std::make_pair(scale_t(c_scales, c_scales + count), c_mask);
  }

  // Helper factory
  static attr_t fuse_sum(float scale = 1.0) {
    attr_t attr;
    post_ops po;
    po.append_sum(scale);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t fuse_relu(
      float scale = 1.0,
      float alpha = 0.f,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_eltwise(scale, algorithm::eltwise_relu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t residual(
      float sum_scale = 1.0,
      float relu_scale = 1.0,
      float alpha = 0.f,
      float beta = 0.f) {
    attr_t attr;
    post_ops po;
    po.append_sum(sum_scale);
    po.append_eltwise(relu_scale, algorithm::eltwise_relu, alpha, beta);
    attr.set_post_ops(po);
    return attr;
  }

  static attr_t attr_post_ops(post_ops po) {
    attr_t attr;
    attr.set_post_ops(po);
    return attr;
  }

  bool has_op_kind(kind op_kind) const {
    auto po = get_post_ops();
    for (int i = 0; i < po.len(); i++)
      if (op_kind == po.kind(i))
        return true;
    return false;
  }

  std::tuple<kind, float, float, float, algorithm> get_params(int index) const {
    auto po = get_post_ops();
    ZENDNN_ENFORCE(index < po.len(), "post_ops index is out of range");

    algorithm alg;
    float scale = 1.0, alpha = 1.0, beta = 0.0;

    auto akind = po.kind(index);
    switch (akind) {
      case kind::sum:
        po.get_params_sum(index, scale);
        break;
      case kind::eltwise:
        po.get_params_eltwise(index, scale, alg, alpha, beta);
        break;
      default:
        error::wrap_c_api(zendnn_invalid_arguments, "could not get params");
        break;
    }

    return std::make_tuple(akind, scale, alpha, beta, alg);
  }

  bool non_negitive_output() const {
    auto po = get_post_ops();
    auto last = po.len() - 1;
    if (last < 0) {
      return false;
    }

    auto params = get_params(last);
    if (std::get<0>(params) != kind::eltwise || std::get<1>(params) <= 0.f ||
        std::get<2>(params) != 0.f || std::get<3>(params) != 0.f ||
        std::get<4>(params) != algorithm::eltwise_relu)
      return false;

    return true;
  }

  void to_bytes(utils::bytestring& bytes) const {
    // encode post ops
    auto num_ops = get_post_ops().len();
    for (int i = 0; i < num_ops; i++) {
      kind akind;
      algorithm alg;
      float scale = 1.0, alpha = 1.0, beta = 0.0;
      std::tie(akind, scale, alpha, beta, alg) = get_params(i);

      switch (akind) {
        case kind::sum:
          utils::to_bytes(bytes, akind);
          bytes.append(1, '.');
          utils::to_bytes(bytes, scale);
          break;
        case kind::eltwise:
          utils::to_bytes(bytes, akind);
          bytes.append(1, '.');
          utils::to_bytes(bytes, scale);
          bytes.append(1, '.');
          utils::to_bytes(bytes, alpha);
          bytes.append(1, '.');
          utils::to_bytes(bytes, beta);
          bytes.append(1, '.');
          utils::to_bytes(bytes, alg);
        default:
          break;
      }
    }

    // encode output scales
    auto scales = get_output_scales();
    utils::to_bytes(bytes, scales.first);
    utils::to_bytes(bytes, scales.second);

    // Note: depthwise/binary post op, zero points, scales, rnn params are
    // not encoded so far. PD cache is supposed to use in convolution only
    // as a temporary workaround for gemm-based conv pd overhead
  }
};

} // namespace zendnn
