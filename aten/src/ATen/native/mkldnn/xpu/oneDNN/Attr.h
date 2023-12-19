#pragma once

#include <ATen/ATen.h>
// #include <core/MemoryFormat.h>
// #include <core/detail/TensorInfo.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_types.h>
// #include <tensor/Context.h>
// #include <utils/Macros.h>
#include "Utils.h"

using namespace dnnl;
// using namespace xpu::dpcpp;

namespace xpu {
namespace oneDNN {
/* oneDNN quantization usage:
   https://oneapi-src.github.io/oneDNN/dev_guide_attributes_quantization.html#

   src_fp32 = scale_src * (src_int8 - zero_point)
   wei_fp32 = scale_wei * (wei_int8 - zero_point)
   dst_fp32 = scale_dst * (dst_int8 - zero_point)
   fp32 Convolution: dst_fp32 = src_fp32 * wei_fp32
   Int8 Convolution: dst_fp32 = (src_int8 * wei_int8) * (scale_src * scale_wei)
   Int8 Convolution: dst_int8 = 1 / scale_dst * dst_fp32;

   Considering zero-point (asymmetric):
   dst_fp32 = (src_int8 - src_zp) * src_sc * wei_int8 * wei_sc
   dst_sc * (dst_int8 - dst_zp) = (src_int8 - src_zp) * wei_int8  * src_sc *
                                 wei_sc
   dst_int8 = (src_int8 - src_zp) * wei_int8 * src_sc * wei_sc / dst_sc +
              dst_zp

   considering bias:
   fp32 Convolution: dst_fp32 = src_fp32 * wei_fp32 + bias
   Int8 Convolution: dst_fp32 = (src_int8 * wei_int8) * (scale_src * scale_wei)
   + bias Int8 Convolution: dst_fp32 = (src_int8 * wei_int8 + bias/(scale_src *
   scale_wei)) * (scale_src * scale_wei) Int8 Convolution: dst_int8 = 1 /
   scale_dst * dst_fp32;
*/

/*
   oneDNN postops usage:
   Currently, oneDNN supports 5 kinds of post ops. More details can be refered
to oneDNN doc.
   https://oneapi-src.github.io/oneDNN/dev_guide_attributes_post_ops.html#doxid-dev-guide-attributes-post-ops-1dev-guide-attributes-post-ops-eltwise

0. without post ops
   dst = Conv(src, wei) + bias;
   dst_int8 = 1/q_scale * dst; q_scale is the op output quantization scale
   fp32 API: Attr attr;
   int8 API: Attr attr(q_scale);

1. append eltwise post op
   dst = elt_scale * Eltwise{conv_scale * [Conv(src, wei) + bias], alpha, beta}
   dst_int8 = 1/q_scale * dst;
   fp32 API:
   Attr attr;
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_eltwise(elt_scale, alpha, beta, eltwise_algorithm)
   int8 API:
   Attr attr(q_scale);
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_eltwise(elt_scale, alpha, beta, eltwise_algorithm)

2. append sum post op
   dst = conv_scale * Conv(src, wei) + sum_scale * (dst - zp)
   dst_int8 = 1/q_scale * dst;
   fp32 API:
   Attr attr;
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_sum(sum_scale)
   int8 API:
   Attr attr(q_scale);
   attr.append_post_eltwise(1.f, conv_scale, 0.f, kind_with_linear)
   attr.append_post_sum(sum_scale)

3. append binary post op
   dst = Binary[Conv(src, wei)]

4. append prelu post op
   // TODO:
   fusion_dst = prelu(Conv(src, wei), weights[:])

5. append depthwise conv post op
   // TODO:
   fusion_dst = Convdw(Conv1x1(...))
*/
using kind_t = dnnl::primitive::kind;
struct PostOpParam {
  // eltwise post op constructor
  PostOpParam(float scale, float alpha, float beta, algorithm algo, kind_t kind)
      : scale_(scale), alpha_(alpha), beta_(beta), algo_(algo), kind_(kind) {}
  // sum post op constructor
  PostOpParam(float scale, kind_t kind) : scale_(scale), kind_(kind) {}
  // binary post op constructor
  PostOpParam(
      at::Tensor& binary,
      dnnl::memory::desc& binary_md,
      dnnl::memory::desc& expected_md,
      algorithm algo,
      kind_t kind)
      : binary_(binary),
        meta_(binary_md),
        expected_meta_(expected_md),
        algo_(algo),
        kind_(kind) {}
  // prelu post op constructor
  PostOpParam(int mask, kind_t kind) : mask_(mask), kind_(kind) {}

  // post sum or binary with scale post op constructor
  PostOpParam(at::Tensor& binary, float scale, algorithm algo, kind_t kind)
      : scale_(scale), binary_(binary), algo_(algo), kind_(kind) {}

  // for int8 sum/eltwise
  float scale_ = 1.0;
  // for eltwise
  float alpha_ = 0.0;
  float beta_ = 0.0;
  // for binary
  at::Tensor binary_ = at::Tensor();
  at::Tensor expected_binary_ = at::Tensor();
  void* binary_ptr_ = nullptr;
  dnnl::memory::desc meta_ = memory::desc();
  dnnl::memory::desc expected_meta_ = memory::desc();
  // for prelu
  int mask_ = 0;
  // common
  algorithm algo_ = algorithm::eltwise_relu;
  kind_t kind_ = kind_t::eltwise;
};

class Attr {
 public:
  Attr() : q_scale_(1.f), q_zero_point_(0) {}
  Attr(float q_scale, int64_t zp = 0) : q_scale_(q_scale), q_zero_point_(zp) {}

  /***** eltwise *****/
  algorithm kind_with_relu = algorithm::eltwise_relu;
  algorithm kind_with_sigmoid = algorithm::eltwise_logistic;
  algorithm kind_with_gelu_tanh = algorithm::eltwise_gelu_tanh;
  algorithm kind_with_gelu_erf = algorithm::eltwise_gelu_erf;
  algorithm kind_with_mish = algorithm::eltwise_mish;
  algorithm kind_with_linear = algorithm::eltwise_linear;
  algorithm kind_with_swish = algorithm::eltwise_swish;
  algorithm kind_with_sqrt = algorithm::eltwise_sqrt;
  algorithm kind_with_tanh = algorithm::eltwise_tanh;
  algorithm kind_with_square = algorithm::eltwise_square;
  algorithm kind_with_abs = algorithm::eltwise_abs;
  algorithm kind_with_exp = algorithm::eltwise_exp;
  algorithm kind_with_log = algorithm::eltwise_log;
  algorithm kind_with_round = algorithm::eltwise_round;
  algorithm kind_with_hardswish = algorithm::eltwise_hardswish;
  algorithm kind_with_soft_relu = algorithm::eltwise_soft_relu;
  algorithm kind_with_elu = algorithm::eltwise_elu;
  algorithm kind_with_pow = algorithm::eltwise_pow;
  algorithm kind_with_clip = algorithm::eltwise_clip;
  // note: hardsigmoid seems oneDNN still not support
  algorithm kind_with_hardsigmoid = algorithm::eltwise_hardsigmoid;

  /***** binary *****/
  algorithm kind_with_binary_mul = algorithm::binary_mul;
  algorithm kind_with_binary_add = algorithm::binary_add;
  algorithm kind_with_binary_sub = algorithm::binary_sub;
  algorithm kind_with_binary_div = algorithm::binary_div;
  algorithm kind_with_binary_eq = algorithm::binary_eq;
  algorithm kind_with_binary_ne = algorithm::binary_ne;
  algorithm kind_with_binary_ge = algorithm::binary_ge;
  algorithm kind_with_binary_gt = algorithm::binary_gt;
  algorithm kind_with_binary_le = algorithm::binary_le;
  algorithm kind_with_binary_lt = algorithm::binary_lt;
  algorithm kind_with_binary_max = algorithm::binary_max;
  algorithm kind_with_binary_min = algorithm::binary_min;

  // append sum post op
  Attr& append_post_sum(
      float sum_scale,
      float sum_q_scale = 1.f,
      int64_t zp = 0) {
    ops_params_.push_back(
        PostOpParam(/*scale_sum*/ sum_scale * sum_q_scale, kind_t::sum));
    return *this;
  }

  // append eltwise post op
  Attr& append_post_eltwise(
      float scale,
      float alpha,
      float beta,
      algorithm algo) {
    ops_params_.push_back(
        PostOpParam(scale, alpha, beta, algo, kind_t::eltwise));
    return *this;
  }

  // append binary post op
  Attr& append_post_binary(algorithm algo, const at::Tensor& binary) {
    // auto binary_ = binary.is_quantized() ? at::dequantize(binary) : binary;
    // auto ctx = DPCPPTensorContext::get_tensor_ctx(binary_);
    // memory::desc md;
    // if (ctx.is_plain()) {
      // binary_ = is_smf_channels_last(binary_) ? binary_ : binary_.contiguous();
      // md = get_onednn_md(binary_);
    // } else {
    //   md = ctx.meta();
    // }

    // Zhiwei modified
    auto binary_ = binary.is_quantized() ? at::dequantize(binary) : binary;
    bool binary_is_channels_last = (binary_.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
                                      binary_.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d);

    binary_ = binary_is_channels_last ? binary_ : binary_.contiguous();
    memory::desc md = get_onednn_md(binary_);
    auto expected_md = memory::desc(
        md.get_dims(), md.get_data_type(), memory::format_tag::any);
    ops_params_.push_back(
        PostOpParam(binary_, md, expected_md, algo, kind_t::binary));
    return *this;
  }

  Attr& append_scale_binary(
      algorithm algo,
      at::Tensor binary,
      float scale,
      float sum_q_scale = 1.f,
      int64_t zp = 0) {
    ops_params_.push_back(PostOpParam(
        binary, /*scale_sum*/ scale * sum_q_scale, algo, kind_t::binary));
    return *this;
  }

  // append bias with binary_add method (only used for QConv now)
  template <int N>
  Attr& append_bias(const at::Tensor& binary) {
    // In PyTorch, bias are in shape of [OC],
    // we expand its shape according to Conv dimension
    // Conv1d [OC, 1, 1], Conv2d [1, OC, 1, ,1], Conv3d [1, OC, 1, 1, 1]
    at::Tensor binary_ = binary.contiguous();
    memory::desc binary_md;
    switch (N) {
      case 1:
        binary_md = memory::desc(
            {binary.size(0), 1, 1},
            memory::data_type::f32,
            memory::format_tag::abc);
        break;
      case 2:
        binary_md = memory::desc(
            {1, binary.size(0), 1, 1},
            memory::data_type::f32,
            memory::format_tag::abcd);
        break;
      case 3:
        binary_md = memory::desc(
            {1, binary.size(0), 1, 1, 1},
            memory::data_type::f32,
            memory::format_tag::abcde);
        break;
      default:
        AT_ERROR(
            "IPEX only supports append_bias for Conv1d, Conv2d and Conv3d.");
    }
    // In this case, expected_md = binary_md
    ops_params_.push_back(PostOpParam(
        binary_, binary_md, binary_md, kind_with_binary_add, kind_t::binary));
    return *this;
  }

  // append prelu post op
  Attr& append_post_prelu(int mask) {
    ops_params_.push_back(PostOpParam(mask, kind_t::prelu));
    return *this;
  }

  void extract_post_ops(post_ops& dnnl_post_ops, const at::Tensor& dst) {
    // this function is used to extract post ops params from the ops_params_
    // and put them into onednn post ops
    for (int i = 0; i < ops_params_.size(); ++i) {
      kind_t kind = ops_params_[i].kind_;
      switch (kind) {
        case kind_t::eltwise: {
          algorithm algo = ops_params_[i].algo_;
          float alpha = ops_params_[i].alpha_;
          float beta = ops_params_[i].beta_;
          dnnl_post_ops.append_eltwise(algo, alpha, beta);
          break;
        }
        case kind_t::sum: {
          float scale = ops_params_[i].scale_;
          // TODO [Asymmetric]:
          // Post-sum zp for gpu is not supported currently
          dnnl_post_ops.append_sum(scale);
          break;
        }
        case kind_t::binary: {
          algorithm algo = ops_params_[i].algo_;
          auto expected_md = ops_params_[i].expected_meta_;
          // In this case user may create src1 memory descriptor with
          // format_tag::any or set a specific tag. However, in later case if
          // tags mismatch with dst, it would result in suboptimal performance.
          // So here we use format_tag::any to make sure the fast can be
          // selected.
          // Thus we use expected_md (with format_any) here to create pd instead
          // of original md
          dnnl_post_ops.append_binary(algo, expected_md);
          break;
        }
        default:
          break;
      }
    }

    // if output is quantized, then append the eltwise linear to adjust the
    // output scale/zero_point
    if (dst.is_quantized()) {
      // [Note: Gap of u8 qtensor scale between oneDNN and PyTorch]
      // The /2 here is for output_scale collected by observer is different
      // from quantization requirements in oneDNN.
      // For Observer, the conv_scale (activation scale in other case) is
      // computed through 2max_v/(qmax - qmin). The max_v is collected
      // from the tensor to be observerd.
      // (https://pytorch.org/docs/stable/generated/torch.quantization.observer.MinMaxObserver.html#torch.quantization.observer.MinMaxObserver)
      // On the other hand, for u8 in oneDNN, the scale for quantization is
      // defined as max_v/(qmax-qmin). Hence, we need to divide by 2 here.
      // (https://oneapi-src.github.io/oneDNN/dev_guide_inference_int8.html)
      dnnl_post_ops.append_eltwise(
          kind_with_linear, 1.f / q_scale_, q_zero_point_);
    }
  }

  bool with_sum() {
    for (int i = 0; i < ops_params_.size(); ++i) {
      if (ops_params_[i].kind_ == kind_t::sum) {
        return true;
      }
    }
    return false;
  }

  bool with_binary() {
    for (int i = 0; i < ops_params_.size(); ++i) {
      if (ops_params_[i].kind_ == kind_t::binary) {
        return true;
      }
    }
    return false;
  }

  void construct_post_binary(
      primitive_desc& pd,
      post_ops& dnnl_post_ops,
      std::unordered_map<int, memory>& args) {
    // This function is used to construct binary memory desc in binary post ops.
    // According to oneDNN doc, the binary tensor can be in shape of
    // [1, 1, 1, 1], tensor broadcast
    // [1, C, 1, 1], channel broadcast
    // [dst.shape], no broadcast and eltwise-wise binary operations on dst

    // Zhiwei modified
    // auto engine =
    //     GpuEngineManager::Instance().get_engine({c10::kXPU, current_device()});
    for (int i = 0; i < ops_params_.size(); ++i) {
      kind_t kind = ops_params_[i].kind_;
      if (kind == kind_t::binary) {
        memory binary_m;
        auto binary = ops_params_[i].binary_;
        auto md = ops_params_[i].meta_;
        // qeury expected_md to achieve peak performance
        auto expected_md = pd.query_md(
            query::exec_arg_md,
            DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1);

        // if (md != expected_md) {
        //   ops_params_[i].expected_binary_ =
        //       empty_opaque_tensor(expected_md, binary.options(), c10::nullopt);
        //   binary_m = dpcpp_onednn_memory(
        //       expected_md, engine, ops_params_[i].expected_binary_.data_ptr());
        //   xpu::oneDNN::reorder(binary, ops_params_[i].expected_binary_);
        // } else {
        //   binary_m = dpcpp_onednn_memory(md, engine, binary.data_ptr());
        // }
        args.insert(
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, binary_m});
      }
    }
  }

#ifdef USE_PRIMITIVE_CACHE
  void to_bytes(bytestring& bytes) {
    xpu::dpcpp::to_bytes(bytes, q_scale_);
    xpu::dpcpp::to_bytes(bytes, q_zero_point_);
    for (int i = 0; i < ops_params_.size(); ++i) {
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].scale_);
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].alpha_);
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].beta_);
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].algo_);
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].kind_);
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].mask_);
      xpu::dpcpp::to_bytes(bytes, ops_params_[i].meta_);
    }
  }
#endif

  float q_scale_ = 1.0; // the scale used to quantize the fused result from fp32
                        // to int8, only works for int8 case
  int64_t q_zero_point_ = 0;
  std::vector<PostOpParam> ops_params_; // series of post ops
}; // namespace oneDNN

} // namespace oneDNN
} // namespace xpu
