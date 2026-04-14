#pragma once

#include <ATen/ATen.h>

#include <cstdint>
#include <cstring>
#include <ATen/native/mkldnn/xpu/detail/DnnlExt.h>
#include <ATen/native/mkldnn/xpu/detail/Utils.h>
#include <ATen/native/mkldnn/xpu/detail/oneDNNContext.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_types.h>

namespace at::native::onednn {
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
   Currently, oneDNN supports 5 kinds of post ops. More details can be referred
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

*/
using kind_t = dnnl::primitive::kind;
struct PostOpParam {
  // eltwise post op constructor
  PostOpParam(
      float scale,
      float alpha,
      float beta,
      dnnl::algorithm algo,
      kind_t kind)
      : scale_(scale), alpha_(alpha), beta_(beta), algo_(algo), kind_(kind) {}
  // sum post op constructor
  PostOpParam(float scale, kind_t kind) : scale_(scale), kind_(kind) {}
  // sum post op with zp
  PostOpParam(float scale, int64_t zero_point, kind_t kind)
      : scale_(scale), zero_point_(zero_point), kind_(kind) {}
  // binary post op constructor
  PostOpParam(
      at::Tensor& binary,
      dnnl::memory::desc& binary_md,
      dnnl::memory::desc& expected_md,
      dnnl::algorithm algo,
      kind_t kind)
      : binary_(binary),
        meta_(binary_md),
        expected_meta_(expected_md),
        algo_(algo),
        kind_(kind) {}
  // prelu post op constructor
  PostOpParam(int mask, kind_t kind) : mask_(mask), kind_(kind) {}

  // post sum or binary with scale post op constructor
  PostOpParam(
      at::Tensor& binary,
      float scale,
      dnnl::algorithm algo,
      kind_t kind)
      : scale_(scale), binary_(binary), algo_(algo), kind_(kind) {}

  // for int8 sum/eltwise
  float scale_ = 1.0;
  int64_t zero_point_ = 0;
  // for eltwise
  float alpha_ = 0.0;
  float beta_ = 0.0;
  // for binary
  at::Tensor binary_ = at::Tensor();
  at::Tensor expected_binary_ = at::Tensor();
  void* binary_ptr_ = nullptr;
  dnnl::memory::desc meta_ = dnnl::memory::desc();
  dnnl::memory::desc expected_meta_ = dnnl::memory::desc();
  // for prelu
  int mask_ = 0;
  // common
  dnnl::algorithm algo_ = dnnl::algorithm::eltwise_relu;
  kind_t kind_ = kind_t::eltwise;
};

namespace detail {

struct PostOpsMatmulKeySink {
  dnnl::memory::dims& key;

  void push_op_kind(kind_t kind) {
    key.push_back(static_cast<dnnl::memory::dim>(kind));
  }

  static void push_f32(dnnl::memory::dims& d, float f) {
    uint32_t u = 0;
    static_assert(sizeof(float) == sizeof(uint32_t));
    std::memcpy(&u, &f, sizeof(float));
    d.push_back(static_cast<dnnl::memory::dim>(u));
  }

  void append_eltwise(
      dnnl::algorithm aalgorithm, float alpha, float beta) {
    push_f32(key, alpha);
    push_f32(key, beta);
    key.push_back(static_cast<dnnl::memory::dim>(static_cast<int>(aalgorithm)));
  }

  void append_sum(float scale, std::int32_t zero_point) {
    push_f32(key, scale);
    key.push_back(static_cast<dnnl::memory::dim>(zero_point));
  }

  void append_binary(
      dnnl::algorithm aalgorithm, const dnnl::memory::desc& src1_desc) {
    key.push_back(static_cast<dnnl::memory::dim>(static_cast<int>(aalgorithm)));
    dnnl::memory::desc md = src1_desc;
    const std::vector<uint8_t> blob = md.get_blob();
    key.push_back(static_cast<dnnl::memory::dim>(blob.size()));
    constexpr size_t kPack = sizeof(dnnl::memory::dim);
    size_t off = 0;
    while (off < blob.size()) {
      const size_t rest = blob.size() - off;
      const size_t chunk = rest < kPack ? rest : kPack;
      dnnl::memory::dim packed = 0;
      std::memcpy(&packed, blob.data() + off, chunk);
      key.push_back(packed);
      off += chunk;
    }
  }

  void append_prelu(int mask) {
    key.push_back(static_cast<dnnl::memory::dim>(mask));
  }
};

constexpr std::size_t kExtractPostOpsLruCapacity = 512;

inline lru_cache<dnnl::memory::dims, dnnl::post_ops>&
get_extract_post_ops_lru_cache() {
  static thread_local lru_cache<dnnl::memory::dims, dnnl::post_ops> cache;
  if (cache.max_size() == 0) {
    cache.resize(kExtractPostOpsLruCapacity);
  }
  return cache;
}

} // namespace detail

inline void fp_matmul_post_sink_push_kind(
    dnnl::post_ops& /*sink*/,
    kind_t /*kind*/) noexcept {}

inline void fp_matmul_post_sink_push_kind(
    detail::PostOpsMatmulKeySink& sink,
    kind_t kind) {
  sink.push_op_kind(kind);
}

class Attr {
 public:
  Attr() : q_scale_(1.f) {}
  Attr(float q_scale, int64_t zp = 0) : q_scale_(q_scale), q_zero_point_(zp) {}

  float q_scale() const noexcept {
    return q_scale_;
  }
  int64_t q_zero_point() const noexcept {
    return q_zero_point_;
  }

  /***** eltwise *****/
  dnnl::algorithm kind_with_relu = dnnl::algorithm::eltwise_relu;
  dnnl::algorithm kind_with_sigmoid = dnnl::algorithm::eltwise_logistic;
  dnnl::algorithm kind_with_gelu_tanh = dnnl::algorithm::eltwise_gelu_tanh;
  dnnl::algorithm kind_with_gelu_erf = dnnl::algorithm::eltwise_gelu_erf;
  dnnl::algorithm kind_with_mish = dnnl::algorithm::eltwise_mish;
  dnnl::algorithm kind_with_linear = dnnl::algorithm::eltwise_linear;
  dnnl::algorithm kind_with_swish = dnnl::algorithm::eltwise_swish;
  dnnl::algorithm kind_with_sqrt = dnnl::algorithm::eltwise_sqrt;
  dnnl::algorithm kind_with_tanh = dnnl::algorithm::eltwise_tanh;
  dnnl::algorithm kind_with_square = dnnl::algorithm::eltwise_square;
  dnnl::algorithm kind_with_abs = dnnl::algorithm::eltwise_abs;
  dnnl::algorithm kind_with_exp = dnnl::algorithm::eltwise_exp;
  dnnl::algorithm kind_with_log = dnnl::algorithm::eltwise_log;
  dnnl::algorithm kind_with_round = dnnl::algorithm::eltwise_round;
  dnnl::algorithm kind_with_hardswish = dnnl::algorithm::eltwise_hardswish;
  dnnl::algorithm kind_with_soft_relu = dnnl::algorithm::eltwise_soft_relu;
  dnnl::algorithm kind_with_elu = dnnl::algorithm::eltwise_elu;
  dnnl::algorithm kind_with_pow = dnnl::algorithm::eltwise_pow;
  dnnl::algorithm kind_with_clip = dnnl::algorithm::eltwise_clip;
  // note: hardsigmoid seems oneDNN still not support
  dnnl::algorithm kind_with_hardsigmoid = dnnl::algorithm::eltwise_hardsigmoid;

  /***** binary *****/
  dnnl::algorithm kind_with_binary_mul = dnnl::algorithm::binary_mul;
  dnnl::algorithm kind_with_binary_add = dnnl::algorithm::binary_add;
  dnnl::algorithm kind_with_binary_sub = dnnl::algorithm::binary_sub;
  dnnl::algorithm kind_with_binary_div = dnnl::algorithm::binary_div;
  dnnl::algorithm kind_with_binary_eq = dnnl::algorithm::binary_eq;
  dnnl::algorithm kind_with_binary_ne = dnnl::algorithm::binary_ne;
  dnnl::algorithm kind_with_binary_ge = dnnl::algorithm::binary_ge;
  dnnl::algorithm kind_with_binary_gt = dnnl::algorithm::binary_gt;
  dnnl::algorithm kind_with_binary_le = dnnl::algorithm::binary_le;
  dnnl::algorithm kind_with_binary_lt = dnnl::algorithm::binary_lt;
  dnnl::algorithm kind_with_binary_max = dnnl::algorithm::binary_max;
  dnnl::algorithm kind_with_binary_min = dnnl::algorithm::binary_min;

  // append sum post op
  Attr& append_post_sum(
      float sum_scale,
      float sum_q_scale = 1.f,
      int64_t zp = 0) {
    ops_params_.push_back(
        PostOpParam(/*scale_sum*/ sum_scale * sum_q_scale, zp, kind_t::sum));
    return *this;
  }

  // append eltwise post op
  Attr& append_post_eltwise(
      float scale,
      float alpha,
      float beta,
      dnnl::algorithm algo) {
    ops_params_.push_back(
        PostOpParam(scale, alpha, beta, algo, kind_t::eltwise));
    return *this;
  }

  // append binary post op
  template <bool is_matmul = false>
  Attr& append_post_binary(dnnl::algorithm algo, const at::Tensor& binary) {
    auto binary_ = binary.is_quantized() ? at::dequantize(binary) : binary;
    bool binary_is_channels_last =
        (binary_.suggest_memory_format() == at::MemoryFormat::ChannelsLast ||
         binary_.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d);

    if constexpr (!is_matmul) {
      binary_ = binary_is_channels_last ? binary_ : binary_.contiguous();
    }
    dnnl::memory::desc md = get_onednn_md(binary_);
    auto expected_md = dnnl::memory::desc(
        md.get_dims(), md.get_data_type(), dnnl::memory::format_tag::any);
    if constexpr (is_matmul) {
      ops_params_.push_back(PostOpParam(binary_, md, md, algo, kind_t::binary));
    } else {
      ops_params_.push_back(
          PostOpParam(binary_, md, expected_md, algo, kind_t::binary));
    }

    return *this;
  }

  Attr& append_scale_binary(
      dnnl::algorithm algo,
      at::Tensor binary,
      float scale,
      float sum_q_scale = 1.f,
      int64_t zp = 0) {
    ops_params_.push_back(PostOpParam(
        binary, /*scale_sum*/ scale * sum_q_scale, algo, kind_t::binary));
    return *this;
  }

  // append bias with binary_add method (only used for QConv now)
  Attr& append_bias(const at::Tensor& binary, const int ndimension) {
    // In PyTorch, bias are in shape of [OC],
    // we expand its shape according to Conv dimension
    // Conv1d [OC, 1, 1], Conv2d [1, OC, 1, ,1], Conv3d [1, OC, 1, 1, 1]
    at::Tensor binary_ = binary.contiguous();
    dnnl::memory::desc binary_md;
    switch (ndimension) {
      case 1:
        binary_md = dnnl::memory::desc(
            {binary.size(0), 1, 1},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::abc);
        break;
      case 2:
        binary_md = dnnl::memory::desc(
            {1, binary.size(0), 1, 1},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::abcd);
        break;
      case 3:
        binary_md = dnnl::memory::desc(
            {1, binary.size(0), 1, 1, 1},
            dnnl::memory::data_type::f32,
            dnnl::memory::format_tag::abcde);
        break;
      default:
        TORCH_INTERNAL_ASSERT(
            0, "XPU only supports append_bias for Conv1d, Conv2d and Conv3d.");
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

  dnnl::memory::dims get_post_ops_key() const {
    dnnl::memory::dims key;
    detail::PostOpsMatmulKeySink key_sink{key};
    emit_post_ops_for_matmul_cache_and_dnnl(key_sink);
    return key;
  }

  dnnl::post_ops extract_post_ops() {
    dnnl::memory::dims cache_key = get_post_ops_key();
    auto& cache = detail::get_extract_post_ops_lru_cache();
    auto pos = cache.find(cache_key);
    if (pos != cache.end()) {
      return pos->second;
    }
    dnnl::post_ops po;
    emit_post_ops_for_matmul_cache_and_dnnl(po);
    auto [it, inserted] =
        cache.insert({std::move(cache_key), std::move(po)});
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
    return it->second;
  }

  bool with_sum() {
    for (size_t i = 0; i < ops_params_.size(); ++i) {
      if (ops_params_[i].kind_ == kind_t::sum) {
        return true;
      }
    }
    return false;
  }

  bool with_binary() {
    for (size_t i = 0; i < ops_params_.size(); ++i) {
      if (ops_params_[i].kind_ == kind_t::binary) {
        return true;
      }
    }
    return false;
  }

  void construct_post_binary(
      const dnnl::primitive_desc& pd,
      std::unordered_map<int, dnnl::memory>& args) {
    // This function is used to construct binary memory desc in binary post ops.
    // According to oneDNN doc, the binary tensor can be in shape of
    // [1, 1, 1, 1], tensor broadcast
    // [1, C, 1, 1], channel broadcast
    // [dst.shape], no broadcast and eltwise-wise binary operations on dst

    auto& engine = GpuEngineManager::Instance().get_engine();
    for (size_t i = 0; i < ops_params_.size(); ++i) {
      kind_t kind = ops_params_[i].kind_;
      if (kind == kind_t::binary) {
        dnnl::memory binary_m;
        auto binary = ops_params_[i].binary_;
        auto md = ops_params_[i].meta_;
        // query expected_md to achieve peak performance
        auto expected_md = pd.query_md(
            dnnl::query::exec_arg_md,
            DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1);

        binary_m = at::native::onednn::make_onednn_memory(
            md, engine, binary.data_ptr());

        args.insert(
            {DNNL_ARG_ATTR_MULTIPLE_POST_OP(i) | DNNL_ARG_SRC_1, binary_m});
      }
    }
  }

 private:
  template <typename PostOpsSink>
  void emit_post_ops_for_matmul_cache_and_dnnl(PostOpsSink&& sink) const {
    for (const auto& op : ops_params_) {
      fp_matmul_post_sink_push_kind(sink, op.kind_);
      switch (op.kind_) {
        case kind_t::eltwise:
          sink.append_eltwise(op.algo_, op.alpha_, op.beta_);
          break;
        case kind_t::sum:
          sink.append_sum(
              op.scale_, static_cast<std::int32_t>(op.zero_point_));
          break;
        case kind_t::binary:
          sink.append_binary(op.algo_, op.expected_meta_);
          break;
        case kind_t::prelu:
          sink.append_prelu(op.mask_);
          break;
        default:
          break;
      }
    }
  }

  float q_scale_ = 1.0; // the scale used to quantize the fused result from fp32
                        // to int8, only works for int8 case
  int64_t q_zero_point_ = 0;
  std::vector<PostOpParam> ops_params_; // series of post ops
};

static inline void construct_attr_for_unary(
    const std::string_view& unary_post_op,
    const torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    const std::string_view& unary_post_op_algorithm,
    at::native::onednn::Attr& attr) {
  if (unary_post_op == "relu") {
    attr = attr.append_post_eltwise(
        /* eltwise_scale */ 1.f,
        /* alpha */ 0.f,
        /* beta */ 0.f,
        attr.kind_with_relu);
  } else if (unary_post_op == "leaky_relu") {
    auto alpha = unary_post_op_args[0].value().to<float>();
    attr = attr.append_post_eltwise(1.0, alpha, 0.f, attr.kind_with_relu);
  } else if (unary_post_op == "tanh") {
    attr = attr.append_post_eltwise(1.0f, 0.0f, 0.0f, attr.kind_with_tanh);
  } else if (unary_post_op == "gelu") {
    auto post_algorithm = unary_post_op_algorithm == "none"
        ? attr.kind_with_gelu_erf
        : attr.kind_with_gelu_tanh;
    attr = attr.append_post_eltwise(1.0f, 0.0f, 0.0f, post_algorithm);
  } else if (unary_post_op == "hardtanh") {
    auto alpha = unary_post_op_args[0].value().to<float>();
    auto beta = unary_post_op_args[1].value().to<float>();
    attr = attr.append_post_eltwise(1.0, alpha, beta, attr.kind_with_clip);
  } else if (unary_post_op == "hardswish") {
    attr = attr.append_post_eltwise(
        1.0f, 1.f / 6.f, 1.f / 2.f, attr.kind_with_hardswish);
  } else if (unary_post_op == "swish") {
    attr = attr.append_post_eltwise(1.0f, 1.0f, 0.0f, attr.kind_with_swish);
  } else {
    TORCH_CHECK(
        unary_post_op == "none",
        "onednn qlinear: unsupported unary post op",
        unary_post_op);
  }
}

static inline void construct_attr_by_post_op(
    const std::string_view& binary_post_op,
    double binary_alpha,
    double input1_scale,
    int64_t input1_zero_point,
    std::optional<at::Tensor> accum,
    const std::string_view& unary_post_op,
    const torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    const std::string_view& unary_post_op_algorithm,
    at::native::onednn::Attr& attr) {
  bool is_none_post_op =
      (binary_post_op == "none" && unary_post_op == "none"); // not post-ops
  bool is_unary_post_op_only =
      (binary_post_op == "none" && unary_post_op != "none"); // ex., conv + relu
  bool is_valid_binary_combination =
      (binary_post_op == "add" || binary_post_op == "sum") &&
      (unary_post_op == "none" || unary_post_op == "relu");
  TORCH_INTERNAL_ASSERT(
      is_unary_post_op_only || is_none_post_op || is_valid_binary_combination,
      "Please provide valid combination of unary post operators and binary post operators");

  if (binary_post_op == "none") {
    construct_attr_for_unary(
        unary_post_op, unary_post_op_args, unary_post_op_algorithm, attr);
  } else if (binary_post_op == "sum") {
    if (unary_post_op == "none") {
      if (input1_zero_point != 0)
        attr = attr.append_post_eltwise(
            /*scale*/ 1.f,
            /*alpha*/ 1.f,
            -input1_zero_point * input1_scale,
            attr.kind_with_linear);
      attr = attr.append_post_sum(1, input1_scale, /*input1_zero_point*/ 0);
    } else if (unary_post_op == "relu") {
      if (input1_zero_point != 0)
        attr = attr.append_post_eltwise(
            /*scale*/ 1.f,
            /*alpha*/ 1.f,
            -input1_zero_point * input1_scale,
            attr.kind_with_linear);
      attr = attr.append_post_sum(1, input1_scale, /*input1_zero_point*/ 0);
      attr = attr.append_post_eltwise(
          /* scale */ 1.f,
          /* alpha */ 0.f,
          /* beta */ 0.f,
          attr.kind_with_relu);
    }
  } else if (binary_post_op == "add") {
    TORCH_CHECK(accum.has_value());
    attr = attr.append_post_binary(attr.kind_with_binary_add, accum.value());
    if (unary_post_op == "relu") {
      attr = attr.append_post_eltwise(1.f, 0.f, 0.f, attr.kind_with_relu);
    }
  }
}

} // namespace at::native::onednn
