#include <ATen/native/mkldnn/detail/InnerProduct.h>
#include <ATen/native/mkldnn/detail/Utils.h>

#if AT_MKLDNN_ENABLED()

#include <unordered_map>
#include <utility>

namespace at::native::mkldnn {

namespace {

// Ported from ideep::inner_product_forward_params.
struct inner_product_forward_params {
  dnnl::inner_product_forward::primitive_desc pd;
  dnnl::inner_product_forward primitive;
  ideep::attr_t op_attr;
  ideep::attr_t src_attr;
  ideep::attr_t weights_attr;
  ideep::attr_t bias_attr;
};

// Ported from ideep::inner_product_forward::get_primitive_desc. On x86 the
// descriptor comes from ideep's primitive cache, shared with ideep's own
// inner_product_forward.
#ifdef __aarch64__
std::pair<
    dnnl::inner_product_forward::primitive_desc,
    dnnl::inner_product_forward>
get_primitive_desc(
    const ideep::tensor::desc& src_desc,
    const ideep::tensor::desc& weights_desc,
    const ideep::tensor::desc& dst_desc,
    const ideep::tensor::desc& bias_desc = ideep::tensor::desc(),
    const bool with_bias = false,
    const ideep::attr_t& attr = ideep::attr_t(),
    const dnnl::prop_kind prop_kind = dnnl::prop_kind::forward) {
  const auto& engine = ideep::engine::cpu_engine();
  dnnl::inner_product_forward::primitive_desc pd;
  if (with_bias) {
    pd = dnnl::inner_product_forward::primitive_desc(
        engine, prop_kind, src_desc, weights_desc, bias_desc, dst_desc, attr);
  } else {
    pd = dnnl::inner_product_forward::primitive_desc(
        engine, prop_kind, src_desc, weights_desc, dst_desc, attr);
  }
  return std::make_pair(pd, dnnl::inner_product_forward(pd));
}
#else
dnnl::inner_product_forward::primitive_desc get_primitive_desc(
    const ideep::tensor::desc& src_desc,
    const ideep::tensor::desc& weights_desc,
    const ideep::tensor::desc& dst_desc,
    const ideep::tensor::desc& bias_desc = ideep::tensor::desc(),
    const bool with_bias = false,
    const ideep::attr_t& attr = ideep::attr_t(),
    const dnnl::prop_kind prop_kind = dnnl::prop_kind::forward) {
  const auto& engine = ideep::engine::cpu_engine();
  auto key = ideep::utils::create_key(
      prop_kind,
      src_desc,
      weights_desc,
      bias_desc,
      dst_desc,
      attr,
      with_bias,
      omp_get_max_threads());
  return ideep::utils::computation_cache<
      dnnl::inner_product_forward::primitive_desc>::
      fetch_or_create(key, [&]() {
        if (with_bias) {
          return dnnl::inner_product_forward::primitive_desc(
              engine,
              prop_kind,
              src_desc,
              weights_desc,
              bias_desc,
              dst_desc,
              attr);
        } else {
          return dnnl::inner_product_forward::primitive_desc(
              engine, prop_kind, src_desc, weights_desc, dst_desc, attr);
        }
      });
}
#endif

// Ported from ideep::inner_product_forward::do_prepare_.
template <bool with_bias, bool reorder_src, bool reorder_weight>
void do_prepare_(
    inner_product_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const ideep::attr_t& attr,
    const dnnl::prop_kind prop_kind) {
  ideep::tensor::desc src_desc, weights_desc, bias_desc, dst_desc;
  ideep::attr_t& op_attr = param.op_attr;
  ideep::attr_t& src_attr = param.src_attr;
  dnnl::memory::data_type dst_data_type;
  auto dst_dims = {src.get_dim(0), weights.get_dim(0)};

  op_attr = attr;
  // Below is used by int8 FC in Caffe2
  if (src.has_scale()) {
    auto src_scale = src.get_scale();
    src_scale[0] = 1.f / src_scale[0];
    src_attr = {0, src_scale};
  }

#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      one_of(
          weights.get_data_type(),
          dnnl::memory::data_type::f32,
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)
          ? dnnl_success
          : dnnl_invalid_arguments,
      "Incorrect data type in weights");
#endif
  // align weights data type with src
  dst_data_type = src.get_data_type() == dnnl::memory::data_type::bf16
      ? dnnl::memory::data_type::bf16
      : ((src.get_data_type() == dnnl::memory::data_type::f16)
             ? dnnl::memory::data_type::f16
             : dnnl::memory::data_type::f32);
  if (!reorder_weight) {
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        weights.get_data_type() == src.get_data_type() ? dnnl_success
                                                       : dnnl_invalid_arguments,
        "weights' data type should be same with input's data type when reorder_weight is false");
#endif
  }
  src_desc = reorder_src
      ? ideep::tensor::desc(
            src.get_dims(), dst_data_type, dnnl::memory::format_tag::any)
      : src.get_desc();
  weights_desc = reorder_weight
      ? ideep::tensor::desc(
            weights.get_dims(), dst_data_type, dnnl::memory::format_tag::any)
      : weights.get_desc();
  if (with_bias) {
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        one_of(
            bias.get_data_type(),
            dnnl::memory::data_type::f32,
            dnnl::memory::data_type::bf16,
            dnnl::memory::data_type::f16)
            ? dnnl_success
            : dnnl_invalid_arguments,
        "Incorrect data type in bias");
#endif
    bias_desc =
        reorder_weight ? bias.get_desc().to_format_any() : bias.get_desc();
  }

  if (!reorder_src) {
#ifndef NDEBUG
    dnnl::error::wrap_c_api(
        !dst.is_empty() && dst.get_data_type() == src.get_data_type()
            ? dnnl_success
            : dnnl_invalid_arguments,
        "dst can't be a empty tensor and data type should be same with input when reorder_src is false");
#endif
  }
  dst_desc = reorder_src
      ? ideep::tensor::desc(
            dst_dims, dst_data_type, dnnl::memory::format_tag::any)
      : dst.get_desc();

  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

#ifdef __aarch64__
  auto pd_pair = get_primitive_desc(
      src_desc,
      weights_desc,
      dst_desc,
      bias_desc,
      with_bias,
      op_attr,
      prop_kind);
  param.pd = std::move(pd_pair.first);
  param.primitive = std::move(pd_pair.second);
#else
  param.pd = get_primitive_desc(
      src_desc,
      weights_desc,
      dst_desc,
      bias_desc,
      with_bias,
      op_attr,
      prop_kind);
  param.primitive = dnnl::inner_product_forward(param.pd);
#endif
}

// Ported from ideep::inner_product_forward::do_compute_binary.
template <bool with_bias, bool reorder_src, bool reorder_weight>
void do_compute_binary(
    const inner_product_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& other,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst) {
  auto& pd = param.pd;
  auto& primitive = param.primitive;
  auto& src_attr = param.src_attr;
  auto& weights_attr = param.weights_attr;
  auto& bias_attr = param.bias_attr;

  auto& expected_src =
      reorder_src ? src.reorder_if_differ_in(pd.src_desc(), src_attr) : src;
  // make sure other has contiguous format.
  auto other_desc = ideep::tensor::desc(other.get_dims(), other.get_data_type());
  auto& expected_other =
      reorder_src ? other.reorder_if_differ_in(other_desc) : other;
  auto& expected_weights = reorder_weight
      ? weights.reorder_if_differ_in(pd.weights_desc(), weights_attr)
      : weights;
  ideep::tensor scratchpad(pd.scratchpad_desc());

  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, expected_src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weights});
  if (with_bias) {
    auto& expected_bias = reorder_weight
        ? bias.reorder_if_differ_in(pd.bias_desc(), bias_attr)
        : bias;
    args.insert({DNNL_ARG_BIAS, expected_bias});
  }
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  args.insert(
      {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, expected_other});
  if (reorder_src) {
    ideep::tensor expected_dst;
    if (dst.is_empty() || dst.get_desc() != pd.dst_desc()) {
      // If dst buffer are not given by user or user given dst buffer are not
      // under expected format We need init a new one. "dst.get_desc() !=
      // pd.dst_desc()" conditional is setting for caffe2 caller, it might
      // given a non-empty but incorrect dst (maybe the size is incorrect)
      expected_dst.init(pd.dst_desc());
    } else {
      // The format of given dst buffer is expected
      expected_dst = dst;
    }
    args.insert({DNNL_ARG_DST, expected_dst});
    primitive.execute(ideep::stream::default_stream(), args);

    // reorder back to dst's buffer if needed
    if (dst.is_empty() ||
        // when dst is empty, expect return buffer allocate by ideep
        dst.get_desc() == expected_dst.get_desc() ||
        // dst and expected_dst is the same under this case
        !dst.get_desc().has_same_shape_as(expected_dst.get_desc())) {
      // for caffe2 caller, get an incorrect size dst from caller, can return
      // buffer allocate by ideep
      dst = expected_dst;
    } else {
      dst.feed_from(expected_dst);
    }
  } else { // reorder_src
    if (!reorder_src) {
#ifndef NDEBUG
      dnnl::error::wrap_c_api(
          !dst.is_empty() ? dnnl_success : dnnl_invalid_arguments,
          "dst can't be a empty tensor when reorder_src is false");
#endif
    }
    args.insert({DNNL_ARG_DST, dst});
    primitive.execute(ideep::stream::default_stream(), args);
  }
}

// Ported from ideep::inner_product_forward::do_compute_.
template <bool with_bias, bool reorder_src, bool reorder_weight>
void do_compute_(
    const inner_product_forward_params& param,
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst) {
  auto& pd = param.pd;
  auto& primitive = param.primitive;
  auto& op_attr = param.op_attr;
  auto& src_attr = param.src_attr;
  auto& weights_attr = param.weights_attr;
  auto& bias_attr = param.bias_attr;

  auto& expected_src =
      reorder_src ? src.reorder_if_differ_in(pd.src_desc(), src_attr) : src;
  auto& expected_weights = reorder_weight
      ? weights.reorder_if_differ_in(pd.weights_desc(), weights_attr)
      : weights;
  ideep::tensor scratchpad(pd.scratchpad_desc());

  std::unordered_map<int, dnnl::memory> args;
  args.insert({DNNL_ARG_SRC, expected_src});
  args.insert({DNNL_ARG_WEIGHTS, expected_weights});
  if (with_bias) {
    auto& expected_bias = reorder_weight
        ? bias.reorder_if_differ_in(pd.bias_desc(), bias_attr)
        : bias;
    args.insert({DNNL_ARG_BIAS, expected_bias});
  }
  args.insert({DNNL_ARG_SCRATCHPAD, scratchpad});
  if (reorder_src) {
    ideep::tensor expected_dst;
    if (dst.is_empty() || dst.get_desc() != pd.dst_desc()) {
      // If dst buffer are not given by user or user given dst buffer are not
      // under expected format We need init a new one. "dst.get_desc() !=
      // pd.dst_desc()" conditional is setting for caffe2 caller, it might
      // given a non-empty but incorrect dst (maybe the size is incorrect)
      expected_dst.init(pd.dst_desc());
      if (!dst.is_empty() && op_attr.has_op_kind(dnnl::primitive::kind::sum)) {
        // We need copy the content of given buffer if ip is fused with sum
        expected_dst.feed_from(dst);
      }
    } else {
      // The format of given dst buffer is expected
      expected_dst = dst;
    }
    args.insert({DNNL_ARG_DST, expected_dst});
    primitive.execute(ideep::stream::default_stream(), args);

    // reorder back to dst's buffer if needed
    if (dst.is_empty() ||
        // when dst is empty, expect return buffer allocate by ideep
        dst.get_desc() == expected_dst.get_desc() ||
        // dst and expected_dst is the same under this case
        !dst.get_desc().has_same_shape_as(expected_dst.get_desc())) {
      // for caffe2 caller, get an incorrect size dst from caller, can return
      // buffer allocate by ideep
      dst = expected_dst;
    } else {
      dst.feed_from(expected_dst);
    }
  } else { // reorder_src
    if (!reorder_src) {
#ifndef NDEBUG
      dnnl::error::wrap_c_api(
          !dst.is_empty() ? dnnl_success : dnnl_invalid_arguments,
          "dst can't be a empty tensor when reorder_src is false");
#endif
    }
    args.insert({DNNL_ARG_DST, dst});
    primitive.execute(ideep::stream::default_stream(), args);
  }
}

// Ported from ideep::inner_product_forward::compute_impl.
template <bool with_bias, bool reorder_src, bool reorder_weight>
void compute_impl(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const ideep::attr_t& attr,
    const dnnl::prop_kind prop_kind) {
  inner_product_forward_params param;
  // workaround: src and weights from caffe2 may have different dims.
  // It would be better for caffe2 to do this reshape anyway.
  if (src.ndims() != weights.ndims()) {
    auto src_ = src;
    auto new_dims = weights.get_dims();
    new_dims[0] = src.get_dim(0);
    src_.reshape(new_dims);
    do_prepare_<with_bias, reorder_src, reorder_weight>(
        param, src_, weights, bias, dst, attr, prop_kind);
    do_compute_<with_bias, reorder_src, reorder_weight>(
        param, src_, weights, bias, dst);
  } else {
    do_prepare_<with_bias, reorder_src, reorder_weight>(
        param, src, weights, bias, dst, attr, prop_kind);
    do_compute_<with_bias, reorder_src, reorder_weight>(
        param, src, weights, bias, dst);
  }
}

// Ported from ideep::inner_product_backward_weights::compute_impl.
template <bool with_diff_bias>
void backward_weights_compute_impl(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    ideep::tensor& diff_weights,
    ideep::tensor& diff_bias) {
  if (is_avx2_vnni_2() &&
      one_of(
          diff_dst.get_data_type(),
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)) {
    dnnl::error::wrap_c_api(
        dnnl_invalid_arguments,
        "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
  }
  const auto& engine = ideep::engine::cpu_engine();
  auto src_desc = src.get_desc().to_format_any();
  auto diff_dst_desc = diff_dst.get_desc().to_format_any();
  auto diff_weights_dims = src.get_dims();
  diff_weights_dims[0] = diff_dst.get_dim(1);
  dnnl::memory::data_type diff_dst_type = diff_dst.get_data_type();
  auto diff_weights_desc = ideep::tensor::desc(
      diff_weights_dims, diff_dst_type, dnnl::memory::format_tag::any);

  auto diff_bias_desc = ideep::tensor::desc(
      {diff_dst.get_dim(1)}, diff_dst_type, dnnl::memory::format_tag::any);

  // for forward hint, weights_desc should have same data_type
  // with other input desc, expect for bias_desc
  auto weights_desc = diff_weights_desc;
  ideep::attr_t op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto forward_hints = get_primitive_desc(
      src_desc,
      weights_desc,
      diff_dst_desc,
      diff_bias_desc,
      with_diff_bias,
      op_attr);

#ifdef __aarch64__
  auto pd = with_diff_bias
      ? dnnl::inner_product_backward_weights::primitive_desc(
            engine,
            src_desc,
            diff_weights_desc,
            diff_bias_desc,
            diff_dst_desc,
            forward_hints.first,
            op_attr)
      : dnnl::inner_product_backward_weights::primitive_desc(
            engine,
            src_desc,
            diff_weights_desc,
            diff_dst_desc,
            forward_hints.first,
            op_attr);
#else
  auto pd = with_diff_bias
      ? dnnl::inner_product_backward_weights::primitive_desc(
            engine,
            src_desc,
            diff_weights_desc,
            diff_bias_desc,
            diff_dst_desc,
            forward_hints,
            op_attr)
      : dnnl::inner_product_backward_weights::primitive_desc(
            engine,
            src_desc,
            diff_weights_desc,
            diff_dst_desc,
            forward_hints,
            op_attr);
#endif

  auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
  auto expected_src = src.reorder_if_differ_in(pd.src_desc());
  ideep::tensor expected_diff_weights;
  if (diff_weights.is_empty() ||
      diff_weights.get_desc() != pd.diff_weights_desc()) {
    // If diff_weights buffer are not given by user or user given diff_weights
    // buffer are not under expected format We need init a new one
    expected_diff_weights.init(pd.diff_weights_desc());
  } else {
    // The format of given diff_weights buffer is expected
    expected_diff_weights = diff_weights;
  }

  ideep::tensor scratchpad(pd.scratchpad_desc());

  std::unordered_map<int, dnnl::memory> args{
      {DNNL_ARG_DIFF_DST, expected_diff_dst},
      {DNNL_ARG_SRC, expected_src},
      {DNNL_ARG_DIFF_WEIGHTS, expected_diff_weights},
      {DNNL_ARG_SCRATCHPAD, scratchpad}};

  if (with_diff_bias) {
    diff_bias.reinit_if_possible(pd.diff_bias_desc());
    args.insert({DNNL_ARG_DIFF_BIAS, diff_bias});
  }

  dnnl::inner_product_backward_weights(pd).execute(
      ideep::stream::default_stream(), args);
  // reorder back to diff_weights's buffer if needed
  if (diff_weights.is_empty() ||
      diff_weights.get_desc() == expected_diff_weights.get_desc() ||
      !diff_weights.get_desc().has_same_shape_as(
          expected_diff_weights.get_desc())) {
    diff_weights = expected_diff_weights;
  } else {
    diff_weights.feed_from(expected_diff_weights);
  }
}

} // namespace

template <bool reorder_src, bool reorder_weight>
void inner_product_forward(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const ideep::attr_t& attr,
    dnnl::prop_kind prop_kind) {
  compute_impl</*with_bias=*/true, reorder_src, reorder_weight>(
      src, weights, bias, dst, attr, prop_kind);
}

template <bool reorder_src, bool reorder_weight>
void inner_product_forward(
    const ideep::tensor& src,
    const ideep::tensor& weights,
    ideep::tensor& dst,
    const ideep::attr_t& attr,
    dnnl::prop_kind prop_kind) {
  static ideep::tensor dummy_bias;
  compute_impl</*with_bias=*/false, reorder_src, reorder_weight>(
      src, weights, dummy_bias, dst, attr, prop_kind);
}

template <bool reorder_src, bool reorder_weight>
void inner_product_forward_binary(
    const ideep::tensor& src,
    const ideep::tensor& other,
    const ideep::tensor& weights,
    const ideep::tensor& bias,
    ideep::tensor& dst,
    const ideep::attr_t& attr,
    dnnl::prop_kind prop_kind) {
  inner_product_forward_params param;
  do_prepare_<true, reorder_src, reorder_weight>(
      param, src, weights, bias, dst, attr, prop_kind);
  do_compute_binary<true, reorder_src, reorder_weight>(
      param, src, other, weights, bias, dst);
}

template <bool reorder_src, bool reorder_weight>
void inner_product_forward_binary(
    const ideep::tensor& src,
    const ideep::tensor& other,
    const ideep::tensor& weights,
    ideep::tensor& dst,
    const ideep::attr_t& attr,
    dnnl::prop_kind prop_kind) {
  static ideep::tensor dummy_bias;
  inner_product_forward_params param;
  do_prepare_<false, reorder_src, reorder_weight>(
      param, src, weights, dummy_bias, dst, attr, prop_kind);
  do_compute_binary<false, reorder_src, reorder_weight>(
      param, src, other, weights, dummy_bias, dst);
}

ideep::tensor::desc inner_product_forward_expected_weights_desc(
    const dnnl::memory::dims& weights_dims,
    const dnnl::memory::dims& src_dims,
    dnnl::memory::data_type dtype,
    dnnl::memory::data_type x_dtype,
    dnnl::prop_kind prop_kind) {
  const auto& engine = ideep::engine::cpu_engine();
  auto x_dims = weights_dims;
  // 128 is default batch size for inner product
  x_dims[0] = src_dims.empty() ? 128 : src_dims[0];
  auto y_dims = {x_dims[0], weights_dims[0]};
  auto y_dtype = (dtype != dnnl::memory::data_type::s8)
      ? dtype
      : dnnl::memory::data_type::s32;

#ifndef NDEBUG
  dnnl::error::wrap_c_api(
      x_dims.size() == weights_dims.size() ? dnnl_success
                                           : dnnl_invalid_arguments,
      "Invalid dims for data and weights");
#endif
  ideep::tensor::desc src_desc(x_dims, x_dtype, dnnl::memory::format_tag::any);
  ideep::tensor::desc dst_desc(y_dims, y_dtype, dnnl::memory::format_tag::any);
  ideep::tensor::desc weights_desc(
      weights_dims, dtype, dnnl::memory::format_tag::any);
  auto pd = dnnl::inner_product_forward::primitive_desc(
      engine, prop_kind, src_desc, weights_desc, dst_desc);
  return pd.weights_desc();
}

void inner_product_backward_data(
    const ideep::tensor& diff_dst,
    const ideep::tensor& weights,
    const dnnl::memory::dims& diff_src_dims,
    ideep::tensor& diff_src) {
  if (is_avx2_vnni_2() &&
      one_of(
          diff_dst.get_data_type(),
          dnnl::memory::data_type::bf16,
          dnnl::memory::data_type::f16)) {
    dnnl::error::wrap_c_api(
        dnnl_invalid_arguments,
        "DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2");
  }
  const auto& engine = ideep::engine::cpu_engine();
  auto weights_ = weights;

  // workaround: diff_src and weights from caffe2 may have different dims.
  // It would be better for caffe2 to do this reshape anyway.
  if (diff_src_dims.size() != weights.ndims()) {
    auto new_dims = diff_src_dims;
    new_dims[0] = weights.get_dim(0);
    weights_.reshape(new_dims);
  }

  auto diff_dst_desc = diff_dst.get_desc().to_format_any();
  auto weights_desc = ideep::tensor::desc(
      weights_.get_dims(),
      diff_dst.get_data_type(),
      dnnl::memory::format_tag::any);
  auto diff_src_desc = ideep::tensor::desc(
      diff_src_dims, diff_dst.get_data_type(), dnnl::memory::format_tag::any);

  ideep::attr_t op_attr;
  op_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

  auto forward_hints = get_primitive_desc(
      diff_src_desc,
      weights_desc,
      diff_dst_desc,
      ideep::tensor::desc(),
      false,
      op_attr);

#ifdef __aarch64__
  auto pd = dnnl::inner_product_backward_data::primitive_desc(
      engine,
      diff_src_desc,
      weights_desc,
      diff_dst_desc,
      forward_hints.first,
      op_attr);
#else
  auto pd = dnnl::inner_product_backward_data::primitive_desc(
      engine,
      diff_src_desc,
      weights_desc,
      diff_dst_desc,
      forward_hints,
      op_attr);
#endif

  auto expected_diff_dst = diff_dst.reorder_if_differ_in(pd.diff_dst_desc());
  auto expected_weights = weights_.reorder_if_differ_in(pd.weights_desc());
  ideep::tensor expected_diff_src;
  if (diff_src.is_empty() || diff_src.get_desc() != pd.diff_src_desc()) {
    // If diff_src buffer are not given by user or user given diff_src buffer
    // are not under expected format We need init a new one
    expected_diff_src.init(pd.diff_src_desc());
  } else {
    // The format of given diff_src buffer is expected
    expected_diff_src = diff_src;
  }

  ideep::tensor scratchpad(pd.scratchpad_desc());

  dnnl::inner_product_backward_data(pd).execute(
      ideep::stream::default_stream(),
      {{DNNL_ARG_DIFF_DST, expected_diff_dst},
       {DNNL_ARG_WEIGHTS, expected_weights},
       {DNNL_ARG_DIFF_SRC, expected_diff_src},
       {DNNL_ARG_SCRATCHPAD, scratchpad}});
  // reorder back to diff_src's buffer if needed
  if (diff_src.is_empty() ||
      diff_src.get_desc() == expected_diff_src.get_desc() ||
      !diff_src.get_desc().has_same_shape_as(expected_diff_src.get_desc())) {
    diff_src = expected_diff_src;
  } else {
    diff_src.feed_from(expected_diff_src);
  }
}

void inner_product_backward_weights(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    ideep::tensor& diff_weights,
    ideep::tensor& diff_bias) {
  backward_weights_compute_impl</*with_diff_bias=*/true>(
      src, diff_dst, diff_weights, diff_bias);
}

void inner_product_backward_weights(
    const ideep::tensor& src,
    const ideep::tensor& diff_dst,
    ideep::tensor& diff_weights) {
  static ideep::tensor dummy_diff_bias;
  backward_weights_compute_impl</*with_diff_bias=*/false>(
      src, diff_dst, diff_weights, dummy_diff_bias);
}

// The template arguments used by ATen callers.
template void inner_product_forward<true, true>(
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&,
    const ideep::attr_t&,
    dnnl::prop_kind);
template void inner_product_forward<true, true>(
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&,
    const ideep::attr_t&,
    dnnl::prop_kind);
template void inner_product_forward<false, false>(
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&,
    const ideep::attr_t&,
    dnnl::prop_kind);
template void inner_product_forward<false, false>(
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&,
    const ideep::attr_t&,
    dnnl::prop_kind);
template void inner_product_forward_binary<false, false>(
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&,
    const ideep::attr_t&,
    dnnl::prop_kind);
template void inner_product_forward_binary<false, false>(
    const ideep::tensor&,
    const ideep::tensor&,
    const ideep::tensor&,
    ideep::tensor&,
    const ideep::attr_t&,
    dnnl::prop_kind);

} // namespace at::native::mkldnn

#endif // AT_MKLDNN_ENABLED()
