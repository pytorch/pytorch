#include <type_traits>
#include <limits>
#include <c10/core/DeviceType.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIndexing.h>
#include <ATen/cpu/vec/vec256/vec256.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <utility>
#include <c10/util/typeid.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/Logging.h>
#include <c10/util/Exception.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <ATen/TensorSubclassLikeUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cat.h>
#endif

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
namespace at {

namespace native {

DEFINE_DISPATCH(_fused_sdp_choice_stub);
REGISTER_NO_CPU_DISPATCH(_fused_sdp_choice_stub);

namespace {

Tensor gemm_nt(const Tensor& self, const Tensor& other) {
  if (self.is_nested()) {
    return NestedTensor_matmul(self, other.t());
  } else {
    return at::native::matmul(self, other.t());
  }
}

template <typename scalar_t>
void transform_bias_rescale_qkv_inner_loop(
    int64_t B,
    int64_t T,
    int64_t _3D,
    int64_t D,
    int64_t num_head,
    int64_t dim_per_head,
    scalar_t* qkv_data,
    scalar_t* qkv_bias_data,
    scalar_t* q_k_v_data,
    scalar_t inv_sqrt_dim_per_head,
    int64_t begin,
    int64_t end) {
  for (auto i : c10::irange(begin, end)) {
    auto t = i % T;
    i /= T;
    auto nh = i % num_head;
    i /= num_head;
    auto b = i;
    using Vec = vec::Vectorized<scalar_t>;
    auto V = vec::Vectorized<scalar_t>::size();
    auto dh = 0;
    auto d = nh * dim_per_head;
    for (; dh + V <= dim_per_head; dh += V, d += V) {
      // load
      auto q_bias_data = Vec::loadu(&qkv_bias_data[d + 0 * D]);
      auto k_bias_data = Vec::loadu(&qkv_bias_data[d + 1 * D]);
      auto v_bias_data = Vec::loadu(&qkv_bias_data[d + 2 * D]);

      auto q_data = Vec::loadu(&qkv_data[b * _3D * T + t * _3D + d + 0 * D]) +
          q_bias_data;
      auto k_data = Vec::loadu(&qkv_data[b * _3D * T + t * _3D + d + 1 * D]) +
          k_bias_data;
      auto v_data = Vec::loadu(&qkv_data[b * _3D * T + t * _3D + d + 2 * D]) +
          v_bias_data;

      q_data = q_data * Vec(inv_sqrt_dim_per_head);

      q_data.store(&q_k_v_data
                       [0 * B * num_head * T * dim_per_head +
                        b * num_head * T * dim_per_head +
                        nh * T * dim_per_head + t * dim_per_head + dh]);
      k_data.store(&q_k_v_data
                       [1 * B * num_head * T * dim_per_head +
                        b * num_head * T * dim_per_head +
                        nh * T * dim_per_head + t * dim_per_head + dh]);
      v_data.store(&q_k_v_data
                       [2 * B * num_head * T * dim_per_head +
                        b * num_head * T * dim_per_head +
                        nh * T * dim_per_head + t * dim_per_head + dh]);
    }
    for (; dh < dim_per_head; dh++) {
      auto d = nh * dim_per_head + dh;
      auto q_bias = qkv_bias_data[d + 0 * D];
      auto k_bias = qkv_bias_data[d + 1 * D];
      auto v_bias = qkv_bias_data[d + 2 * D];
      auto q_data = qkv_data[b * _3D * T + t * _3D + d + 0 * D] + q_bias;
      auto k_data = qkv_data[b * _3D * T + t * _3D + d + 1 * D] + k_bias;
      auto v_data = qkv_data[b * _3D * T + t * _3D + d + 2 * D] + v_bias;
      q_data = q_data * inv_sqrt_dim_per_head;
      q_k_v_data
          [0 * B * num_head * T * dim_per_head +
           b * num_head * T * dim_per_head + nh * T * dim_per_head +
           t * dim_per_head + dh] = q_data;
      q_k_v_data
          [1 * B * num_head * T * dim_per_head +
           b * num_head * T * dim_per_head + nh * T * dim_per_head +
           t * dim_per_head + dh] = k_data;
      q_k_v_data
          [2 * B * num_head * T * dim_per_head +
           b * num_head * T * dim_per_head + nh * T * dim_per_head +
           t * dim_per_head + dh] = v_data;
    }
  }
}

Tensor transform_0213(const Tensor& a) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.size(1));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.size(3));
  return a.permute({0, 2, 1, 3})
      .contiguous()
      .view({a.size(0), a.size(2), a.size(1) * a.size(3)});
}

} // namespace


Tensor bmm_nt(const Tensor& a, const Tensor& b) {
  auto a_ = a.view({a.size(0) * a.size(1), a.size(2), a.size(3)});
  auto b_ = b.view({b.size(0) * b.size(1), b.size(2), b.size(3)});
  auto bt_ = b_.transpose(2, 1);
  auto c_ = at::bmm(a_, bt_);
  return c_.view({a.size(0), a.size(1), a.size(2), b.size(2)});
}

Tensor masked_softmax(
    Tensor& attn_scores,
    c10::optional<Tensor> attn_mask,
    const Tensor& query,
    c10::optional<int64_t> mask_type) {
  if (query.is_nested() && !attn_mask) {
    return at::_nested_tensor_softmax_with_shape(attn_scores, query);
  }
  if (attn_mask && attn_mask->dtype() != at::kBool) {
    attn_mask = attn_mask->to(at::kBool);
  }
  if (attn_mask) {
    return _masked_softmax(attn_scores, *attn_mask, attn_scores.dim() - 1, mask_type);
  } else {
    return _softmax_out(attn_scores, attn_scores, attn_scores.dim() - 1, false);
  }
}

Tensor bmm_nn(Tensor& out, const Tensor& a, const Tensor& b) {
  const std::array<int64_t, 3> newAShape = {
      a.sizes()[0] * a.sizes()[1], a.sizes()[2], a.sizes()[3]};
  auto a_ = a.view(newAShape);
  const std::array<int64_t, 3> newBShape = {
      b.sizes()[0] * b.sizes()[1], b.sizes()[2], b.sizes()[3]};
  auto b_ = b.view(newBShape);
  auto out_ = out.reshape({newAShape[0], newAShape[1], newBShape[2]});
  auto c_ = at::bmm_out(out_, a_, b_);
  return c_.view({a.size(0), a.size(1), a.size(2), b.size(3)});
}


Tensor transform0213_gemm_nt_bias(
    const Tensor& a,
    const Tensor& b,
    const Tensor& c,
    const Tensor& query) {
  if (query.is_nested()) {
    at::Tensor nested_a = _nested_from_padded(
        a, get_nested_tensor_impl(query)->get_nested_sizes(), true);
    return NestedTensor_times_Tensor_plus_Tensor_addmm(
        c, nested_a, b.t(), 1, 1);
  } else {
    const Tensor a_0213 = transform_0213(a);
    auto a_ = a_0213.view({a_0213.size(0) * a_0213.size(1), a_0213.size(2)});
    auto r_ = at::native::linear(a_, b, c);
    return r_.view({a_0213.size(0), a_0213.size(1), r_.size(1)});
  }
}

void debug_assert_shape(int line, const Tensor& t, c10::IntArrayRef shape) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      (size_t)t.dim() == shape.size(),
      "(called from line ",
      line,
      ") ",
      "expected ",
      shape.size(),
      "-D tensor but got ",
      t.dim());
  if (t.is_nested()) {
    return;
  }
  for (auto idx : c10::irange(shape.size())) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        shape[idx] == 0 || t.sizes()[idx] == shape[idx],
        "(called from line ",
        line,
        ") ",
        "expected dim ",
        idx,
        " to be ",
        shape[idx],
        " but got ",
        t.sizes()[idx]);
  }
}

Tensor qkv_projection(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const Tensor& qkv_weight) {
  // shape: [B, T, 3 x D]
  Tensor qkv;

  if (key.is_same(value)) {
    if (query.is_same(key)) {
      // self-attention
      qkv = gemm_nt(query, qkv_weight);
    } else {
      // encoder-decoder attention
      // TODO: is there a more efficient way to set this up?
      // TODO: can we stay nested insted of using cat? Probably just make a
      // NestedTensor out of the matmul results or something?
      auto q_kv_weight_s =
          at::native::split_with_sizes(qkv_weight, {embed_dim, embed_dim * 2}, 0);
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          q_kv_weight_s.size() == 2,
          "expected split to produce 2 tensors but it produced ",
          q_kv_weight_s.size());
      auto q = gemm_nt(query, q_kv_weight_s[0]);
      auto kv = gemm_nt(key, q_kv_weight_s[1]);
      qkv = at::cat({std::move(q), std::move(kv)}, 2);
    }
  } else {
    auto q_k_v_weight_s = at::native::chunk(qkv_weight, 3, 0);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        q_k_v_weight_s.size() == 3,
        "expected chunk to produce 3 tensors but it produced ",
        q_k_v_weight_s.size());
    // TODO: can we stay nested instead of using cat?
    auto q = gemm_nt(query, q_k_v_weight_s[0]);
    auto k = gemm_nt(key, q_k_v_weight_s[1]);
    auto v = gemm_nt(value, q_k_v_weight_s[2]);
    qkv = at::cat({std::move(q), std::move(k), std::move(v)}, 2);
  }

  return qkv;
}

// compute q = (q + q_bias) / sqrt(dim_per_head), k = k + k_bias, v = v + v_bias
std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_cpu(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head) {
  auto qkv_ = qkv.is_nested()
    ? c10::MaybeOwned<Tensor>::owned(qkv.to_padded_tensor(0))
    : c10::MaybeOwned<Tensor>::borrowed(qkv);
  auto B = qkv_->size(0);
  auto T = qkv_->size(1);
  auto _3D = qkv_->size(2);
  auto D = _3D / 3;
  TORCH_CHECK(D % num_head == 0);
  TORCH_CHECK(_3D % 3 == 0);
  const auto dim_per_head = D / num_head;
  auto q_k_v = at::empty({3, B, num_head, T, dim_per_head}, qkv_->options());

  const auto qkv_contig = qkv_->expect_contiguous();
  const auto qkv_bias_contig = qkv_bias.expect_contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      qkv_->scalar_type(),
      "transform_bias_rescale_qkv",
      [&] {
        scalar_t* qkv_data = qkv_contig->data_ptr<scalar_t>();
        scalar_t* qkv_bias_data = qkv_bias_contig->data_ptr<scalar_t>();
        scalar_t* q_k_v_data = q_k_v.data_ptr<scalar_t>();
        const scalar_t inv_sqrt_dim_per_head =
            1.0 / std::sqrt(static_cast<scalar_t>(dim_per_head));

        int64_t grain_size =
            std::max(internal::GRAIN_SIZE / (3 * dim_per_head), (int64_t)1);
        parallel_for(
            0, B * num_head * T, grain_size, [&](int64_t begin, int64_t end) {
              transform_bias_rescale_qkv_inner_loop(
                  B,
                  T,
                  _3D,
                  D,
                  num_head,
                  dim_per_head,
                  qkv_data,
                  qkv_bias_data,
                  q_k_v_data,
                  inv_sqrt_dim_per_head,
                  begin,
                  end);
            });
      });
  auto q_k_v_s =
      at::native::split(q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(q_k_v_s.size() == 3);
  return std::make_tuple(q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]);
}

std::tuple<Tensor, Tensor> native_multi_head_attention_cpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const int64_t num_head,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const c10::optional<Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    const c10::optional<int64_t> mask_type) {
  // query shape: [B, T, D]
  // qkv_weight shape: [3 * D, D]

  TORCH_CHECK(
      !mask || !query.is_nested(),
      "NestedTensor with mask is not supported yet");
  const auto D = embed_dim;
  TORCH_CHECK(
      query.dim() == 3,
      "expected 3-D `query`, got ",
      query.dim(),
      "-D tensor");
  TORCH_CHECK(
      query.is_nested() || query.sizes()[2] == embed_dim,
      "passed-in embed_dim ",
      embed_dim,
      " didn't match last dim of query ",
      query.sizes()[2]);
  TORCH_CHECK(
      key.dim() == 3,
      "expected 3-D `key`, got ",
      key.dim(),
      "-D tensor");
  TORCH_CHECK(
      value.dim() == 3,
      "expected 3-D `value`, got ",
      value.dim(),
      "-D tensor");
  TORCH_CHECK(
      query.is_nested() || key.is_nested() || value.is_nested() ||
          (query.sizes() == key.sizes() && key.sizes() == value.sizes()),
      "expected `query`/`key`/`value` shapes to match");
  TORCH_CHECK(
      qkv_weight.dim() == 2,
      "expected 2-D `qkv_weight`, got ",
      qkv_weight.dim(),
      "-D tensor");
  TORCH_CHECK(
      D * 3 == qkv_weight.sizes()[0],
      "expected `qkv_weight` first dim to be 3x embed_dim");
  TORCH_CHECK(
      D == qkv_weight.sizes()[1],
      "expected `qkv_weight` second dim to be embed_Dim");
  TORCH_CHECK(
      qkv_bias.dim() == 1,
      "expected 1-D `qkv_bias`, got ",
      qkv_bias.dim(),
      "-D tensor");
  TORCH_CHECK(
      qkv_bias.sizes()[0] == 3 * D,
      "expected `qkv_bias` first dim and first dim of query to be equal");
  TORCH_CHECK(D % num_head == 0, "`embed_dim` must divide evenly by `num_heads`");

#ifndef NDEBUG
  const auto B = query.is_nested()
      ? get_nested_tensor_impl(query)->get_nested_sizes().size(0)
      : query.sizes()[0];
  auto T = query.is_nested() ? 0 : query.sizes()[1];
  const auto dim_per_head = D / num_head;
#endif

  // shape: [B, T, 3 x D]
  auto qkv = qkv_projection(query, key, value, embed_dim, qkv_weight);

  if (!qkv.is_nested() && qkv.numel() == 0) {
    if (query.is_nested()) {
      return std::make_tuple(Tensor(), Tensor());
    }
    return std::make_tuple(at::empty_like(query), Tensor());
  }

#ifndef NDEBUG
  if (!query.is_nested() || !qkv.is_nested()) {
    if (query.is_nested()) {
      T = qkv.size(1);
    }
    debug_assert_shape(__LINE__, qkv, {B, T, 3 * D});
  }
#endif

#ifdef DEBUG_PRINT_EACH_STEP
  if (!qkv.is_nested()) {
    std::cerr << "qkv: " << qkv << std::endl;
  }
#endif
  // shape: 3 x [B, num_head, T, dim_per_head]
  auto q_k_v = _transform_bias_rescale_qkv(qkv, qkv_bias, num_head);
  qkv = Tensor(); // Not used any more, allow free
  auto& q = std::get<0>(q_k_v);
  const auto& k = std::get<1>(q_k_v);
  const auto& v = std::get<2>(q_k_v);
#ifndef NDEBUG
  debug_assert_shape(__LINE__, q, {B, num_head, T, dim_per_head});
  debug_assert_shape(__LINE__, k, {B, num_head, T, dim_per_head});
  debug_assert_shape(__LINE__, v, {B, num_head, T, dim_per_head});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "q: " << q << std::endl;
  std::cerr << "k: " << k << std::endl;
  std::cerr << "v: " << v << std::endl;
#endif

  // shape: [B, num_head, T, T]
  auto qkt = bmm_nt(q, k);
  // q & k are dead but cannot be freed because they were packed with v
#ifndef NDEBUG
  debug_assert_shape(__LINE__, qkt, {B, num_head, T, T});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "qkt: " << qkt << std::endl;
#endif

  // shape: [B, num_head, T, T]
  // TODO: long-term, have a kernel that works with
  // NestedTensor directly if there is no mask passed
  qkt = masked_softmax(qkt, mask, query, mask_type);
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "qkt after softmax: " << qkt << std::endl;
#endif

  // shape: [B, num_head, T, dim_per_head]
  // reuse storage for q; we're done with it
  auto attn_ctx = bmm_nn(q, qkt, v);
  // qkv is not dead; we just reused storage for q!
  if (!need_weights) {
    qkt = Tensor();
  }
#ifndef NDEBUG
  debug_assert_shape(__LINE__, attn_ctx, {B, num_head, T, dim_per_head});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "attn_ctx: " << attn_ctx << std::endl;
#endif

  // shape: [B, T, D]
  // Fuse transform_0213 inside
  auto proj = transform0213_gemm_nt_bias(
      attn_ctx, proj_weight, proj_bias, query);
#ifndef NDEBUG
  debug_assert_shape(__LINE__, proj, {B, T, D});
#endif
  if (need_weights && average_attn_weights) {
    // weights are not needed for full transformer, so don't worry too
    // much about performance -- we implement this just to make use
    // cases that don't disable need_weights still get some speedup.
    qkt = qkt.sum(1);
    qkt /= num_head;
  }
  return std::make_tuple(std::move(proj), std::move(qkt));
}

int64_t _fused_sdp_choice_cpp(const Tensor& query_, const Tensor& key, const Tensor& value,
        const c10::optional<Tensor>& attn_mask_, double dropout_p, bool is_causal, c10::optional<double> scale){
  return static_cast<int64_t>(sdp::SDPBackend::math);
}

int64_t _fused_sdp_choice_meta(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  auto query_key_set = query_.key_set();
  bool has_cuda = query_key_set.has(c10::DispatchKey::CUDA);
  if (has_cuda) {
    auto choice_int = _fused_sdp_choice_stub(
        at::kCUDA,
        query_,
        key,
        value,
        attn_mask_,
        dropout_p,
        is_causal,
        scale);
    return choice_int;
  }
  return static_cast<int64_t>(sdp::SDPBackend::math);
}
namespace {

inline void validate_sdpa_input(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  TORCH_CHECK(
      query_.dtype() == key.dtype() && query_.dtype() == value.dtype(),
      "Expected query, key, and value to have the same dtype, but got query.dtype: ",
      query_.dtype(), " key.dtype: ", key.dtype(), " and value.dtype: ", value.dtype(), " instead.");
  TORCH_CHECK(
      query_.device() == key.device() && query_.device() == value.device(),
      "Expected query, key, and value to have the same device type, but got query.device: ",
      query_.device(), " key.device: ", key.device(), " and value.device: ", value.device(), " instead.");
  TORCH_CHECK(
      query_.dim() >= 2 && key.dim() >= 2 && value.dim() >= 2,
      "Expected query, key, and value to all be  at least 2 dimensional, but got query.dim: ",
      query_.dim(), " key.dim: ", key.dim(), " and value.dim: ", value.dim(), " instead.");
  if (attn_mask_.has_value()){
    auto mask_dtype = attn_mask_->dtype();
    TORCH_CHECK(mask_dtype == at::kBool || mask_dtype == query_.dtype(),
      "Expected attn_mask dtype to be bool or to match query dtype, but got attn_mask.dtype: ",
      mask_dtype, " and  query.dtype: ", query_.dtype(), " instead.");
    TORCH_CHECK(
      !query_.is_nested() && !key.is_nested(),
      "Scaled_dot_product_attention: Nested tensors for query / key are not supported "
      "when an explicit attn_mask is set");
  }
  return;
}
// This function is used to produce an attn_mask
// in a standard format that can be consumed by both
// the math and memory efficient attn_mask implementation
//  Args:
//    attn_mask: attn_mask of shape (B, L, S) or (L, S) or (B, N_heads, L, S)
c10::optional<Tensor> convert_boolean_attn_mask(const c10::optional<Tensor>& attn_mask, caffe2::TypeMeta dtype) {
  // Pass through
  if(!attn_mask.has_value()){
    return c10::nullopt;
  }
  // Convert boolean mask to additive mask; need to invert mask to indicate what
  // to mask *out*.
  if (attn_mask->dtype() == at::kBool) {
    auto new_attn_mask = at::zeros_like(attn_mask.value(), dtype);
    // TODO Use the max type of the input and output
    new_attn_mask.masked_fill_(
        attn_mask->logical_not(), -std::numeric_limits<double>::infinity());
    return new_attn_mask;
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}
// Memory Efficient Attention requires a padded attn mask bias
// This function pads the attn_mask bias to be a multiple of 16
// Then slices the padded bias to the original size
// We apply this function to the top level SDPA so that
// if padding is done it will be tracked for backward automatically
at::Tensor pad_bias(const at::Tensor& attn_bias) {
  int align_to = 16;
  auto last_dim_size = attn_bias.sym_size(-1);
  if (last_dim_size % align_to == 0) {
    return attn_bias;
  }
  auto pad_count = align_to - (last_dim_size % align_to);
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  return padded_bias.slice_symint(-1, 0, last_dim_size);
}

} // namespace

// Computes scaled dot product attention on query, key and value tensors, using
// an optional attention mask if passed, and applying dropout if a probability
// greater than 0.0 is specified.
//
// Args:
//     query (Tensor): Query tensor; shape (N, ..., L, E)
//     key (Tensor): Key tensor; shape (N, ..., S, E)
//     value (Tensor): Value tensor; shape (N, ..., S, E)
//     attn_mask (optional Tensor): Attention mask; shape (N, ..., L, S) or (L, S). Currently, only a boolean mask
//         is supported, where a value of True indicates that the element *should* take part in attention.
//     dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
//     need_attn_weights (bool): If true, the second return value will contain the attention weights used;
//         otherwise, the second return value is unspecified
//     is_causal (bool): If true, assumes causal attention masking; for this case, attn_mask should not be set.
//         TODO: Consider removing this flag before promoting this function to the public API. It's possible
//         to get specialized support for causal masks (and other types of masking e.g. local attention / block
//         sparse masks) via tensor subclassing, allowing for a leaner API.
//
// Returns a tuple containing:
//     output (Tensor): Attention output; shape (N, ..., L, E)
//     attn_weights (Tensor): Attention weighting; shape (N, ..., L, S)
//
// Shape legend:
//     N: Batch size
//     ...: Any number of other batch dimensions (optional)
//     S: Source sequence length
//     L: Target sequence length
//     E: Embedding dimension
Tensor scaled_dot_product_attention(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const c10::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    c10::optional<double> scale) {
  validate_sdpa_input(query_, key, value, attn_mask_, dropout_p, is_causal, scale);
  int64_t choice_int = static_cast<int64_t>(sdp::SDPBackend::math);
  if (query_.device().type() == DeviceType::CUDA){
    choice_int = _fused_sdp_choice_stub(query_.device().type(),
      query_, key, value, attn_mask_, dropout_p, is_causal, scale);
  }
  sdp::SDPBackend backend = static_cast<sdp::SDPBackend>(choice_int);
  c10::optional<Tensor> attn_mask = convert_boolean_attn_mask(attn_mask_, query_.dtype());
  switch (backend) {
    case sdp::SDPBackend::flash_attention: {
      auto out_lse_softmax = at::_scaled_dot_product_flash_attention(
          query_, key, value, dropout_p, is_causal, false /*return_debug_mask*/, scale);
      return std::get<0>(out_lse_softmax);
    }
    case sdp::SDPBackend::efficient_attention: {
      bool compute_logsumexp =
          (query_.requires_grad() || key.requires_grad() ||
           value.requires_grad());
      if (attn_mask.has_value()) {
        // Expand to 4d case
        attn_mask = attn_mask.value().expand_symint(
            {query_.sym_size(0),
             query_.sym_size(1),
             query_.sym_size(2),
             key.sym_size(2)});
        attn_mask = pad_bias(attn_mask.value());
      }
      auto out_and_lse = at::_scaled_dot_product_efficient_attention(
          query_, key, value, attn_mask, compute_logsumexp, dropout_p, is_causal, scale);
      return std::get<0>(out_and_lse);
    }
    case sdp::SDPBackend::math:
      return std::get<0>(at::_scaled_dot_product_attention_math(
          query_,
          key,
          value,
          attn_mask,
          dropout_p,
          is_causal,
          c10::nullopt, /*dropout_mask*/
          scale));
    default:
      TORCH_CHECK(
          false,
          "No viable backend for scaled_dot_product_attention was found.");
      return Tensor();
  }
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math(
        const Tensor& query_, const Tensor& key, const Tensor& value,
        const c10::optional<Tensor>& attn_mask_, double dropout_p, bool is_causal,
        const c10::optional<Tensor>& dropout_mask, c10::optional<double> scale) {
  C10_LOG_API_USAGE_ONCE("torch.sdpa.math_fallback");
  if (query_.is_nested() || key.is_nested() || value.is_nested()) {
    TORCH_CHECK(
        query_.is_contiguous() && key.is_contiguous() &&
            value.is_contiguous(),
        "scaled_dot_product_attention: If inputs are nested tensors they must be contiguous");
  }
    auto attn_mask = attn_mask_;
    // Naive, composite implementation defined here.

    // Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for math
    bool is_negative_scaling = scale.has_value() && scale.value() < 0.0;
    const auto scaling_factor = sdp::calculate_scale(query_, is_negative_scaling ? std::abs(scale.value()) : scale).sqrt();

    const auto query = query_ * (is_negative_scaling ? c10::SymFloat(0.0) - scaling_factor: scaling_factor);
    if (is_causal) {
        TORCH_CHECK(!attn_mask.has_value(),
                "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
        TORCH_CHECK(!query.is_nested() && !key.is_nested(),
                "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");

        // Replace attn_mask with causal mask; lower triangular elements take part in attention.
        const auto L = query.sym_size(-2), S = key.sym_size(-2);
        attn_mask = at::ones_symint({L, S}, query.options().dtype(at::kBool)).tril();
        attn_mask = convert_boolean_attn_mask(attn_mask, query.dtype());
    }
    auto attn = at::matmul(query, key.transpose(-2, -1) * scaling_factor);
    if (attn_mask.has_value()) {
      if (at::areAnyTensorSubclassLike({attn, *attn_mask})) {
        attn = attn.add(*attn_mask);
      } else {
        attn.add_(*attn_mask);
      }
    }
    attn = at::softmax(attn, -1);
    if (dropout_p > 0.0) {
      if (dropout_mask.has_value()) {
        // In order to validate the correctness of the fused kernels, we need to
        // use the same dropout mask in order to compare the results.
        TORCH_WARN_ONCE("Dropout mask should only be used for testing purposes.");
        attn = attn.masked_fill(dropout_mask->logical_not(), 0.0);
        auto dropout_scaling = 1.0 / (1 - dropout_p);
        return std::make_tuple(at::matmul(attn, value * dropout_scaling), attn);
      } else {
        attn = at::dropout(attn, dropout_p, true);
      }
    }

    return std::make_tuple(at::matmul(attn, value), attn);
}

Tensor triton_multi_head_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const int64_t embed_dim,
    const int64_t num_head,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const c10::optional<Tensor>& mask) {
  // query shape: [B, T, D]
  // qkv_weight shape: [3 * D, D]
  TORCH_CHECK(!mask, "Only causal mask is supported for Triton.");

  const auto D = embed_dim;
  TORCH_CHECK(
      query.dim() == 3,
      "expected 3-D `query`, got ",
      query.dim(),
      "-D tensor");
  TORCH_CHECK(
      query.sizes()[2] == embed_dim,
      "passed-in embed_dim ",
      embed_dim,
      " didn't match last dim of query ",
      query.sizes()[2]);
  TORCH_CHECK(
      key.dim() == 3,
      "expected 3-D `key`, got ",
      key.dim(),
      "-D tensor");
  TORCH_CHECK(
      value.dim() == 3,
      "expected 3-D `value`, got ",
      value.dim(),
      "-D tensor");
  TORCH_CHECK(
          query.sizes() == key.sizes() && key.sizes() == value.sizes(),
      "expected `query`/`key`/`value` shapes to match");
  TORCH_CHECK(
      qkv_weight.dim() == 2,
      "expected 2-D `qkv_weight`, got ",
      qkv_weight.dim(),
      "-D tensor");
  TORCH_CHECK(
      D * 3 == qkv_weight.sizes()[0],
      "expected `qkv_weight` first dim to be 3x embed_dim");
  TORCH_CHECK(
      D == qkv_weight.sizes()[1],
      "expected `qkv_weight` second dim to be embed_Dim");

#ifndef NDEBUG
  const auto B = query.is_nested()
      ? get_nested_tensor_impl(query)->get_nested_sizes().size(0)
      : query.sizes()[0];
  auto T = query.is_nested() ? 0 : query.sizes()[1];
  const auto dim_per_head = D / num_head;
#endif

  // shape: [B, T, 3 x D]
  auto qkv = qkv_projection(query, key, value, embed_dim, qkv_weight);

  // shape: 3 x [B, num_head, T, dim_per_head]
  auto q_k_v = _transform_bias_rescale_qkv(qkv, qkv_bias, num_head);
  qkv = Tensor(); // Not used any more, allow free
  auto& q = std::get<0>(q_k_v);
  const auto& k = std::get<1>(q_k_v);
  const auto& v = std::get<2>(q_k_v);
#ifndef NDEBUG
  debug_assert_shape(__LINE__, q, {B, num_head, T, dim_per_head});
  debug_assert_shape(__LINE__, k, {B, num_head, T, dim_per_head});
  debug_assert_shape(__LINE__, v, {B, num_head, T, dim_per_head});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "q: " << q << std::endl;
  std::cerr << "k: " << k << std::endl;
  std::cerr << "v: " << v << std::endl;
#endif

  auto attn_ctx = at::_triton_scaled_dot_attention(q, k, v);

#ifndef NDEBUG
  debug_assert_shape(__LINE__, attn_ctx, {B, num_head, T, dim_per_head});
#endif
#ifdef DEBUG_PRINT_EACH_STEP
  std::cerr << "attn_ctx: " << attn_ctx << std::endl;
#endif

  // shape: [B, T, D]
  // Fuse transform_0213 inside
  auto proj = transform0213_gemm_nt_bias(
      attn_ctx, proj_weight, proj_bias, query);
#ifndef NDEBUG
  debug_assert_shape(__LINE__, proj, {B, T, D});
#endif
  return proj;
}
} // namespace native
} // namespace at
