#include <ATen/core/TensorBody.h>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/TensorOperators.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorIndexing.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <c10/util/typeid.h>
#include <c10/core/DeviceType.h>
#include <c10/core/SymInt.h>
#include <c10/core/SymIntArrayRef.h>
#include <c10/util/Logging.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>

#include <limits>
#include <utility>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_sdp_choice_native.h>
#include <ATen/ops/_masked_softmax.h>
#include <ATen/ops/_native_multi_head_attention_native.h>
#include <ATen/ops/_nested_from_padded.h>
#include <ATen/ops/_nested_tensor_softmax_with_shape.h>
#include <ATen/ops/_scaled_dot_product_attention_math.h>
#include <ATen/ops/_scaled_dot_product_attention_math_for_mps.h>
#include <ATen/ops/_scaled_dot_product_attention_math_for_mps_native.h>
#include <ATen/ops/_scaled_dot_product_attention_math_native.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_backward_native.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_native.h>
#include <ATen/ops/_scaled_dot_product_cudnn_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_native.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_backward.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_for_cpu_backward_native.h>
#include <ATen/ops/_scaled_dot_product_fused_attention_overrideable.h>
#include <ATen/ops/_scaled_dot_product_fused_attention_overrideable_native.h>
#include <ATen/ops/_scaled_dot_product_fused_attention_overrideable_backward.h>
#include <ATen/ops/_scaled_dot_product_fused_attention_overrideable_backward_native.h>
#include <ATen/ops/_softmax.h>
#include <ATen/ops/_transform_bias_rescale_qkv.h>
#include <ATen/ops/_transform_bias_rescale_qkv_native.h>
#include <ATen/ops/_triton_multi_head_attention_native.h>
#include <ATen/ops/_triton_scaled_dot_attention.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/chunk_native.h>
#include <ATen/ops/dropout.h>
#include <ATen/ops/linear_native.h>
#include <ATen/ops/matmul.h>
#include <ATen/ops/matmul_native.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <ATen/ops/softmax.h>
#include <ATen/ops/split_native.h>
#include <ATen/ops/split_with_sizes_native.h>
#include <ATen/ops/where.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/_safe_softmax.h>
#include <ATen/ops/_safe_softmax_native.h>
#include <ATen/ops/all.h>
#endif

#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
namespace at::native {

DEFINE_DISPATCH(_fused_sdp_choice_stub);

DEFINE_DISPATCH(transform_bias_rescale_qkv_stub);
DEFINE_DISPATCH(flash_attention_kernel);
DEFINE_DISPATCH(flash_attention_backward_kernel);

namespace {

Tensor gemm_nt(const Tensor& self, const Tensor& other) {
  if (self.is_nested()) {
    return NestedTensor_matmul(self, other.t());
  } else {
    return at::native::matmul(self, other.t());
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
    std::optional<Tensor> attn_mask,
    const Tensor& query,
    std::optional<int64_t> mask_type) {
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
  transform_bias_rescale_qkv_stub(
      kCPU,
      qkv_->scalar_type(),
      q_k_v.data_ptr(),
      qkv_contig->const_data_ptr(),
      qkv_bias_contig->const_data_ptr(),
      B, T, D, num_head);
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
    const std::optional<Tensor>& mask,
    bool need_weights,
    bool average_attn_weights,
    const std::optional<int64_t> mask_type) {
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
// TODO: Remove me when https://github.com/pytorch/pytorch/issues/130073 is fixed
#if !defined(NDEBUG) && 0
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
        const std::optional<Tensor>& attn_mask_, double dropout_p, bool is_causal, std::optional<double> scale, bool enable_gqa){
  sdp::sdp_params kernel_params{query_, key, value, attn_mask_, dropout_p, is_causal, enable_gqa};
  auto backend = sdp::select_sdp_backend_cpp(kernel_params);
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the fused kernels.");
  }
  return static_cast<int64_t>(backend);
}

REGISTER_ARCH_DISPATCH(_fused_sdp_choice_stub, DEFAULT, &_fused_sdp_choice_cpp);
REGISTER_AVX2_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_cpp);
REGISTER_AVX512_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_cpp);
REGISTER_VSX_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_cpp);
REGISTER_ZVECTOR_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_cpp);
REGISTER_SVE_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_cpp);

int64_t _fused_sdp_choice_meta(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  auto query_key_set = query_.key_set();
#if defined(USE_ROCM)
  bool has_rocm = query_key_set.has(c10::DispatchKey::HIP);
  if (has_rocm) {
    auto choice_int = _fused_sdp_choice_stub(at::kHIP, query_, key, value, attn_mask_, dropout_p, is_causal, scale, enable_gqa);
    return choice_int;
  }
#else
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
        scale,
        enable_gqa);
    return choice_int;
  }
#endif
  return static_cast<int64_t>(sdp::SDPBackend::math);
}
namespace {

inline void validate_sdpa_input(
    const Tensor& query_,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
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
    TORCH_CHECK(mask_dtype == at::kBool || mask_dtype == at::kFloat || mask_dtype == query_.dtype(),
      "Expected attn_mask dtype to be bool or float or to match query dtype, but got attn_mask.dtype: ",
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
std::optional<Tensor> convert_boolean_attn_mask(const std::optional<Tensor>& attn_mask, caffe2::TypeMeta dtype) {
  // Pass through
  if(!attn_mask.has_value()){
    return std::nullopt;
  }
  // Convert boolean mask to additive mask; need to invert mask to indicate what
  // to mask *out*.
  if (attn_mask->dtype() == at::kBool) {
    return at::where(attn_mask->logical_not(), -std::numeric_limits<double>::infinity(), at::scalar_tensor(0.0, at::TensorOptions().dtype(dtype).device(attn_mask->device())));
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}

// alternate version to workaround -inf issue with cuDNN
// TODO(eqy): delete this when cuDNN -inf issue is resolved
std::optional<Tensor> convert_boolean_attn_mask_cudnn(const std::optional<Tensor>& attn_mask, caffe2::TypeMeta dtype) {
  // Pass through
  if(!attn_mask.has_value()){
    return std::nullopt;
  }
  // Convert boolean mask to additive mask; need to invert mask to indicate what
  // to mask *out*.
  if (attn_mask->dtype() == at::kBool) {
    // TODO Use the max type of the input and output
    return at::where(attn_mask->logical_not(), -65504.0, at::scalar_tensor(0.0, at::TensorOptions().dtype(dtype)));
  }
  // Otherwise, attn_mask represents an additive attention tensor
  return attn_mask;
}

// Memory Efficient Attention requires a padded attn mask bias
// This function pads the attn_mask bias to be a multiple of 16
// Then slices the padded bias to the original size
// We apply this function to the top level SDPA so that
// if padding is done it will be tracked for backward automatically

template<int alignment>
bool aligned_tensor(const at::Tensor& tensor){
  for(const auto i : c10::irange(tensor.dim() - 1)){
    if(tensor.sym_stride(i) % alignment != 0){
      return false;
    }
  }
  return tensor.sym_stride(-1) == 1;
}

template <int alignment>
at::Tensor pad_bias(const at::Tensor& attn_bias) {
  auto last_dim_size = attn_bias.sym_size(-1);
  auto pad_count = alignment - (last_dim_size % alignment);
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  return padded_bias.slice_symint(-1, 0, last_dim_size);
}

at::Tensor preprocess_mask(
    const at::Tensor& mask,
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value) {
  constexpr int mem_eff_alignment = 8;
  at::Tensor result_mask = mask;
  if (!aligned_tensor<mem_eff_alignment>(mask)) {
    result_mask = pad_bias<mem_eff_alignment>(mask);
  }
  return result_mask.expand_symint(
      {query.sym_size(0),
       query.sym_size(1),
       query.sym_size(2),
       key.sym_size(2)});
}
// FlashAttentionV2 requires that head dimension be a multiple of 8
// This was previously done within the kernel, however
// This causes the kernel to maybe alias query, key, value
// So instead we pad the head_dimensions to be a multiple of 8 in the composite
// region
template <int alignment_size, bool slice>
at::Tensor pad_last_dim(const at::Tensor& attn_bias) {
  auto last_dim_size = attn_bias.sym_size(-1);
  if (last_dim_size % alignment_size == 0) {
    return attn_bias;
  }
  auto pad_count = alignment_size - (last_dim_size % alignment_size);
  auto padded_bias = at::pad_symint(attn_bias, {c10::SymInt(0), pad_count});
  if (slice) {
    return padded_bias.slice_symint(-1, 0, last_dim_size);
  }
  return padded_bias;
}

at::Tensor post_process_flash_output(
    at::Tensor out,
    c10::SymInt const& og_size) {
  if (!out.is_nested() && out.sym_size(-1) != og_size) {
    out = out.slice_symint(-1, 0, og_size);
  }
  return out;
}

bool should_compute_logsumexp(const Tensor& query, const Tensor& key, const Tensor& value) {
  const bool any_inputs_require_grad = query.requires_grad() || key.requires_grad() || value.requires_grad();
  const bool gradmode_enabled = at::GradMode::is_enabled();
  return any_inputs_require_grad && gradmode_enabled;
}

std::tuple<at::Tensor, at::Tensor> pre_process_group_query_attention_input(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const bool enable_gqa) {

  if (!enable_gqa) {
    return std::make_tuple(key, value);
  }
  const auto q_num_heads = query.sym_size(-3);
  const auto k_num_heads = key.sym_size(-3);
  const auto v_num_heads = value.sym_size(-3);

  bool all_equal = q_num_heads == k_num_heads && k_num_heads == v_num_heads;
  bool key_divisible = q_num_heads % k_num_heads == 0;
  bool value_divisible = q_num_heads % v_num_heads == 0;
  TORCH_CHECK(all_equal || (key_divisible && value_divisible),
              "Number of heads in key and value must divide the number of heads in ");

  if (all_equal){
    return std::make_tuple(key, value);
  }
  auto repeat_key_shape = query.sym_size(-3) / key.sym_size(-3);
  auto repeat_value_shape = query.sym_size(-3) / value.sym_size(-3);

  at::Tensor key_repeated = key.repeat_interleave_symint(repeat_key_shape, -3);
  at::Tensor value_repeated = value.repeat_interleave_symint(repeat_value_shape, -3);
  return std::make_tuple(std::move(key_repeated), std::move(value_repeated));
}

} // namespace

Tensor _safe_softmax(
    const Tensor& self,
    int64_t dim,
    std::optional<ScalarType> dtype) {
  auto out = at::softmax(self, dim, dtype);
  const auto neg_inf = at::scalar_tensor(-std::numeric_limits<float>::infinity(), at::TensorOptions().dtype(out.dtype()).device(out.device()));
  const auto masked = self.eq(neg_inf);
  const auto masked_rows = all(masked, dim, true);
  const auto zero = at::scalar_tensor(0.0, at::TensorOptions().dtype(out.dtype()).device(out.device()));
  return at::where(masked_rows, zero, out);
}
// Computes scaled dot product attention on query, key and value tensors, using
// an optional attention mask if passed, and applying dropout if a probability
// greater than 0.0 is specified.
//
// Args:
//     query (Tensor): Query tensor; shape (N, ..., L, E)
//     key (Tensor): Key tensor; shape (N, ..., S, E)
//     value (Tensor): Value tensor; shape (N, ..., S, E)
//     attn_mask (optional Tensor): Attention mask; shape must be broadcastable to the shape of attention weights,
//         which is (N,..., L, S). Two types of masks are supported.
//         A boolean mask where a value of True indicates that the element *should* take part in attention.
//         A float mask of the same type as query, key, value that is added to the attention score.
//     dropout_p (float): Dropout probability; if greater than 0.0, dropout is applied
//     need_attn_weights (bool): If true, the second return value will contain the attention weights used;
//         otherwise, the second return value is unspecified
//     is_causal (bool): If true, assumes causal attention masking; for this case, attn_mask should not be set.
//         TODO: Consider removing this flag before promoting this function to the public API. It's possible
//         to get specialized support for causal masks (and other types of masking e.g. local attention / block
//         sparse masks) via tensor subclassing, allowing for a leaner API.
//
// Returns a tensor:
//     output (Tensor): Attention output; shape (N, ..., L, E)
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
    const std::optional<Tensor>& attn_mask_,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale,
    bool enable_gqa) {
  validate_sdpa_input(query_, key, value, attn_mask_, dropout_p, is_causal, scale);
  int64_t choice_int = static_cast<int64_t>(sdp::SDPBackend::math);
  if (_fused_sdp_choice_stub.is_device_supported(query_.device().type())) {
    choice_int = _fused_sdp_choice_stub(query_.device().type(),
          query_, key, value, attn_mask_, dropout_p, is_causal, scale, enable_gqa);
  }
  sdp::SDPBackend backend = static_cast<sdp::SDPBackend>(choice_int);
  switch (backend) {
    case sdp::SDPBackend::cudnn_attention: {
      std::optional<Tensor> attn_mask = convert_boolean_attn_mask_cudnn(attn_mask_, query_.dtype());
      bool compute_logsumexp = should_compute_logsumexp(query_, key, value);
      auto out_lse_softmax = at::_scaled_dot_product_cudnn_attention(
          query_, key, value, attn_mask, compute_logsumexp, dropout_p, is_causal, false /*return_debug_mask*/, scale);
      return std::get<0>(out_lse_softmax);
    }
    case sdp::SDPBackend::flash_attention: {
      std::optional<Tensor> attn_mask = convert_boolean_attn_mask(attn_mask_, query_.dtype());
      if(query_.device().type() == DeviceType::CUDA){
        c10::SymInt og_size = query_.sym_size(-1);
        Tensor query_padded = pad_last_dim<8, false>(query_);
        Tensor key_padded = pad_last_dim<8, false>(key);
        Tensor value_padded = pad_last_dim<8, false>(value);
        // We need to calculate the scale based off the OG head dim size
        auto og_scale = sdp::calculate_scale(query_, scale);
        auto out_lse_softmax = at::_scaled_dot_product_flash_attention(
            query_padded, key_padded, value_padded, dropout_p, is_causal, false /*return_debug_mask*/, og_scale.as_float_unchecked());
        return post_process_flash_output(std::get<0>(out_lse_softmax), og_size);
      }
      // For the CPU case we do not need to pad the last dim
      return std::get<0>(at::_scaled_dot_product_flash_attention_for_cpu(
          query_, key, value, dropout_p, is_causal, attn_mask, scale));
    }
    case sdp::SDPBackend::efficient_attention: {
      std::optional<Tensor> attn_mask = convert_boolean_attn_mask(attn_mask_, query_.dtype());
      bool compute_logsumexp = should_compute_logsumexp(query_, key, value);
      if (attn_mask.has_value()) {
        attn_mask.value() = preprocess_mask(attn_mask.value(), query_, key, value);;
      }
      auto out_and_lse = at::_scaled_dot_product_efficient_attention(
          query_, key, value, attn_mask, compute_logsumexp, dropout_p, is_causal, scale);
      return std::get<0>(out_and_lse);
    }
    case sdp::SDPBackend::overrideable: {
      std::optional<Tensor> attn_mask = convert_boolean_attn_mask(attn_mask_, query_.dtype());
      auto out_lse_softmax = at::_scaled_dot_product_fused_attention_overrideable(
          query_, key, value, attn_mask, dropout_p, is_causal, false /*return_debug_mask*/, scale);
      return std::get<0>(out_lse_softmax);
    }
    case sdp::SDPBackend::math: {
      std::optional<Tensor> attn_mask = convert_boolean_attn_mask(attn_mask_, query_.dtype());
      if ((!GradMode::is_enabled() || (!query_.requires_grad() && !key.requires_grad() && !value.requires_grad()))
          && query_.device().type() == DeviceType::MPS && dropout_p == 0.0
          && query_.is_contiguous() && key.is_contiguous() && value.is_contiguous()
          && !query_.is_nested() && !key.is_nested() && !value.is_nested()) {
        return std::get<0>(at::_scaled_dot_product_attention_math_for_mps(
            query_,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            std::nullopt, /*dropout_mask*/
            scale));
      }
      return std::get<0>(at::_scaled_dot_product_attention_math(
          query_,
          key,
          value,
          attn_mask,
          dropout_p,
          is_causal,
          std::nullopt, /*dropout_mask*/
          scale,
          enable_gqa));
    }
    default:
      TORCH_CHECK(
          false,
          "No viable backend for scaled_dot_product_attention was found.");
      return Tensor();
  }
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math(
        const Tensor& query_, const Tensor& key, const Tensor& value,
        const std::optional<Tensor>& attn_mask_, double dropout_p, bool is_causal,
        const std::optional<Tensor>& dropout_mask, std::optional<double> scale, bool enable_gqa) {
  C10_LOG_API_USAGE_ONCE("torch.sdpa.math_fallback");
  if (query_.is_nested() || key.is_nested() || value.is_nested()) {
    TORCH_CHECK(
        query_.is_contiguous() && key.is_contiguous() &&
            value.is_contiguous(),
        "scaled_dot_product_attention: If inputs are nested tensors they must be contiguous");
  }
  auto& ctx = at::globalContext();
  auto origin_dtype = query_.scalar_type();
  // Keep query, key, value in high precision for accuracy
  // NestedTensor reports issues for backward with autograd so disabled: must be
  // contiguous to get buffer.
  auto query_acc = !ctx.allowFP16BF16ReductionMathSDP() &&
          (query_.scalar_type() == at::kHalf ||
           query_.scalar_type() == at::kBFloat16) &&
          !query_.is_nested()
      ? query_.to(at::kFloat)
      : query_;
  auto key_acc = !ctx.allowFP16BF16ReductionMathSDP() &&
          (key.scalar_type() == at::kHalf ||
           key.scalar_type() == at::kBFloat16) &&
          !key.is_nested()
      ? key.to(at::kFloat)
      : key;
  auto value_acc = !ctx.allowFP16BF16ReductionMathSDP() &&
          (value.scalar_type() == at::kHalf ||
           value.scalar_type() == at::kBFloat16) &&
          !value.is_nested()
      ? value.to(at::kFloat)
      : value;
  auto attn_mask = attn_mask_;
  // Naive, composite implementation defined here.

  // Scale q, k before matmul for stability see https://tinyurl.com/sudb9s96 for
  // math
  bool is_negative_scaling = scale.has_value() && scale.value() < 0.0;
  const auto scaling_factor =
      sdp::calculate_scale(
          query_acc, is_negative_scaling ? std::abs(scale.value()) : scale)
          .sqrt();

  const auto query = query_acc *
      (is_negative_scaling ? c10::SymFloat(0.0) - scaling_factor
                           : scaling_factor);
  if (is_causal) {
    TORCH_CHECK(
        !attn_mask.has_value(),
        "_scaled_dot_product_attention: Explicit attn_mask should not be set when is_causal=True");
    TORCH_CHECK(
        !query.is_nested() && !key_acc.is_nested(),
        "_scaled_dot_product_attention: Nested tensors for query / key are not supported when is_causal=True");

    // Replace attn_mask with causal mask; lower triangular elements take part
    // in attention.
    const auto L = query.sym_size(-2), S = key_acc.sym_size(-2);
    attn_mask =
        at::ones_symint({L, S}, query.options().dtype(at::kBool)).tril();
    attn_mask = convert_boolean_attn_mask(attn_mask, query.dtype());
    }


    // MQA/GQA handling
    auto [key_expanded, value_expanded] = pre_process_group_query_attention_input(query, key_acc, value_acc, enable_gqa);
    auto attn = at::matmul(query, key_expanded.transpose(-2, -1) * scaling_factor);
    if (attn_mask.has_value()) {
      if (at::areAnyTensorSubclassLike({attn, *attn_mask})) {
        attn = attn.add(*attn_mask);
      } else {
        attn.add_(*attn_mask);
      }
    }
    attn = at::_safe_softmax(attn, -1);
    if (dropout_p > 0.0) {
      if (dropout_mask.has_value()) {
        // In order to validate the correctness of the fused kernels, we need to
        // use the same dropout mask in order to compare the results.
        TORCH_WARN_ONCE("Dropout mask should only be used for testing purposes.");
        attn = attn.masked_fill(dropout_mask->logical_not(), 0.0);
        auto dropout_scaling = 1.0 / (1 - dropout_p);
        return std::make_tuple(at::matmul(attn, value_expanded * dropout_scaling).to(origin_dtype), attn.to(origin_dtype));
      } else {
        attn = at::dropout(attn, dropout_p, true);
      }
    }

    return std::make_tuple(at::matmul(attn, value_expanded).to(origin_dtype), attn.to(origin_dtype));
}

std::tuple<at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_cpu(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& attn_mask,
    std::optional<double> scale) {
  const auto dtype = query.scalar_type();
  int64_t batchSize = query.size(0);
  int64_t qSize = query.size(2);
  int64_t num_head = query.size(1);

  TORCH_CHECK(c10::isFloatingType(dtype),
    "scaled_dot_product_attention_flash_attention: Expected data type in FP32, FP64, BF16, FP16, but got ", dtype, " instead.");
  TORCH_CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4,
    "scaled_dot_product_attention_flash_attention: Accept only 4 dims inputs shape of {B, H, T, K}");
  TORCH_CHECK(dropout_p == 0.0,
    "scaled_dot_product_attention_flash_attention: Currently do not support dropout > 0");
  TORCH_CHECK((query.size(3) == value.size(3)) && (key.size(3) == value.size(3)),
    "scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");
  TORCH_CHECK(!attn_mask.has_value() ||
          attn_mask.value().scalar_type() == at::kFloat ||
          dtype == attn_mask.value().scalar_type(),
    "scaled_dot_product_attention_flash_attention: Attention mask is the same data type as query");
  TORCH_CHECK(!attn_mask.has_value() ||
          (attn_mask.value().dim() == 2 || attn_mask.value().dim() == 4),
    "scaled_dot_product_attention_flash_attention: Attention mask dim in {2, 4}");

  at::Tensor output = at::empty_like(query, query.options()).transpose(1, 2);
  const auto accumulate_dtype = toOpMathType(dtype);
  at::Tensor logsumexp = at::empty({batchSize, qSize, num_head},
      query.options().dtype(accumulate_dtype));

  flash_attention_kernel(kCPU, output, logsumexp,
      query, key, value, dropout_p, is_causal, attn_mask, scale);

  output = output.transpose(1, 2);
  logsumexp = logsumexp.transpose(1, 2);

  return std::make_tuple(std::move(output), std::move(logsumexp));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_flash_attention_cpu_backward(
    const Tensor& grad_out,
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& out,
    const Tensor& logsumexp,
    double dropout_p,
    bool is_causal,
    const std::optional<Tensor>& attn_mask,
    std::optional<double> scale) {
  if (!grad_out.defined()) {
    return std::make_tuple(Tensor{}, Tensor{}, Tensor{});
  }
  auto grad_out_t = grad_out.transpose(1, 2);
  auto q_t = query.transpose(1, 2);
  auto k_t = key.transpose(1, 2);
  auto v_t = value.transpose(1, 2);
  auto o_t = out.transpose(1, 2);
  auto lse_t = logsumexp.transpose(1, 2);

  auto grad_q = at::zeros(q_t.sizes(), query.options());
  auto grad_k = at::zeros(k_t.sizes(), key.options());
  auto grad_v = at::zeros(v_t.sizes(), value.options());

  flash_attention_backward_kernel(kCPU, grad_q, grad_k, grad_v,
      grad_out_t, q_t, k_t, v_t, o_t, lse_t,
      dropout_p, is_causal, attn_mask, scale);

  grad_q = grad_q.transpose(1, 2);
  grad_k = grad_k.transpose(1, 2);
  grad_v = grad_v.transpose(1, 2);

  return std::make_tuple(std::move(grad_q), std::move(grad_k), std::move(grad_v));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable(
    const at::Tensor & query,
    const at::Tensor & key,
    const at::Tensor & value,
    const std::optional<at::Tensor> & attn_bias,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "_scaled_dot_product_fused_attention_overrideable not implemented. This is an operator for privateuse1 backends, please use TORCH_LIBRARY_IMPL to override this function ");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_scaled_dot_product_fused_attention_overrideable_backward(
    const at::Tensor & grad_out,
    const at::Tensor & query,
    const at::Tensor & key,
    const at::Tensor & value,
    const at::Tensor & attn_bias,
    std::array<bool,4> grad_input_mask,
    const at::Tensor & out,
    const at::Tensor & logsumexp,
    const at::Tensor & cum_seq_q,
    const at::Tensor & cum_seq_k,
    int64_t max_q,
    int64_t max_k,
    double dropout_p,
    bool is_causal,
    const at::Tensor & philox_seed,
    const at::Tensor & philox_offset,
    std::optional<double> scale) {
  TORCH_CHECK_NOT_IMPLEMENTED(false, "_scaled_dot_product_fused_attention_overrideable_backward not implemented: This is an operator for privateuse1 backends, please use TORCH_LIBRARY_IMPL to override this function ");
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(
          at::empty_like(query),
          at::empty_like(key),
          at::empty_like(value),
          at::empty_like(attn_bias));
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
    const std::optional<Tensor>& mask) {
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
} // namespace at::native
