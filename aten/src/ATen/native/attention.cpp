#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec256/vec256.h>

namespace at {

namespace native {

namespace {

Tensor gemm_nt(const Tensor& a, const Tensor& b) {
  return at::native::matmul(a, b.t());
}

// compute q = (q + q_bias) / sqrt(dim_per_head), k = k + k_bias, v = v + v_bias
std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head) {
  auto B = qkv.size(0);
  auto T = qkv.size(1);
  auto _3D = qkv.size(2);
  auto D = _3D / 3;
  TORCH_CHECK(D % num_head == 0);
  const auto dim_per_head = D / num_head;
  auto q_k_v = at::empty({3, B, num_head, T, dim_per_head}, qkv.options());

  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      qkv.scalar_type(),
      "transform_bias_rescale_qkv",
      [&] {
        scalar_t* qkv_data = qkv.data_ptr<scalar_t>();
        scalar_t* qkv_bias_data = qkv_bias.data_ptr<scalar_t>();
        scalar_t* q_k_v_data = q_k_v.data_ptr<scalar_t>();
        const scalar_t sqrt_dim_per_head = std::sqrt(static_cast<scalar_t>(dim_per_head));

        int64_t grain_size =
            std::min(internal::GRAIN_SIZE / (3 * dim_per_head), (int64_t)1);
        parallel_for(
            0, B * num_head * T, grain_size, [&](int64_t begin, int64_t end) {
              for (auto i : c10::irange(begin, end)) {
                auto t = i % T;
                i /= T;
                auto nh = i % num_head;
                i /= num_head;
                auto b = i;
                using Vec = vec::Vectorized<scalar_t>;
                auto V = vec::Vectorized<scalar_t>::size();
                // TODO: handle epilogue
                TORCH_INTERNAL_ASSERT(dim_per_head % V == 0, "epilogue not implemented yet");
                for (auto dh = 0; dh < dim_per_head; dh += V) {
                  auto d = nh * dim_per_head + dh;
                  // load
                  auto q_bias_data = Vec::loadu(&qkv_bias_data[d + 0 * D]);
                  auto k_bias_data = Vec::loadu(&qkv_bias_data[d + 1 * D]);
                  auto v_bias_data = Vec::loadu(&qkv_bias_data[d + 2 * D]);

                  auto q_data =
                      Vec::loadu(&qkv_data[b * _3D * T + t * _3D + d + 0 * D]) +
                      q_bias_data;
                  auto k_data =
                      Vec::loadu(&qkv_data[b * _3D * T + t * _3D + d + 1 * D]) +
                      k_bias_data;
                  auto v_data =
                      Vec::loadu(&qkv_data[b * _3D * T + t * _3D + d + 2 * D]) +
                      v_bias_data;

                  q_data = q_data / Vec(sqrt_dim_per_head);

                  q_data.store(&q_k_v_data
                                   [0 * B * num_head * T * dim_per_head +
                                    b * num_head * T * dim_per_head +
                                    nh * T * dim_per_head +
                                    t * dim_per_head + dh]);
                  k_data.store(&q_k_v_data
                                   [1 * B * num_head * T * dim_per_head +
                                    b * num_head * T * dim_per_head +
                                    nh * T * dim_per_head +
                                    t * dim_per_head + dh]);
                  v_data.store(&q_k_v_data
                                   [2 * B * num_head * T * dim_per_head +
                                    b * num_head * T * dim_per_head +
                                    nh * T * dim_per_head +
                                    t * dim_per_head + dh]);
                }
              }
            });
      });
  auto q_k_v_s =
      at::native::split(q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);
  return std::make_tuple(q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]);
}

Tensor bmm_nt(const Tensor& a, const Tensor& b) {
  auto a_ = a.view({a.size(0) * a.size(1), a.size(2), a.size(3)});
  auto b_ = b.view({b.size(0) * b.size(1), b.size(2), b.size(3)});
  auto bt_ = b_.transpose(2, 1);
  // TODO: are these a single call to cublas batched matmul?
  auto c_ = at::matmul(a_, bt_);
  return c_.view({a.size(0), a.size(1), a.size(2), b.size(2)});
}

void masked_softmax_dropout(
    const Tensor& attn_scores,
    const c10::optional<Tensor>& attn_mask) {
  auto B = attn_scores.size(0);
  auto num_heads = attn_scores.size(1);
  auto T = attn_scores.size(2);
  if (attn_mask) {
    TORCH_CHECK(attn_mask->is_contiguous());
  }
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      attn_scores.scalar_type(),
      "masked_softmax_dropout",
      [&] {
        using accscalar_t = acc_type<scalar_t, false>;
        // TODO: proper implementation with masking.
        scalar_t* attn_scores_data = attn_scores.data_ptr<scalar_t>();
        int64_t grain_size = std::min(internal::GRAIN_SIZE / T, (int64_t)1);
        parallel_for(
            0, B * num_heads * T, grain_size, [&](int64_t begin, int64_t end) {
              for (const auto i : c10::irange(begin, end)) {
                using Vec = vec::Vectorized<scalar_t>;
                auto V = vec::Vectorized<scalar_t>::size();

                scalar_t* input_data = attn_scores_data + i;
                auto max_input = Vec(std::numeric_limits<scalar_t>::lowest());
                // TODO: handle epilogue
                for (auto t = 0; t < T; t += V) {
                  auto v = Vec::loadu(&input_data[t]);
                  max_input = vec::maximum(max_input, v);
                }

                auto hmax = std::numeric_limits<scalar_t>::lowest();
                for (auto i = 0; i < V; ++i) {
                  hmax = std::max(max_input[i], hmax);
                }
                accscalar_t hsum = 0;
                for (auto t = 0; t < T; t += V) {
                  auto v = Vec::loadu(&input_data[t]);
                  // TODO: vectorize in accscalar_t?
                  for (auto i = 0; i < V; ++i) {
                    hsum += std::exp(static_cast<accscalar_t>(v[i]) - hmax);
                  }
                }
                auto inv_denominator = 1.0 / hsum;
                for (auto t = 0; t < T; t += V) {
                  Vec v = Vec::loadu(&input_data[t]);

                  // TODO: vectorize in accscalar_t?
                  // TODO this faster solution does not work on Android build
                  /*
                  for (auto i = 0; i < V; ++i) {
                    v[i] = static_cast<scalar_t>(std::exp(static_cast<accscalar_t>(v[i]) - hmax) * inv_denominator);
                  }
                  v.store(&input_data[t]);
                  */
                  for (auto i = 0; i < V; ++i) {
                    input_data[t + i] = static_cast<scalar_t>(std::exp(static_cast<accscalar_t>(v[i]) - hmax) * inv_denominator);
                  }
                }
              }
            });
      });
}

Tensor bmm_nn(const Tensor& a, const Tensor& b) {
  auto a_ = a.view({a.size(0) * a.size(1), a.size(2), a.size(3)});
  auto b_ = b.view({b.size(0) * b.size(1), b.size(2), b.size(3)});
  // TODO: are these a single call to cublas batched matmul?
  auto c_ = at::matmul(a_, b_);
  return c_.view({a.size(0), a.size(1), a.size(2), b.size(3)});
}

Tensor transform_0213(const Tensor& a) {
  // TODO: check perf vs dedicated kernel.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.size(1));
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(a.size(3));
  return a.permute({0, 2, 1, 3})
      .contiguous()
      .view({a.size(0), a.size(2), a.size(1) * a.size(3)});
}

Tensor gemm_nt_bias(const Tensor& a, const Tensor& b, const Tensor& c) {
  auto a_ = a.view({a.size(0) * a.size(1), a.size(2)});
  auto r_ = at::native::linear(a_, b, c);
  return r_.view({a.size(0), a.size(1), r_.size(1)});
}

void debug_assert_shape(const Tensor& t, c10::IntArrayRef shape) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY((size_t)t.dim() == shape.size(), "expected ", shape.size(), "-D tensor but got ", t.dim());
  for (auto idx : c10::irange(shape.size())) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t.sizes()[idx] == shape[idx], "expected dim ", idx, " to be ", shape[idx], " but got ", t.sizes()[idx]);
  }
}

} // namespace

Tensor multi_head_self_attention_cpu(
    const Tensor& query,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const int64_t num_head,
    const c10::optional<Tensor>& mask) {
  // query shape: [B, T, D]
  // qkv_weight shape: [3 * D, D]

  const auto D = query.sizes()[2];

  TORCH_CHECK(query.dim() == 3, "expected 3-dimensional query, got ", query.dim(), "-D tensor");
  TORCH_CHECK(qkv_weight.dim() == 2, "expected 2-dimensional qkv_weight, got ", qkv_weight.dim(), "-D tensor");
  TORCH_CHECK(D * 3 == qkv_weight.sizes()[0], "expected qkv_weight first dim to be 3x last dim of query");
  TORCH_CHECK(D == qkv_weight.sizes()[1], "expected qkv_weight second dim and last dim of query to be equal");
  TORCH_CHECK(D % num_head == 0, "D must divide evenly by num_head");

#ifndef NDEBUG
  const auto B = query.sizes()[0];
  const auto T = query.sizes()[1];
  const auto dim_per_head = D / num_head;
#endif

  // shape: [B, T, 3 x D]
  auto qkv = gemm_nt(query, qkv_weight);
#ifndef NDEBUG
  debug_assert_shape(qkv, {B, T, 3 * D});
#endif

  // shape: 3 x [B, num_head, T, dim_per_head]
  auto q_k_v = transform_bias_rescale_qkv(qkv, qkv_bias, num_head);
  const auto& q = std::get<0>(q_k_v);
  const auto& k = std::get<1>(q_k_v);
  const auto& v = std::get<2>(q_k_v);
#ifndef NDEBUG
  debug_assert_shape(q, {B, num_head, T, dim_per_head});
  debug_assert_shape(k, {B, num_head, T, dim_per_head});
  debug_assert_shape(v, {B, num_head, T, dim_per_head});
#endif

  // shape: [B, num_head, T, T]
  auto qkt = bmm_nt(q, k);
#ifndef NDEBUG
  debug_assert_shape(qkt, {B, num_head, T, T});
#endif

  // shape: [B, num_head, T, T]
  masked_softmax_dropout(qkt, mask);

  // shape: [B, num_head, T, dim_per_head]
  auto attn_ctx = bmm_nn(qkt, v);
#ifndef NDEBUG
  debug_assert_shape(attn_ctx, {B, num_head, T, dim_per_head});
#endif

  // shape: [B, T, D]
  auto attn = transform_0213(attn_ctx);
#ifndef NDEBUG
  debug_assert_shape(attn, {B, T, D});
#endif

  // shape: [B, T, D]
  auto proj = gemm_nt_bias(attn, proj_weight, proj_bias);
#ifndef NDEBUG
  debug_assert_shape(proj, {B, T, D});
#endif
  return proj;
}

} // namespace native
} // namespace at
