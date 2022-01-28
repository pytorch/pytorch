#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorAccessor.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>

#include <c10/cuda/CUDAMathCompat.h>

namespace at {

namespace native {

namespace {

Tensor gemm_nt(const Tensor& a, const Tensor& b) {
  auto a_ = a.view({a.size(0) * a.size(1), a.size(2)});
  auto b_ = b.transpose(1, 0);
  auto c_ = at::native::matmul(a_, b_);
  return c_.view({a.size(0), a.size(1), b.size(0)});
}

template <typename scalar_t, typename accscalar_t>
__global__ void transform_bias_rescale_qkv_kernel(
    // [B, T, 3 * D]
    const PackedTensorAccessor64<scalar_t, 3, RestrictPtrTraits> qkv,
    // [3 * D]
    const PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> qkv_bias,
    // [3, B, NH, T, DH]
    PackedTensorAccessor64<scalar_t, 5, RestrictPtrTraits> q_k_v) {
  // warp per DH.
  // so launch B * NH * T warps.
  auto B = q_k_v.size(1);
  auto NH = q_k_v.size(2);
  auto T = q_k_v.size(3);
  auto DH = q_k_v.size(4);

  auto t = blockIdx.x % T;
  auto b = blockIdx.x / T;

  auto D = NH * DH;
  constexpr int VEC = 4;
  using LoadT = memory::aligned_vector<scalar_t, VEC>;

  // FIXME: assert ((D % VEC) == 0)

  for (int32_t d_v = threadIdx.x; d_v < D / VEC; d_v += blockDim.x) {
    auto d = d_v * VEC;
    auto nh = d / DH;
    auto dh = d % DH;
    scalar_t qkv_bias_q[VEC];
    scalar_t qkv_bias_k[VEC];
    scalar_t qkv_bias_v[VEC];
    scalar_t qkv_q[VEC];
    scalar_t qkv_k[VEC];
    scalar_t qkv_v[VEC];

    *reinterpret_cast<LoadT*>(&qkv_bias_q) =
        *reinterpret_cast<const LoadT*>(&qkv_bias[d + 0 * D]);
    *reinterpret_cast<LoadT*>(&qkv_bias_k) =
        *reinterpret_cast<const LoadT*>(&qkv_bias[d + 1 * D]);
    *reinterpret_cast<LoadT*>(&qkv_bias_v) =
        *reinterpret_cast<const LoadT*>(&qkv_bias[d + 2 * D]);

    *reinterpret_cast<LoadT*>(&qkv_q) =
        *reinterpret_cast<const LoadT*>(&qkv[b][t][d + 0 * D]);
    *reinterpret_cast<LoadT*>(&qkv_k) =
        *reinterpret_cast<const LoadT*>(&qkv[b][t][d + 1 * D]);
    *reinterpret_cast<LoadT*>(&qkv_v) =
        *reinterpret_cast<const LoadT*>(&qkv[b][t][d + 2 * D]);

#pragma unroll
    // TODO: specialize for float2half2/half2float2?
    for (auto ii = 0; ii < VEC; ++ii) {
      qkv_q[ii] = static_cast<scalar_t>(
          (static_cast<accscalar_t>(qkv_q[ii]) +
           static_cast<accscalar_t>(qkv_bias_q[ii])) /
          static_cast<accscalar_t>(8));
      qkv_k[ii] = static_cast<scalar_t>(
          (static_cast<accscalar_t>(qkv_k[ii]) +
           static_cast<accscalar_t>(qkv_bias_k[ii])));
      qkv_v[ii] = static_cast<scalar_t>(
          (static_cast<accscalar_t>(qkv_v[ii]) +
           static_cast<accscalar_t>(qkv_bias_v[ii])));
    }
    *reinterpret_cast<LoadT*>(&q_k_v[0][b][nh][t][dh]) =
        *reinterpret_cast<const LoadT*>(&qkv_q);
    *reinterpret_cast<LoadT*>(&q_k_v[1][b][nh][t][dh]) =
        *reinterpret_cast<const LoadT*>(&qkv_k);
    *reinterpret_cast<LoadT*>(&q_k_v[2][b][nh][t][dh]) =
        *reinterpret_cast<const LoadT*>(&qkv_v);
  }
}

// compute q = (q + q_bias) / sqrt(dim_per_head), k = k + k_bias, v = v + v_bias
std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv(
    const Tensor& qkv,
    const Tensor& qkv_bias) {
  auto B = qkv.size(0);
  auto T = qkv.size(1);
  auto _3D = qkv.size(2);
  auto D = _3D / 3;
  auto dim_per_head = 64;
  auto num_head = D / dim_per_head;
  auto q_k_v = at::empty({3, B, num_head, T, dim_per_head}, qkv.options());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      qkv.scalar_type(),
      "transform_bias_rescale_qkv",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        auto threads = std::min<int32_t>(1024, D / 4);
        auto blocks = B * T;
        transform_bias_rescale_qkv_kernel<scalar_t, accscalar_t>
            <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
                qkv.packed_accessor64<scalar_t, 3, RestrictPtrTraits>(),
                qkv_bias.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),
                q_k_v.packed_accessor64<scalar_t, 5, RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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

template <typename T>
__inline__ __device__ T WarpReduceMax(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val = std::max(val, WARP_SHFL_DOWN(val, offset));
  }
  return val;
}

template <typename T>
__inline__ __device__ T WarpReduceSum(T val) {
#pragma unroll
  for (int offset = (C10_WARP_SIZE >> 1); offset > 0; offset >>= 1) {
    val += WARP_SHFL_DOWN(val, offset);
  }
  return val;
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
        using accscalar_t = acc_type<scalar_t, true>;
        // TODO: proper implementation with masking.
        dispatch_softmax_forward<scalar_t, scalar_t, accscalar_t, false, false>(
          attn_scores.data_ptr<scalar_t>(),
          attn_scores.data_ptr<scalar_t>(),
          T,
          T,
          B * num_heads * T
        );
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
  return a.permute({0, 2, 1, 3})
      .contiguous()
      .view({a.size(0), a.size(2), a.size(1) * a.size(3)});
}

Tensor gemm_nt_bias(const Tensor& a, const Tensor& b, const Tensor& c) {
  auto a_ = a.view({a.size(0) * a.size(1), a.size(2)});
  // TODO: should be b.transpose(1, 0)?
  auto r_ = at::native::linear(a_, b, c);
  return r_.view({a.size(0), a.size(1), r_.size(1)});
}

} // namespace

Tensor multi_head_self_attention_cuda(
    const Tensor& query,
    const Tensor& qkv_weight,
    const Tensor& qkv_bias,
    const Tensor& proj_weight,
    const Tensor& proj_bias,
    const c10::optional<Tensor>& mask) {
  // query shape: [B, T, D]
  // qkv_weight shape: [3 * D, D]

  // shape: [B, T, 3 x D]
  auto qkv = gemm_nt(query, qkv_weight);

  // shape: 3 x [B, num_head, T, dim_per_head]
  auto q_k_v = transform_bias_rescale_qkv(qkv, qkv_bias);
  auto q = std::get<0>(q_k_v);
  auto k = std::get<1>(q_k_v);
  auto v = std::get<2>(q_k_v);

  // shape: [B, num_head, T, T]
  auto qkt = bmm_nt(q, k);

  // shape: [B, num_head, T, T]
  masked_softmax_dropout(qkt, mask);

  // shape: [B, num_head, T, dim_per_head]
  auto attn_ctx = bmm_nn(qkt, v);

  // shape: [B, T, D]
  auto attn = transform_0213(attn_ctx);

  // shape: [B, T, D]
  auto proj = gemm_nt_bias(attn, proj_weight, proj_bias);

  return proj;
}

} // namespace native
} // namespace at
