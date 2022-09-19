#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorAccessor.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>

#ifdef USE_FLASH_ATTENTION
#include <ATen/native/transformers/cuda/flash_attn/fmha_api.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#endif
namespace at {

namespace native {

namespace {

static constexpr int TRANSFORM_BIAS_RESCALE_VEC = 4;

template <typename scalar_t, typename accscalar_t, bool assume_aligned>
__global__ void transform_bias_rescale_qkv_kernel(
    // [B, T, 3 * D]
    const PackedTensorAccessor64<scalar_t, 3, RestrictPtrTraits> qkv,
    // [3 * D]
    const PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> qkv_bias,
    // [3, B, NH, T, DH]
    PackedTensorAccessor64<scalar_t, 5, RestrictPtrTraits> q_k_v,
    const scalar_t inv_sqrt_dim_per_head) {
  // warp per DH.
  // so launch B * NH * T warps.
  auto NH = q_k_v.size(2);
  auto T = q_k_v.size(3);
  auto DH = q_k_v.size(4);

  auto t = blockIdx.x % T;
  auto b = blockIdx.x / T;

  auto D = NH * DH;

  if (assume_aligned) {
    constexpr int VEC = TRANSFORM_BIAS_RESCALE_VEC;
    using LoadT = memory::aligned_vector<scalar_t, VEC>;
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

      // Here we require D % VEC == 0 for these vectorized loads.
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
             static_cast<accscalar_t>(qkv_bias_q[ii])) *
            static_cast<accscalar_t>(inv_sqrt_dim_per_head));
        qkv_k[ii] = static_cast<scalar_t>(
            (static_cast<accscalar_t>(qkv_k[ii]) +
             static_cast<accscalar_t>(qkv_bias_k[ii])));
        qkv_v[ii] = static_cast<scalar_t>(
            (static_cast<accscalar_t>(qkv_v[ii]) +
             static_cast<accscalar_t>(qkv_bias_v[ii])));
      }

      // Here we require DH % VEC == 0 for these vectorized stores.
      *reinterpret_cast<LoadT*>(&q_k_v[0][b][nh][t][dh]) =
          *reinterpret_cast<const LoadT*>(&qkv_q);
      *reinterpret_cast<LoadT*>(&q_k_v[1][b][nh][t][dh]) =
          *reinterpret_cast<const LoadT*>(&qkv_k);
      *reinterpret_cast<LoadT*>(&q_k_v[2][b][nh][t][dh]) =
          *reinterpret_cast<const LoadT*>(&qkv_v);
    }
  } else {
    // Same as above, but we can't vectorize memory access.
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      auto nh = d / DH;
      auto dh = d % DH;
      scalar_t qkv_bias_q = qkv_bias[d + 0 * D];
      scalar_t qkv_bias_k = qkv_bias[d + 1 * D];
      scalar_t qkv_bias_v = qkv_bias[d + 2 * D];
      scalar_t qkv_q = qkv[b][t][d + 0 * D];
      scalar_t qkv_k = qkv[b][t][d + 1 * D];
      scalar_t qkv_v = qkv[b][t][d + 2 * D];
      qkv_q = static_cast<scalar_t>(
          (static_cast<accscalar_t>(qkv_q) +
           static_cast<accscalar_t>(qkv_bias_q)) *
          static_cast<accscalar_t>(inv_sqrt_dim_per_head));
      qkv_k = static_cast<scalar_t>(
          (static_cast<accscalar_t>(qkv_k) +
           static_cast<accscalar_t>(qkv_bias_k)));
      qkv_v = static_cast<scalar_t>(
          (static_cast<accscalar_t>(qkv_v) +
           static_cast<accscalar_t>(qkv_bias_v)));

      q_k_v[0][b][nh][t][dh] = qkv_q;
      q_k_v[1][b][nh][t][dh] = qkv_k;
      q_k_v[2][b][nh][t][dh] = qkv_v;
    }
  }
}

template <typename scalar_t, typename accscalar_t, bool assume_aligned = false>
__global__ void transform_bias_rescale_qkv_add_padding_kernel(
    // [B, T, 3 * D], but it's a NestedTensor buffer
    const PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> qkv,
    // [3 * D]
    const PackedTensorAccessor64<scalar_t, 1, RestrictPtrTraits> qkv_bias,
    const int* offsets,
    const int* input_sizes,
    // [3, B, NH, T, DH]
    PackedTensorAccessor64<scalar_t, 5, RestrictPtrTraits> q_k_v,
    const scalar_t inv_sqrt_dim_per_head) {
  // warp per DH.
  // so launch B * NH * T warps.
  const auto NH = q_k_v.size(2);
  const auto T = q_k_v.size(3);
  const auto DH = q_k_v.size(4);

  const auto t = blockIdx.x % T;
  const auto b = blockIdx.x / T;

  const auto D = NH * DH;
  const auto _3D = 3 * D;

  const auto offset_for_batch = offsets[b];
  const auto input_dim = 1;
  const auto* sizes_i = input_sizes + b * input_dim;
  if (assume_aligned) {
    constexpr int VEC = TRANSFORM_BIAS_RESCALE_VEC;
    using LoadT = memory::aligned_vector<scalar_t, VEC>;
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

      const auto first_item_offset = t * _3D + d;
      const auto last_item_offset = first_item_offset + VEC - 1;
      const bool first_item_in_bounds = first_item_offset < sizes_i[0];
      const bool entire_vec_in_bounds = last_item_offset < sizes_i[0];

      // Here we require D % VEC == 0 for these vectorized loads.
      *reinterpret_cast<LoadT*>(&qkv_bias_q) =
          *reinterpret_cast<const LoadT*>(&qkv_bias[d + 0 * D]);
      *reinterpret_cast<LoadT*>(&qkv_bias_k) =
          *reinterpret_cast<const LoadT*>(&qkv_bias[d + 1 * D]);
      *reinterpret_cast<LoadT*>(&qkv_bias_v) =
          *reinterpret_cast<const LoadT*>(&qkv_bias[d + 2 * D]);

      if (entire_vec_in_bounds) {
        const auto offset = offset_for_batch + first_item_offset;
        *reinterpret_cast<LoadT*>(&qkv_q) =
            *reinterpret_cast<const LoadT*>(&qkv[offset + 0 * D]);
        *reinterpret_cast<LoadT*>(&qkv_k) =
            *reinterpret_cast<const LoadT*>(&qkv[offset + 1 * D]);
        *reinterpret_cast<LoadT*>(&qkv_v) =
            *reinterpret_cast<const LoadT*>(&qkv[offset + 2 * D]);
#pragma unroll
        // TODO: specialize for float2half2/half2float2?
        for (auto ii = 0; ii < VEC; ++ii) {
          qkv_q[ii] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_q[ii]) +
               static_cast<accscalar_t>(qkv_bias_q[ii])) *
              static_cast<accscalar_t>(inv_sqrt_dim_per_head));
          qkv_k[ii] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_k[ii]) +
               static_cast<accscalar_t>(qkv_bias_k[ii])));
          qkv_v[ii] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_v[ii]) +
               static_cast<accscalar_t>(qkv_bias_v[ii])));
        }
      } else if (first_item_in_bounds) {
        const auto offset = offset_for_batch + first_item_offset;
        qkv_q[0] = qkv[offset + 0 * D];
        qkv_k[0] = qkv[offset + 1 * D];
        qkv_v[0] = qkv[offset + 2 * D];
        qkv_q[0] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_q[0]) +
               static_cast<accscalar_t>(qkv_bias_q[0])) *
              static_cast<accscalar_t>(inv_sqrt_dim_per_head));
        qkv_k[0] = static_cast<scalar_t>(
            (static_cast<accscalar_t>(qkv_k[0]) +
               static_cast<accscalar_t>(qkv_bias_k[0])));
          qkv_v[0] = static_cast<scalar_t>(
              (static_cast<accscalar_t>(qkv_v[0]) +
               static_cast<accscalar_t>(qkv_bias_v[0])));
#pragma unroll
        for (auto ii = 1; ii < VEC; ++ii) {
          const auto loop_offset = offset + ii;
          if (loop_offset < sizes_i[0]) {
            qkv_q[ii] = qkv[loop_offset + 0 * D];
            qkv_k[ii] = qkv[loop_offset + 1 * D];
            qkv_v[ii] = qkv[loop_offset + 2 * D];
            qkv_q[ii] = static_cast<scalar_t>(
                (static_cast<accscalar_t>(qkv_q[ii]) +
                 static_cast<accscalar_t>(qkv_bias_q[ii])) *
                static_cast<accscalar_t>(inv_sqrt_dim_per_head));
            qkv_k[ii] = static_cast<scalar_t>(
                (static_cast<accscalar_t>(qkv_k[ii]) +
                 static_cast<accscalar_t>(qkv_bias_k[ii])));
            qkv_v[ii] = static_cast<scalar_t>(
                (static_cast<accscalar_t>(qkv_v[ii]) +
                 static_cast<accscalar_t>(qkv_bias_v[ii])));
          } else {
            qkv_q[ii] = 0;
            qkv_k[ii] = 0;
            qkv_v[ii] = 0;
          }
        }
      } else {
#pragma unroll
        for (auto ii = 0; ii < VEC; ++ii) {
          qkv_q[ii] = 0;
          qkv_k[ii] = 0;
          qkv_v[ii] = 0;
        }
      }

      // Here we require DH % VEC == 0 for these vectorized stores.
      *reinterpret_cast<LoadT*>(&q_k_v[0][b][nh][t][dh]) =
          *reinterpret_cast<const LoadT*>(&qkv_q);
      *reinterpret_cast<LoadT*>(&q_k_v[1][b][nh][t][dh]) =
          *reinterpret_cast<const LoadT*>(&qkv_k);
      *reinterpret_cast<LoadT*>(&q_k_v[2][b][nh][t][dh]) =
          *reinterpret_cast<const LoadT*>(&qkv_v);
    }
  } else {
    for (int32_t d = threadIdx.x; d < D; d += blockDim.x) {
      auto nh = d / DH;
      auto dh = d % DH;
      scalar_t qkv_bias_q = qkv_bias[d + 0 * D];
      scalar_t qkv_bias_k = qkv_bias[d + 1 * D];
      scalar_t qkv_bias_v = qkv_bias[d + 2 * D];

      const auto item_offset = t * _3D + d;
      const bool in_bounds = item_offset < sizes_i[0];
      scalar_t qkv_q, qkv_k, qkv_v;
      if (in_bounds) {
        const auto qkv_offset = offset_for_batch + item_offset;
        qkv_q = qkv[qkv_offset + 0 * D];
        qkv_k = qkv[qkv_offset + 1 * D];
        qkv_v = qkv[qkv_offset + 2 * D];
        qkv_q = static_cast<scalar_t>(
            (static_cast<accscalar_t>(qkv_q) +
             static_cast<accscalar_t>(qkv_bias_q)) *
            static_cast<accscalar_t>(inv_sqrt_dim_per_head));
        qkv_k = static_cast<scalar_t>(
            (static_cast<accscalar_t>(qkv_k) +
             static_cast<accscalar_t>(qkv_bias_k)));
        qkv_v = static_cast<scalar_t>(
            (static_cast<accscalar_t>(qkv_v) +
             static_cast<accscalar_t>(qkv_bias_v)));
      } else {
        qkv_q = 0;
        qkv_k = 0;
        qkv_v = 0;
      }

      q_k_v[0][b][nh][t][dh] = qkv_q;
      q_k_v[1][b][nh][t][dh] = qkv_k;
      q_k_v[2][b][nh][t][dh] = qkv_v;
    }
  }
}

Tensor collapse_dims_1_and_2(const Tensor& sizes) {
  auto sizes_dim1 = at::native::narrow(sizes, 1, 0, 1);
  auto sizes_dim2 = at::native::narrow(sizes, 1, 1, 1);

  return (sizes_dim1 * sizes_dim2).contiguous();
}

} // namespace
// compute q = (q + q_bias) / sqrt(dim_per_head), k = k + k_bias, v = v + v_bias
__host__ std::tuple<Tensor, Tensor, Tensor> transform_bias_rescale_qkv_cuda(
    const Tensor& qkv,
    const Tensor& qkv_bias,
    const int64_t num_head) {
  auto B = qkv.is_nested()
      ? get_nested_tensor_impl(qkv)->get_nested_size_tensor().size(0)
      : qkv.size(0);
  // TODO: calculate this without the std::vector -- NestedTensor_to_mask wants
  // this too
  auto T = qkv.is_nested()
      ? NestedTensor_get_max_size(*get_nested_tensor_impl(qkv))[0]
      : qkv.size(1);
  if (qkv.is_nested()) {
    // Don't mess with non-nested case for now since it's not set up to fiddle
    // with mask size.

    // Round T up to next multiple of 8 so as to be able to utilize Tensor
    // cores. Otherwise, sometimes with padding, *no* row will have the maximum
    // sequence length and so we'll have a non-divisible-by-8 dimension even if
    // the model author chose a multiple of 8.
    T = T + (8 - (T % 8)) % 8;
  }
  auto _3D = qkv_bias.size(0);
  auto D = _3D / 3;
  TORCH_CHECK(D % num_head == 0);
  const auto dim_per_head = D / num_head;
  auto q_k_v = at::empty({3, B, num_head, T, dim_per_head}, qkv_bias.options());
#define CALL_KERNEL(assume_aligned)                                        \
  transform_bias_rescale_qkv_kernel<scalar_t, accscalar_t, assume_aligned> \
      <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(          \
          qkv.packed_accessor64<scalar_t, 3, RestrictPtrTraits>(),         \
          qkv_bias.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),    \
          q_k_v.packed_accessor64<scalar_t, 5, RestrictPtrTraits>(),       \
          1.0 / std::sqrt(static_cast<scalar_t>(dim_per_head)))
#define CALL_ADD_PADDING_KERNEL(assume_aligned)                         \
  transform_bias_rescale_qkv_add_padding_kernel<                        \
      scalar_t,                                                         \
      accscalar_t,                                                      \
      assume_aligned>                                                   \
      <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(       \
          nt_qkv_buffer                                          \
              .packed_accessor64<scalar_t, 1, RestrictPtrTraits>(),     \
          qkv_bias.packed_accessor64<scalar_t, 1, RestrictPtrTraits>(), \
          offsets_ptr,                                                  \
          sizes_ptr,                                                    \
          q_k_v.packed_accessor64<scalar_t, 5, RestrictPtrTraits>(),    \
          1.0 / std::sqrt(static_cast<scalar_t>(dim_per_head)))

  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      qkv.scalar_type(),
      "transform_bias_rescale_qkv",
      [&] {
        using accscalar_t = acc_type<scalar_t, true>;
        auto threads = std::max(
            std::min<int32_t>(1024, D / TRANSFORM_BIAS_RESCALE_VEC), 1);
        auto blocks = B * T;
        const bool aligned =
            ((dim_per_head % TRANSFORM_BIAS_RESCALE_VEC) == 0) &&
            ((reinterpret_cast<intptr_t>(qkv_bias.data_ptr()) %
              TRANSFORM_BIAS_RESCALE_VEC) == 0);
        if (aligned) {
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
              D % TRANSFORM_BIAS_RESCALE_VEC == 0,
              "D = num_heads * dim_per_head, so we should have dim_per_head % "
              "TRANSFORM_BIAS_RESCALE_VEC == 0 => "
              "D % TRANSFORM_BIAS_RESCALE_VEC == 0");
        }
        if (qkv.is_nested()) {
          auto* nt_qkv = get_nested_tensor_impl(qkv);
          const at::Tensor& nt_qkv_buffer = nt_qkv->get_buffer();
          auto sizes = collapse_dims_1_and_2(nt_qkv->get_nested_size_tensor());
          auto offsets =
              NestedTensor_batch_offsets_from_size_tensor(sizes, sizes.numel());
          at::native::narrow(offsets, 0, sizes.numel() + 1, sizes.numel())
              .copy_(sizes.reshape({-1}));
          auto metadata = offsets.to(at::Device(kCUDA), at::kInt, true, true);
          const auto offsets_ptr = metadata.data_ptr<int>();
          const auto sizes_ptr = offsets_ptr + sizes.numel() + 1;
          const auto input_dim = sizes.sizes()[1];
          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(input_dim == 1);
          if (aligned &&
              ((reinterpret_cast<intptr_t>(qkv.data_ptr()) %
                TRANSFORM_BIAS_RESCALE_VEC) == 0)) {
            CALL_ADD_PADDING_KERNEL(true);
          } else {
            CALL_ADD_PADDING_KERNEL(false);
          }
        } else if (aligned) {
          CALL_KERNEL(true);
        } else {
          CALL_KERNEL(false);
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
#undef CALL_ADD_PADDING_KERNEL
#undef CALL_KERNEL
  auto q_k_v_s =
      at::native::split(q_k_v.view({3 * B, num_head, T, dim_per_head}), B, 0);
  return std::make_tuple(q_k_v_s[0], q_k_v_s[1], q_k_v_s[2]);
}

Tensor triton_scaled_dot_attention(const Tensor& q, const Tensor& k, const Tensor& v, double dropout_p){
  TORCH_CHECK(false, "This operator should be overridden in python before use");
  return at::Tensor();
}

Tensor flash_scaled_dot_product_attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const Tensor& cumulative_sequence_length_q,
    const Tensor& cumulative_sequence_length_k,
    const int64_t max_seqlen_batch_q,
    const int64_t max_seqlen_batch_k,
    double dropout_p,
    bool causal) {
#if defined(USE_FLASH_ATTENTION)
  auto softmax_scale = std::pow(query.size(-1), -0.5);
  std::vector<Tensor> output = fmha::mha_fwd(
      query,
      key,
      value,
      cumulative_sequence_length_q,
      cumulative_sequence_length_k,
      max_seqlen_batch_q,
      max_seqlen_batch_k,
      dropout_p,
      softmax_scale,
      false,
      causal,
      false,
      c10::nullopt);
  return output[0];
#endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.")
  return Tensor{};
}

std::tuple<at::Tensor, at::Tensor> efficient_attention_forward_generic(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    bool compute_logsumexp,
    const c10::optional<at::Tensor>& attn_bias_,
    double p,
    bool causal) {
#if defined(USE_FLASH_ATTENTION)
  TORCH_CHECK(p == 0.0, "Dropout is not supported at the moment");
  TORCH_CHECK(
      !attn_bias_.has_value(), "attn_bias is not supported at the moment");

  TORCH_CHECK(query.dim() == 3);
  TORCH_CHECK(key.dim() == 3);
  TORCH_CHECK(value.dim() == 3);

  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(0) == key.size(0));

  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");

  TORCH_CHECK(!query.is_sparse(), "query must be a dense tensor");
  TORCH_CHECK(!key.is_sparse(), "key must be a dense tensor");
  TORCH_CHECK(!value.is_sparse(), "value must be a dense tensor");

  TORCH_CHECK(query.is_contiguous());
  TORCH_CHECK(key.is_contiguous());
  TORCH_CHECK(value.is_contiguous());

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t K = query.size(2);

  at::Tensor res;
  at::Tensor logsumexp;

  cudaDeviceProp* properties =
      at::cuda::getDeviceProperties(query.device().index());
  const int computeCapability = properties->major * 10 + properties->minor;

#define DISPATCH_ARCHTAG(func)                                            \
  {                                                                       \
    if (computeCapability >= 80) {                                        \
      using ArchTag = cutlass::arch::Sm80;                                \
      func();                                                             \
    } else if (computeCapability >= 75) {                                 \
      using ArchTag = cutlass::arch::Sm75;                                \
      func();                                                             \
    } else if (computeCapability >= 70) {                                 \
      using ArchTag = cutlass::arch::Sm70;                                \
      func();                                                             \
    } else if (computeCapability >= 50) {                                 \
      using ArchTag = cutlass::arch::Sm50;                                \
      func();                                                             \
    } else {                                                              \
      TORCH_CHECK(                                                        \
          false,                                                          \
          "Your device is too old. We require compute capability >= 50"); \
    }                                                                     \
  }
// Dispatch to the right kernel
#define DISPATCH_TYPES(func)                                          \
  {                                                                   \
    if (query.scalar_type() == at::ScalarType::Float) {               \
      using scalar_t = float;                                         \
      func();                                                         \
    } else if (query.scalar_type() == at::ScalarType::Half) {         \
      using scalar_t = cutlass::half_t;                               \
      func();                                                         \
    } else {                                                          \
      TORCH_CHECK(false, "Only fp32 & half supported at the moment"); \
    }                                                                 \
  }

#define DISPATCH_BLOCKSIZE(VALUE_HEAD_DIM, BLOCK_6464, SINGLE_VALUE_ITER, FN) \
  {                                                                           \
    if (VALUE_HEAD_DIM <= 64) {                                               \
      constexpr bool BLOCK_6464 = true;                                       \
      constexpr bool SINGLE_VALUE_ITER = true;                                \
      FN();                                                                   \
    } else {                                                                  \
      constexpr bool BLOCK_6464 = false;                                      \
      if (VALUE_HEAD_DIM <= 128) {                                            \
        constexpr bool SINGLE_VALUE_ITER = true;                              \
        FN();                                                                 \
      } else {                                                                \
        constexpr bool SINGLE_VALUE_ITER = false;                             \
        FN();                                                                 \
      }                                                                       \
    }                                                                         \
  }

  DISPATCH_BLOCKSIZE(
      value.size(2), kIs64x64, kSingleValueIteration, ([&]() {
        static constexpr int64_t kQueriesPerBlock = kIs64x64 ? 64 : 32;
        static constexpr int64_t kKeysPerBlock = kIs64x64 ? 64 : 128;
        DISPATCH_TYPES(([&]() {
          DISPATCH_ARCHTAG(([&]() {
            // Run a more efficient kernel (with `isAligned=True`) if
            // memory is correctly aligned
            bool isAligned;
            using AlignedAK = AttentionKernel<
                scalar_t,
                ArchTag,
                true,
                kQueriesPerBlock,
                kKeysPerBlock,
                kSingleValueIteration>;
            isAligned =
                (query.stride(1) % AlignedAK::kAlignmentQ == 0 &&
                 key.stride(1) % AlignedAK::kAlignmentK == 0 &&
                 value.stride(1) % AlignedAK::kAlignmentV == 0);
            // TODO: Should we warn or log somewhere when we use a less
            // efficient kernel due to wrong alignment?
            DISPATCH_BOOL(
                isAligned, kIsAligned, ([&]() {
                  using Kernel = AttentionKernel<
                      scalar_t,
                      ArchTag,
                      kIsAligned,
                      kQueriesPerBlock,
                      kKeysPerBlock,
                      kSingleValueIteration>;
                  // Might happen on Sm80/half, where the minimum
                  // alignment is 32bits
                  TORCH_CHECK(
                      query.stride(1) % Kernel::kAlignmentQ == 0,
                      "query is not correctly aligned");
                  TORCH_CHECK(
                      key.stride(1) % Kernel::kAlignmentK == 0,
                      "key is not correctly aligned");
                  TORCH_CHECK(
                      value.stride(1) % Kernel::kAlignmentV == 0,
                      "value is not correctly aligned");

                  res = at::zeros(
                      {B, M, K},
                      query.options().dtype(
                          TypeTraits<
                              typename Kernel::output_t>::atScalarType()));
                  // NOTE: Should be aligned (by padding) in case M is not
                  // a good number for loading during backward
                  constexpr decltype(M) kAlignLSE =
                      32; // block size of backward
                  logsumexp = at::empty(
                      {B,
                       compute_logsumexp ? ceil_div(M, kAlignLSE) * kAlignLSE
                                         : 0},
                      query.options().dtype(at::ScalarType::Float));

                  constexpr auto kernel_fn = attention_kernel_batched<Kernel>;
                  size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
                  if (smem_bytes > 0xc000) {
                    TORCH_INTERNAL_ASSERT(
                        computeCapability >= 70,
                        "This kernel requires too much shared memory on this machine!");
                    AT_CUDA_CHECK(cudaFuncSetAttribute(
                        kernel_fn,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        smem_bytes));
                  }

                  typename Kernel::Params p;
                  p.query_ptr = (scalar_t*)query.data_ptr();
                  p.key_ptr = (scalar_t*)key.data_ptr();
                  p.value_ptr = (scalar_t*)value.data_ptr();
                  p.logsumexp_ptr = compute_logsumexp
                      ? (typename Kernel::lse_scalar_t*)logsumexp.data_ptr()
                      : nullptr;
                  p.output_ptr = (typename Kernel::output_t*)res.data_ptr();
                  p.head_dim = query.size(2);
                  p.head_dim_value = value.size(2);
                  p.num_queries = query.size(1);
                  p.num_keys = key.size(1);
                  p.num_batches = B;
                  p.causal = causal;
                  kernel_fn<<<
                      p.getBlocksGrid(),
                      p.getThreadsGrid(),
                      smem_bytes>>>(p);
                }));
          }));
        }));
      }));

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(res, logsumexp);
#endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.")
  return std::make_tuple(Tensor{}, Tensor{});
}

} // namespace native
} // namespace at
