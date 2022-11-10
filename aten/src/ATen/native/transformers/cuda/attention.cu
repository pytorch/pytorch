#include <type_traits>

#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorAccessor.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>
#include <ATen/native/cuda/block_reduce.cuh>

#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/native/transformers/attention.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>

#ifdef USE_FLASH_ATTENTION
#include <ATen/native/transformers/cuda/flash_attn/fmha_api.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#endif

namespace at {

namespace native {

namespace {

#define DISPATCH_BLOCKSIZE(VALUE_HEAD_DIM, FN)        \
  {                                                   \
    if (VALUE_HEAD_DIM <= 64) {                       \
      constexpr bool kIs64x64 = true;                 \
      constexpr bool kSingleValueIteration = true;    \
      FN();                                           \
    } else {                                          \
      constexpr bool kIs64x64 = false;                \
      if (VALUE_HEAD_DIM <= 128) {                    \
        constexpr bool kSingleValueIteration = true;  \
        FN();                                         \
      } else {                                        \
        constexpr bool kSingleValueIteration = false; \
        FN();                                         \
      }                                               \
    }                                                 \
  }

#define DISPATCH_KERNEL(QUERY, KEY, VALUE, FUNC)                              \
  {                                                                           \
    cudaDeviceProp* properties =                                              \
        at::cuda::getDeviceProperties(QUERY.device().index());                \
    const int computeCapability = properties->major * 10 + properties->minor; \
    DISPATCH_BLOCKSIZE(                                                       \
        VALUE.size(-1), ([&]() {                                              \
          static constexpr int64_t kQueriesPerBlock = kIs64x64 ? 64 : 32;     \
          static constexpr int64_t kKeysPerBlock = kIs64x64 ? 64 : 128;       \
          DISPATCH_TYPES(                                                     \
              QUERY, ([&]() {                                                 \
                DISPATCH_ARCHTAG(                                             \
                    computeCapability, ([&]() {                               \
                      using AlignedAK = AttentionKernel<                      \
                          scalar_t,                                           \
                          ArchTag,                                            \
                          true,                                               \
                          kQueriesPerBlock,                                   \
                          kKeysPerBlock,                                      \
                          kSingleValueIteration>;                             \
                      /* Run a more efficient kernel (with `isAligned=True`)  \
                      if memory is correctly aligned*/                        \
                      bool isAligned =                                        \
                          (QUERY.stride(2) % AlignedAK::kAlignmentQ == 0 &&   \
                           KEY.stride(2) % AlignedAK::kAlignmentK == 0 &&     \
                           VALUE.stride(2) % AlignedAK::kAlignmentV == 0);    \
                      /* TODO: Should we warn or log somewhere when we use a  \
                      less efficient kernel due to wrong alignment? */        \
                      DISPATCH_BOOL(isAligned, kIsAligned, ([&]() {           \
                                      using Kernel = AttentionKernel<         \
                                          scalar_t,                           \
                                          ArchTag,                            \
                                          kIsAligned,                         \
                                          kQueriesPerBlock,                   \
                                          kKeysPerBlock,                      \
                                          kSingleValueIteration>;             \
                                      FUNC();                                 \
                                    }))                                       \
                    }))                                                       \
              }));                                                            \
        }));                                                                  \
  }


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
  auto sizes_dim1 = at::native::narrow_symint(sizes, 1, 0, 1);
  auto sizes_dim2 = at::native::narrow_symint(sizes, 1, 1, 1);

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
          at::native::narrow_symint(offsets, 0, sizes.numel() + 1, sizes.numel())
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

std::tuple<Tensor, Tensor> native_multi_head_attention_cuda(
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
      "expected 2-D `qkv_bias`, got ",
      qkv_bias.dim(),
      "-D tensor");
  TORCH_CHECK(
      qkv_bias.sizes()[0] == 3 * D,
      "expected `qkv_bias` first dim and first dim of query to be equal");
  TORCH_CHECK(D % num_head == 0, "`embed_dim` must divide evenly by `num_heads`");

#ifndef NDEBUG
  const auto B = query.is_nested()
      ? get_nested_tensor_impl(query)->get_nested_size_tensor().size(0)
      : query.sizes()[0];
  auto T = query.is_nested() ? 0 : query.sizes()[1];

#endif
  const auto dim_per_head = D / num_head;
  if ((query.is_same(key) && key.is_same(value)) && dim_per_head % 8 == 0 ) {

    // We have not done linear projection yet but the input for SDP
    // Is expected to be 4 dimensional. We "cheaply" create view tensors
    // That will then be used for checking hot path conditions with select_sd_backend
    auto q = query.view({query.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto k = key.view({key.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto v = value.view({value.size(0), -1, num_head, dim_per_head}).transpose(1, 2);

    sdp::sdp_params kernel_params{q, k, v, mask.has_value(), 0.0, need_weights, false};
    auto backend = select_sdp_backend(kernel_params);
    if (backend == sdp::SDPBackend::flash_attention || backend == sdp::SDPBackend::efficient_attention) {
      auto x = at::linear(query, qkv_weight, qkv_bias);
      auto chunks = x.chunk(3, -1);
      auto x_size_0 = x.size(0);

      chunks[0] = (chunks[0].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[1] = (chunks[1].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[2] = (chunks[2].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);

      auto y = at::_scaled_dot_product_attention(
          chunks[0], chunks[1], chunks[2], mask, 0.0, need_weights, false);
      auto past_sdp =
          std::get<0>(y).transpose(1, 2).reshape({x_size_0, -1, embed_dim});
      return std::make_tuple(
          at::linear(past_sdp, proj_weight, proj_bias), Tensor());
    }
    // Returned math or error lets not use it
  }

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

std::tuple<Tensor, Tensor> flash_attention_helper_dense_unpacked(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool need_atten_weights,
    bool is_causal) {
  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)
  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t head_dim = query.size(3);

  const int64_t max_seqlen_batch_k = key.size(2);
  const int64_t max_seqlen_batch_v = value.size(2);
  TORCH_CHECK(
      max_seqlen_batch_k == max_seqlen_batch_v,
      "Key and Value must have the same sequence length");

  // Query -> Query(Batch x Q_seq_len x Num_heads x Dim_per_head)
  // Key   -> Key(Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value -> Value(Batch x KV_seq_len x  Num_heads x Dim_per_head)
  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  Tensor cumulative_sequence_length_q = at::arange(
      0,
      (batch_size + 1) * max_seqlen_batch_q,
      max_seqlen_batch_q,
      TensorOptions().device(at::kCUDA).dtype(at::kInt));

  Tensor cumulative_sequence_length_k = at::arange(
      0,
      (batch_size + 1) * max_seqlen_batch_k,
      max_seqlen_batch_k,
      TensorOptions().device(at::kCUDA).dtype(at::kInt));

  int64_t Nnz_q{batch_size * max_seqlen_batch_q};
  int64_t Nnz_kv{batch_size * max_seqlen_batch_k};

  // For the standard MHA these will actually be views
  Tensor query_reshaped = q_t.reshape({Nnz_q, num_heads, head_dim});
  Tensor key_reshaped = k_t.reshape({Nnz_kv, num_heads, head_dim});
  Tensor value_reshaped = v_t.reshape({Nnz_kv, num_heads, head_dim});

  Tensor attention =
      at::_flash_scaled_dot_product_attention(
          query_reshaped,
          key_reshaped,
          value_reshaped,
          cumulative_sequence_length_q,
          cumulative_sequence_length_k,
          max_seqlen_batch_q,
          max_seqlen_batch_k,
          dropout_p,
          is_causal);
  // Reshape output to convert nnz to batch_size and seq_len
  attention =
      attention.view({batch_size, max_seqlen_batch_q, num_heads, head_dim}).transpose(1,2);

  return std::tuple<Tensor, Tensor>(attention, Tensor());
}
std::tuple<Tensor, Tensor> mem_eff_helper(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    bool compute_log_sumexp,
    bool is_causal) {
  // Query -> Query(Batch x Q_seq_len x Num_heads x Dim_per_head)
  // Key   -> Key(Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value -> Value(Batch x KV_seq_len x  Num_heads x Dim_per_head)
  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  Tensor attention, log_sumexp;
  std::tie(attention, log_sumexp) = at::_efficient_attention_forward(
      q_t,
      k_t,
      v_t,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      compute_log_sumexp,
      is_causal);
  attention = attention.transpose(1,2);
  return std::make_tuple(std::move(attention), std::move(log_sumexp));
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_forward_cuda(
        const Tensor& query_, const Tensor& key, const Tensor& value,
        const c10::optional<Tensor>& attn_mask_, double dropout_p, bool need_attn_weights, bool is_causal) {
    // Determine which efficient kernel to use
    sdp::sdp_params kernel_params{query_, key, value, attn_mask_.has_value(), dropout_p, need_attn_weights, is_causal};
    auto backend = select_sdp_backend(kernel_params);
    switch(backend){
      case sdp::SDPBackend::flash_attention:
          return flash_attention_helper_dense_unpacked(query_, key, value, dropout_p, need_attn_weights, is_causal);
      case sdp::SDPBackend::efficient_attention:
          return mem_eff_helper(query_, key , value, need_attn_weights, is_causal);
      case sdp::SDPBackend::math:
        return at::_scaled_dot_product_attention_math(query_, key, value, attn_mask_, dropout_p, need_attn_weights, is_causal);
      default:
        TORCH_CHECK(false, "No viable backend for scaled_dot_product_attention was found.");
        return std::make_tuple(Tensor(), Tensor());
    }
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
    bool is_causal) {
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
      is_causal,
      false,
      c10::nullopt);
  return output[0];
#endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.")
  return Tensor();
}

std::tuple<at::Tensor, at::Tensor> _efficient_attention_forward(
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const c10::optional<at::Tensor>& cu_seqlens_q,
    // (Mode 1MHK only) [b+1]: cu_seqlens_k[b] contains the
    // position of the first key token for batch $b
    const c10::optional<at::Tensor>& cu_seqlens_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const c10::optional<int64_t> max_seqlen_q_,
    bool compute_logsumexp,
    bool causal) {
#if defined(USE_FLASH_ATTENTION)
// TODO In theory it is possible to compile with _CUDA_ARCH < 5.0 and run on a
// machine that is >= 5.0. In practice, this is not a problem but since
// this would avoid runtime architecture checks, we should look into it
  TORCH_CHECK(query.dim() == 4);
  TORCH_CHECK(key.dim() == 4);
  TORCH_CHECK(value.dim() == 4);

  // Batch sizes
  TORCH_CHECK(query.size(0) == key.size(0));
  TORCH_CHECK(query.size(0) == value.size(0));

  // Sequence length
  TORCH_CHECK(key.size(1) == value.size(1));

  // Num heads
  TORCH_CHECK(query.size(2) == key.size(2));
  TORCH_CHECK(query.size(2) == value.size(2));

  // Embedding per head
  TORCH_CHECK(query.size(3) == key.size(3));

  int64_t max_seqlen_q = 0, max_seqlen_k=0;
  TORCH_CHECK(cu_seqlens_q.has_value() == cu_seqlens_k.has_value());
  if (cu_seqlens_q.has_value()) {
    TORCH_CHECK(cu_seqlens_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(cu_seqlens_q->dim() == 1 && cu_seqlens_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*cu_seqlens_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*cu_seqlens_k));
    TORCH_CHECK(cu_seqlens_q->size(0) == cu_seqlens_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q_.has_value());
    max_seqlen_q = *max_seqlen_q_;
    max_seqlen_k = 0; // Will be set inside the kernel
  } else {
    max_seqlen_q = query.size(1);
    max_seqlen_k = key.size(1);
  }

  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(query);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(key);
  CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(value);

  at::cuda::CUDAGuard device_guard(query.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int64_t B = query.size(0);
  int64_t M = query.size(1);
  int64_t N = key.size(1);
  int64_t num_heads = query.size(-2);
  int64_t K = query.size(-1);
  int64_t Kv = value.size(-1);

  at::Tensor res;
  at::Tensor logsumexp;

  auto launchKernel = [&](auto _k, int computeCapability) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    res = at::empty(
        {B, M, num_heads, Kv},
        query.options().dtype(
            TypeTraits<typename Kernel::output_t>::atScalarType()));

    // NOTE: Should be aligned (by padding) in case M is
    // not a good number for loading during backward
    constexpr decltype(M) kAlignLSE = Kernel::kAlignLSE;
    logsumexp = at::empty(
        {B,
         num_heads,
         compute_logsumexp ? ceil_div(max_seqlen_q, kAlignLSE) * kAlignLSE : 0},
        query.options().dtype(at::ScalarType::Float));

    typename Kernel::Params p;
    p.query_ptr = (scalar_t*)query.data_ptr();
    p.key_ptr = (scalar_t*)key.data_ptr();
    p.value_ptr = (scalar_t*)value.data_ptr();
    p.logsumexp_ptr = compute_logsumexp
        ? (typename Kernel::lse_scalar_t*)logsumexp.data_ptr()
        : nullptr;
    at::Tensor output_accum;
    if (Kernel::kNeedsOutputAccumulatorBuffer) {
      output_accum = at::empty(
          {B, M, num_heads, Kv},
          query.options().dtype(
              TypeTraits<typename Kernel::output_accum_t>::atScalarType()));
      p.output_accum_ptr =
          (typename Kernel::output_accum_t*)output_accum.data_ptr();
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.output_ptr = (typename Kernel::output_t*)res.data_ptr();

    if (cu_seqlens_q.has_value()) {
      p.cu_seqlens_q_ptr = (int32_t*)cu_seqlens_q->data_ptr();
      p.cu_seqlens_k_ptr = (int32_t*)cu_seqlens_k->data_ptr();
    }

#define ASSIGN_CHECK_OVERFLOW(A, B)                                            \
  {                                                                            \
    A = B;                                                                     \
    TORCH_CHECK(B < std::numeric_limits<decltype(A)>::max(), #B " overflows"); \
  }

    p.num_heads = num_heads;
    p.head_dim = query.size(3);
    p.head_dim_value = value.size(3);
    p.num_queries = max_seqlen_q;
    p.num_keys = max_seqlen_k;
    p.num_batches = cu_seqlens_q.has_value() ? cu_seqlens_q->size(0) - 1 : B;
    p.causal = causal;

    ASSIGN_CHECK_OVERFLOW(p.q_strideB, query.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.k_strideB, key.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.v_strideB, value.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.q_strideM, query.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.k_strideM, key.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.v_strideM, value.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.q_strideH, query.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.k_strideH, key.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.v_strideH, value.stride(2));

    constexpr auto kernel_fn = attention_kernel_batched<Kernel>;
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > 0xc000) {
      TORCH_INTERNAL_ASSERT(
          computeCapability >= 70,
          "This kernel requires too much shared memory on this machine!");
      AT_CUDA_CHECK(cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    }
    Kernel::check_supported(p);
    kernel_fn<<<p.getBlocksGrid(), p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };
  // Dispatch to the right kernel
  DISPATCH_KERNEL(query, key, value, ([&]() {
                    launchKernel(Kernel{}, computeCapability);
                  }));

  AT_CUDA_CHECK(cudaGetLastError());
  return std::make_tuple(res, logsumexp);
#endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.")
  return std::make_tuple(Tensor{}, Tensor{});
}

Tensor triton_scaled_dot_attention(const Tensor& q, const Tensor& k, const Tensor& v, double dropout_p){
  TORCH_CHECK(false, "This operator should be overridden in python before use");
  return at::Tensor();
}
} // namespace native
} // namespace at
