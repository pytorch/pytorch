#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <type_traits>

#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/NestedTensorImpl.h>
#include <ATen/TensorAccessor.h>
#include <ATen/TensorOperators.h>
#include <c10/util/Logging.h>
#include <c10/util/bit_cast.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/native/NonSymbolicBC.h>
#include <ATen/native/cuda/Loops.cuh>
#include <ATen/native/cuda/MemoryAccess.cuh>
#include <ATen/native/cuda/PersistentSoftmax.cuh>
#include <ATen/native/cuda/block_reduce.cuh>
#include <optional>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_cudnn_attention_forward.h>
#include <ATen/ops/_cudnn_attention_forward_native.h>
#include <ATen/ops/_efficient_attention_forward.h>
#include <ATen/ops/_efficient_attention_forward_native.h>
#include <ATen/ops/_fill_mem_eff_dropout_mask_native.h>
#include <ATen/ops/_flash_attention_forward.h>
#include <ATen/ops/_flash_attention_forward_native.h>
#include <ATen/ops/_fused_sdp_choice_native.h>
#include <ATen/ops/_masked_softmax.h>
#include <ATen/ops/_native_multi_head_attention_native.h>
#include <ATen/ops/scaled_dot_product_attention_native.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention.h>
#include <ATen/ops/_scaled_dot_product_efficient_attention_native.h>
#include <ATen/ops/_scaled_dot_product_flash_attention.h>
#include <ATen/ops/_scaled_dot_product_flash_attention_native.h>
#include <ATen/ops/_softmax.h>
#include <ATen/ops/_transform_bias_rescale_qkv.h>
#include <ATen/ops/_triton_multi_head_attention_native.h>
#include <ATen/ops/_triton_scaled_dot_attention.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/linear.h>
#include <ATen/ops/narrow_native.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <ATen/ops/split_native.h>
#include <ATen/ops/zeros.h>
#endif

#ifdef __HIP_PLATFORM_AMD__
#include <ATen/native/cudnn/hip/MHA.h>
#else
#include <ATen/native/cudnn/MHA.h>
#endif

#include <c10/cuda/CUDAMathCompat.h>

#include <ATen/native/transformers/attention.h>
#include <ATen/native/nested/NestedTensorUtils.h>
#include <ATen/native/nested/NestedTensorTransformerUtils.h>
#include <ATen/native/nested/NestedTensorTransformerFunctions.h>
#include <ATen/native/transformers/cuda/sdp_utils.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#ifdef USE_FLASH_ATTENTION
// FlashAttention Specific Imports
#include <ATen/native/transformers/cuda/flash_attn/flash_api.h>
#if !defined(__HIP_PLATFORM_AMD__)
#include <namespace_config.h>
#endif
#endif
#ifdef USE_MEM_EFF_ATTENTION
#ifndef USE_ROCM
// MemoryEfficient Attention Specific Imports for CUDA
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/kernels/cutlassF.h>
#include <ATen/native/transformers/cuda/mem_eff_attention/pytorch_utils.h>
#else
// MemoryEfficient Attention Specific Imports for ROCM
#ifndef DISABLE_AOTRITON
#include <ATen/native/transformers/hip/aotriton_adapter.h>
#include <aotriton/flash.h>
#include <aotriton/runtime.h>
#endif
#include <ATen/native/transformers/hip/flash_attn/ck/me_ck_api.h>
#endif
#endif

namespace at {

namespace cuda::philox {

__global__ void unpack_cudnn(at::PhiloxCudaState arg, int64_t* seed_ptr, int64_t* offset_ptr) {
  if (arg.captured_) {
    *seed_ptr = static_cast<int64_t>(*arg.seed_.ptr);
    *offset_ptr = static_cast<int64_t>(
                    *(arg.offset_.ptr) + static_cast<int64_t>(arg.offset_intragraph_));
  } else {
    *seed_ptr = static_cast<int64_t>(arg.seed_.val);
    *offset_ptr = static_cast<int64_t>(arg.offset_.val);
  }
}

void unpack_cudnn_wrapper(at::PhiloxCudaState arg, int64_t* seed_ptr, int64_t* offset_ptr, cudaStream_t stream) {
at::cuda::philox::unpack_cudnn<<<1, 1, 0, stream>>>(arg, seed_ptr, offset_ptr);
}

} // namespace cuda::philox

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
      ? get_nested_tensor_impl(qkv)->get_nested_sizes().size(0)
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
          auto sizes = collapse_dims_1_and_2(nt_qkv->get_nested_sizes());
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

#endif
  const auto dim_per_head = D / num_head;
  if ((query.is_same(key) && key.is_same(value)) && dim_per_head % 8 == 0 && !need_weights) {

    // We have not done linear projection yet but the input for SDP
    // Is expected to be 4 dimensional. We "cheaply" create view tensors
    // That will then be used for checking hot path conditions with select_sd_backend
    auto q = query.view({query.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto k = key.view({key.size(0), -1, num_head, dim_per_head}).transpose(1, 2);
    auto v = value.view({value.size(0), -1, num_head, dim_per_head}).transpose(1, 2);

    sdp::sdp_params kernel_params{q, k, v, mask, 0.0, false, false};
    auto backend = select_sdp_backend(kernel_params);
    // strides from packed projection for nested tensors when seq_len is 1 will be
    // and will trigger a contiguous call in the kernel, so we prevent this
    bool no_seq_len_1_nested = query.is_nested() ? check_for_seq_len_1_nested_tensor(kernel_params, false) : true;
    // The API for transformer_encoder is a mask of shape (Batch_Size, Seq_len_q)
    // For mem-eff attention this will cause the expand call to error
    // For now I am going to turn of that path not have to deal with all the annoying
    // Mask type shape grossness
    if (!mask.has_value() && no_seq_len_1_nested &&
        (backend == sdp::SDPBackend::flash_attention || backend == sdp::SDPBackend::efficient_attention ||
         backend == sdp::SDPBackend::cudnn_attention)) {
      auto x = at::linear(query, qkv_weight, qkv_bias);
      auto chunks = x.chunk(3, -1);
      auto x_size_0 = x.size(0);

      chunks[0] = (chunks[0].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[1] = (chunks[1].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      chunks[2] = (chunks[2].view({x_size_0, -1, num_head, dim_per_head}))
                      .transpose(1, 2);
      auto y = at::scaled_dot_product_attention(
          chunks[0], chunks[1], chunks[2], mask, 0.0, false, std::nullopt);

      auto past_sdp = y.transpose(1, 2).reshape({x_size_0, -1, embed_dim});
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
  auto [q, k, v] = _transform_bias_rescale_qkv(qkv, qkv_bias, num_head);
  qkv = Tensor(); // Not used any more, allow free
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
std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt, Tensor, Tensor, Tensor> _scaled_dot_product_flash_attention_cuda(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("torch.sdpa.flash_attention");
  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)

  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t max_seqlen_batch_k = key.size(2);
  const int64_t max_seqlen_batch_v = value.size(2);
  TORCH_CHECK(
      max_seqlen_batch_k == max_seqlen_batch_v,
      "Key and Value must have the same sequence length");

  // Query -> Query(Batch x Q_seq_len  x Num_heads x Dim_per_head)
  // Key   -> Key  (Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value -> Value(Batch x KV_seq_len x Num_heads x Dim_per_head)
  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  auto
      [output,
       logsumexp,
       philox_seed,
       philox_offset,
       debug_attn_mask] =
          at::_flash_attention_forward(
              q_t,
              k_t,
              v_t,
              std::nullopt,
              std::nullopt,
              max_seqlen_batch_q,
              max_seqlen_batch_k,
              dropout_p,
              is_causal,
              return_debug_mask,
              scale,
              std::nullopt,
              std::nullopt);
  // Reshape output to convert nnz to batch_size and seq_len
  Tensor attention = output.transpose(1,2);

  return std::make_tuple(attention, logsumexp, Tensor(), Tensor(), max_seqlen_batch_q, max_seqlen_batch_k, philox_seed, philox_offset, debug_attn_mask);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt, Tensor, Tensor, Tensor> _cudnn_attention_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_bias,
    const std::optional<Tensor>& cumulative_sequence_length_q,
    const std::optional<Tensor>& cumulative_sequence_length_kv,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_kv,
    bool compute_logsumexp,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  // TODO(eqy): debug mask support
  // Query (Batch x Num_heads x Q_seq_len  x Dim_per_head)
  // Key   (Batch x Num_heads x KV_seq_len x Dim_per_head)
  // Value (Batch x Num_heads x KV_seq_len x Dim_per_head)
  const bool is_nested = cumulative_sequence_length_q.has_value();
  if (!is_nested) {
    const int64_t batch_size = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t head_dim_qk = query.size(3);
    const int64_t head_dim_v = value.size(3);
    auto attn_bias_ = attn_bias;
    if (attn_bias_.has_value()) {
      const auto bias_dim = attn_bias_.value().dim();
      if (bias_dim == 2) {
        attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_kv});
      } else if (bias_dim == 3) {
        attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_kv});
      } else {
        TORCH_CHECK(bias_dim == 4, "cuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ", attn_bias_.value().dim(), "D");
        attn_bias_ = attn_bias_.value().expand({batch_size, attn_bias_.value().size(1), max_seqlen_batch_q, max_seqlen_batch_kv});
      }
    }

    Tensor attention, log_sumexp;
    at::Tensor cudnn_seed, cudnn_offset;
    cudnn_seed = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
    cudnn_offset = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));

    const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

    // See Note [Seed and Offset Device] in _efficient_attention_forward
    at::PhiloxCudaState philox_state;
    const bool in_capture_stream =
        at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None;
    if (use_dropout) {
      // Device
      auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
          std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);
      // if using dropout, we produce 1 random number for each element of the
      // attention tensor
      // TODO(eqy): should state be advanced per thread (local) amount or per call/launch (global) amount
      philox_state = gen->philox_cuda_state(batch_size * num_heads * max_seqlen_batch_q * max_seqlen_batch_kv);
      at::cuda::philox::unpack_cudnn_wrapper(
                                        philox_state, static_cast<int64_t*>(cudnn_seed.data_ptr()), static_cast<int64_t*>(cudnn_offset.data_ptr()), at::cuda::getCurrentCUDAStream());
    }

    const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
    Tensor debugmask;

    run_cudnn_SDP_fprop(batch_size/*int64_t b*/,
                        num_heads/*int64_t h*/,
                        max_seqlen_batch_q/*int64_t s_q*/,
                        max_seqlen_batch_kv/*int64_t s_kv*/,
                        head_dim_qk/*int64_t d_qk*/,
                        head_dim_v/*int64_t d_v*/,
                        softmax_scale/*float scaling_factor*/,
                        compute_logsumexp/* bool */,
                        is_causal/* bool */,
                        dropout_p/*double dropout_probability*/,
                        query/* Tensor q*/,
                        key/* Tensor k*/,
                        value/* Tensor v*/,
                        attn_bias_ /* std::optional<Tensor> */,
                        log_sumexp/*Tensor softmaxstats*/,
                        attention/*Tensor o*/,
                        cudnn_seed/*Tensor dropoutseed*/,
                        cudnn_offset/*Tensor dropoutoffset*/);

    // TODO(eqy): support debug_attn_mask
    return std::make_tuple(std::move(attention), std::move(log_sumexp), Tensor(), Tensor(), max_seqlen_batch_q, max_seqlen_batch_kv, std::move(cudnn_seed), std::move(cudnn_offset), Tensor());
  } else {
    //auto [
    //    query_buffer_reshaped,
    //    key_buffer_reshaped,
    //    value_buffer_reshaped,
    //    cumulative_sequence_length_q,
    //    cumulative_sequence_length_kv,
    //    max_seqlen_batch_q,
    //    max_seqlen_batch_kv,
    //    output_shape] = preprocessing::sdpa_nested_preprocessing(query, key, value);
    // C10_LOG_API_USAGE_ONCE("torch.sdpa.flash_attention_cudnn");
    // TODO(eqy): debug mask support
    // BHSD ...
    const int64_t batch_size = cumulative_sequence_length_q.value().size(0) - 1;
    const int64_t num_heads_q = query.size(-2);
    const int64_t num_heads_k = key.size(-2);
    const int64_t num_heads_v = value.size(-2);
    const int64_t head_dim_qk = query.size(-1);
    const int64_t head_dim_v = value.size(-1);
    auto attn_bias_ = attn_bias;
    if (attn_bias_.has_value()) {
      const auto bias_dim = attn_bias_.value().dim();
      if (bias_dim == 2) {
        attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_kv});
      } else if (bias_dim == 3) {
        attn_bias_ = attn_bias_.value().expand({batch_size, 1, max_seqlen_batch_q, max_seqlen_batch_kv});
      } else {
        attn_bias_ = attn_bias_.value().expand({batch_size, attn_bias_.value().size(1), max_seqlen_batch_q, max_seqlen_batch_kv});
        TORCH_CHECK(bias_dim == 4, "cuDNN SDPA expects either a 2D, 3D, or 4D attn_bias but got ", attn_bias_.value().dim(), "D");
      }
    }

    Tensor attention, log_sumexp;

    at::Tensor cudnn_seed, cudnn_offset;
    cudnn_seed = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));
    cudnn_offset = at::empty({}, at::dtype(at::kLong).device(at::kCUDA));

    const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

    // See Note [Seed and Offset Device] in _efficient_attention_forward
    at::PhiloxCudaState philox_state;
    const bool in_capture_stream =
        at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None;
    if (use_dropout) {
      // Device
      auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
          std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

      // See Note [Acquire lock when using random generators]
      std::lock_guard<std::mutex> lock(gen->mutex_);
      // if using dropout, we produce 1 random number for each element of the
      // attention tensor
      // TODO(eqy): should state be advanced per thread (local) amount or per call/launch (global) amount
      philox_state = gen->philox_cuda_state(batch_size * num_heads_q * max_seqlen_batch_q * max_seqlen_batch_kv);
      at::cuda::philox::unpack_cudnn_wrapper(philox_state, static_cast<int64_t*>(cudnn_seed.data_ptr()), static_cast<int64_t*>(cudnn_offset.data_ptr()), at::cuda::getCurrentCUDAStream());
    }

    const auto softmax_scale = sdp::calculate_scale(query, scale).as_float_unchecked();

    run_cudnn_SDP_fprop_nestedtensor(batch_size/*int64_t b*/,
                                     num_heads_q/*int64_t h*/,
                                     num_heads_k,
                                     num_heads_v,
                                     max_seqlen_batch_q/*int64_t s_q*/,
                                     max_seqlen_batch_kv/*int64_t s_kv*/,
                                     head_dim_qk/*int64_t d_qk*/,
                                     head_dim_v/*int64_t d_v*/,
                                     softmax_scale/*float scaling_factor*/,
                                     compute_logsumexp/* bool */,
                                     is_causal/* bool */,
                                     dropout_p/*double dropout_probability*/,
                                     cumulative_sequence_length_q.value(),
                                     cumulative_sequence_length_kv.value(),
                                     query/* Tensor q*/,
                                     key/* Tensor k*/,
                                     value/* Tensor v*/,
                                     attn_bias_ /* std::optional<Tensor> */,
                                     log_sumexp/*Tensor softmaxstats*/,
                                     attention/*Tensor o*/,
                                     cudnn_seed/*Tensor dropoutseed*/,
                                     cudnn_offset/*Tensor dropoutoffset*/);
    //attention = wrap_buffer(attention.view(-1), output_shape).transpose(1, 2);
    return std::make_tuple(std::move(attention), std::move(log_sumexp), cumulative_sequence_length_q.value(), cumulative_sequence_length_kv.value(), max_seqlen_batch_q, max_seqlen_batch_kv, std::move(cudnn_seed), std::move(cudnn_offset), Tensor());
  }
}

std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt, Tensor, Tensor, Tensor> _scaled_dot_product_cudnn_attention_cuda(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& attn_bias,
    bool compute_logsumexp,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale) {
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("torch.sdpa.flash_attention_cudnn");
  const int64_t max_seqlen_batch_q = query.size(2);
  const int64_t max_seqlen_batch_k = key.size(2);

  return at::_cudnn_attention_forward(query, key, value, attn_bias, std::nullopt, std::nullopt, max_seqlen_batch_q, max_seqlen_batch_k, compute_logsumexp, dropout_p, is_causal, return_debug_mask, scale);
}

std::tuple<Tensor, Tensor, Tensor, Tensor> _scaled_dot_product_efficient_attention_cuda(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<at::Tensor>& attn_bias,
    bool compute_log_sumexp,
    double dropout_p,
    bool is_causal,
    std::optional<double> scale) {
  // Used for tracking usage statistics
  C10_LOG_API_USAGE_ONCE("torch.sdpa.mem_efficient_attention");
  // Query -> Query(Batch x Q_seq_len x Num_heads x Dim_per_head)
  // Key   -> Key(Batch x KV_seq_len x Num_heads x Dim_per_head)
  // Value -> Value(Batch x KV_seq_len x  Num_heads x Dim_per_head)
  Tensor q_t = query.transpose(1, 2);
  Tensor k_t = key.transpose(1, 2);
  Tensor v_t = value.transpose(1, 2);

  sdp::CustomMaskType custom_mask_type = is_causal
      ? sdp::CustomMaskType::CausalFromTopLeft
      : sdp::CustomMaskType::NoCustomMask;

  auto [attention, log_sumexp, seed, offset, max_seqlen_batch_q, max_seqlen_batch_kv] = at::_efficient_attention_forward(
      q_t,
      k_t,
      v_t,
      attn_bias,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      std::nullopt,
      dropout_p,
      static_cast<int64_t>(custom_mask_type),
      compute_log_sumexp,
      scale);

  attention = attention.transpose(1, 2);
  return std::make_tuple(std::move(attention), std::move(log_sumexp), std::move(seed), std::move(offset));
}

int64_t _fused_sdp_choice_cuda(const Tensor& query_, const Tensor& key, const Tensor& value,
        const std::optional<Tensor>& attn_mask_, double dropout_p, bool is_causal, std::optional<double> scale, bool enable_gqa){
  sdp::sdp_params kernel_params{query_, key, value, attn_mask_, dropout_p, is_causal, enable_gqa};
  auto backend = select_sdp_backend(kernel_params);
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the fused kernels.");
  }
  return static_cast<int64_t>(backend);
}

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor>
_flash_attention_forward(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    const std::optional<Tensor>& cumulative_sequence_length_q,
    const std::optional<Tensor>& cumulative_sequence_length_k,
    int64_t max_seqlen_batch_q,
    int64_t max_seqlen_batch_k,
    double dropout_p,
    bool is_causal,
    bool return_debug_mask,
    std::optional<double> scale,
    std::optional<int64_t> window_size_left,
    std::optional<int64_t> window_size_right,
    const std::optional<Tensor>& _seqused_k,
    const std::optional<Tensor>& _alibi_slopes
    ) {
#if defined(USE_FLASH_ATTENTION)
  const auto softmax_scale =
      sdp::calculate_scale(query, scale).expect_float();
  std::optional<Tensor> out = std::nullopt;

  std::optional<Tensor> seqused_k = _seqused_k;
  std::optional<at::Tensor> block_table = std::nullopt;  // we are not using the block table yet
  std::optional<Tensor> alibi_slopes = _alibi_slopes;
  const float softcap = 0.0;

  const int non_null_window_left = window_size_left.has_value() ? window_size_left.value() : -1;
  const int non_null_window_right = window_size_right.has_value() ? window_size_right.value() : -1;

  // We are going to have two paths:
  // 1. The standard MHA path for dense tensors
  // 2. The Varseqlen path
  TORCH_CHECK(
      cumulative_sequence_length_q.has_value() ==
          cumulative_sequence_length_k.has_value(),
      "cumulative_sequence_length_q and cumulative_sequence_length_k must be both set or both not set");
  Tensor output, q_padded, k_padded, v_padded, logsumexp, output_shape,
      philox_seed, philox_offset, debug_attn_mask;
  if (cumulative_sequence_length_q.has_value()) {
    std::tie(
        output,
        q_padded,
        k_padded,
        v_padded,
        logsumexp,
        philox_seed,
        philox_offset,
        debug_attn_mask) =
        FLASH_NAMESPACE::mha_varlen_fwd(
            query,
            key,
            value,
            out,
            cumulative_sequence_length_q.value(),
            cumulative_sequence_length_k.value(),
            seqused_k, /*seqused_k*/
            block_table, /*block_table*/
            alibi_slopes, /*alibi_slopes*/
            max_seqlen_batch_q,
            max_seqlen_batch_k,
            dropout_p,
            softmax_scale,
            false /*zero_tensors*/,
            is_causal,
            non_null_window_left,
            non_null_window_right,
            softcap,
            return_debug_mask,
            std::nullopt /*gen_*/);
  } else {
    std::tie(
        output,
        q_padded,
        k_padded,
        v_padded,
        logsumexp,
        philox_seed,
        philox_offset,
        debug_attn_mask) =
        FLASH_NAMESPACE::mha_fwd(
            query,
            key,
            value,
            out,
            alibi_slopes,
            dropout_p,
            softmax_scale,
            is_causal,
            non_null_window_left,
            non_null_window_right,
            softcap,
            return_debug_mask, /*return_softmax (this is used for testing)*/
            std::nullopt);
  }
  debug_attn_mask =
      return_debug_mask ? debug_attn_mask : at::empty({0}, query.options());
  return std::make_tuple(
      std::move(output),
      std::move(logsumexp),
      std::move(philox_seed),
      std::move(philox_offset),
      std::move(debug_attn_mask));

#endif
  TORCH_CHECK(false, "USE_FLASH_ATTENTION was not enabled for build.")
  return std::make_tuple(
      Tensor(),
      Tensor(),
      Tensor(),
      Tensor(),
      Tensor());
}

std::tuple<Tensor, Tensor, Tensor, Tensor, c10::SymInt, c10::SymInt> _efficient_attention_forward(
    const at::Tensor& query, // [b, seqlen, num_heads, K]
    const at::Tensor& key, // [b, seqlen, num_heads, K]
    const at::Tensor& value, // [b, seqlen, num_heads, Kv]
    const std::optional<at::Tensor>& bias, // [b, num_heads, seqlen, seqlen]
    // (Mode 1MHK only) [b+1]: cu_seqlens_q[b] contains the
    // position of the first query token for batch $b
    const std::optional<at::Tensor>& seqstart_q,
    // (Mode 1MHK only) [b+1]: cu_seqlen_k[b] contains the
    // position of the first key token for batch $b
    const std::optional<at::Tensor>& seqstart_k,
    // (Mode 1MHK only) Maximum sequence length across batches
    const std::optional<int64_t> max_seqlen_q_,
    const std::optional<int64_t> max_seqlen_k_,
    double dropout_p, // attention matrix dropout probability
    int64_t custom_mask_type,
    bool compute_logsumexp,
    std::optional<double> scale,
    const std::optional<at::Tensor>& seqlen_k,
    const std::optional<int64_t> window_size) {
#if defined(USE_MEM_EFF_ATTENTION)
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

  int64_t max_seqlen_q = 0, max_seqlen_k = 0;
  TORCH_CHECK(seqstart_q.has_value() == seqstart_k.has_value());
  if (seqstart_q.has_value()) {
    TORCH_CHECK(seqstart_q->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_k->scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(seqstart_q->dim() == 1 && seqstart_k->dim() == 1);
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_q));
    CHECK_NOSPARSE_CONTIGUOUS_CUDA((*seqstart_k));
    TORCH_CHECK(seqstart_q->size(0) == seqstart_k->size(0));
    TORCH_CHECK(query.size(0) == 1, "cu_seqlen only supports batch_size=1");
    TORCH_CHECK(max_seqlen_q_.has_value());
    max_seqlen_q = *max_seqlen_q_;
    max_seqlen_k = 0; // TODO: is this actually being set inside the kernel anywhere?
                      // see https://github.com/pytorch/pytorch/issues/115590s
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
  at::Tensor seed_t, offset_t;

  const bool use_dropout = std::fpclassify(dropout_p) != FP_ZERO;

  // Note [Seed and Offset Device]
  // If we are currently in graph capture mode, we need to create the seed and offset tensors on the device.
  // This is necessary for CUDA graph-safe random number generation, which requires the seed and offset tensors
  // to be single element tensors on device. During graph capture, when the seed and offset tensors are passed
  // the pointers act as scratch space for storing the RNG state for the backwards pass.
  // When calling backwards, we either construct a PhiloxState with the pointers or the actual values.
  // For more information on CUDA graph-safe RNG states, see Note [CUDA Graph-safe RNG states].

  at::PhiloxCudaState philox_state;
  const bool in_capture_stream =
      at::cuda::currentStreamCaptureStatus() != at::cuda::CaptureStatus::None;
  auto device = in_capture_stream ? at::kCUDA : at::kCPU;
  if (use_dropout) {
    auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
        std::nullopt, at::cuda::detail::getDefaultCUDAGenerator());

    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen->mutex_);
    // if using dropout, we produce 1 random number for each element of the
    // attention tensor
    philox_state = gen->philox_cuda_state(B * num_heads * M * N);

    if (in_capture_stream) {
      // The seed and offset will be populated by the kernel
      seed_t = at::empty({}, at::dtype(at::kLong).device(device));
      offset_t = at::empty({}, at::dtype(at::kLong).device(device));
    } else {
      auto [seed, offset] = at::cuda::philox::unpack(philox_state);
#ifdef USE_ROCM
      const auto options = at::dtype(at::kLong).device(at::kCUDA);
#else
      const auto options = at::dtype(at::kLong);
#endif
      seed_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(seed)), options);
      offset_t = at::scalar_tensor(at::Scalar(static_cast<int64_t>(offset)), options);
    }
  } else {
    // Not using dropout
    seed_t = at::empty({}, at::dtype(at::kLong).device(device));
    offset_t = at::empty({}, at::dtype(at::kLong).device(device));
  }

#ifdef USE_ROCM
  // ROCM Implementation

  // Need this in both aot and CK case
  const auto softmax_scale = sdp::calculate_scale(query, scale).expect_float();
  res = at::empty({B, M, num_heads, Kv}, query.options());

  if(at::globalContext().getROCmFAPreferredBackend() ==
    at::ROCmFABackend::Ck) {

#if defined(USE_CK_FLASH_ATTENTION)
    std::optional<Tensor> out(res);
    std::optional<Tensor> seqused_k = std::nullopt;
    std::optional<Tensor> alibi_slopes = std::nullopt;
    auto
        [out_,
         q,
         k,
         v,
         lse,
         seed_t,
         offset_t,
         p] =
            pytorch_flash::mem_eff_forward_ck(
                                    query,
                                    key,
                                    value,
                                    dropout_p,
                                    false,                                // return dropout_randval
                                    custom_mask_type == 0 ? false : true, // is_causal
                                    softmax_scale,
                                    bias,
                                    out,
                                    std::nullopt,                         // cu_seqlens_q
                                    std::nullopt,                         // cu_seqlens_k
                                    seqstart_q,
                                    seqstart_k,
                                    std::nullopt,                         // gen_
                                    seqused_k);                           // seqused_k_

    logsumexp = lse;
#else
    TORCH_CHECK(false, "Attempting to use CK mem_eff_forward backend in a build that has not built CK");
#endif
  } else { // use aotriton
#ifndef DISABLE_AOTRITON
    auto ret = aotriton::v2::flash::check_gpu(stream);
    if (hipSuccess != ret) {
        TORCH_CHECK(false,
                  "[AOTriton] Accelerated SDPA only supports MI200/MI300X/Navi31 GPUs"
                  " (gfx90a:sramecc+:xnack-/gfx942:sramecc+:xnack-/gfx1100)")
    }

    // AOTriton may accept aligned on logsumexp tensor in the future for better
    // performance, but for now it requires compact logsumexp tensor, even if
    // compute_logsumexp is false
    constexpr int kAlignLSE = 1;
    res = at::empty({B, M, num_heads, Kv}, query.options());
    at::Tensor softmax_lse;
    logsumexp = at::empty(
      { B, num_heads, compute_logsumexp ? max_seqlen_q : 0},
      query.options().dtype(at::ScalarType::Float));
    if (compute_logsumexp) {
      softmax_lse = logsumexp.view({B * num_heads, max_seqlen_q});
    }
    at::Tensor q_t = query.transpose(1, 2);
    at::Tensor k_t = key.transpose(1, 2);
    at::Tensor v_t = value.transpose(1, 2);
    at::Tensor output_t = res.transpose(1, 2);
    bool is_causal;
    if (static_cast<int64_t>(sdp::CustomMaskType::CausalFromTopLeft) == custom_mask_type) {
      is_causal = true;
    } else if (static_cast<int64_t>(sdp::CustomMaskType::NoCustomMask) == custom_mask_type) {
      is_causal = false;
    } else {
      TORCH_CHECK(false, "[_efficient_attention_forward] Unsupported mask type on ROCM, for now");
    }

    at::Tensor atomic_counter;
    if (is_causal) {
      atomic_counter = at::zeros({1}, query.options().dtype(at::kInt));
    }

    using aotriton::v2::flash::attn_fwd;
    using aotriton::v2::flash::attn_fwd_compact_varlen;
    using sdp::aotriton_adapter::mk_aotensor;
    using sdp::aotriton_adapter::mk_aoscalartensor;
    using sdp::aotriton_adapter::mk_philoxtensor;
    using sdp::aotriton_adapter::mk_atomictensor;
    aotriton::TensorView<4> empty_t4(0, {0, 0, 0, 0}, {0, 0, 0, 0}, aotriton::DType::kFloat16);
    aotriton::TensorView<2> empty_t2(0, {0, 0}, {0, 0}, aotriton::DType::kFloat32);
    at::Tensor softmax_fa_t = at::empty({ 0, 0, 0, 0 }, query.options());
    const bool use_philox_state = in_capture_stream;
    auto seed = use_philox_state ? mk_philoxtensor(philox_state.seed_.ptr) : mk_aoscalartensor(seed_t);
    auto offset1 = use_philox_state ? mk_philoxtensor(philox_state.offset_.ptr) : mk_aoscalartensor(offset_t);
    auto offset2 = use_philox_state ? philox_state.offset_intragraph_ : 0;
    auto seed_output = mk_philoxtensor(use_philox_state ? seed_t.data_ptr<int64_t>() : nullptr);
    auto offset_output = mk_philoxtensor(use_philox_state ? offset_t.data_ptr<int64_t>() : nullptr);
    auto persistent_counter = mk_atomictensor(is_causal ? atomic_counter.data_ptr<int32_t>() : nullptr);
    hipError_t err; // TODO: Error handling
    if (seqstart_q.has_value()) {
      // varlen aka nested tensor
      err = attn_fwd_compact_varlen(mk_aotensor(q_t, "q"),
                                    mk_aotensor(k_t, "k"),
                                    mk_aotensor(v_t, "v"),
                                    bias.has_value() ? mk_aotensor(bias.value(), "bias"): empty_t4,
                                    mk_aotensor<1>(seqstart_q.value(), "cu_seqlens_q"),
                                    mk_aotensor<1>(seqstart_k.value(), "cu_seqlens_k"),
                                    max_seqlen_q,
                                    max_seqlen_k,
                                    softmax_scale,
                                    compute_logsumexp ? mk_aotensor<2>(softmax_lse, "M") : empty_t2,
                                    mk_aotensor(output_t, "Out"),
                                    dropout_p,
                                    seed,
                                    offset1,
                                    offset2,
                                    seed_output,
                                    offset_output,
                                    mk_aotensor(softmax_fa_t, "encoded_softmax"),
                                    is_causal,
                                    persistent_counter,
                                    stream);
    } else {
      err = attn_fwd(mk_aotensor(q_t, "q"),
                     mk_aotensor(k_t, "k"),
                     mk_aotensor(v_t, "v"),
                     bias.has_value() ? mk_aotensor(bias.value(), "bias"): empty_t4,
                     softmax_scale,
                     compute_logsumexp ? mk_aotensor<2>(softmax_lse, "M") : empty_t2,
                     mk_aotensor(output_t, "Out"),
                     dropout_p,
                     seed,
                     offset1,
                     offset2,
                     seed_output,
                     offset_output,
                     mk_aotensor(softmax_fa_t, "encoded_softmax"),
                     is_causal,
                     persistent_counter,
                     stream);
    }
#else
    TORCH_CHECK(false, "Attempting to use AOTriton mem_eff_forward backend in a build that has not built AOTriton");
#endif
  } // CK BACKEND
#else
  // CUDA Implementation
  cudaDeviceProp* p = at::cuda::getDeviceProperties(query.device().index());
  int computeCapability = p->major * 10 + p->minor;
  if (computeCapability == 121) {
    computeCapability = 120;
  }

  bool kernel_launched = false;
  const auto maxShmem = p->sharedMemPerBlockOptin;

  auto launchKernel = [&](auto _k, auto kernel_fn) {
    using Kernel = decltype(_k);
    using scalar_t = typename Kernel::scalar_t;
    (void)_k;

    if (kernel_launched) {
      return;
    }
    // Check if this kernel is compatible
    if (!Kernel::kSupportsDropout && use_dropout) {
      return;
    }
    if (!Kernel::kSupportsBias && bias.has_value()) {
      return;
    }

    if (value.size(3) > Kernel::kMaxK || key.size(3) > Kernel::kMaxK) {
      return;
    }
    // Alignment
    if ((query.stride(2) % Kernel::kAlignmentQ) ||
        (key.stride(2) % Kernel::kAlignmentK) ||
        (value.stride(2) % Kernel::kAlignmentV)) {
      return;
    }
    // Uses too much shmem
    size_t smem_bytes = sizeof(typename Kernel::SharedStorage);
    if (smem_bytes > maxShmem) {
      return;
    }
    kernel_launched = true;

    res = at::empty(
        {B, M, num_heads, Kv},
        query.options().dtype(
            CutlassToAtenDtype<typename Kernel::output_t>::atScalarType()));

    // NOTE: Should be aligned (by padding) in case M is
    // not a good number for loading during backward
    constexpr decltype(M) kAlignLSE = Kernel::kAlignLSE;
    logsumexp = at::empty(
        {seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B,
         num_heads,
         compute_logsumexp ? ceil_div(max_seqlen_q, kAlignLSE) * kAlignLSE : 0},
        query.options().dtype(at::ScalarType::Float));
    typename Kernel::Params p;
    p.query_ptr = (const scalar_t*)query.const_data_ptr();
    p.key_ptr = (const scalar_t*)key.const_data_ptr();
    p.value_ptr = (const scalar_t*)value.const_data_ptr();
    p.logsumexp_ptr = compute_logsumexp
        ? (typename Kernel::lse_scalar_t*)logsumexp.data_ptr()
        : nullptr;
    at::Tensor output_accum;
    if (Kernel::kNeedsOutputAccumulatorBuffer) {
      output_accum = at::empty(
          {B, M, num_heads, Kv},
          query.options().dtype(
              CutlassToAtenDtype<
                  typename Kernel::output_accum_t>::atScalarType()));
      p.output_accum_ptr =
          (typename Kernel::output_accum_t*)output_accum.data_ptr();
    } else {
      p.output_accum_ptr = nullptr;
    }
    p.output_ptr = (typename Kernel::output_t*)res.data_ptr();

    if (seqstart_q.has_value()) {
      p.seqstart_q_ptr = (const int32_t*)seqstart_q->const_data_ptr();
      p.seqstart_k_ptr = (const int32_t*)seqstart_k->const_data_ptr();
    }

    p.num_heads = num_heads;
    p.head_dim = query.size(3);
    p.head_dim_value = value.size(3);
    p.num_queries = max_seqlen_q;
    p.num_keys = max_seqlen_k;
    p.num_batches = seqstart_q.has_value() ? seqstart_q->size(0) - 1 : B;
    p.custom_mask_type = custom_mask_type;

    p.seqlen_k_ptr = nullptr;
    if (seqlen_k.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA(seqlen_k.value());
      TORCH_CHECK(seqlen_k->scalar_type() == at::ScalarType::Int);
      p.seqlen_k_ptr = (const int32_t*)seqlen_k->const_data_ptr();
    }
    if (window_size.has_value()) {
      p.window_size = *window_size;
    }
    p.scale = sdp::calculate_scale(query, scale).expect_float();

    ASSIGN_CHECK_OVERFLOW(p.q_strideB, query.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.k_strideB, key.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.v_strideB, value.stride(0));
    ASSIGN_CHECK_OVERFLOW(p.q_strideM, query.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.k_strideM, key.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.v_strideM, value.stride(1));
    ASSIGN_CHECK_OVERFLOW(p.q_strideH, query.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.k_strideH, key.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.v_strideH, value.stride(2));
    ASSIGN_CHECK_OVERFLOW(p.o_strideM, res.stride(1));

    if (bias.has_value()) {
      CHECK_NOSPARSE_LASTCONTIGUOUS_CUDA((*bias));
      TORCH_CHECK(
          bias->scalar_type() == CutlassToAtenDtype<scalar_t>::atScalarType(),
          "invalid dtype for bias - should match query's dtype");
      p.attn_bias_ptr = (const scalar_t*)bias->const_data_ptr();

      TORCH_CHECK(bias->dim() == 4, "Bias expected in BMHK format");
      TORCH_CHECK(
          bias->size(0) == query.size(0),
          "attn_bias: wrong shape (batch dimension)");
      TORCH_CHECK(
          bias->size(1) == query.size(2),
          "attn_bias: wrong shape (head dimension)");
      TORCH_CHECK(
          bias->size(2) == query.size(1),
          "attn_bias: wrong shape (seqlenQ dimension)");
      TORCH_CHECK(
          bias->size(3) == key.size(1),
          "attn_bias: wrong shape (seqlenKV dimension)");
      ASSIGN_CHECK_OVERFLOW(p.bias_strideB, bias->stride(0));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideH, bias->stride(1));
      ASSIGN_CHECK_OVERFLOW(p.bias_strideM, bias->stride(2));
      TORCH_CHECK(
          bias->stride(3) == 1,
          "attn_bias: wrong alignment (last dimension must be contiguous)");
    }

    p.use_dropout = use_dropout;
    if (p.use_dropout) {
      p.rng_engine_inputs = philox_state;
      p.dropout_prob = dropout_p;
      p.seed = seed_t.data_ptr<int64_t>();
      p.extragraph_offset = offset_t.data_ptr<int64_t>();
    }

    if (smem_bytes > 0xc000) {
      auto err = cudaFuncSetAttribute(
          kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
      TORCH_CHECK(
          err != cudaErrorInvalidValue,
          "This GPU does not have enough shared-memory (kernel requires ",
          smem_bytes / 1024,
          " kb)");
      AT_CUDA_CHECK(err);
    }
    auto blocks = p.getBlocksGrid();
    if (blocks.x * blocks.y * blocks.z == 0 || key.size(1) == 0) {
      res.zero_();
      return;
    }
    Kernel::check_supported(p);
    kernel_fn<<<blocks, p.getThreadsGrid(), smem_bytes, stream>>>(p);
  };

  // Dispatch to the right kernel
  DISPATCH_TYPES(query, ([&]() {
                   dispatch_cutlassF<scalar_t>(launchKernel, computeCapability);
                 }));
  TORCH_CHECK(kernel_launched, "cutlassF: no kernel found to launch!");
  AT_CUDA_CHECK(cudaGetLastError());

#endif // USE_ROCM
  return std::make_tuple(
      std::move(res),
      std::move(logsumexp),
      std::move(seed_t),
      std::move(offset_t),
      max_seqlen_q,
      // TODO: why isn't this being set in the kernel?
      max_seqlen_k_.has_value() ? max_seqlen_k_.value() : max_seqlen_k);
#endif
  TORCH_CHECK(false, "USE_MEM_EFF_ATTENTION was not enabled for build.")
  return std::make_tuple(Tensor{}, Tensor{}, Tensor{}, Tensor{}, 0, 0);
}

Tensor triton_scaled_dot_attention(const Tensor& q, const Tensor& k, const Tensor& v, double dropout_p){
  TORCH_CHECK(false, "This operator should be overridden in python before use");
  return at::Tensor();
}

REGISTER_CUDA_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_cuda)

#if defined(USE_MEM_EFF_ATTENTION) and !defined(USE_ROCM)
namespace {
/**
 * simple kernel that populates a tensor with rand uniform values.
 * currently only used for testing purposes, not much attention
 * is paid to performance.
 *
 * problem is partitioned as follows:
 * - (batch, head) is given by block coordinates
 * - each thread handles a row for a given (batch, head)
 */
template <typename mask_t>
__global__ void rand_uniform_kernel(
    int64_t n_heads,
    int64_t n_queries,
    int64_t n_keys,
    float dropout_prob,
    at::PhiloxCudaState rng_engine_inputs,
    mask_t* mask_out,
    int64_t mask_numel) {
  const int64_t batch_id = blockIdx.x;
  const int64_t head_id = blockIdx.y;
  const int64_t query_idx = threadIdx.x;

  const auto [seed, offset] = at::cuda::philox::unpack(rng_engine_inputs);

  const int dropout_seq_start = batch_id * (n_heads * n_queries * n_keys) +
      head_id * (n_queries * n_keys);
  const int64_t query_start_idx = query_idx * n_keys;

  curandStatePhilox4_32_10_t curand_state;
  curand_init(
      seed,
      0,
      offset + dropout_seq_start + query_start_idx,
      &curand_state);

  for (int key_start_idx = 0; key_start_idx < n_keys; key_start_idx += 4) {
    float4 rand_quad = curand_uniform4(&curand_state);

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t linear_idx = dropout_seq_start + query_start_idx + key_start_idx + i;
      if (linear_idx < mask_numel) {
        mask_out[linear_idx] = (&rand_quad.x)[i];
      }
    }
  }
}
} // namespace
#endif // defined(USE_MEM_EFF_ATTENTION) and !defined(USE_ROCM)
/**
 * fill tensor with random uniform values. only used for testing, not much
 * attention is paid to performance
 */
at::Tensor& _fill_mem_eff_dropout_mask_(
    Tensor& self,
    double dropout_p,
    const int64_t seed,
    const int64_t offset) {
  TORCH_CHECK(self.is_contiguous());
  TORCH_CHECK(self.dtype() == at::ScalarType::Float);
  const int64_t batch_sz = self.size(0);
  const int64_t n_heads = self.size(1);
  const int64_t n_queries = self.size(2);
  const int64_t n_keys = self.size(3);
#if defined(USE_MEM_EFF_ATTENTION)

#ifdef USE_ROCM
#ifndef DISABLE_AOTRITON
  using aotriton::v2::flash::debug_simulate_encoded_softmax;
  using sdp::aotriton_adapter::mk_aotensor;
  using sdp::aotriton_adapter::mk_aoscalartensor;
  at::cuda::CUDAGuard device_guard(self.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  at::Tensor seed_t, offset_t;
  const auto options = at::dtype(at::kLong).device(at::kCUDA);
  seed_t = at::scalar_tensor(at::Scalar(seed), options);
  offset_t = at::scalar_tensor(at::Scalar(offset), options);
  hipError_t err; // TODO: Error handling

  err = debug_simulate_encoded_softmax(mk_aotensor(self, "r"),
                                       dropout_p,
                                       mk_aoscalartensor(seed_t),
                                       mk_aoscalartensor(offset_t),
                                       0,
                                       stream);
#else
  TORCH_CHECK(false, "_fill_mem_eff_dropout_mask_ is only enabled with aotriton");
#endif
#else
  at::PhiloxCudaState rng_engine_inputs;
  rng_engine_inputs = at::PhiloxCudaState(seed, offset);
  at::cuda::CUDAGuard device_guard(self.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  rand_uniform_kernel<float><<<dim3(batch_sz, n_heads), n_queries, 0, stream>>>(
      n_heads,
      n_queries,
      n_keys,
      dropout_p,
      rng_engine_inputs,
      reinterpret_cast<float*>(self.data_ptr()),
      self.numel());
#endif

  return self;
#endif
  TORCH_CHECK(false, "USE_MEM_EFF_ATTENTION was not enabled for build.")
  return self;
}

} // namespace native
} // namespace at
