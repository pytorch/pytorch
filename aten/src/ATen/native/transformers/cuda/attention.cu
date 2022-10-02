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
} // namespace native
} // namespace at
