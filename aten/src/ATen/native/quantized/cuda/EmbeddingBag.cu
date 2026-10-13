#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

// BEGIN QUANTIZE HELPER FUNCTIONS
// Get the bit field of [pos, pos+len) bits from val, i.e.
// (val >> pos) & ((1u << len) - 1u), and return it as a float.
__device__ __forceinline__ float bfe(uint32_t val, uint32_t pos, uint32_t len) {
  uint32_t ret;
#ifdef USE_ROCM
  // Bitwise-AND (`&`), not logical-AND (`&&`): the logical form yields a bool,
  // which the previous implementation reinterpret_cast to float* and
  // dereferenced, faulting on ROCm.
  ret = (val >> pos) & ((1u << len) - 1u);
#else
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
#endif
  return __uint2float_rn(ret);
}

// FMA with constant scale/bias for all 4 floats in fa
__forceinline__ __device__ float4
fma4sb(const float4 fa, const float fscale, const float fbias) {
  float4 res;
#ifdef USE_ROCM
  res.x = fa.x * fscale + fbias;
  res.y = fa.y * fscale + fbias;
  res.z = fa.z * fscale + fbias;
  res.w = fa.w * fscale + fbias;
#else
  res.x = fmaf(fa.x, fscale, fbias);
  res.y = fmaf(fa.y, fscale, fbias);
  res.z = fmaf(fa.z, fscale, fbias);
  res.w = fmaf(fa.w, fscale, fbias);
#endif
  return res;
}

template <uint8_t bits_per_dim>
__forceinline__ __device__ float4
dequantize_intx(uint32_t packedVals, float2 scale_bias, uint8_t offset_bits) {
  float4 res;

  res.x = bfe(packedVals, offset_bits + (0 * bits_per_dim), bits_per_dim);
  res.y = bfe(packedVals, offset_bits + (1 * bits_per_dim), bits_per_dim);
  res.z = bfe(packedVals, offset_bits + (2 * bits_per_dim), bits_per_dim);
  res.w = bfe(packedVals, offset_bits + (3 * bits_per_dim), bits_per_dim);

  return fma4sb(res, scale_bias.x, scale_bias.y);
}

template <uint8_t bits_per_dim>
__forceinline__ __device__ void
accumulate_packed_intx(float4* acc, uint32_t packedVals, float2 scale_bias, float sample_weight) {
  constexpr uint8_t dims_per_byte = 8 / bits_per_dim;
  for (uint8_t i = 0; i < dims_per_byte; i++) {
    float4 res = dequantize_intx<bits_per_dim>(packedVals, scale_bias, 4 * bits_per_dim * i /* offset_bits */);
    // Accumulate in float32.
    acc[i].x += (res.x * sample_weight);
    acc[i].y += (res.y * sample_weight);
    acc[i].z += (res.z * sample_weight);
    acc[i].w += (res.w * sample_weight);
  }
}

// END QUANTIZE HELPER FUNCTIONS

// UN-OPTIMIZED kernel, doesn't even avoid warp divergence!
template <typename index_t, uint8_t bits_per_dim>
__global__ void embedding_bag_nbits_rowwise_offsets_kernel(
    const PackedTensorAccessor64<uint8_t, 2, RestrictPtrTraits> weight,
    const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
    const bool /* pruned_weights */,
    const PackedTensorAccessor32<float, 1, RestrictPtrTraits> per_sample_weights_,
    const std::optional<Tensor>& compressed_indices_mapping,
    const bool include_last_offset,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> output) {
  static_assert(bits_per_dim == 4 || bits_per_dim == 8, "the current embedding_bag_nbits_rowwise_offsets_kernel only has been tested for 4 and 8 bits per dim");
  constexpr uint8_t dims_per_byte = 8 / bits_per_dim;
  constexpr bool fp32_scale_bias = bits_per_dim == 8;

  int32_t B = output.size(0);
  int32_t D = output.size(1);
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (b_t >= B * D) {
    return;
  }
  int32_t t = b_t / B;
  int32_t b = b_t % B;

  const int32_t D_bytes = weight.size(1);

  bool use_per_sample = per_sample_weights_.size(0) > 0;

  int64_t indices_start = offsets[t * B + b];
  int64_t indices_end;
  if (include_last_offset) {
    indices_end = offsets[t * B + b + 1];
  } else {
    indices_end = (t * B + b + 1) < offsets.size(0) ? offsets[t * B + b + 1]
                                                    : indices.size(0);
  }

  int32_t L = indices_end - indices_start;
  const uint8_t* __restrict__ weights = &weight[0][0];

  if (L == 0) {
    for (int32_t d = 0; d < D; d += 4) {
      *(float4*)(&output[b][d]) = make_float4(0, 0, 0, 0);
    }
    return;
  }


  float4 accumulator[dims_per_byte];
  int32_t byte_offset = 0;
  for (int32_t d = 0; d < D; d += dims_per_byte * 4, byte_offset += 4) {
    for (int32_t i = 0; i < dims_per_byte; ++i) {
        accumulator[i] = make_float4(0, 0, 0, 0);
    }
    for (int32_t l = indices_start; l < indices_end; ++l) {
      int64_t idx = indices[l];
      float sample_weight = use_per_sample ? per_sample_weights_[l] : 1.0f;
      const uint8_t* __restrict__ row = &weights[idx * D_bytes];
      float2 scale_bias;
      if (fp32_scale_bias) {
        scale_bias = make_float2(
            reinterpret_cast<const float*>(&row[D_bytes - 8])[0],
            reinterpret_cast<const float*>(&row[D_bytes - 4])[0]);
      } else {
        scale_bias = make_float2(
            __half2float(reinterpret_cast<const __half*>(&row[D_bytes - 4])[0]),
            __half2float(reinterpret_cast<const __half*>(&row[D_bytes - 2])[0]));
      }

      uint32_t v0 = reinterpret_cast<const uint32_t*>(&row[byte_offset])[0];

      accumulate_packed_intx<bits_per_dim>(accumulator, v0, scale_bias, sample_weight);
    }


    for (int32_t i = 0; i < dims_per_byte; ++i) {
      *(float4*)(&output[b][d + (i * 4)]) = accumulator[i];
    }
  }
}

inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::ScalarType dtype) {
  return at::native::empty_cuda({0}, dtype, t.layout(), t.device(), false);
}

Tensor qembeddingbag_byte_unpack(const Tensor& packed_weight) {
  const auto packed_weight_sizes = packed_weight.sizes();
  const auto col_dim = packed_weight_sizes.size() - 1;
  const int32_t input_rows = c10::size_to_dim_(col_dim, packed_weight_sizes);
  const int32_t input_columns = packed_weight_sizes[col_dim];
  const int32_t output_columns = input_columns - 2 * sizeof(float);

  std::vector<int64_t> output_shape = packed_weight_sizes.vec();
  output_shape[col_dim] = output_columns;

  return at::empty(
      output_shape,
      packed_weight.options().dtype(kFloat),
      packed_weight.suggest_memory_format());
}

template <typename IndexType, typename OffsetType>
at::Tensor& embedding_bag_byte_impl(
    at::Tensor& output,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool pruned_weights,
    const std::optional<at::Tensor>& per_sample_weights_,
    const std::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) {
  TORCH_CHECK(weight.is_cuda());
  TORCH_CHECK(indices.is_cuda());
  TORCH_CHECK(offsets.is_cuda());
  TORCH_CHECK(indices.device() == weight.device())
  TORCH_CHECK(offsets.device() == weight.device());
  if (per_sample_weights_.has_value()) {
    TORCH_CHECK(per_sample_weights_.value().device() == weight.device());
  }
  TORCH_CHECK(weight.dtype() == at::kByte);
  TORCH_CHECK(weight.dim() == 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weight.get_device());

  const auto weight_sizes = weight.sizes();
  const int64_t N = weight_sizes[0];
  const int D = weight_sizes[1] - 8; // NB: -8 to account for scale and bias
  const int64_t M = offsets.sizes()[0];
  TORCH_CHECK(D % 4 == 0);
  if(per_sample_weights_.has_value()) {
      TORCH_CHECK(per_sample_weights_.value().scalar_type() == at::kFloat,
              "Per sample weights expected scalar type ", at::kFloat, " but got ",
              per_sample_weights_.value().scalar_type());
  }
  const auto maxThreads = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

  int64_t output_size = include_last_offset ? M - 1 : M;

  at::Tensor sample_weights;
  if (per_sample_weights_.has_value()) {
      sample_weights = per_sample_weights_.value();
  } else {
      sample_weights = create_empty_from(output, kFloat);
  }

  const std::vector<int64_t> shape = {output_size, D};
  at::native::resize_(output, shape, std::nullopt);
  const at::Tensor offsets_k = offsets.scalar_type() == indices.scalar_type()
      ? offsets
      : offsets.to(indices.scalar_type());
  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "embedding_bag_byte_rowwise_offsets_kernel", ([&] {
        embedding_bag_nbits_rowwise_offsets_kernel<index_t, 8><<<
            output_size,
            dim3(1, 1, 1),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            weight.packed_accessor64<uint8_t, 2, RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            offsets_k.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            false /* pruned_weights */,
            sample_weights.packed_accessor32<float, 1, RestrictPtrTraits>(),
            compressed_indices_mapping,
            include_last_offset,
            output.packed_accessor32<float, 2, RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));

  TORCH_CHECK(output.is_cuda());

  return output;
}

Tensor embedding_bag_byte_rowwise_offsets(
    const Tensor& weight,
    const Tensor& indices,
    const std::optional<Tensor>& offsets_in,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool pruned_weights,
    const std::optional<Tensor>& per_sample_weights_,
    const std::optional<Tensor>& compressed_indices_mapping,
    bool include_last_offset) {
  // Screen the caller's tensors before anything below can rewrite them. The
  // pruned path re-enters this function with translated indices and synthesized
  // per-sample weights, and returns early when every row was pruned, so a check
  // placed further down would either inspect the rewritten tensors or be skipped
  // altogether.
  TORCH_CHECK(weight.is_cuda());
  TORCH_CHECK(indices.is_cuda());
  TORCH_CHECK(indices.device() == weight.device());
  TORCH_CHECK(weight.dtype() == at::kByte);
  TORCH_CHECK(weight.dim() == 2);
  // NB: -8 to account for scale and bias. The lower bound is not implied by the
  // remainder: (0 - 8) % 4 and (4 - 8) % 4 are both 0, and the negative D that
  // follows only surfaces much later as an overflow or a negative dimension.
  TORCH_CHECK(weight.size(1) >= 8 && (weight.size(1) - 8) % 4 == 0);
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "qembedding/qembedding_bag operator supports 1 or 2d indices, got ",
      indices.dim());
  TORCH_CHECK(
      indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
      "Expect 32 or 64 bit indices, but found ",
      indices.scalar_type(),
      " instead.");
  if (offsets_in.has_value()) {
    const auto& offsets_arg = offsets_in.value();
    TORCH_CHECK(offsets_arg.is_cuda());
    TORCH_CHECK(offsets_arg.device() == weight.device());
    TORCH_CHECK(
        offsets_arg.dim() == 1, "Expect 1D offsets, got ", offsets_arg.dim());
    TORCH_CHECK(
        offsets_arg.scalar_type() == at::kInt ||
            offsets_arg.scalar_type() == at::kLong,
        "Expect 32 or 64 bit offsets, but found ",
        offsets_arg.scalar_type(),
        " instead.");
  }
  TORCH_CHECK(
      weight.is_contiguous() && indices.is_contiguous() &&
          (!offsets_in.has_value() || offsets_in.value().is_contiguous()),
      "Expect weight, indices, and offsets to be contiguous.");
  if (per_sample_weights_.has_value()) {
    TORCH_CHECK(per_sample_weights_.value().device() == weight.device());
    TORCH_CHECK(
        per_sample_weights_.value().scalar_type() == at::kFloat,
        "Per sample weights expected scalar type ",
        at::kFloat,
        " but got ",
        per_sample_weights_.value().scalar_type());
  }

  // The kernel takes compressed_indices_mapping but ignores it, so remap the
  // indices here and re-enter as a dense lookup, mirroring the CPU op.
  if (compressed_indices_mapping.has_value()) {
    const auto& mapping = compressed_indices_mapping.value();
    Tensor eff_indices = indices;
    std::optional<Tensor> eff_per_sample_weights = per_sample_weights_;
    // A single-entry mapping is the "not pruned" sentinel: {0} by CPU's
    // spelling, {-1} by that of at least one out-of-tree producer. CPU honours
    // only {0}; it reads any other single value as a real one-row mapping and
    // indexes through it, rejecting ids >= 1 as out of bounds. Telling the two
    // apart needs a device read, i.e. a sync on every call, so every
    // single-entry mapping is taken as the sentinel and looked up densely.
    if (pruned_weights && mapping.numel() != 1) {
      TORCH_CHECK_TYPE(
          mapping.scalar_type() == at::kInt,
          "compressed_indices_mapping must have dtype Int, got ",
          mapping.scalar_type());
      TORCH_CHECK(
          mapping.device() == weight.device(),
          "compressed_indices_mapping must be on the same device as weight; got ",
          mapping.device(),
          " vs ",
          weight.device());
      // Not redundant: index_select would fault device-side, unrecoverably.
      TORCH_CHECK_VALUE(
          mapping.numel() > 0, "compressed_indices_mapping must not be empty");
      // CPU accepts 2-D indices by synthesizing offsets for them; on CUDA that
      // path is already dead (host-side at::arange), so reject rather than
      // carry reshape logic for it.
      TORCH_CHECK(
          indices.dim() == 1,
          "compressed_indices_mapping requires 1D indices, got ",
          indices.dim());
      // Out-of-range ids fault device-side (a bare abort() on ROCm) rather than
      // raising CPU's clean error; screening on the host would sync every call.
      const auto remapped = mapping.reshape({-1}).index_select(0, indices);
      // Every row pruned away, so the result is all zeros, as on CPU; clamp_min
      // below would instead read row 0 of an empty table. The caller's tensors
      // were screened at the top of the function, so returning here skips none
      // of the checks above.
      if (weight.size(0) == 0) {
        TORCH_CHECK(
            offsets_in.has_value(),
            "embedding_bag_byte expects offsets to be set for 1D indices.");
        const auto num_bags = offsets_in.value().size(0);
        return at::zeros(
            {include_last_offset ? num_bags - 1 : num_bags, weight.size(1) - 8},
            weight.options().dtype(at::kFloat));
      }
      // CPU skips pruned rows (-1); the dense kernel cannot, so cancel them with a
      // zero per-sample weight. Exact for finite rows and weights: 0 * inf = NaN.
      const auto keep = remapped.ge(0).to(at::kFloat);
      eff_per_sample_weights = per_sample_weights_.has_value()
          ? per_sample_weights_.value().mul(keep)
          : keep;
      eff_indices = remapped.clamp_min(0).to(indices.scalar_type());
    }
    return embedding_bag_byte_rowwise_offsets(
        weight,
        eff_indices,
        offsets_in,
        scale_grad_by_freq,
        mode,
        /*pruned_weights=*/false,
        eff_per_sample_weights,
        /*compressed_indices_mapping=*/std::nullopt,
        include_last_offset);
  }
  bool is_embedding_op = false;
  auto output = create_empty_from(weight, at::kFloat);

  c10::MaybeOwned<at::Tensor> offsets;
  // For embedding_bag operator with 2D indices, we set the offsets explicitly
  // here.
  if (indices.dim() == 2 && !is_embedding_op) {
    TORCH_CHECK(
        !offsets_in.has_value(),
        "embedding_bag_byte operator: input is 2D, then offsets has to be None, as input is treated is a mini-batch of fixed length sequences.");

    offsets = c10::MaybeOwned<at::Tensor>::owned(at::arange(
        0, indices.numel(), indices.sizes()[1], indices.scalar_type()));

  } else {
    TORCH_CHECK(
        offsets_in.has_value(),
        "embedding_bag_byte expects offsets to be set for 1D indices.");
    offsets = c10::MaybeOwned<at::Tensor>::borrowed(offsets_in.value());
  }

  // Only the synthesized 2-D offsets are unchecked above; they are Int/Long and
  // contiguous by construction, but at::arange builds them on the host, so the
  // device check in embedding_bag_byte_impl is what catches that (pre-existing;
  // the same device-less at::arange call sits in the 4-bit function).

  // IndexType/OffsetType are unused in embedding_bag_byte_impl -- all four
  // instantiations are identical, and the impl casts offsets to the index type
  // before launching.
  if (indices.scalar_type() == at::kInt && offsets->scalar_type() == at::kInt) {
    return embedding_bag_byte_impl<int, int>(
        output,
        weight,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  } else if (
      indices.scalar_type() == at::kInt &&
      offsets->scalar_type() == at::kLong) {
    return embedding_bag_byte_impl<int, int64_t>(
        output,
        weight,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  } else if (
      indices.scalar_type() == at::kLong &&
      offsets->scalar_type() == at::kInt) {
    return embedding_bag_byte_impl<int64_t, int>(
        output,
        weight,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  }

  // default case given the TORCH_CHECK above
  return embedding_bag_byte_impl<int64_t, int64_t>(
      output,
      weight,
      indices,
      *offsets,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      is_embedding_op);
}

template <typename IndexType, typename OffsetType>
at::Tensor& embedding_bag_4bit_impl(
    at::Tensor& output,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool pruned_weights,
    const std::optional<at::Tensor>& per_sample_weights_,
    const std::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset) {
  TORCH_CHECK(weight.is_cuda());
  TORCH_CHECK(indices.is_cuda());
  TORCH_CHECK(offsets.is_cuda());
  TORCH_CHECK(indices.device() == weight.device())
  TORCH_CHECK(offsets.device() == weight.device());
  if (per_sample_weights_.has_value()) {
    TORCH_CHECK(per_sample_weights_.value().device() == weight.device());
  }
  if (compressed_indices_mapping.has_value()) {
    TORCH_CHECK(compressed_indices_mapping.value().device() == weight.device());
  }

  TORCH_CHECK(weight.dtype() == at::kByte);
  TORCH_CHECK(weight.dim() == 2);

  at::cuda::OptionalCUDAGuard device_guard;
  device_guard.set_index(weight.get_device());

  const auto weight_sizes = weight.sizes();
  const int64_t N = weight_sizes[0];
  const int D = 2*(weight_sizes[1] - 4); // NB: -4 to account for scale and bias @fp16
  const int64_t M = offsets.sizes()[0];
  TORCH_CHECK(D % 8 == 0);
  if(per_sample_weights_.has_value()) {
      TORCH_CHECK(per_sample_weights_.value().scalar_type() == at::kFloat,
              "Per sample weights expected scalar type ", at::kFloat, " but got ",
              per_sample_weights_.value().scalar_type());
  }
  TORCH_CHECK(
      !compressed_indices_mapping.has_value(),
      "Compressed indices mapping not yet implemented for embedding_bag_4bit_rowwise_offsets_cuda");

  const auto maxThreads = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

  int64_t output_size = include_last_offset ? M - 1 : M;

  at::Tensor sample_weights;
  if (per_sample_weights_.has_value()) {
      sample_weights = per_sample_weights_.value();
  } else {
      sample_weights = create_empty_from(output, kFloat);
  }

  const std::vector<int64_t> shape = {output_size, D};
  at::native::resize_(output, shape, std::nullopt);
  const at::Tensor offsets_k = offsets.scalar_type() == indices.scalar_type()
      ? offsets
      : offsets.to(indices.scalar_type());
  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "embedding_bag_4bit_rowwise_offsets_kernel", ([&] {
        embedding_bag_nbits_rowwise_offsets_kernel<index_t, 4><<<
            output_size,
            dim3(1, 1, 1),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            weight.packed_accessor64<uint8_t, 2, RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            offsets_k.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            false /* pruned_weights */,
            sample_weights.packed_accessor32<float, 1, RestrictPtrTraits>(),
            compressed_indices_mapping,
            include_last_offset,
            output.packed_accessor32<float, 2, RestrictPtrTraits>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));

  TORCH_CHECK(output.is_cuda());

  return output;
}

Tensor embedding_bag_4bit_rowwise_offsets(
    const Tensor& weight,
    const Tensor& indices,
    const std::optional<Tensor>& offsets_in,
    const bool /* scale_grad_by_freq */,
    const int64_t /* mode */,
    bool pruned_weights,
    const std::optional<Tensor>& per_sample_weights_,
    const std::optional<Tensor>& compressed_indices_mapping,
    bool include_last_offset) {
  auto output = create_empty_from(weight, at::kFloat);

  c10::MaybeOwned<at::Tensor> offsets;
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "qembedding/qembedding_bag operator supports 1 or 2d indices, got ",
      indices.dim());

  // For embedding_bag operator with 2D indices, we need to set the offsets
  // explicitly here.
  if (indices.dim() == 2) {
    TORCH_CHECK(
        !offsets_in.has_value(),
        "embedding_bag_4bit operator: input is 2D, then offsets has to be None, as input is treated is a mini-batch of fixed length sequences.");

    offsets = c10::MaybeOwned<at::Tensor>::owned(at::arange(
        0, indices.numel(), indices.sizes()[1], indices.scalar_type()));
  } else {
    TORCH_CHECK(
        offsets_in.has_value(),
        "embedding_bag_4bit operator expects offsets to be set for 1D indices.");
    offsets = c10::MaybeOwned<at::Tensor>::borrowed(offsets_in.value());
  }

  TORCH_CHECK(
      indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
      "Expect 32 or 64 bit indices, but found ",
      indices.scalar_type(),
      " instead.");
  TORCH_CHECK(
      offsets->scalar_type() == at::kInt || offsets->scalar_type() == at::kLong,
      "Expect 32 or 64 bit offsets, but found ",
      offsets->scalar_type(),
      " instead.");
  TORCH_CHECK(
      weight.is_contiguous() && indices.is_contiguous() &&
          offsets->is_contiguous(),
      "Expect weight, indices, and offsets to be contiguous.");

  if (indices.scalar_type() == at::kInt && offsets->scalar_type() == at::kInt) {
    return embedding_bag_4bit_impl<int, int>(
        output,
        weight,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset);
  } else if (
      indices.scalar_type() == at::kInt &&
      offsets->scalar_type() == at::kLong) {
    return embedding_bag_4bit_impl<int, int64_t>(
        output,
        weight,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset);
  } else if (
      indices.scalar_type() == at::kLong &&
      offsets->scalar_type() == at::kInt) {
    return embedding_bag_4bit_impl<int64_t, int>(
        output,
        weight,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset);
  }
  return embedding_bag_4bit_impl<int64_t, int64_t>(
      output,
      weight,
      indices,
      *offsets,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset);
}

Tensor qembeddingbag_4bit_unpack(const Tensor& packed_weight) {
  int BIT_RATE = 4;
  const auto input_rows = packed_weight.size(0);
  const auto input_columns = packed_weight.size(1);
  const auto* input_data = packed_weight.const_data_ptr<uint8_t>();
  int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;

  // The last 4 bytes per row are two fp16 scale and zero_point.
  // The rest of input_columns is the number of values in the original row.
  std::vector<int64_t> output_dimensions = {
      input_rows,
      static_cast<std::int64_t>(input_columns - 2 * sizeof(at::Half)) *
          NUM_ELEM_PER_BYTE};

  auto output = at::empty(
      output_dimensions,
      packed_weight.options().dtype(kFloat),
      packed_weight.suggest_memory_format());
  return output;
}

TORCH_LIBRARY_IMPL(quantized, CUDA, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_unpack"),
      TORCH_FN(qembeddingbag_byte_unpack));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_rowwise_offsets"),
      TORCH_FN(embedding_bag_byte_rowwise_offsets));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_unpack"),
      TORCH_FN(qembeddingbag_4bit_unpack));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_rowwise_offsets"),
      TORCH_FN(embedding_bag_4bit_rowwise_offsets));
}

} // namespace at::native
