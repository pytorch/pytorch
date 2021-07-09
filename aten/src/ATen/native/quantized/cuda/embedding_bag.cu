#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

namespace at {
namespace native {

// BEGIN QUANTIZE HELPER FUNCTIONS
__device__ __forceinline__ float bfe(uint32_t val, uint32_t pos, uint32_t len) {
#ifdef USE_ROCM
  return *reinterpret_cast<float*>((val >> pos) && ((1u << len) - 1u ));
#else
  uint32_t ret;
  // Get the bit field of [pos, pos+len) bits from val:
  // (val >> pos) && ( (1u << len) - 1u )
  asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
  return __uint2float_rn(ret);
#endif
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

__forceinline__ __device__ float4
dequantize_int8(uint32_t packedVals, float2 scale_bias) {
  float4 res;

  res.x = bfe(packedVals, 0, 8);
  res.y = bfe(packedVals, 8, 8);
  res.z = bfe(packedVals, 16, 8);
  res.w = bfe(packedVals, 24, 8);

  return fma4sb(res, scale_bias.x, scale_bias.y);
}

__forceinline__ __device__ float4
accumulate_packed_int8(float4 acc, uint32_t packedVals, float2 scale_bias) {
  float4 res = dequantize_int8(packedVals, scale_bias);

  // Accumulate in float32.
  acc.x += res.x;
  acc.y += res.y;
  acc.z += res.z;
  acc.w += res.w;
  return acc;
}

// END QUANTIZE HELPER FUNCTIONS

// UN-OPTIMIZED kernel, doesn't even avoid warp divergence!
template <typename index_t>
__global__ void embedding_bag_byte_rowwise_offsets_kernel(
    const PackedTensorAccessor64<uint8_t, 2, RestrictPtrTraits> weight,
    const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> indices,
    const PackedTensorAccessor32<index_t, 1, RestrictPtrTraits> offsets,
    const bool /* pruned_weights */,
    const c10::optional<Tensor>& per_sample_weights_,
    const c10::optional<Tensor>& compressed_indices_mapping,
    const bool include_last_offset,
    PackedTensorAccessor32<float, 2, RestrictPtrTraits> output) {
  int32_t B = output.size(0);
  int32_t T = output.size(1);
  int32_t b_t = blockIdx.x * blockDim.y + threadIdx.y;
  if (b_t >= B * T) {
    return;
  }
  int32_t t = b_t / B;
  int32_t b = b_t % B;

  const int32_t D_bytes = T + 8;

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
    for (int32_t d = 0; d < T; d += 4) {
      *(float4*)(&output[b][d]) = make_float4(0, 0, 0, 0);
    }
    return;
  }

  float4 accumulator;
  for (int32_t d = 0; d < T; d += 4) {
    accumulator = make_float4(0, 0, 0, 0);
    for (int32_t l = indices_start; l < indices_end; ++l) {
      int64_t idx = indices[l];
      const uint8_t* __restrict__ row = &weights[idx * D_bytes];
      float2 scale_bias = make_float2(
          reinterpret_cast<const float*>(&row[D_bytes - 8])[0],
          reinterpret_cast<const float*>(&row[D_bytes - 4])[0]);

      uint32_t v0 = reinterpret_cast<const uint32_t*>(&row[d])[0];

      accumulator = accumulate_packed_int8(accumulator, v0, scale_bias);
    }

    *(float4*)(&output[b][d]) = accumulator;
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
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
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
  if (compressed_indices_mapping.has_value()) {
    TORCH_CHECK(compressed_indices_mapping.value().device() == weight.device());
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
  TORCH_CHECK(
      !per_sample_weights_.has_value(),
      "Per sample weights not yet implemented for embedding_bag_byte_rowwise_offsets_cuda");
  TORCH_CHECK(
      !compressed_indices_mapping.has_value(),
      "Compressed indices mapping not yet implemented for embedding_bag_byte_rowwise_offsets_cuda");

  const auto maxThreads = at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;

  int64_t output_size = include_last_offset ? M - 1 : M;

  const std::vector<int64_t> shape = {output_size, D};
  at::native::resize_(output, shape, c10::nullopt);
  AT_DISPATCH_INDEX_TYPES(
      indices.scalar_type(), "embedding_bag_byte_rowwise_offsets_kernel", ([&] {
        embedding_bag_byte_rowwise_offsets_kernel<<<
            output_size,
            dim3(1, 1, 1),
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            weight.packed_accessor64<uint8_t, 2, RestrictPtrTraits>(),
            indices.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            offsets.packed_accessor32<index_t, 1, RestrictPtrTraits>(),
            false /* pruned_weights */,
            per_sample_weights_,
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
    const c10::optional<Tensor>& offsets_in,
    const bool /* scale_grad_by_freq */,
    const int64_t /* mode */,
    bool pruned_weights,
    const c10::optional<Tensor>& per_sample_weights_,
    const c10::optional<Tensor>& compressed_indices_mapping,
    bool include_last_offset) {
  bool is_embedding_op = false;
  auto output = create_empty_from(weight, at::kFloat);

  c10::MaybeOwned<at::Tensor> offsets;
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "qembedding/qembedding_bag operator supports 1 or 2d indices, got ",
      indices.dim());
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

  // Using helper function to support different type combination without the
  // need to cast, which can be additional performance overhead
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
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset) {
  TORCH_CHECK(weight.dim() == 2);
  TORCH_CHECK(offsets.dim() == 1);

  const auto weight_data = weight.data_ptr<uint8_t>();
  const auto indices_data = indices.data_ptr<IndexType>();
  auto offsets_data = offsets.data_ptr<OffsetType>();

  // Get compressed indices for pruned_weights op.
  int32_t* compressed_indices_mapping_data = nullptr;
  int compressed_index_size = 0;
  bool fallback_to_no_sparse = false;
  if (pruned_weights) {
    compressed_index_size = compressed_indices_mapping.value().numel();
    compressed_indices_mapping_data =
        compressed_indices_mapping.value().data_ptr<int32_t>();

    // if compressed_indices_mapping is [0], it is a indicator that
    // we should fallback to non sparse embedding look up kernel.
    if ((compressed_index_size == 1 &&
         compressed_indices_mapping_data[0] == 0)) {
      fallback_to_no_sparse = true;
    }
  }

  const auto weight_sizes = weight.sizes();
  const int64_t N = weight_sizes[0];
  const int64_t weight_size = weight_sizes[1];
  const int64_t D =
      (weight_size - 4) * 2; // NB: 2-byte fp16 scale and 2-byte zero_offset
  const int64_t M = offsets.sizes()[0];

  int64_t output_size = M - 1;
  std::vector<OffsetType> offsets_include_last_val;
  if (!include_last_offset) {
    output_size = M;
    offsets_include_last_val.resize(M + 1);
    // Avoid `null pointer passed as argument 2` ASAN violation when offsets
    // tensor is empty.
    if (M > 0) {
      std::memcpy(
          offsets_include_last_val.data(),
          offsets_data,
          sizeof(OffsetType) * M);
    }
    offsets_include_last_val[M] = indices.numel();
    offsets_data = offsets_include_last_val.data();
  }

  const std::vector<int64_t> shape = {output_size, D};
  at::native::resize_(output, shape, c10::nullopt);
  TORCH_CHECK(output.is_cuda());
  return output;
}

Tensor embedding_bag_4bit_rowwise_offsets(
    const Tensor& weight,
    const Tensor& indices,
    const c10::optional<Tensor>& offsets_in,
    const bool /* scale_grad_by_freq */,
    const int64_t /* mode */,
    bool pruned_weights,
    const c10::optional<Tensor>& per_sample_weights_,
    const c10::optional<Tensor>& compressed_indices_mapping,
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
  const auto* input_data = packed_weight.data_ptr<uint8_t>();
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

} // namespace native
} // namespace at
