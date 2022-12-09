#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/native/quantized/cpu/EmbeddingPackedParams.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <ATen/native/quantized/cpu/qembeddingbag.h>
#include <torch/library.h>
#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmEmbedding.h>
#endif

#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <c10/util/irange.h>

#include <array>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/resize_native.h>
#endif

int register_embedding_params();

namespace {

// Fallback implementation when FBGEMM is not available.
template <
    typename IndexType,
    typename OffsetType,
    int BIT_RATE,
    int NUM_ELEM_PER_BYTE>
at::Tensor& embedding_lookup_fallback_impl(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    at::Tensor& output,
    const int64_t block_size,
    const int64_t output_size,
    bool include_last_offset,
    bool pruned) {
  auto* output_data = output.data_ptr<float>();
  const auto weight_data = weight.data_ptr<uint8_t>();
  const auto indices_data = indices.data_ptr<IndexType>();
  int32_t* compressed_indices_mapping_data = nullptr;
  const auto weight_sizes = weight.sizes();
  const int64_t N = weight_sizes[0];
  const int64_t weight_size = weight_sizes[1];
  const int index_size = indices.numel();

  auto accessor = offsets.accessor<OffsetType, 1>();
  std::vector<OffsetType> lengths_data;

  int64_t lower = accessor[0];
  for (const auto i : c10::irange(1, offsets.numel())) {
    lengths_data.push_back(accessor[i] - lower);
    lower = accessor[i];
  }
  if (!include_last_offset) {
    lengths_data.push_back(indices.numel() - lower);
  }

  int64_t current = 0;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  float* per_sample_weights_data;
  if (per_sample_weights_.has_value()) {
    per_sample_weights_data = per_sample_weights_.value().data_ptr<float>();
  }
  for (const auto m : c10::irange(output_size)) {
    memset(output_data, 0, block_size * sizeof(float));
    TORCH_CHECK(
        current + lengths_data[m] <= index_size,
        "Expect the lengths data to be less than indices size");

    for (int i = 0; i < lengths_data[m]; ++i, ++current) {
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t idx;
      if (!pruned) {
        idx = indices_data[current];
        TORCH_CHECK((idx >= 0 && idx < N), "Invalid indices data");
      } else {
        int64_t uncompressed_idx = indices_data[current];
        int compressed_index_size = compressed_indices_mapping.value().numel();
        compressed_indices_mapping_data =
            compressed_indices_mapping.value().data_ptr<int32_t>();
        TORCH_CHECK(
            uncompressed_idx >= 0 && uncompressed_idx < compressed_index_size,
            "Invalid indices data for Sparse Op.")
        idx = compressed_indices_mapping_data[uncompressed_idx];
        if (idx == -1) {
          continue;
        }
      }

      float weight_val = 1.0f;
      if (per_sample_weights_.has_value()) {
        weight_val = per_sample_weights_data[current];
      }
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      float scale, bias;
      if (BIT_RATE == 8) {
        const uint8_t* scale_bias =
            weight_data + (idx + 1) * weight_size - 2 * sizeof(float);
        uint32_t scale_val_int32 = 0;
        scale_val_int32 = scale_val_int32 |
          (scale_bias[0]) |
          (scale_bias[1] << 8) |
          (scale_bias[2] << 16) |
          (scale_bias[3] << 24);
        float scale_val = (reinterpret_cast<float*>(&scale_val_int32))[0];
        uint32_t bias_val_int32 = 0;
        bias_val_int32 = bias_val_int32 |
          (scale_bias[4]) |
          (scale_bias[5] << 8) |
          (scale_bias[6] << 16) |
          (scale_bias[7] << 24);
        float bias_val = (reinterpret_cast<float*>(&bias_val_int32))[0];
        scale = weight_val * scale_val;
        bias = weight_val * bias_val;
      } else {
        const uint8_t* scale_bias =
            weight_data + (idx + 1) * weight_size - 2 * sizeof(at::Half);
        uint16_t scale_val_int16 = 0;
        scale_val_int16 = scale_val_int16 |
          (scale_bias[0]) |
          (scale_bias[1] << 8);
        at::Half scale_val = (reinterpret_cast<at::Half*>(&scale_val_int16))[0];
        uint16_t bias_val_int16 = 0;
        bias_val_int16 = bias_val_int16 |
          (scale_bias[2]) |
          (scale_bias[3] << 8);
        at::Half bias_val = (reinterpret_cast<at::Half*>(&bias_val_int16))[0];
        scale = weight_val * scale_val;
        bias = weight_val * bias_val;
      }

      for (const auto j : c10::irange(block_size)) {
        uint8_t quantized =
            weight_data[idx * weight_size + j / NUM_ELEM_PER_BYTE];
        quantized >>= (j % NUM_ELEM_PER_BYTE) * BIT_RATE;
        quantized &= (1 << BIT_RATE) - 1;

        output_data[j] = fma(scale, quantized, output_data[j] + bias);
      }
    } // for each i
    output_data += block_size;
  } // for each m
  return output;
}

namespace {
template <typename IndexType, typename OffsetType>
void fbgemm_spmdm_report_error_(
    int64_t output_size,
    int index_size,
    int64_t N,
    const OffsetType* offsets,
    const IndexType* indices) {
  for (const auto m : c10::irange(output_size)) {
    for (OffsetType i = offsets[m]; i < offsets[m + 1]; ++i) {
      TORCH_CHECK(i < index_size);
      IndexType idx = indices[i];
      TORCH_CHECK(
          0 <= idx && idx < N,
          "Index ",
          i,
          " is out of bounds: ",
          idx,
          ", range 0 to ",
          N);
    }
  }
  TORCH_CHECK(
      offsets[output_size] == index_size,
      "Yout input seems to be incorrect: the last offset value should be "
      "the size of the indices tensor, but it appears not.");
}
} // namespace

template <typename IndexType, typename OffsetType>
at::Tensor& embedding_bag_nbit_impl(
    at::Tensor& output,
    const at::Tensor& weight,
    const int bit_width,
    const at::Tensor& indices,
    const at::Tensor& offsets,
    bool pruned_weights,
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) {
  TORCH_CHECK(weight.dim() == 2);
  TORCH_CHECK(offsets.dim() == 1);

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
  const int64_t weight_size = weight_sizes[1];
  int NUM_ELEM_PER_BYTE = 8 / bit_width;
  const int64_t D =
      (weight_size - 2 * sizeof(at::Half)) * NUM_ELEM_PER_BYTE; // NB: 2-byte fp16 scale and 2-byte zero_offset
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
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<int64_t, 3> shape_arr;
    c10::IntArrayRef shape;
    if(indices.dim() == 2 && is_embedding_op) {
      const auto indices_sizes = indices.sizes();
      shape_arr[0] = indices_sizes[0];
      shape_arr[1] = indices_sizes[1];
      shape_arr[2] = D;
      shape = shape_arr;
    } else {
      shape_arr[0] = output_size;
      shape_arr[1] = D;
      shape = c10::IntArrayRef(&shape_arr[0], 2);
    }
    at::native::resize_(output, shape, c10::nullopt);
  }
#ifdef USE_FBGEMM
  const auto indices_data = indices.data_ptr<IndexType>();
  const auto weight_data = weight.data_ptr<uint8_t>();
  auto* output_data = output.data_ptr<float>();
  const int64_t N = weight_sizes[0];

  const int64_t block_size = D;
  const int index_size = indices.numel();
  constexpr int prefetch_distance = 16;
  if (!pruned_weights || fallback_to_no_sparse) {
    // Generate the fbgemm kernel
    auto kernel = fbgemm::GenerateEmbeddingSpMDMNBit<IndexType, OffsetType>(
        /*bit rate=*/bit_width,
        /*block size=*/block_size,
        /*has weights=*/per_sample_weights_.has_value(),
        /*normalize_by_lengths=*/false,
        /*prefetch distance=*/prefetch_distance,
        /*is_weight_positional=*/false,
        /*use_offsets=*/true);

    bool success = kernel(
        /*output_size=*/output_size,
        /*index_size=*/index_size,
        /*data_size=*/N,
        /*input=*/weight_data,
        /*indices=*/indices_data,
        /*offsets=*/offsets_data,
        /*weights=*/
        per_sample_weights_.has_value()
            ? per_sample_weights_.value().data_ptr<float>()
            : nullptr,
        /*output=*/output_data);

    if (!success) {
      fbgemm_spmdm_report_error_(
          output_size, index_size, N, offsets_data, indices_data);
    }
  } else {
    auto kernel =
        fbgemm::GenerateEmbeddingSpMDMNBitRowWiseSparse<IndexType, OffsetType>(
            /*bit rate=*/bit_width,
            /*block_size=*/block_size,
            /*has weights=*/per_sample_weights_.has_value(),
            /*normalize_by_lengths=*/false,
            /*prefetch distance*/ prefetch_distance,
            /*is_weight_positional*/ false,
            /*use_offsets*/ true);
    bool success = kernel(
        /*output_size=*/output_size,
        /*index_size=*/index_size,
        /*data_size=*/compressed_index_size,
        /*input=*/weight_data,
        /*indices=*/indices_data,
        /*offsets=*/offsets_data,
        /*weights=*/
        per_sample_weights_.has_value()
            ? per_sample_weights_.value().data_ptr<float>()
            : nullptr,
        /*output=*/output_data,
        /*compressed_indices_table=*/compressed_indices_mapping_data);
    if (!success) {
      fbgemm_spmdm_report_error_(
          output_size,
          index_size,
          compressed_index_size,
          offsets_data,
          indices_data);
    }
  }
  return output;
#else
  if (bit_width == 4) {
    return embedding_lookup_fallback_impl<IndexType, OffsetType, 4, 2>(
      weight,
      indices,
      offsets,
      per_sample_weights_,
      compressed_indices_mapping,
      output,
      D,
      output_size,
      include_last_offset,
      (pruned_weights && !fallback_to_no_sparse));
  }
  // bit_width == 2
  return embedding_lookup_fallback_impl<IndexType, OffsetType, 2, 4>(
    weight,
    indices,
    offsets,
    per_sample_weights_,
    compressed_indices_mapping,
    output,
    D,
    output_size,
    include_last_offset,
    (pruned_weights && !fallback_to_no_sparse));
#endif
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
  TORCH_CHECK(weight.scalar_type() == at::kByte);
  TORCH_CHECK(weight.dim() == 2);
  TORCH_CHECK(offsets.dim() == 1);
  auto offsets_data = offsets.data_ptr<OffsetType>();

  // Get compressed indices for pruned_weights.
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
  const int64_t D = weight_sizes[1] - 8; // NB: -8 to account for scale and bias
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
  {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    std::array<int64_t, 3> shape_arr;
    c10::IntArrayRef shape;
    if (indices.dim() == 2 && is_embedding_op) {
      const auto indices_sizes = indices.sizes();
      shape_arr[0] = indices_sizes[0];
      shape_arr[1] = indices_sizes[1];
      shape_arr[2] = D;
      shape = shape_arr;
    } else {
      shape_arr[0] = output_size;
      shape_arr[1] = D;
      shape = c10::IntArrayRef(&shape_arr[0], 2);
    }
    at::native::resize_(output, shape, c10::nullopt);
  }
#ifdef USE_FBGEMM
  const int64_t N = weight_sizes[0];
  const auto weight_data = weight.data_ptr<uint8_t>();
  const auto indices_data = indices.data_ptr<IndexType>();
  auto* output_data = output.data_ptr<float>();
  const int index_size = indices.numel();

  if (!pruned_weights || fallback_to_no_sparse) {
    auto kernel_i8 =
        fbgemm::GenerateEmbeddingSpMDM<uint8_t, IndexType, OffsetType, /*OutType=*/float, /*TRHEAD_LOCAL=*/true>(
            /*block_size=*/D,
            /*has_weight=*/per_sample_weights_.has_value(),
            /*normalize_by_lengths=*/false,
            /*prefetch=*/16, // NOLINT(cppcoreguidelines-avoid-magic-numbers)
            /*is_weight_positional=*/false,
            /*use_offsets=*/true);

    at::parallel_for(
        0, output_size, 1, [&](int64_t start_idx, int64_t end_idx) {
          bool success = kernel_i8(
              /*output_size=*/end_idx - start_idx,
              /*index_size=*/offsets_data[end_idx] - offsets_data[start_idx],
              /*data_size=*/N,
              /*input=*/weight_data,
              /*indices=*/indices_data + offsets_data[start_idx],
              /*offsets_or_lengths=*/offsets_data + start_idx,
              /*weights=*/
              per_sample_weights_
                  ? per_sample_weights_.value().data_ptr<float>() +
                      offsets_data[start_idx]
                  : nullptr,
              /*out=*/output_data + start_idx * D);

          if (!success) {
            fbgemm_spmdm_report_error_(
                end_idx - start_idx,
                offsets_data[end_idx] - offsets_data[start_idx],
                N,
                offsets_data + start_idx,
                indices_data + offsets_data[start_idx]);
          }
        });
  } else {
    // pruned weights
    auto kernel_i8_sparse = fbgemm::
        GenerateEmbeddingSpMDMRowWiseSparse<uint8_t, IndexType, OffsetType>(
            /*block_size=*/D,
            /*has_weight=*/per_sample_weights_.has_value(),
            /*normalize_by_lengths=*/false,
            /*prefetch=*/16, // NOLINT(cppcoreguidelines-avoid-magic-numbers)
            /*is_weight_positional=*/false,
            /*use_offsets=*/true);

    auto success = kernel_i8_sparse(
        /*output_size=*/output_size,
        /*index_size=*/index_size,
        /*data_size=*/compressed_index_size,
        /*input=*/weight_data,
        /*indices=*/indices_data,
        /*offsets=*/offsets_data,
        /*weights=*/
        per_sample_weights_.has_value()
            ? per_sample_weights_.value().data_ptr<float>()
            : nullptr,
        /*output=*/output_data,
        /*compressed_indices_table=*/compressed_indices_mapping_data);
    if (!success) {
      fbgemm_spmdm_report_error_(
          output_size,
          index_size,
          compressed_index_size,
          offsets_data,
          indices_data);
    }
  }
  return output;
#else
  return embedding_lookup_fallback_impl<IndexType, OffsetType, 8, 1>(
      weight,
      indices,
      offsets,
      per_sample_weights_,
      compressed_indices_mapping,
      output,
      D,
      output_size,
      include_last_offset,
      (pruned_weights && !fallback_to_no_sparse));
#endif
}

at::Tensor& embedding_bag_byte_helper(
    at::Tensor& output,
    const at::Tensor& weight,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& offsets_in,
    bool pruned_weights,
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) {
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

    offsets = c10::MaybeOwned<at::Tensor>::owned(at::arange(0, indices.numel(), indices.sizes()[1], indices.scalar_type()));
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
      indices.scalar_type() == at::kInt && offsets->scalar_type() == at::kLong) {
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
      indices.scalar_type() == at::kLong && offsets->scalar_type() == at::kInt) {
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

at::Tensor& _embedding_bag_nbit_helper(
    at::Tensor& output,
    const at::Tensor& weight,
    const int bit_width,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& offsets_in,
    bool pruned_weights,
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) {
  c10::MaybeOwned<at::Tensor> offsets;
  TORCH_CHECK(
      bit_width == 4 || bit_width == 2,
      "qembedding/qembedding_bag operator supports bit_width 2 or 4, got ",
      bit_width);
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "qembedding/qembedding_bag operator supports 1 or 2d indices, got ",
      indices.dim());

  // For embedding_bag operator with 2D indices, we need to set the offsets
  // explicitly here.
  if (indices.dim() == 2 && !is_embedding_op) {
    TORCH_CHECK(
        !offsets_in.has_value(),
        "embedding_bag_4bit/embedding_bag_2bit operator: input is 2D, then offsets has to be None, as input is treated is a mini-batch of fixed length sequences.");

    offsets = c10::MaybeOwned<at::Tensor>::owned(at::arange(
        0, indices.numel(), indices.sizes()[1], indices.scalar_type()));
  } else {
    TORCH_CHECK(
        offsets_in.has_value(),
        "embedding_bag_4bit/embedding_bag_2bit operator expects offsets to be set for 1D indices.");
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
    return embedding_bag_nbit_impl<int, int>(
        output,
        weight,
        bit_width,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  } else if (
      indices.scalar_type() == at::kInt && offsets->scalar_type() == at::kLong) {
    return embedding_bag_nbit_impl<int, int64_t>(
        output,
        weight,
        bit_width,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  } else if (
      indices.scalar_type() == at::kLong && offsets->scalar_type() == at::kInt) {
    return embedding_bag_nbit_impl<int64_t, int>(
        output,
        weight,
        bit_width,
        indices,
        *offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  }
  return embedding_bag_nbit_impl<int64_t, int64_t>(
      output,
      weight,
      bit_width,
      indices,
      *offsets,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      is_embedding_op);
}
} // namespace

at::Tensor PackedEmbeddingBagWeight::embeddingbag_byte(
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& offsets_in,
    bool pruned_weights,
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) {
  auto output = at::empty({0}, packed_w.options().dtype(at::kFloat));
  return embedding_bag_byte_helper(
      output,
      packed_w,
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      is_embedding_op);
}

at::Tensor PackedEmbeddingBagWeight::embeddingbag_4bit(
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& offsets_in,
    bool pruned_weights,
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) {
  if (per_sample_weights_.has_value()) {
    TORCH_CHECK(
        (per_sample_weights_.value().scalar_type() == at::kFloat ||
         per_sample_weights_.value().scalar_type() == at::kHalf),
        "Expect fp32 or fp16 weights, but found",
        per_sample_weights_.value().scalar_type(),
        " instead")
  }

  auto output = at::empty({0}, packed_w.options().dtype(at::kFloat));
  return _embedding_bag_nbit_helper(
    output,
    packed_w,
    4,
    indices,
    offsets_in,
    pruned_weights,
    per_sample_weights_.has_value()
        ? per_sample_weights_.value().to(at::kFloat)
        : per_sample_weights_,
    compressed_indices_mapping,
    include_last_offset,
    is_embedding_op);
}

namespace at {
namespace native {

Tensor& embedding_bag_byte_rowwise_offsets_out(
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const c10::optional<Tensor>& offsets_in,
    const bool /* scale_grad_by_freq */,
    const int64_t /* mode */,
    bool pruned_weights,
    const c10::optional<Tensor>& per_sample_weights_,
    const c10::optional<Tensor>& compressed_indices_mapping,
    bool include_last_offset) {
  return embedding_bag_byte_helper(
      output,
      weight,
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      false /* is_embedding_op */);
}

Tensor& embedding_bag_4bit_rowwise_offsets_out(
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const c10::optional<Tensor>& offsets_in,
    const bool /* scale_grad_by_freq */,
    const int64_t /* mode */,
    bool pruned_weights,
    const c10::optional<Tensor>& per_sample_weights_,
    const c10::optional<Tensor>& compressed_indices_mapping,
    bool include_last_offset) {

  if (per_sample_weights_.has_value()) {
    TORCH_CHECK(
        (per_sample_weights_.value().scalar_type() == at::kFloat ||
         per_sample_weights_.value().scalar_type() == at::kHalf),
        "Expect fp32 or fp16 weights, but found",
        per_sample_weights_.value().scalar_type(),
        " instead")
  }
  return _embedding_bag_nbit_helper(
      output,
      weight,
      4,
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_.has_value()
          ? per_sample_weights_.value().to(at::kFloat)
          : per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      false);
}

Tensor& embedding_bag_2bit_rowwise_offsets_out(
    Tensor& output,
    const Tensor& weight,
    const Tensor& indices,
    const c10::optional<Tensor>& offsets_in,
    const bool /* scale_grad_by_freq */,
    const int64_t /* mode */,
    bool pruned_weights,
    const c10::optional<Tensor>& per_sample_weights_,
    const c10::optional<Tensor>& compressed_indices_mapping,
    bool include_last_offset) {

  if (per_sample_weights_.has_value()) {
    TORCH_CHECK(
        (per_sample_weights_.value().scalar_type() == at::kFloat ||
         per_sample_weights_.value().scalar_type() == at::kHalf),
        "Expect fp32 or fp16 weights, but found",
        per_sample_weights_.value().scalar_type(),
        " instead")
  }
  return _embedding_bag_nbit_helper(
      output,
      weight,
      2,
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_.has_value()
          ? per_sample_weights_.value().to(at::kFloat)
          : per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      false);
}

namespace {


inline at::Tensor create_empty_from(
    const at::Tensor& t,
    c10::ScalarType dtype) {
  return at::detail::empty_cpu(
      {0}, dtype, t.layout(), t.device(), c10::nullopt, c10::nullopt);
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
  auto output = create_empty_from(weight, at::kFloat);
  embedding_bag_byte_rowwise_offsets_out(
      output,
      weight,
      indices,
      offsets_in,
      false /*unused scale_grad_by_freq*/,
      0 /*unused mode*/,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset);
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
  embedding_bag_4bit_rowwise_offsets_out(
    output,
    weight,
    indices,
    offsets_in,
    false, // unused scale_grad_by_freq
    0, // unused mode
    pruned_weights,
    per_sample_weights_,
    compressed_indices_mapping,
    include_last_offset);
  return output;
}

Tensor embedding_bag_2bit_rowwise_offsets(
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
  embedding_bag_2bit_rowwise_offsets_out(
    output,
    weight,
    indices,
    offsets_in,
    false, // unused scale_grad_by_freq
    0, // unused mode
    pruned_weights,
    per_sample_weights_,
    compressed_indices_mapping,
    include_last_offset);
  return output;
}

template <int bit_rate>
class QEmbeddingBag final {
 public:
  static at::Tensor run(
      const c10::intrusive_ptr<EmbeddingPackedParamsBase>& packed_weight,
      const Tensor& indices,
      const c10::optional<Tensor>& offsets,
      const bool /* scale_grad_by_freq */,
      const int64_t /* mode */,
      bool pruned_weights,
      const c10::optional<Tensor>& per_sample_weights_,
      const c10::optional<Tensor>& compressed_indices_mapping,
      bool include_last_offset) {
    if (bit_rate == 8) {
      return packed_weight->embeddingbag_byte(
          indices,
          offsets,
          pruned_weights,
          per_sample_weights_,
          compressed_indices_mapping,
          include_last_offset,
          false /* is_embedding_op */);
    } else if (bit_rate == 4) {
      return packed_weight->embeddingbag_4bit(
          indices,
          offsets,
          pruned_weights,
          per_sample_weights_,
          compressed_indices_mapping,
          include_last_offset,
          false);
    } else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Currently only support 8-bit embedding_bag quantization");
    }
  }
};

template <int bit_rate>
class QEmbedding final {
 public:
  static at::Tensor run(
      const c10::intrusive_ptr<EmbeddingPackedParamsBase>& packed_weight,
      const Tensor& indices,
      bool pruned_weights) {
    // Set default offsets here since the FBGEMM lookup op expects it.
    const auto offsets_size = indices.numel();
    at::Tensor offsets = at::arange(0, offsets_size, indices.scalar_type());
    at::Tensor output;
    if (bit_rate == 8) {
      return packed_weight->embeddingbag_byte(
          indices,
          offsets,
          pruned_weights,
          c10::nullopt,
          c10::nullopt,
          false /* include_last_offset */,
          true /* is_embedding_op */);
    } else if (bit_rate == 4) {
      return packed_weight->embeddingbag_4bit(
          indices,
          offsets,
          pruned_weights,
          c10::nullopt,
          c10::nullopt,
          false,
          true);
    } else {
      TORCH_INTERNAL_ASSERT(
          false,
          "Currently only support 8-bit embedding quantization");
    }
    return output;
  }
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  // Function that works on TorchBind packed weights.
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte"),
      TORCH_FN(QEmbeddingBag<8>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit"),
      TORCH_FN(QEmbeddingBag<4>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_byte"),
      TORCH_FN(QEmbedding<8>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_4bit"),
      TORCH_FN(QEmbedding<4>::run));

  // Functions that work on at::Tensor packed weight.
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_rowwise_offsets"),
      embedding_bag_byte_rowwise_offsets);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_rowwise_offsets"),
      embedding_bag_4bit_rowwise_offsets);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_2bit_rowwise_offsets"),
      embedding_bag_2bit_rowwise_offsets);
}
} // namespace
} // namespace native
} // namespace at
