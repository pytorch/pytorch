#include <ATen/ATen.h>
#include <ATen/native/quantized/cpu/embedding_packed_params.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <torch/library.h>
#ifdef USE_FBGEMM
#include <fbgemm/Fbgemm.h>
#include <fbgemm/FbgemmEmbedding.h>
#endif

#include <ATen/Parallel.h>

torch::class_<EmbeddingPackedParamsBase> register_embedding_params();

namespace {
template <typename IndexType, typename OffsetType>
at::Tensor embedding_bag_4bit_impl(
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
  auto output = at::empty(shape, weight.options().dtype(at::kFloat));
  auto* output_data = output.data_ptr<float>();

  const int64_t block_size = D;
  const int index_size = indices.numel();
  constexpr int prefetch_distance = 16;

#ifdef USE_FBGEMM
  if (!pruned_weights || fallback_to_no_sparse) {
    // Generate the fbgemm kernel
    auto kernel = fbgemm::GenerateEmbeddingSpMDMNBit<IndexType, OffsetType>(
        /*bit rate=*/4,
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

    TORCH_CHECK(
        success,
        "FBGEMM GenerateEmbeddingSpMDMNBit kernel failed for 4-bit input");
  } else {
    auto kernel =
        fbgemm::GenerateEmbeddingSpMDMNBitRowWiseSparse<IndexType, OffsetType>(
            /*bit rate=*/4,
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
    TORCH_CHECK(
        success,
        "FBGEMM GenerateEmbeddingSpMDMNBitRowWiseSparse kernel failed for 4-bit input");
  }
#else

  auto accessor = offsets.accessor<OffsetType, 1>();
  std::vector<OffsetType> lengths_data;

  int64_t lower = accessor[0];
  for (int64_t i = 1; i < offsets.numel(); ++i) {
    lengths_data.push_back(accessor[i] - lower);
    lower = accessor[i];
  }
  if (!include_last_offset) {
    lengths_data.push_back(indices.numel() - lower);
  }

  int64_t current = 0;
  float* per_sample_weights_data;
  if (per_sample_weights_.has_value()) {
    per_sample_weights_data = per_sample_weights_.value().data_ptr<float>();
  }
  for (int m = 0; m < output_size; ++m) {
    memset(output_data, 0, block_size * sizeof(float));
    TORCH_CHECK(
        current + lengths_data[m] <= index_size,
        "Expect the lengths data to be less than indices size");

    for (int i = 0; i < lengths_data[m]; ++i, ++current) {
      int64_t idx;
      if (!pruned_weights) {
        idx = indices_data[current];
        TORCH_CHECK((idx >= 0 && idx < N), "Invalid indices data");
      } else {
        int64_t uncompressed_idx = indices_data[current];
        TORCH_CHECK(
            uncompressed_idx >= 0 && uncompressed_idx < compressed_index_size,
            "Invalid indices data for Sparse Op.")
        idx = compressed_indices_mapping_data[uncompressed_idx];
        if (idx == -1) {
          continue;
        }
      }
      const at::Half* scale_bias = reinterpret_cast<const at::Half*>(
          weight_data + (idx + 1) * weight_size - 2 * sizeof(at::Half));

      float weight_val = 1.0f;
      if (per_sample_weights_.has_value()) {
        weight_val = per_sample_weights_data[current];
      }
      const float scale = weight_val * scale_bias[0];
      const float bias = weight_val * scale_bias[1];

      for (int j = 0; j < block_size; ++j) {
        uint8_t quantized =
            weight_data[idx * weight_size + j / /*NUM_ELEM_PER_BYTE*/ 2];
        quantized >>= (j % 2) * 4;
        quantized &= (1 << 4) - 1;

        output_data[j] = fma(scale, quantized, output_data[j] + bias);
      }
    } // for each i
    output_data += block_size;
  } // for each m

#endif
  return output;
}

template <typename IndexType, typename OffsetType>
at::Tensor embedding_bag_byte_impl(
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
  const auto weight_data = weight.data_ptr<uint8_t>();
  const auto indices_data = indices.data_ptr<IndexType>();
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
  const int64_t N = weight_sizes[0];
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
  std::vector<int64_t> shape;
  if (indices.dim() == 2 && is_embedding_op) {
    const auto indices_sizes = indices.sizes();
    shape = {indices_sizes[0], indices_sizes[1], D};
  } else {
    shape = {output_size, D};
  }
  auto output = at::empty(shape, weight.options().dtype(at::kFloat));
  auto* output_data = output.data_ptr<float>();

  const int index_size = indices.numel();
#ifdef USE_FBGEMM
  if (!pruned_weights || fallback_to_no_sparse) {
    auto kernel_i8 =
        fbgemm::GenerateEmbeddingSpMDM<uint8_t, IndexType, OffsetType>(
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

          TORCH_CHECK(
              success,
              "FBGEMM GenerateEmbeddingSpMDM kernel failed for 8-bit input");
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
    TORCH_CHECK(
        success,
        "FBGEMM GenerateEmbeddingSpMDMRowWiseSparse kernel failed for 8-bit input");
  }
  return output;
#endif
  // TODO add default (non-FBGEMM) implementation.
  TORCH_CHECK(
      false,
      "embedding_bag_byte expects FBGEMM support. This PyTorch installation was not built with FBGEMM operators");
}

at::Tensor embedding_bag_byte_helper(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& offsets_in,
    bool pruned_weights,
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset,
    bool is_embedding_op) {
  at::Tensor offsets;
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

    offsets = at::arange(
        0, indices.numel(), indices.sizes()[1], indices.scalar_type());
  } else {
    TORCH_CHECK(
        offsets_in.has_value(),
        "embedding_bag_byte expects offsets to be set for 1D indices.");
    offsets = offsets_in.value();
  }

  TORCH_CHECK(
      indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
      "Expect 32 or 64 bit indices, but found ",
      indices.scalar_type(),
      " instead.");
  TORCH_CHECK(
      offsets.scalar_type() == at::kInt || offsets.scalar_type() == at::kLong,
      "Expect 32 or 64 bit offsets, but found ",
      offsets.scalar_type(),
      " instead.");
  TORCH_CHECK(
      weight.is_contiguous() && indices.is_contiguous() &&
          offsets.is_contiguous(),
      "Expect weight, indices, and offsets to be contiguous.");

  // Using helper function to support different type combination without the
  // need to cast, which can be additional performance overhead
  if (indices.scalar_type() == at::kInt && offsets.scalar_type() == at::kInt) {
    return embedding_bag_byte_impl<int, int>(
        weight,
        indices,
        offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  } else if (
      indices.scalar_type() == at::kInt && offsets.scalar_type() == at::kLong) {
    return embedding_bag_byte_impl<int, int64_t>(
        weight,
        indices,
        offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  } else if (
      indices.scalar_type() == at::kLong && offsets.scalar_type() == at::kInt) {
    return embedding_bag_byte_impl<int64_t, int>(
        weight,
        indices,
        offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset,
        is_embedding_op);
  }

  // default case given the TORCH_CHECK above
  return embedding_bag_byte_impl<int64_t, int64_t>(
      weight,
      indices,
      offsets,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      is_embedding_op);
}

at::Tensor embedding_bag_4bit_helper(
    const at::Tensor& weight,
    const at::Tensor& indices,
    const c10::optional<at::Tensor>& offsets_in,
    bool pruned_weights,
    const c10::optional<at::Tensor>& per_sample_weights_,
    const c10::optional<at::Tensor>& compressed_indices_mapping,
    bool include_last_offset) {
  at::Tensor offsets;
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

    offsets = at::arange(
        0, indices.numel(), indices.sizes()[1], indices.scalar_type());
  } else {
    TORCH_CHECK(
        offsets_in.has_value(),
        "embedding_bag_4bit operator expects offsets to be set for 1D indices.");
    offsets = offsets_in.value();
  }

  TORCH_CHECK(
      indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong,
      "Expect 32 or 64 bit indices, but found ",
      indices.scalar_type(),
      " instead.");
  TORCH_CHECK(
      offsets.scalar_type() == at::kInt || offsets.scalar_type() == at::kLong,
      "Expect 32 or 64 bit offsets, but found ",
      offsets.scalar_type(),
      " instead.");
  TORCH_CHECK(
      weight.is_contiguous() && indices.is_contiguous() &&
          offsets.is_contiguous(),
      "Expect weight, indices, and offsets to be contiguous.");

  // Using helper function to support different type combination without the
  // need to cast, which can be additional performance overhead
  if (indices.scalar_type() == at::kInt && offsets.scalar_type() == at::kInt) {
    return embedding_bag_4bit_impl<int, int>(
        weight,
        indices,
        offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset);
  } else if (
      indices.scalar_type() == at::kInt && offsets.scalar_type() == at::kLong) {
    return embedding_bag_4bit_impl<int, int64_t>(
        weight,
        indices,
        offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset);
  } else if (
      indices.scalar_type() == at::kLong && offsets.scalar_type() == at::kInt) {
    return embedding_bag_4bit_impl<int64_t, int>(
        weight,
        indices,
        offsets,
        pruned_weights,
        per_sample_weights_,
        compressed_indices_mapping,
        include_last_offset);
  }
  return embedding_bag_4bit_impl<int64_t, int64_t>(
      weight,
      indices,
      offsets,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset);
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
  return embedding_bag_byte_helper(
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
    bool include_last_offset) {
  if (per_sample_weights_.has_value()) {
    TORCH_CHECK(
        (per_sample_weights_.value().scalar_type() == at::kFloat ||
         per_sample_weights_.value().scalar_type() == at::kHalf),
        "Expect fp32 or fp16 weights, but found",
        per_sample_weights_.value().scalar_type(),
        " instead")
  }

  return embedding_bag_4bit_helper(
      packed_w,
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_.has_value()
          ? per_sample_weights_.value().to(at::kFloat)
          : per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset);
}

namespace at {
namespace native {
namespace {

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
  return embedding_bag_byte_helper(
      weight,
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset,
      false /* is_embedding_op */);
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
  if (per_sample_weights_.has_value()) {
    TORCH_CHECK(
        (per_sample_weights_.value().scalar_type() == at::kFloat ||
         per_sample_weights_.value().scalar_type() == at::kHalf),
        "Expect fp32 or fp16 weights, but found",
        per_sample_weights_.value().scalar_type(),
        " instead")
  }

  return embedding_bag_4bit_helper(
      weight,
      indices,
      offsets_in,
      pruned_weights,
      per_sample_weights_.has_value()
          ? per_sample_weights_.value().to(at::kFloat)
          : per_sample_weights_,
      compressed_indices_mapping,
      include_last_offset);
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
          include_last_offset);
    } else {
      TORCH_INTERNAL_ASSERT(
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

    } else {
      TORCH_INTERNAL_ASSERT(
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

  // Functions that work on at::Tensor packed weight.
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_rowwise_offsets"),
      embedding_bag_byte_rowwise_offsets);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_rowwise_offsets"),
      embedding_bag_4bit_rowwise_offsets);
}
} // namespace
} // namespace native
} // namespace at
