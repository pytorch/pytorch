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
      // TODO: uncomment an implement
      /*std::memcpy(
          offsets_include_last_val.data(),
          offsets_data,
          sizeof(OffsetType) * M);
      */
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

  at::native::resize_(output, shape, c10::nullopt);

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
      TORCH_SELECTIVE_NAME("quantized::qembeddingbag_4bit_unpack"),
      TORCH_FN(qembeddingbag_4bit_unpack));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_rowwise_offsets"),
      TORCH_FN(embedding_bag_4bit_rowwise_offsets));
}

} // namespace native
} // namespace at
