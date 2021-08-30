#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/embedding_packed_params.h>
#include <ATen/native/quantized/cpu/fbgemm_utils.h>
#include <torch/library.h>

torch::class_<EmbeddingPackedParamsBase> register_embedding_params();


namespace at {
namespace native {
namespace {

Tensor qembeddingbag_byte_unpack(const Tensor& packed_weight) {
  // The "last" dimension of an N-Dimensioned batch of embedding bags is
  // quantization channel. E.g. for a 2D embedding bag, this has
  // [ row, col ] dimensions, for batched of embedding bags, dimensions might be
  // [ batch, row, col ].
  //
  // Python Batched Embedding Example:
  // weights = torch.from_numpy((np.random.random_sample((
  //          2, 10, 3)).squeeze() + 1).astype(np.float32))
  // assert(weights.size() == torch.Size([2, 10, 3]))
  // # NOTE: 8 bytes (columns) are added due to fp32 zero_point and scales
  // packed_weights = torch.ops.quantized.embedding_bag_byte_prepack(weights)
  // assert(packed_weights.size() == torch.Size([2, 10, 11]))
  // unpacked_weights = torch.ops.quantized.embedding_bag_byte_unpack(packed_weights)
  // assert(unpacked_weights.size() == torch.Size([2, 10, 3]))
  const auto packed_weight_sizes = packed_weight.sizes();
  const auto col_dim = packed_weight_sizes.size() - 1;
  const int32_t input_rows = c10::size_to_dim_(col_dim, packed_weight_sizes);
  const int32_t input_columns = packed_weight_sizes[col_dim];
  // The last 2 values are used to store the FP32 scale and zero_point values
  // per row.
  const int32_t output_columns = input_columns - 2 * sizeof(float);
  const auto* input_data = packed_weight.data_ptr<uint8_t>();

  std::vector<int64_t> output_shape = packed_weight_sizes.vec();
  output_shape[col_dim] = output_columns;
  at::Tensor output = at::empty(
      output_shape,
      packed_weight.options().dtype(kFloat),
      packed_weight.suggest_memory_format());
  float* output_data = output.data_ptr<float>();

#ifdef USE_FBGEMM
    at::parallel_for(
      0, input_rows, 1, [&](int32_t start_idx, int32_t end_idx) {
        for (int64_t row = start_idx; row < end_idx; ++row) {
          fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloatOrHalf<float>(
            input_data + row * input_columns,
            1,
            input_columns,
            output_data + row * output_columns);
        }
      });
#else
  for (auto row: c10::irange(input_rows)) {
    const std::uint8_t* input_row = input_data + row * input_columns;
    const float* input_row_scale_zp =
        reinterpret_cast<const float*>(input_row + output_columns);
    float* output_row = output_data + row * output_columns;

    for(auto col: c10::irange(output_columns)) {
      output_row[col] =
          input_row[col] * input_row_scale_zp[0] + input_row_scale_zp[1];
    } // output_columns
  } // input_rows
#endif // USE_FBGEMM
  return output;
}

Tensor _qembeddingbag_nbit_unpack_helper(
    const Tensor& packed_weight,
    int BIT_RATE) {
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
  float* output_data = output.data_ptr<float>();
#ifdef USE_FBGEMM
    at::parallel_for(
      0, input_rows, 1, [&](int32_t start_idx, int32_t end_idx) {
        for (int64_t row = start_idx; row < end_idx; ++row) {
          fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloatOrHalf<float>(BIT_RATE,
            input_data + row * input_columns,
            1,
            input_columns,
            output_data + row * output_dimensions[1]);
        }
      });
#else
  auto output_columns = output_dimensions[1];
  for (auto row: c10::irange(input_rows)) {
    float* output_row = output_data + row * output_columns;
    const std::uint8_t* input_row = input_data + row * input_columns;
    const at::Half* input_row_scale_zp = reinterpret_cast<const at::Half*>(
        input_row +
        (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
    float scale = input_row_scale_zp[0];
    float zero_point = input_row_scale_zp[1];

    for (int col = 0; col < output_columns; ++col) {
      std::uint8_t quantized = input_row[col / NUM_ELEM_PER_BYTE];
      quantized >>= (col % NUM_ELEM_PER_BYTE) * BIT_RATE;
      quantized &= (1 << BIT_RATE) - 1;
      output_row[col] = scale * quantized + zero_point;
    } // output_columns
  } // input_rows
#endif // USE_FBGEMM

  return output;
}

// De-quantizes the result of the qembeddingbag_4bit_prepack operator.
// The input is expected to first have quantized values,
// then 2-byte fp16 scale and 2-byte zero_offset.
// The output is a matrix containing only the values, but de-quantized.
// De-quantization is performed by multiplying each value by its
// row's scale and zero_point parameters. The de-quantized values
// will thus not be exactly equal to the original, un-quantized
// floating point values.
Tensor qembeddingbag_4bit_unpack(const Tensor& packed_weight) {
  return _qembeddingbag_nbit_unpack_helper(packed_weight, 4 /*BIT_RATE*/);
}

// De-quantizes the result of the qembeddingbag_2bit_prepack operator.
// The input is expected to first have quantized values,
// then 2-byte fp16 scale and 2-byte zero_offset.
// The output is a matrix containing only the values, but de-quantized.
// De-quantization is performed by multiplying each value by its
// row's scale and zero_point parameters. The de-quantized values
// will thus not be exactly equal to the original, un-quantized
// floating point values.
Tensor qembeddingbag_2bit_unpack(const Tensor& packed_weight) {
  return _qembeddingbag_nbit_unpack_helper(packed_weight, 2 /*BIT_RATE*/);
}

class QEmbeddingUnpackWeights final {
 public:
  
};

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_byte_unpack"),
      qembeddingbag_byte_unpack);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_4bit_unpack"),
      qembeddingbag_4bit_unpack);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_2bit_unpack"),
      qembeddingbag_2bit_unpack);
}

TORCH_LIBRARY_IMPL(quantized, CatchAll, m) {
  // Unpack the packed embedding_bag weights using TorchBind custom class.
  // TODO extend to support 4-bit qtensor.
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::embedding_bag_unpack"),
      TORCH_FN(QEmbeddingUnpackWeights::run));
}

} // namespace
} // namespace native
} // namespace at
