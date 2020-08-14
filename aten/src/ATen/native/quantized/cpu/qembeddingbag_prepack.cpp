#include <ATen/ATen.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace {

// Note - This is a temporary pack function for embedding bag which quantizes
// and packs the float weight tensor. In the next step it will be replaced by a
// quantize and pack function once we support FP scale and FP zero_point
Tensor qembeddingbag_byte_prepack(const Tensor& weight) {
  int64_t embedding_rows = weight.size(0);
  int64_t embedding_cols = weight.size(1);
  Tensor weight_contig = weight.contiguous(weight.suggest_memory_format());

  const float* weight_data = weight_contig.data_ptr<float>();
  std::vector<int64_t> output_shape = {
      embedding_rows,
      embedding_cols +
          8}; // extra 8 bytes to store FP scale and zero_point per row.
  size_t output_columns = output_shape[1];
  constexpr float kEpsilon = 1e-8f;

  // Allocate output packed weights
  auto output = at::empty(
      output_shape,
      weight_contig.options().dtype(at::kByte),
      weight_contig.suggest_memory_format());
  auto* output_data = output.data_ptr<uint8_t>();

  for (std::size_t row = 0; row < embedding_rows; ++row) {
    const float* input_row = weight_data + row * embedding_cols;
    std::uint8_t* output_row = output_data + row * output_columns;
    float* output_row_scale_zp =
        reinterpret_cast<float*>(output_row + embedding_cols);

    float minimum_element =
        *std::min_element(input_row, input_row + embedding_cols);
    float maximum_element =
        *std::max_element(input_row, input_row + embedding_cols);
    float range = maximum_element - minimum_element;

    output_row_scale_zp[0] = range / 255.0f;
    output_row_scale_zp[1] = minimum_element;
    const auto inverse_scale = 255.0f / (range + kEpsilon);
    for (std::size_t col = 0; col < embedding_cols; ++col) {
      output_row[col] =
          lrintf((input_row[col] - minimum_element) * inverse_scale);
    } // embedding_cols
  } // embedding_rows
  return output;
}

Tensor qembeddingbag_4bit_prepack(const Tensor& weight) {
  int64_t embedding_rows = weight.size(0);
  int64_t embedding_cols = weight.size(1);

  Tensor weight_contig = weight.contiguous(weight.suggest_memory_format());

  const auto weight_data = weight.data_ptr<float>();
  constexpr int BIT_RATE = 4;
  constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
  TORCH_CHECK(
      weight_contig.size(weight.dim() - 1) % NUM_ELEM_PER_BYTE == 0,
      "FloatToFused4BitRowwiseQuantizedOp only works for the number of "
      "columns a multiple of 2");

  // The "fused" representation stores the scale and bias with the
  // row-wise quantized data in one tensor.
  // Since we represent the scale and bias in 16-bit float, we'll use the
  // last 4 bytes of each row for scale (2 bytes) and bias (2 bytes).
  // | ... quantized data ... | scale | bias |
  // |    number_of_columns   |  2B   |  2B  |
  std::vector<int64_t> output_shape = {
      embedding_rows,
      static_cast<std::int64_t>(
          (embedding_cols + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
          2 * sizeof(at::Half))};
  auto output = at::empty(
      output_shape,
      weight_contig.options().dtype(at::kByte),
      weight_contig.suggest_memory_format());
  auto* output_data = output.data_ptr<uint8_t>();
  const auto output_columns = output.size(output.dim() - 1);

  for (int row = 0; row < embedding_rows; ++row) {
    const float* input_row = weight_data + row * embedding_cols;
    std::uint8_t* output_row = output_data + row * output_columns;

    float Xmin = *std::min_element(input_row, input_row + embedding_cols);
    float Xmax = *std::max_element(input_row, input_row + embedding_cols);

    Xmin = static_cast<at::Half>(Xmin);
    const float range = Xmax - Xmin;

    // Set scale to 1.0f for the corner case of Xmax == Xmin .
    // Any non-zero scale would work because during quantization
    // (X - Xmin) / scale will be 0 for all X unless scale is 0.
    at::Half scale = range == 0 ? 1.0f : range / ((1 << BIT_RATE) - 1);
    float inverse_scale = scale == 0 ? 1.0f : 1.0f / scale;
    if (scale == 0 || std::isinf(inverse_scale)) {
      // Corner case handling when Xmax == Xmin
      // Any scale would work because X - Xmin will be 0 for all X
      scale = 1.0f;
      inverse_scale = 1.0f;
    }

    // Update the scale and zero_point of each row.
    at::Half* output_row_scale_zp = reinterpret_cast<at::Half*>(
        output_row +
        (embedding_cols + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);

    output_row_scale_zp[0] = scale;
    output_row_scale_zp[1] = Xmin;

    // Pack the weight values.
    for (int col = 0; col < embedding_cols; ++col) {
      float X = input_row[col];
      std::uint8_t quantized = std::max(
          0,
          std::min<int>(
              lrintf((X - Xmin) * inverse_scale), (1 << BIT_RATE) - 1));
      // We pack 2 4-bit values in a byte. Index 0 is packed in the lower 4-bits
      // and index 1 is packed in the upper 4-bits.
      if (col % NUM_ELEM_PER_BYTE == 0) {
        output_row[col / NUM_ELEM_PER_BYTE] = quantized;
      } else {
        output_row[col / NUM_ELEM_PER_BYTE] |=
            (quantized << ((col % NUM_ELEM_PER_BYTE) * BIT_RATE));
      }
    } // embedding_cols
  } // embedding_rows
  return output;
}

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl("embedding_bag_byte_prepack", qembeddingbag_byte_prepack);
  m.impl("embedding_bag_4bit_prepack", qembeddingbag_4bit_prepack);
}

} // namespace
} // namespace native
} // namespace at
