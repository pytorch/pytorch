#include <ATen/ATen.h>
#include <torch/library.h>

namespace at {
namespace native {
namespace {

Tensor qembeddingbag_byte_unpack(const Tensor& packed_weight) {
  const auto input_rows = packed_weight.size(0);
  const auto input_columns = packed_weight.size(1);

  // The last 2 values are used to store the FP32 scale and zero_point values
  // per row.
  int output_columns = input_columns - 2 * sizeof(float);

  const auto* input = packed_weight.data_ptr<uint8_t>();
  std::vector<int64_t> output_shape = {input_rows, output_columns};
  at::Tensor output = at::empty(
      output_shape,
      packed_weight.options().dtype(kFloat),
      packed_weight.suggest_memory_format());
  float* output_data = output.data_ptr<float>();

  for (std::size_t row = 0; row < input_rows; ++row) {
    const std::uint8_t* input_row = input + row * input_columns;
    const float* input_row_scale_zp =
        reinterpret_cast<const float*>(input_row + output_columns);
    float* output_row = output_data + row * output_columns;

    for (std::size_t col = 0; col < output_columns; ++col) {
      output_row[col] =
          input_row[col] * input_row_scale_zp[0] + input_row_scale_zp[1];
    } // output_columns
  } // input_rows
  return output;
}

Tensor qembeddingbag_4bit_unpack(const Tensor& packed_weight) {
  const auto input_rows = packed_weight.size(0);
  const auto input_columns = packed_weight.size(1);
  const auto* input_data = packed_weight.data_ptr<uint8_t>();
  constexpr int NUM_ELEM_PER_BYTE = 2;
  constexpr int BIT_RATE = 4;

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
  auto output_columns = output_dimensions[1];
  for (size_t row = 0; row < input_rows; ++row) {
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
  return output;
}

TORCH_LIBRARY_IMPL(quantized, CPU, m) {
  m.impl("embedding_bag_byte_unpack", qembeddingbag_byte_unpack);
  m.impl("embedding_bag_4bit_unpack", qembeddingbag_4bit_unpack);
}
} // namespace
} // namespace native
} // namespace at
