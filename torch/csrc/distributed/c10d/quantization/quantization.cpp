#include <torch/csrc/distributed/c10d/quantization/quantization.h>
#include <torch/csrc/distributed/c10d/quantization/quantization_utils.h>
#include <torch/library.h>

namespace torch {
namespace distributed {
namespace c10d {
namespace quantization {

// TODO: The kernels are copied from fbgemm_gpu, we should dedup them later

static void FloatToBFloat16Quantized_ref(
    const float* const input,
    const size_t nrows,
    const size_t ncols,
    uint16_t* const output) {
  for (const auto row : c10::irange(nrows)) {
    const float* input_row = input + row * ncols;
    uint16_t* output_row = output + row * ncols;

    for (const auto col : c10::irange(ncols)) {
      output_row[col] =
          (*reinterpret_cast<const uint32_t*>(input_row + col) + (1 << 15)) >>
          16;
    }
  }
}

static void BFloat16QuantizedToFloat_ref(
    const at::BFloat16* const input,
    const size_t nrows,
    const size_t ncols,
    float* const output) {
  const int32_t output_columns = ncols;

  for (const auto row : c10::irange(nrows)) {
    const at::BFloat16* input_row = input + row * ncols;
    float* output_row = output + row * output_columns;

    for (const auto col : c10::irange(ncols)) {
      uint32_t val_fp32 = static_cast<uint32_t>(
                              reinterpret_cast<const uint16_t*>(input_row)[col])
          << 16;
      reinterpret_cast<uint32_t*>(output_row)[col] = val_fp32;
    }
  }
}

at::Tensor _float_to_bfloat16_cpu(const at::Tensor& input) {
  TENSOR_ON_CPU(input);
  // Currently it supports 2D inputs
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const int32_t nrows = input_sizes[0];
  const int32_t ncols = input_sizes[1];
  const int32_t output_columns = ncols;
  auto output =
      at::empty({nrows, output_columns}, input.options().dtype(at::kHalf));

  FloatToBFloat16Quantized_ref(
      input.const_data_ptr<float>(),
      nrows,
      ncols,
      reinterpret_cast<uint16_t*>(output.mutable_data_ptr<at::Half>()));

  return output;
}

at::Tensor _bfloat16_to_float_cpu(const at::Tensor& input) {
  TENSOR_ON_CPU(input);
  // Currently it supports 2D inputs
  TENSOR_NDIM_EQUALS(input, 2);

  const auto input_sizes = input.sizes();
  const int32_t nrows = input_sizes[0];
  const int32_t ncols = input_sizes[1];
  const int32_t output_columns = ncols;

  auto output = at::empty(
      {nrows, output_columns}, // 4 = sizeof(float)
      input.options().dtype(at::kFloat)); //
  BFloat16QuantizedToFloat_ref(
      reinterpret_cast<const at::BFloat16*>(input.const_data_ptr<at::Half>()),
      nrows,
      ncols,
      output.mutable_data_ptr<float>());

  return output;
}

TORCH_LIBRARY(quantization, m) {
  m.def("_Bfloat16QuantizedToFloat(Tensor input) -> Tensor");
  m.def("_FloatToBfloat16Quantized(Tensor input) -> Tensor");
}

TORCH_LIBRARY_IMPL(quantization, CPU, m) {
  m.impl("_Bfloat16QuantizedToFloat", _bfloat16_to_float_cpu);
  m.impl("_FloatToBfloat16Quantized", _float_to_bfloat16_cpu);
}

} // namespace quantization
} // namespace c10d
} // namespace distributed
} // namespace torch
