#include "./fused_nbit_rowwise_conversion.h"

#include <c10/util/Half.h>
#include <algorithm>
#include <cmath>

#include "common.h"
#include "fbgemm/QuantUtils.h"

namespace caffe2 {

void FloatToFused8BitRowwiseQuantized(
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  fbgemm::FloatToFused8BitRowwiseQuantizedSBFloat(
      input, input_rows, input_columns, output);
}

void Fused8BitRowwiseQuantizedToFloat(
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  fbgemm::Fused8BitRowwiseQuantizedSBFloatToFloat(
      input, input_rows, input_columns, output);
}

void FloatToFusedNBitRowwiseQuantizedSBHalf(
    int bit_rate,
    const float* input,
    int input_rows,
    int input_columns,
    std::uint8_t* output) {
  fbgemm::FloatToFusedNBitRowwiseQuantizedSBHalf(
      bit_rate, input, input_rows, input_columns, output);
}

void FusedNBitRowwiseQuantizedSBHalfToFloat(
    int bit_rate,
    const std::uint8_t* input,
    int input_rows,
    int input_columns,
    float* output) {
  fbgemm::FusedNBitRowwiseQuantizedSBHalfToFloat(
      bit_rate, input, input_rows, input_columns, output);
}

} // namespace caffe2
