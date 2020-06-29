#pragma once

#include <algorithm>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
// for param_search_greedy
#include "caffe2/operators/fused_rowwise_nbitfake_conversion_ops.h"
#include "caffe2/perfkernels/fused_nbit_rowwise_conversion.h"

namespace caffe2 {

template <
    int BIT_RATE,
    typename T,
    void (*convert)(float* dst, const T* src, size_t N),
    bool GREEDY = false>
class FloatToFusedNBitRowwiseQuantizedOp final : public Operator<CPUContext> {
 public:
  FloatToFusedNBitRowwiseQuantizedOp(const OperatorDef& def, Workspace* ws)
      : Operator<CPUContext>(def, ws) {}
  ~FloatToFusedNBitRowwiseQuantizedOp() override {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE(internal::is_little_endian(), "Unsupported endianness");

    const auto& input = Input(DATA_FLOAT);

    CAFFE_ENFORCE_GT(input.dim(), 0, "Input's dimension must be at least 1");
    const auto input_rows = input.size_to_dim(input.dim() - 1);
    const auto input_columns = input.size(input.dim() - 1);
    static_assert(8 % BIT_RATE == 0, "BIT_RATE must divide 8");
    constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;
    CAFFE_ENFORCE_EQ(
        input.dim(input.dim() - 1) % NUM_ELEM_PER_BYTE,
        0,
        "FloatToFused" + caffe2::to_string(BIT_RATE) +
            "BitRowwiseQuantizedOp only works for the number of "
            "columns a multiple of " +
            caffe2::to_string(NUM_ELEM_PER_BYTE));

    // The "fused" representation stores the scale and bias with the
    // row-wise quantized data in one tensor.
    // Since we represent the scale and bias in 16-bit float, we'll use the
    // last 4 bytes of each row for scale (2 bytes) and bias (2 bytes).
    // | ... quantized data ... | scale | bias |
    // |    number_of_columns   |  2B   |  2B  |
    auto output_dimensions = input.sizes().vec();
    output_dimensions[input.dim() - 1] = static_cast<std::int64_t>(
        (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
        2 * sizeof(at::Half));
    auto* output = Output(
        DATA_FUSED_SCALE_BIAS, output_dimensions, at::dtype<std::uint8_t>());

    const auto* input_data = input.template data<T>();
    auto* output_data = output->template mutable_data<std::uint8_t>();

    if (!GREEDY && std::is_same<T, float>::value) {
      // fast path
      CAFFE_ENFORCE(
          reinterpret_cast<void (*)(float*, const float*, std::size_t)>(
              convert) == internal::convertfp32fp32,
          "When T == float, convert must be convertfp32fp32");
      FloatToFusedNBitRowwiseQuantizedSBHalf(
          BIT_RATE,
          reinterpret_cast<const float*>(input_data),
          input_rows,
          input_columns,
          output_data);
    } else {
      const auto output_columns = output->size(output->dim() - 1);

#ifdef _OPENMP
      vector<float> tmp_vec(
          input_columns * (GREEDY ? omp_get_max_threads() : 1));
#else
      vector<float> tmp_vec(input_columns);
#endif

#pragma omp parallel for if (GREEDY)
      for (int row = 0; row < input_rows; ++row) {
        float* tmp = tmp_vec.data();
#ifdef _OPENMP
        if (GREEDY) {
          tmp = &tmp_vec[omp_get_thread_num() * input_columns];
        }
#endif
        convert(tmp, input_data + row * input_columns, input_columns);

        std::uint8_t* output_row = output_data + row * output_columns;
        at::Half* output_row_scale = reinterpret_cast<at::Half*>(
            output_row +
            (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
        at::Half* output_row_bias = reinterpret_cast<at::Half*>(
            output_row +
            (input_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
            sizeof(at::Half));

        float Xmin = *std::min_element(tmp, tmp + input_columns);
        float Xmax = *std::max_element(tmp, tmp + input_columns);

        if (GREEDY) {
          internal::param_search_greedy(
              tmp, input_columns, 200, 0.16, Xmin, Xmax, BIT_RATE);
        }

        // Round Xmin to fp16 to match with dequantization that will use fp16
        // for Xmin.
        Xmin = static_cast<at::Half>(Xmin);
        const float range = Xmax - Xmin;
        // Round scale to fp16 to match with dequantization that will use fp16
        // for scale.
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

        *output_row_scale = scale;
        *output_row_bias = Xmin;

        for (int col = 0; col < input_columns; ++col) {
          float X = tmp[col];
          std::uint8_t quantized = std::max(
              0,
              std::min<int>(
                  std::lrintf((X - Xmin) * inverse_scale),
                  (1 << BIT_RATE) - 1));
          if (col % NUM_ELEM_PER_BYTE == 0) {
            output_row[col / NUM_ELEM_PER_BYTE] = quantized;
          } else {
            output_row[col / NUM_ELEM_PER_BYTE] |=
                (quantized << ((col % NUM_ELEM_PER_BYTE) * BIT_RATE));
          }
        }
      }
    } // GREEDY || !std::is_same<T, float>::value

    return true;
  }

 private:
  INPUT_TAGS(DATA_FLOAT);
  OUTPUT_TAGS(DATA_FUSED_SCALE_BIAS);
};

template <
    int BIT_RATE,
    typename T,
    void (*convert)(T* dst, const float* src, size_t N)>
class FusedNBitRowwiseQuantizedToFloatOp final : public Operator<CPUContext> {
 public:
  FusedNBitRowwiseQuantizedToFloatOp(const OperatorDef& def, Workspace* ws)
      : Operator<CPUContext>(def, ws) {}
  ~FusedNBitRowwiseQuantizedToFloatOp() override {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE(internal::is_little_endian(), "Unsupported endianness");

    const auto& input = Input(DATA_FUSED_SCALE_BIAS);

    CAFFE_ENFORCE_GT(input.dim(), 0, "Input's dimension must be at least 1");
    const auto input_rows = input.size_to_dim(input.dim() - 1);
    const auto input_columns = input.size(input.dim() - 1);

    static_assert(8 % BIT_RATE == 0, "BIT_RATE must divide 8");
    constexpr int NUM_ELEM_PER_BYTE = 8 / BIT_RATE;

    // The last 4 bytes per row are two fp16 scale and bias.
    // The rest of input_columns is the number of values in the original row.
    auto output_dimensions = input.sizes().vec();
    output_dimensions[input.dim() - 1] =
        static_cast<std::int64_t>(input_columns - 2 * sizeof(at::Half)) *
        NUM_ELEM_PER_BYTE;
    auto* output = Output(DATA_FLOAT, output_dimensions, at::dtype<T>());
    const auto output_columns = output->size(output->dim() - 1);

    const auto* input_data = input.template data<std::uint8_t>();
    T* output_data = output->template mutable_data<T>();

    if (std::is_same<T, float>::value) {
      // fast path
      CAFFE_ENFORCE(
          reinterpret_cast<void (*)(float*, const float*, std::size_t)>(
              convert) == internal::convertfp32fp32,
          "When T == float, convert must be convertfp32fp32");
      FusedNBitRowwiseQuantizedSBHalfToFloat(
          BIT_RATE,
          input_data,
          input_rows,
          input_columns,
          reinterpret_cast<float*>(output_data));
    } else {
      std::vector<float> tmp(output_columns);

      for (size_t row = 0; row < input_rows; ++row) {
        const std::uint8_t* input_row = input_data + row * input_columns;
        float scale = *reinterpret_cast<const at::Half*>(
            input_row +
            (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE);
        float bias = *reinterpret_cast<const at::Half*>(
            input_row +
            (output_columns + NUM_ELEM_PER_BYTE - 1) / NUM_ELEM_PER_BYTE +
            sizeof(at::Half));

        for (int col = 0; col < output_columns; ++col) {
          std::uint8_t quantized = input_row[col / NUM_ELEM_PER_BYTE];
          quantized >>= (col % NUM_ELEM_PER_BYTE) * BIT_RATE;
          quantized &= (1 << BIT_RATE) - 1;
          tmp[col] = scale * quantized + bias;
        }

        convert(output_data + row * output_columns, tmp.data(), output_columns);
      }
    }

    return true;
  }

 private:
  INPUT_TAGS(DATA_FUSED_SCALE_BIAS);
  OUTPUT_TAGS(DATA_FLOAT);
};
} // namespace caffe2
