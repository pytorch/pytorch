#ifndef CAFFE2_OPERATORS_FUSED_ROWWISE_8BIT_CONVERSION_OPS_H_
#define CAFFE2_OPERATORS_FUSED_ROWWISE_8BIT_CONVERSION_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include <c10/util/irange.h>
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/reducer_functors.h"
#include "caffe2/perfkernels/fused_nbit_rowwise_conversion.h"
#include "caffe2/utils/math.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(Fused8BitRowwiseQuantizedToFloat);

namespace caffe2 {

#define IS_LITTLE_ENDIAN                                           \
  [] {                                                             \
    const int32_t kValue = 1;                                      \
    return reinterpret_cast<const std::uint8_t*>(&kValue)[0] == 1; \
  }()

template <
    typename T,
    typename Tsb, // Type for Scale and Bias
    void (*convert)(float* dst, const T* src, size_t N),
    bool HAS_CONVERT,
    class Context>
class FloatToFused8BitRowwiseQuantizedOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FloatToFused8BitRowwiseQuantizedOp)

  bool RunOnDevice() override {
    const auto& input = Input(DATA_FLOAT);

    CAFFE_ENFORCE_GT(input.dim(), 0, "Input's dimension must be at least 1");
    const auto input_rows = input.size_to_dim(input.dim() - 1);
    const auto input_columns = input.size(input.dim() - 1);

    // The "fused" representation stores the scale and bias with the row-wise
    // quantized data in one tensor. Since we quantize with 8 bits (1 byte) and
    // represent the scale and bias with 32-bit floats, we'll use the last 8
    // bytes of each row for scale (4 bytes) and bias (4 bytes).
    // | ... int8 data ... | scale       | bias       |
    // | number_of_columns |  sizeof(Tsb)| sizeof(Tsb)|
    auto output_dimensions = input.sizes().vec();
    output_dimensions[input.dim() - 1] =
        input_columns + 2 * static_cast<std::int64_t>(sizeof(Tsb));
    auto* output = Output(
        DATA_FUSED_SCALE_BIAS_INT8,
        output_dimensions,
        at::dtype<std::uint8_t>());

    const auto* input_data = input.template data<T>();
    auto* output_data = output->template mutable_data<std::uint8_t>();
    const auto output_columns = output->size(output->dim() - 1);

    bool is_float = std::is_same<T, float>::value;
    bool out_sb_half = std::is_same<Tsb, at::Half>::value;

    if (!HAS_CONVERT) {
      CAFFE_ENFORCE(is_float, "convert can be nullptr only if T is float");
      if (out_sb_half) {
        FloatToFusedNBitRowwiseQuantizedSBHalf(
            8,
            reinterpret_cast<const float*>(input_data),
            input_rows,
            input_columns,
            output_data);
      } else {
        FloatToFused8BitRowwiseQuantized(
            reinterpret_cast<const float*>(input_data),
            input_rows,
            input_columns,
            output_data);
      }
    } else {
      bool is_half = std::is_same<T, at::Half>::value;
      CAFFE_ENFORCE(is_half);

      vector<float> tmp(input_columns);
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (const auto row : c10::irange(input_rows)) {
        convert(tmp.data(), input_data + row * input_columns, input_columns);
        if (out_sb_half) {
          FloatToFusedNBitRowwiseQuantizedSBHalf(
              8,
              tmp.data(),
              1,
              input_columns,
              output_data + row * output_columns);
        } else {
          FloatToFused8BitRowwiseQuantized(
              tmp.data(), 1, input_columns, output_data + row * output_columns);
        }
      }
    }

    return true;
  }

 private:
  INPUT_TAGS(DATA_FLOAT);
  OUTPUT_TAGS(DATA_FUSED_SCALE_BIAS_INT8);
};

template <
    typename T,
    typename Tsb,
    void (*convert)(T* dst, const float* src, size_t N),
    bool HAS_CONVERT,
    class Context>
class Fused8BitRowwiseQuantizedToFloatOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(Fused8BitRowwiseQuantizedToFloatOp)

  bool RunOnDevice() override {
    const auto& input = Input(DATA_FUSED_SCALE_BIAS_INT8);

    CAFFE_ENFORCE_GT(input.dim(), 0, "Input's dimension must be at least 1");
    const auto input_rows = input.size_to_dim(input.dim() - 1);
    const auto input_columns = input.size(input.dim() - 1);

    // The last 2*sizeof(Tsb) bytes per row are the scale and the bias.
    // The rest of input_columns is the number of values in the original row.
    auto output_dimensions = input.sizes().vec();
    output_dimensions[input.dim() - 1] =
        input_columns - 2 * static_cast<std::int64_t>(sizeof(Tsb));
    auto* output = Output(DATA_FLOAT, output_dimensions, at::dtype<T>());
    const auto output_columns = output->size(output->dim() - 1);

    const auto* input_data = input.template data<std::uint8_t>();
    T* output_data = output->template mutable_data<T>();

    bool is_float = std::is_same<T, float>::value;
    bool in_sb_half = std::is_same<Tsb, at::Half>::value;

    if (!HAS_CONVERT) {
      CAFFE_ENFORCE(is_float, "convert can be nullptr only if T is float");

      if (in_sb_half) {
        FusedNBitRowwiseQuantizedSBHalfToFloat(
            8,
            input_data,
            input_rows,
            input_columns,
            reinterpret_cast<float*>(output_data));
      } else {
        Fused8BitRowwiseQuantizedToFloat(
            input_data,
            input_rows,
            input_columns,
            reinterpret_cast<float*>(output_data));
      }
    } else {
      bool is_half = std::is_same<T, at::Half>::value;
      CAFFE_ENFORCE(is_half);

      vector<float> tmp(input_columns);
      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (const auto row : c10::irange(input_rows)) {
        if (in_sb_half) {
          FusedNBitRowwiseQuantizedSBHalfToFloat(
              8,
              input_data + row * input_columns,
              1,
              input_columns,
              tmp.data());
        } else {
          Fused8BitRowwiseQuantizedToFloat(
              input_data + row * input_columns, 1, input_columns, tmp.data());
        }
        convert(output_data + row * output_columns, tmp.data(), output_columns);
      }
    }

    return true;
  }

 private:
  INPUT_TAGS(DATA_FUSED_SCALE_BIAS_INT8);
  OUTPUT_TAGS(DATA_FLOAT);
};

#undef IS_LITTLE_ENDIAN

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FUSED_ROWWISE_8BIT_CONVERSION_OPS_H_
