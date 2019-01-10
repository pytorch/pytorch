#ifndef CAFFE2_OPERATORS_FUSED_ROWWISE_8BIT_CONVERSION_OPS_H_
#define CAFFE2_OPERATORS_FUSED_ROWWISE_8BIT_CONVERSION_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/reducer_functors.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

#define IS_LITTLE_ENDIAN                                      \
  [] {                                                        \
    const int32_t kValue = 1;                                 \
    return reinterpret_cast<const uint8_t*>(&kValue)[0] == 1; \
  }()

template <class Context>
class FloatToFused8BitRowwiseQuantizedOp : public Operator<Context> {
 public:
  static constexpr float kEqualityThreshold = 1e-7f;
  static constexpr float kEpsilon = 1e-8f;

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FloatToFused8BitRowwiseQuantizedOp)

  bool RunOnDevice() override {
    CAFFE_ENFORCE(IS_LITTLE_ENDIAN, "Unsupported endianness");

    const auto& input = Input(DATA_FLOAT);
    auto* output = Output(DATA_FUSED_SCALE_BIAS_INT8);

    const auto input_rows = input.dim(0);
    const auto input_columns = input.dim(1);
    CAFFE_ENFORCE_EQ(input.ndim(), 2, "Expect input to be a matrix");

    // The "fused" representation stores the scale and bias with the row-wise
    // quantized data in one tensor. Since we quantize with 8 bits (1 byte) and
    // represent the scale and bias with 32-bit floats, we'll use the last 8
    // bytes of each row for scale (4 bytes) and bias (4 bytes).
    // | ... int8 data ... | scale | bias |
    // | number_of_columns |  4B   |  4B  |
    const std::vector<TIndex> output_dimensions = {input_rows,
                                                   input_columns + 8};
    output->Resize(output_dimensions);

    const auto* input_data = input.template data<float>();
    auto* output_data = output->template mutable_data<uint8_t>();
    const auto output_columns = output->dim(1);

    for (size_t row = 0; row < input_rows; ++row) {
      ConstEigenVectorArrayMap<float> input_row(
          input_data + row * input_columns, input_columns);

      uint8_t* output_row = output_data + row * output_columns;
      EigenVectorArrayMap<uint8_t> output_row_values(output_row, input_columns);
      EigenVectorArrayMap<float> output_row_scale_bias(
          reinterpret_cast<float*>(output_row + input_columns), 2);

      const float minimum_element = input_row.minCoeff();
      const float maximum_element = input_row.maxCoeff();
      const float range = maximum_element - minimum_element;

      output_row_scale_bias(0) = range / 255.0f;
      output_row_scale_bias(1) = minimum_element;
      const auto inverse_scale = 255.0f / (range + kEpsilon);
      output_row_values = ((input_row - minimum_element) * inverse_scale)
                              .round()
                              .cast<uint8_t>();
    }

    return true;
  }

 private:
  INPUT_TAGS(DATA_FLOAT);
  OUTPUT_TAGS(DATA_FUSED_SCALE_BIAS_INT8);
};

template <class Context>
class Fused8BitRowwiseQuantizedToFloatOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(Fused8BitRowwiseQuantizedToFloatOp)

  bool RunOnDevice() override {
    CAFFE_ENFORCE(IS_LITTLE_ENDIAN, "Unsupported endianness");

    const auto& input = Input(DATA_FUSED_SCALE_BIAS_INT8);
    auto* output = Output(DATA_FLOAT);

    const auto input_rows = input.dim(0);
    const auto input_columns = input.dim(1);
    CAFFE_ENFORCE_EQ(input.ndim(), 2, "Expect input to be a matrix");

    // The last 8 bytes per row are the scale and the bias. The rest of
    // input_columns is the number of values in the original row.
    const std::vector<TIndex> output_dimensions = {input_rows,
                                                   input_columns - 8};
    output->Resize(output_dimensions);
    const auto output_columns = output->dim(1);

    const auto* input_data = input.template data<uint8_t>();
    auto* output_data = output->template mutable_data<float>();

    for (size_t row = 0; row < input_rows; ++row) {
      const uint8_t* input_row = input_data + row * input_columns;
      ConstEigenVectorArrayMap<uint8_t> input_row_values(
          input_row, output_columns);
      ConstEigenVectorArrayMap<float> input_row_scale_bias(
          reinterpret_cast<const float*>(input_row + output_columns), 2);

      EigenVectorArrayMap<float> output_row(
          output_data + row * output_columns, output_columns);

      output_row = input_row_values.cast<float>() * input_row_scale_bias(0) +
          input_row_scale_bias(1);
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
