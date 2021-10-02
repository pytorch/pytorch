#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/reducer_functors.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

namespace internal {
inline bool is_little_endian() {
  constexpr std::int32_t kValue = 1;
  return reinterpret_cast<const std::uint8_t*>(&kValue)[0] == 1;
}

void convertfp32fp32(float* dst, const float* src, size_t N);
void convertfp16fp32(float* dst, const at::Half* src, size_t N);

/**
 * @params Xmin initial solution passed and potentiall better solution returns
 * @params Xmax initial solution passed and potentiall better solution returns
 */
void param_search_greedy(
    const float* X,
    int N,
    const int n_bins, // = 200,
    const float ratio, // = 0.16,
    float& Xmin,
    float& Xmax,
    int bit_rate);
} // namespace internal

// Fake 2/4 bit quantization
// Creates a 2/4bit rowwise quantized blob with scales and biases in fp16
// The storage format is 8 bit rowwise with scales and biases in fp32
template <
    int BIT_RATE,
    typename T,
    void (*convert)(float* dst, const T* src, size_t N),
    bool GREEDY = false>
class FloatToFusedNBitFakeRowwiseQuantizedOp final
    : public Operator<CPUContext> {
 public:
  FloatToFusedNBitFakeRowwiseQuantizedOp(const OperatorDef& def, Workspace* ws)
      : Operator<CPUContext>(def, ws) {}
  ~FloatToFusedNBitFakeRowwiseQuantizedOp() override {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE(internal::is_little_endian(), "Unsupported endianness");

    const auto& input = Input(DATA_FLOAT);

    const auto input_rows = input.size(0);
    const auto input_columns = input.size(1);
    CAFFE_ENFORCE_EQ(input.dim(), 2, "Expect input to be a matrix");

    const std::vector<int64_t> output_dimensions = {input_rows,
                                                    input_columns + 8};
    auto* output = Output(
        DATA_FUSED_SCALE_BIAS_INT8, output_dimensions, at::dtype<uint8_t>());

    const auto* input_data = input.template data<T>();
    auto* output_data = output->template mutable_data<uint8_t>();
    const auto output_columns = output->size(1);

    if (!std::is_same<T, float>::value && !std::is_same<T, at::Half>::value) {
      CAFFE_THROW("Unsupported data type");
    }

    bool use_openmp = GREEDY;
#ifdef _OPENMP
    vector<float> tmp_vec(input_columns * (GREEDY ? omp_get_max_threads() : 1));
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
      uint8_t* output_row = output_data + row * output_columns;
      float* output_row_scale_bias =
          reinterpret_cast<float*>(output_row + input_columns);

      float minimum_element = *std::min_element(tmp, tmp + input_columns);
      float maximum_element = *std::max_element(tmp, tmp + input_columns);

      if (GREEDY) {
        internal::param_search_greedy(
            tmp,
            input_columns,
            200,
            0.16,
            minimum_element,
            maximum_element,
            BIT_RATE);
      }

      minimum_element = static_cast<at::Half>(minimum_element);
      const float range = maximum_element - minimum_element;

      const float scale = range == 0
          ? 1.0f
          : static_cast<float>(static_cast<at::Half>(
                range / static_cast<float>((1 << BIT_RATE) - 1)));
      const float inverse_scale = 1.0f / scale;

      output_row_scale_bias[0] = scale;
      output_row_scale_bias[1] = minimum_element;

      // NOLINTNEXTLINE(clang-diagnostic-sign-compare)
      for (size_t col = 0; col < input_columns; ++col) {
        output_row[col] = std::max(
            0,
            std::min<int>(
                std::lrintf((tmp[col] - minimum_element) * inverse_scale),
                (1 << BIT_RATE) - 1));
      }
    }

    return true;
  }

 private:
  INPUT_TAGS(DATA_FLOAT);
  // INT8 suffix because this is a fake quantization operator whose output
  // type is always 8-bit regardless of BIT_RATE.
  OUTPUT_TAGS(DATA_FUSED_SCALE_BIAS_INT8);
};
} // namespace caffe2
