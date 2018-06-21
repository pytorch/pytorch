#ifndef CAFFE2_OPERATORS_FUSED_ROWWISE_RAND_CONVERSION_OPS_H_
#define CAFFE2_OPERATORS_FUSED_ROWWISE_RAND_CONVERSION_OPS_H_

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

// define a custom template unary functor
// random quantization
template <typename Scalar>
struct CwiseRandQuantizeOp {
  CwiseRandQuantizeOp(
      CPUContext::rand_gen_type& gen,
      const uint8_t& bw,
      const bool& r,
      const Scalar& minv,
      const Scalar& maxv)
      : m_gen(gen),
        m_bitwdith(bw),
        m_random(r),
        m_min(minv),
        m_max(maxv),
        m_gap((maxv - minv) / ((1 << bw) - 1.f)),
        m_maxq((1 << bw) - 1) {}
  uint8_t operator()(const Scalar& fval) const {
    Scalar thetimes = (fval - m_min) / (m_gap + 1e-8f);
    thetimes = thetimes < 0 ? 0 : (thetimes > m_maxq ? m_maxq : thetimes);
    uint8_t floor_times = floor(thetimes);
    // the probability of quantizing to the larger discrete level
    Scalar p = thetimes - floor_times;
    if (!m_random) {
      p = p > 0.5;
    }
    // generate Bernoulli distribution
    std::bernoulli_distribution dist(p);
    uint8_t q = floor_times + dist(m_gen); // quantzied value
    // (1 << m_bitwdith) - 1 is to avoid contamination when q is dirty data
    return m_maxq & q;
  }

  CPUContext::rand_gen_type& m_gen;
  const uint8_t m_bitwdith; // bits per data
  const bool m_random; // false for unittest
  const Scalar m_min; // the minimum value
  const Scalar m_max; // the maximum value
  const Scalar m_gap; // the gap between two discrete levels
  const uint8_t m_maxq; // the max quantized value
};

// define a custom template binary functor
// left shift and or
template <typename T>
struct CwiseLeftShiftOrOp {
  CwiseLeftShiftOrOp(const size_t& s) : m_shift(s) {}
  const T operator()(const T& shiftval, const T& orval) const {
    return (T)(shiftval << m_shift) | orval;
  }
  const size_t m_shift; // bits per data
};

template <class Context>
class FloatToFusedRandRowwiseQuantizedOp : public Operator<Context> {
 public:
  static constexpr float kEqualityThreshold = 1e-7f;
  static constexpr float kEpsilon = 1e-8f;

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  FloatToFusedRandRowwiseQuantizedOp(
      const OperatorDef& operator_def,
      Workspace* ws)
      : Operator<Context>(operator_def, ws),
        bitwidth_(OperatorBase::GetSingleArgument<int32_t>("bitwidth", 8)),
        random_(OperatorBase::GetSingleArgument<bool>("random", true)) {
    CAFFE_ENFORCE(
        bitwidth_ == 1 || bitwidth_ == 2 || bitwidth_ == 4 || bitwidth_ == 8,
        "Unsupported bitwidth");
  }
  ~FloatToFusedRandRowwiseQuantizedOp() {}

  bool RunOnDevice() override {
    CAFFE_ENFORCE(IS_LITTLE_ENDIAN, "Unsupported endianness");

    const auto& input = Input(DATA_FLOAT);
    auto* output = Output(DATA_FUSED_QUANTIZED);

    // TODO extend to higher dimensional blob like conv filters and batched
    // feature maps
    CAFFE_ENFORCE_EQ(input.ndim(), 2, "Expect input to be a matrix");

    const auto input_rows = input.dim(0);
    const auto input_columns = input.dim(1);

    // The "fused" representation stores the [bitwidth][tail][min][max]
    // with the row-wise quantized data in one tensor. Since we store 8/bitwidth
    // quantized data in one byte, the last buckets of some bytes may have
    // unused bits. There are totally tail buckets are unused.
    // We encode *bitwidth* and *tail* at the beginning of
    // each row, following by 32-bit floating data respresenting min and max.
    // | bitwidth | tail | min | max | ... int8 data ... |
    // |    1B    |  1B  |  4B |  4B | ...output_data....|
    // In output_data: the b-th bucket of the i-th byte stores
    // the i-th data of the b-th segment of input row
    size_t data_per_byte = 8 / bitwidth_;
    size_t tail = input_columns % data_per_byte;
    tail = tail ? data_per_byte - tail : 0;
    // How many bytes in the output
    size_t segment_size = (input_columns + data_per_byte - 1) / data_per_byte;
    const std::vector<TIndex> output_dimensions = {input_rows,
                                                   10 + segment_size};
    output->Resize(output_dimensions);

    const auto* input_data = input.template data<float>();
    auto* output_data = output->template mutable_data<uint8_t>();
    const auto output_columns = output->dim(1);
    memset(output_data, 0, output->size());

    for (size_t row = 0; row < input_rows; ++row) {
      // memory pointers
      ConstEigenVectorArrayMap<float> input_row(
          input_data + row * input_columns, input_columns);
      uint8_t* output_row = output_data + row * output_columns;
      EigenVectorArrayMap<uint8_t> output_bitwidth_tail(output_row, 2);
      EigenVectorArrayMap<float> output_row_min_max(
          reinterpret_cast<float*>(output_row + 2), 2);

      // basic info
      const float minimum_element = input_row.minCoeff();
      const float maximum_element = input_row.maxCoeff();
      output_bitwidth_tail(0) = bitwidth_;
      output_bitwidth_tail(1) = tail;
      output_row_min_max(0) = minimum_element;
      output_row_min_max(1) = maximum_element;

      CwiseRandQuantizeOp<float> cwiserand(
          context_.RandGenerator(),
          bitwidth_,
          random_,
          minimum_element,
          maximum_element);
      CwiseLeftShiftOrOp<uint8_t> cwiseshftor(bitwidth_);

      for (int start = 0; start < input_columns; start += segment_size) {
        size_t stride = start + segment_size <= input_columns
            ? segment_size
            : input_columns - start;
        ConstEigenVectorArrayMap<float> sub_input_row(
            input_data + row * input_columns + start, stride);
        auto qvals = sub_input_row.unaryExpr(
            cwiserand); // some last buckets may have unused data
        EigenVectorArrayMap<uint8_t> output_seg(output_row + 10, stride);
        // shift previous bits and concatenate current ones
        output_seg = output_seg.binaryExpr(qvals, cwiseshftor);
      }
    }

    return true;
  }

 private:
  INPUT_TAGS(DATA_FLOAT);
  OUTPUT_TAGS(DATA_FUSED_QUANTIZED);

 protected:
  size_t bitwidth_{8};
  bool random_{true};
};

template <class Context>
class FusedRandRowwiseQuantizedToFloatOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(FusedRandRowwiseQuantizedToFloatOp)

  bool RunOnDevice() override {
    CAFFE_ENFORCE(IS_LITTLE_ENDIAN, "Unsupported endianness");

    // const auto& input = Input(DATA_FUSED_QUANTIZED);
    // auto* output = Output(DATA_FLOAT);

    // TODO out[0].set_data_type(TensorProto_DataType_FLOAT);
    // ...
    CAFFE_ENFORCE(0, "Not implemented yet.");
    return true;
  }

 private:
  INPUT_TAGS(DATA_FUSED_QUANTIZED);
  OUTPUT_TAGS(DATA_FLOAT);
};

#undef IS_LITTLE_ENDIAN

} // namespace caffe2

#endif // CAFFE2_OPERATORS_FUSED_ROWWISE_RAND_CONVERSION_OPS_H_
