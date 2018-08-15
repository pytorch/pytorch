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

#define QEPSILON 1e-8f
// define a custom template unary functor
// stochastic quantization
template <typename Scalar>
struct CwiseRandQuantizeOp {
  CwiseRandQuantizeOp(
      CPUContext::rand_gen_type& gen,
      const uint8_t& bitwidth,
      const bool& r,
      const Scalar& minv,
      const Scalar& maxv)
      : gen_(gen),
        bitwidth_(bitwidth),
        random_(r),
        min_(minv),
        max_(maxv),
        gap_((maxv - minv) / ((1 << bitwidth) - 1.f)),
        maxq_((1 << bitwidth) - 1) {}
  uint8_t operator()(const Scalar& fval) const {
    Scalar thetimes = (fval - min_) / (gap_ + QEPSILON);
    thetimes = thetimes < 0 ? 0 : (thetimes > maxq_ ? maxq_ : thetimes);
    uint8_t floor_times = floor(thetimes);
    // the probability of quantizing to the larger discrete level
    Scalar p = thetimes - floor_times;
    // generate Bernoulli distribution
    std::bernoulli_distribution dist(p);
    uint8_t q =
        floor_times + (random_ ? dist(gen_) : p > 0.5); // quantized value
    // (1 << bitwidth_) - 1 is to avoid contamination when q is dirty data
    return (uint8_t)(maxq_ & q);
  }

  CPUContext::rand_gen_type& gen_;
  const uint8_t bitwidth_; // bits per data
  const bool random_; // false for unittest
  const Scalar min_; // the minimum value
  const Scalar max_; // the maximum value
  const Scalar gap_; // the gap between two discrete levels
  const uint8_t maxq_; // the max quantized value
};

// define a custom template binary functor
// left shift and or
template <typename T>
struct CwiseLeftShiftOrOp {
  CwiseLeftShiftOrOp(const size_t& s) : shift_(s) {}
  const T operator()(const T& orval, const T& shiftval) const {
    return (T)(orval | (shiftval << shift_));
  }
  const size_t shift_; // bits per data
};

// define a custom template unary functor
// select [start, start+bitwidth) bits and convert to a value
template <typename T, typename D>
struct CwiseBitsValueOp {
  CwiseBitsValueOp(const size_t& start, const size_t& bitwidth)
      : start_(start), mask_((1 << bitwidth) - 1) {}
  const T operator()(const D& bytes) const {
    return (T)((bytes >> start_) & mask_);
  }
  const size_t start_; // start index of the bits
  const D mask_;
};

template <class Context>
class FloatToFusedRandRowwiseQuantizedOp : public Operator<Context> {
 public:
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

    CAFFE_ENFORCE_EQ(
        input.ndim(),
        2,
        "Expect input to be a matrix. Reshape the input tensor to a matrix for usage.");

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
    const std::vector<TIndex> output_dimensions = {
        input_rows, static_cast<TIndex>(10 + segment_size)};
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
      size_t bit_start = 0;
      for (int start = 0; start < input_columns; start += segment_size) {
        CwiseLeftShiftOrOp<uint8_t> cwiseshftor(bit_start);
        bit_start += bitwidth_;
        size_t stride = start + segment_size <= input_columns
            ? segment_size
            : input_columns - start;
        ConstEigenVectorArrayMap<float> sub_input_row(
            input_data + row * input_columns + start, stride);
        auto qvals = sub_input_row.unaryExpr(
            cwiserand); // some last buckets may have unused data
        EigenVectorArrayMap<uint8_t> output_seg(output_row + 10, stride);
        // shift and concatenate current ones
        output_seg = output_seg.binaryExpr(qvals.cast<uint8_t>(), cwiseshftor);
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

    const auto& input = Input(DATA_FUSED_QUANTIZED);
    auto* output = Output(DATA_FLOAT);
    CAFFE_ENFORCE_EQ(input.ndim(), 2, "Expect input to be a matrix.");
    CAFFE_ENFORCE_GE(
        input.size(),
        4,
        "Expect input to have size greater than or equal to 4.");

    const auto input_rows = input.dim(0);
    const auto input_columns = input.dim(1);
    const auto* input_data = input.template data<uint8_t>();
    const size_t bitwidth = input_data[0];
    CAFFE_ENFORCE(
        bitwidth == 1 || bitwidth == 2 || bitwidth == 4 || bitwidth == 8,
        "Unsupported bitwidth");
    const size_t tail = input_data[1];
    const size_t output_columns = (input_columns - 10) * (8 / bitwidth) - tail;
    const std::vector<TIndex> output_dimensions = {
        input_rows, static_cast<TIndex>(output_columns)};
    output->Resize(output_dimensions);
    auto* output_data = output->template mutable_data<float>();
    for (size_t row = 0; row < input_rows; ++row) {
      // memory pointers
      const uint8_t* input_row = input_data + row * input_columns;
      ConstEigenVectorArrayMap<uint8_t> input_bitwidth_tail(input_row, 2);
      ConstEigenVectorArrayMap<float> input_row_min_max(
          reinterpret_cast<const float*>(input_row + 2), 2);
      EigenVectorArrayMap<float> output_row(
          output_data + row * output_columns, output_columns);

      // basic info
      const float minimum_element = input_row_min_max(0);
      const float maximum_element = input_row_min_max(1);
      const float gap =
          (maximum_element - minimum_element) / ((1 << bitwidth) - 1.f) +
          QEPSILON; // for exact recovering
      CAFFE_ENFORCE_EQ(
          bitwidth,
          input_bitwidth_tail(0),
          "Expect each row have the same bitwidth");
      CAFFE_ENFORCE_EQ(
          tail, input_bitwidth_tail(1), "Expect each row have the same tail");

      // decoding
      size_t bit_start = 0;
      const size_t segment_size = input_columns - 10;
      for (int start = 0; start < output_columns; start += segment_size) {
        CwiseBitsValueOp<float, uint8_t> cwisedec(bit_start, bitwidth);
        bit_start += bitwidth;
        size_t stride = start + segment_size <= output_columns
            ? segment_size
            : output_columns - start;
        EigenVectorArrayMap<float> sub_output_row(
            output_data + row * output_columns + start, stride);
        ConstEigenVectorArrayMap<uint8_t> input_seg(input_row + 10, stride);
        sub_output_row = input_seg.unaryExpr(cwisedec).cast<float>();
      }
      // scaling and biasing
      output_row = output_row * gap + minimum_element;
    }

    return true;
  }

 private:
  INPUT_TAGS(DATA_FUSED_QUANTIZED);
  OUTPUT_TAGS(DATA_FLOAT);
};

#undef IS_LITTLE_ENDIAN
#undef QEPSILON
} // namespace caffe2

#endif // CAFFE2_OPERATORS_FUSED_ROWWISE_RAND_CONVERSION_OPS_H_
