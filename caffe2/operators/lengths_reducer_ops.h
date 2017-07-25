#pragma once
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/perfkernels/typed_axpy.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

// A templated class that implements SparseLengths[Sum,WeightedSum,Mean].
template <
    typename T, // output type
    class InputTypes, // supported input types, such as TensorTypes<float>
    bool USE_WEIGHT = 0, // Whether it is SparseLengthsWeightedSum
    bool USE_MEAN = 0 // Whether this is SparseLengthsMean
    >
class CPUSparseLengthsReductionOp : public Operator<CPUContext> {
 public:
  USE_OPERATOR_FUNCTIONS(CPUContext);
  CPUSparseLengthsReductionOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {
    static_assert(
        !(USE_WEIGHT & USE_MEAN), "Cannot both specify weight and mean.");
  }

  ~CPUSparseLengthsReductionOp() {}

  // Currently, we support float and float16 inputs for input data type, and
  // int32_t and int64_t for the index type.

  bool RunOnDevice() override {
    return DispatchHelper<InputTypes>::call(this, Input(DATA));
  }

  template <typename InputType>
  bool DoRunWithType() {
    return DispatchHelper<TensorTypes2<int32_t, int64_t>, InputType>::call(
        this, Input(INDICES));
  }

  template <typename InputType, typename IndexType>
  bool DoRunWithType2() {
    auto& dataInput = Input(DATA);
    auto& indicesInput = Input(INDICES);
    auto& lengthsInput = Input(LENGTHS);

    CAFFE_ENFORCE_EQ(1, indicesInput.ndim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.ndim(), "LENGTHS must be a vector");
    const TIndex N = dataInput.dim(0);
    const int D = dataInput.size_from_dim(1);
    const TIndex M = lengthsInput.dim(0);
    const TIndex indices_size = indicesInput.size();

    auto* output = Output(0);
    auto shape = dataInput.dims();
    shape[0] = M;
    output->Resize(shape);
    T* out_data = output->template mutable_data<T>();

    const InputType* in_data = dataInput.template data<InputType>();
    const IndexType* indices = indicesInput.template data<IndexType>();
    const int* lengths = lengthsInput.template data<int>();
    const T* in_weight = nullptr;

    if (USE_WEIGHT) { // static if
      auto& weightInput = Input(WEIGHT);
      CAFFE_ENFORCE_EQ(
          weightInput.size(),
          indices_size,
          "Weight should have the same length as indices.");
      in_weight = weightInput.template data<T>();
    }

    TIndex current = 0;
    for (int m = 0; m < M; ++m) {
      memset(out_data, 0, sizeof(T) * D);
      for (int i = 0; i < lengths[m]; ++i) {
        CAFFE_ENFORCE_LT(current, indices_size);
        CAFFE_ENFORCE_LT(indices[current], N);
#ifdef __GNUC__
        if (current + 1 < indices_size) {
          __builtin_prefetch(in_data + D * indices[current + 1], 0, 1);
        }
#endif // __GNUC__
        TypedAxpy<InputType, T>(
            D,
            USE_WEIGHT ? in_weight[current] : 1.0,
            in_data + D * indices[current],
            out_data);
        ++current;
      }

      if (USE_MEAN && lengths[m]) { // static if
        math::Scale<T, CPUContext>(
            D, 1.f / lengths[m], out_data, out_data, &context_);
      }
      out_data += D;
    }
    CAFFE_ENFORCE_EQ(
        current,
        indicesInput.size(),
        "Your input seems to be incorrect: the sum of lengths values should be "
        "the size of the indices tensor, but it appears not.");
    return true;
  }

 private:
  enum {
    DATA = 0, // Data input.
    WEIGHT = 1, // Weight input used in SparseLengthsWeightedSum
    INDICES = 1 + USE_WEIGHT, // 1 in SparseLengths[Sum,Mean] and
                              // 2 in SparseLengthsWeightedSum
    LENGTHS = 2 + USE_WEIGHT, // 2 in SparseLengths[Sum, Mean],
                              // 3 in SparseLengthsWeightedSum
  };
};

} // namespace caffe2
