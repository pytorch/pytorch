#pragma once
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/perfkernels/embedding_lookup.h"

namespace caffe2 {

// A templated class that implements SparseLengths[Sum,WeightedSum,Mean].
template <
    typename T, // output type
    class InputTypes, // supported input types, such as TensorTypes<float>
    bool USE_WEIGHT = 0, // Whether it is SparseLengthsWeightedSum
    bool USE_MEAN = 0, // Whether this is SparseLengthsMean
    bool USE_POSITIONAL_WEIGHT = 0
    // USE_WEIGHT = 1 and USE_POSITIONAL_WEIGHT = 1
    // -> SparseLengthsPositionalWeightedSum
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

  // Currently, we support float and at::Half inputs for input data type, and
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

    CAFFE_ENFORCE_EQ(1, indicesInput.dim(), "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(1, lengthsInput.dim(), "LENGTHS must be a vector");
    const int64_t N = dataInput.size(0);
    const int D = dataInput.size_from_dim(1);
    const int64_t M = lengthsInput.size(0);
    const int64_t indices_size = indicesInput.numel();

    auto* output = Output(0);
    auto shape = dataInput.sizes().vec();
    shape[0] = M;
    output->Resize(shape);
    T* out_data = output->template mutable_data<T>();

    const InputType* in_data = dataInput.template data<InputType>();
    const IndexType* indices = indicesInput.template data<IndexType>();
    const int* lengths = lengthsInput.template data<int>();
    const T* in_weight = nullptr;

    if (USE_WEIGHT) {
      // static if
      auto& weightInput = Input(WEIGHT);
      CAFFE_ENFORCE_EQ(1, weightInput.dim(), "WEIGHT must be a vector");
      if (!USE_POSITIONAL_WEIGHT) {
        CAFFE_ENFORCE_EQ(
            weightInput.numel(),
            indices_size,
            "Weight should have the same length as indices.");
      }
      in_weight = weightInput.template data<T>();
    }

    // delegate work to perfkernel that branches based on architecture
    EmbeddingLookup<IndexType, InputType, T, USE_POSITIONAL_WEIGHT>(
        D,
        M,
        indices_size,
        N,
        in_data,
        indices,
        lengths,
        in_weight,
        nullptr, // scale_bias field is only used in SparseLengths8BitsRowwiseOp
        USE_MEAN,
        out_data);
    return true;
  }

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
