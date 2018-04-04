#ifndef CAFFE2_OPERATORS_LENGTHS_REDUCER_FUSED_8BIT_ROWWISE_OPS_H_
#define CAFFE2_OPERATORS_LENGTHS_REDUCER_FUSED_8BIT_ROWWISE_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/fused_rowwise_8bit_conversion_ops.h"
#include "caffe2/operators/reducer_functors.h"
#include "caffe2/perfkernels/fused_8bit_rowwise_embedding_lookup.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context, bool with_weights = 0, bool is_mean = 0>
class SparseLengthsFused8BitRowwiseOp : public Operator<Context> {
 public:
  static_assert(
      !(with_weights && is_mean),
      "Cannot have with_weights and is_mean a the same time");

  USE_OPERATOR_CONTEXT_FUNCTIONS;
  USE_SIMPLE_CTOR_DTOR(SparseLengthsFused8BitRowwiseOp)

  bool RunOnDevice() override {
    return DispatchHelper<TensorTypes<int32_t, int64_t>>::call(
        this, Input(INDICES));
  }

  template <typename IndexType>
  bool DoRunWithType() {
    const auto& data = Input(DATA);
    const auto& indices = Input(INDICES);
    const auto& lengths = Input(LENGTHS);
    auto* output = Output(0);

    CAFFE_ENFORCE_EQ(indices.ndim(), 1, "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(lengths.ndim(), 1, "LENGTHS must be a vector");

    const float* weights = nullptr;
    if (with_weights) {
      const auto& weights_input = Input(WEIGHTS);
      CAFFE_ENFORCE_EQ(weights_input.ndim(), 1, "WEIGHTS must be a vector");
      CAFFE_ENFORCE_EQ(
          weights_input.size(),
          indices.size(),
          "WEIGHTS should have the same length as INDICES.");
      weights = weights_input.template data<float>();
    }

    CAFFE_ENFORCE_GT(data.dim(1), 8, "DATA must have more than 8 columns");
    // Subtract 8 from the #columns of data for the 4 bytes for scale and 4
    // bytes for bias that we use in the fused representation (per row).
    const std::vector<TIndex> shape = {lengths.dim(0), data.dim(1) - 8};
    output->Resize(shape);

    Fused8BitRowwiseEmbeddingLookup(
        /*block_size=*/output->dim(1),
        /*output_size=*/output->dim(0),
        /*index_size=*/indices.size(),
        /*data_size=*/data.dim(0),
        /*input=*/data.template data<uint8_t>(),
        /*indices=*/indices.template data<IndexType>(),
        /*lengths=*/lengths.template data<int>(),
        /*weights=*/weights,
        /*normalize_by_lengths=*/is_mean,
        /*out=*/output->template mutable_data<float>());

    return true;
  }

 private:
  enum {
    DATA = 0,
    WEIGHTS = 1,
    INDICES = 1 + with_weights,
    LENGTHS = 2 + with_weights,
  };
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LENGTHS_REDUCER_FUSED_8BIT_ROWWISE_OPS_H_
