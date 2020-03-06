#ifndef CAFFE2_OPERATORS_LENGTHS_REDUCER_FUSED_8BIT_ROWWISE_OPS_H_
#define CAFFE2_OPERATORS_LENGTHS_REDUCER_FUSED_8BIT_ROWWISE_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/operators/fused_rowwise_8bit_conversion_ops.h"
#include "caffe2/operators/reducer_functors.h"
#include "caffe2/perfkernels/fused_8bit_rowwise_embedding_lookup.h"
#include "caffe2/utils/math.h"
#ifdef USE_FBGEMM
#include "fbgemm/Fbgemm.h"
#endif

namespace caffe2 {

template <class Context, bool with_weights = false, bool is_mean = false>
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

    CAFFE_ENFORCE_EQ(indices.dim(), 1, "INDICES must be a vector");
    CAFFE_ENFORCE_EQ(lengths.dim(), 1, "LENGTHS must be a vector");

    const float* weights = nullptr;
    if (with_weights) {
      const auto& weights_input = Input(WEIGHTS);
      CAFFE_ENFORCE_EQ(weights_input.dim(), 1, "WEIGHTS must be a vector");
      CAFFE_ENFORCE_EQ(
          weights_input.numel(),
          indices.numel(),
          "WEIGHTS should have the same length as INDICES.");
      weights = weights_input.template data<float>();
    }

    CAFFE_ENFORCE_GT(data.size(1), 8, "DATA must have more than 8 columns");
    // Subtract 8 from the #columns of data for the 4 bytes for scale and 4
    // bytes for bias that we use in the fused representation (per row).
    const std::vector<int64_t> shape = {lengths.size(0), data.size(1) - 8};
    auto* output = Output(0, shape, at::dtype<float>());

    std::int64_t block_size = output->size(1);
    auto output_size = output->size(0);
    auto index_size = indices.numel();
    auto data_size = data.size(0);
    const std::uint8_t* input_data = data.template data<std::uint8_t>();
    const int* lengths_data = lengths.template data<int>();
    float* output_data = output->template mutable_data<float>();

#ifdef USE_FBGEMM
    // Calling the JITed kernel from FBGEMM
    // Will Remove the call to C2/perfkernels/

    // If this is the first call or block size has changed (should never happen
    // actually), generate a kernel.
    if (block_size != last_block_size) {
      last_block_size = block_size;
      if (std::is_same<IndexType, std::int32_t>::value) {
        kernel32_ = fbgemm::GenerateEmbeddingSpMDM<std::uint8_t, std::int32_t>(
            block_size,
            with_weights,
            is_mean,
            /*prefetch distance*/ 16);
      } else {
        CAFFE_ENFORCE((std::is_same<IndexType, std::int64_t>::value));
        kernel64_ = fbgemm::GenerateEmbeddingSpMDM<std::uint8_t, std::int64_t>(
            block_size,
            with_weights,
            is_mean,
            /*prefetch distance*/ 16);
      }
    }

    bool success;
    if (std::is_same<IndexType, std::int32_t>::value) {
      success = kernel32_(
          output_size,
          index_size,
          data_size,
          input_data,
          indices.template data<std::int32_t>(),
          lengths_data,
          weights,
          output_data);
    } else {
      success = kernel64_(
          output_size,
          index_size,
          data_size,
          input_data,
          indices.template data<std::int64_t>(),
          lengths_data,
          weights,
          output_data);
    }

    if (success) {
      return true;
    }

    auto indices_data = indices.template data<IndexType>();

    int64_t current = 0;
    for (int m = 0; m < output_size; ++m) {
      for (int i = 0; i < lengths_data[m]; ++i) {
        CAFFE_ENFORCE_LT(current, index_size);
        IndexType idx = indices_data[current];
        CAFFE_ENFORCE(
            0 <= idx && idx < data_size,
            "Index ",
            current,
            " is out of bounds: ",
            idx,
            ", range 0 to ",
            data_size);
        ++current;
      }
    }
    CAFFE_ENFORCE_EQ(
        current,
        index_size,
        "Your input seems to be incorrect: the sum of lengths values should be "
        "the size of the indices tensor, but it appears not.");

    return false;
#else
    Fused8BitRowwiseEmbeddingLookup(
        block_size,
        output_size,
        index_size,
        data_size,
        input_data,
        indices.template data<IndexType>(),
        lengths_data,
        weights,
        is_mean,
        output_data);

    return true;
#endif
  }

  enum {
    DATA = 0,
    WEIGHTS = 1,
    INDICES = 1 + with_weights,
    LENGTHS = 2 + with_weights,
  };

#ifdef USE_FBGEMM
 private:
  std::int64_t last_block_size{-1};
  fbgemm::EmbeddingSpMDMKernelSignature<std::uint8_t, std::int32_t>::Type
      kernel32_;
  fbgemm::EmbeddingSpMDMKernelSignature<std::uint8_t, std::int64_t>::Type
      kernel64_;
#endif
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LENGTHS_REDUCER_FUSED_8BIT_ROWWISE_OPS_H_
