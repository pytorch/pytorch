#include <memory>
#include <string>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace {

template <class Context>
class ReservoirSamplingOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ReservoirSamplingOp(const OperatorDef operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        numToCollect_(
            OperatorBase::GetSingleArgument<int>("num_to_collect", -1)) {
    CAFFE_ENFORCE(numToCollect_ > 0);
  }

  bool RunOnDevice() override {
    auto* output = Output(RESERVOIR);
    const auto& input = Input(DATA);

    CAFFE_ENFORCE_GE(input.ndim(), 1);

    bool output_initialized = output->size() > 0;
    if (output_initialized) {
      CAFFE_ENFORCE_EQ(output->ndim(), input.ndim());
      for (size_t i = 1; i < input.ndim(); ++i) {
        CAFFE_ENFORCE_EQ(output->dim(i), input.dim(i));
      }
    }

    auto dims = input.dims();
    auto num_entries = dims[0];

    dims[0] = numToCollect_;
    output->Reserve(dims, &context_);

    if (num_entries == 0) {
      if (!output_initialized) {
        // Get both shape and meta
        output->CopyFrom(input, &context_);
      }
      return true;
    }

    auto num_to_copy = std::min<int32_t>(num_entries, numToCollect_);
    auto output_batch_size = output_initialized ? output->dim(0) : 0;
    dims[0] = std::min<size_t>(numToCollect_, output_batch_size + num_to_copy);
    if (output_batch_size < numToCollect_) {
      output->Resize(dims);
    }
    auto* output_data =
        static_cast<char*>(output->raw_mutable_data(input.meta()));

    auto block_size = input.size_from_dim(1);
    auto block_bytesize = block_size * input.itemsize();
    const auto* input_data = static_cast<const char*>(input.raw_data());

    auto* num_visited_tensor = Output(NUM_VISITED);
    CAFFE_ENFORCE_EQ(0, num_visited_tensor->ndim());
    auto* num_visited = num_visited_tensor->template mutable_data<int64_t>();
    CAFFE_ENFORCE_GE(*num_visited, 0);

    for (int i = 0; i < num_entries; ++i) {
      int64_t pos = -1;
      if (*num_visited < numToCollect_) {
        // append
        pos = *num_visited;
      } else {
        auto& gen = context_.RandGenerator();
        // uniform between [0, num_visited]
        std::uniform_int_distribution<int64_t> uniformDist(0, *num_visited);
        pos = uniformDist(gen);
        if (pos >= numToCollect_) {
          // discard
          pos = -1;
        }
      }

      if (pos < 0) {
        // discard
        CAFFE_ENFORCE_GE(*num_visited, numToCollect_);
      } else {
        // replace
        context_.template CopyItems<Context, Context>(
            input.meta(),
            block_size,
            input_data + i * block_bytesize,
            output_data + pos * block_bytesize);
      }

      ++(*num_visited);
    }
    return true;
  }

 private:
  // number of tensors to collect
  int numToCollect_;

  INPUT_TAGS(RESERVOIR_IN, NUM_VISITED_IN, DATA);
  OUTPUT_TAGS(RESERVOIR, NUM_VISITED);
};

REGISTER_CPU_OPERATOR(ReservoirSampling, ReservoirSamplingOp<CPUContext>);

OPERATOR_SCHEMA(ReservoirSampling)
    .NumInputs(3)
    .NumOutputs(2)
    .EnforceOneToOneInplace()
    .SetDoc(R"DOC(
Collect `DATA` tensor into `RESERVOIR` of size `num_to_collect`. `DATA` is
assumed to be a batch.
)DOC")
    .Arg(
        "num_to_collect",
        "The number of random samples to append for each positive samples")
    .Input(
        0,
        "RESERVOIR",
        "The reservoir; should be initialized to empty tensor")
    .Input(
        1,
        "NUM_VISITED",
        "Number of examples seen so far; should be initialized to 0")
    .Input(
        2,
        "DATA",
        "Tensor to collect from. The first dimension is assumed to be batch "
        "size. If the object to be collected is represented by multiple "
        "tensors, use `PackRecords` to pack them into single tensor.")
    .Output(0, "RESERVOIR", "Same as the input")
    .Output(1, "NUM_VISITED", "Same as the input");

SHOULD_NOT_DO_GRADIENT(ReservoirSampling);
}
}
