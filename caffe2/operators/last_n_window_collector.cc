#include <memory>
#include <string>
#include <vector>
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {
namespace {

template <class Context, class DataType>
class LastNWindowCollectorOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LastNWindowCollectorOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        numToCollect_(
            OperatorBase::GetSingleArgument<int>("num_to_collect", -1)),
        numVisited_(0) {
    CAFFE_ENFORCE_GT(numToCollect_, 0);
  }

  bool RunOnDevice() override {
    auto* output = Output(0);
    const auto& input = Input(0);
    auto dims = input.dims();
    if (dims[0] == 0) {
      return true;
    }

    dims[0] = numToCollect_;
    output->Reserve(dims, &context_);

    dims[0] = std::min<size_t>(numToCollect_, input.dim(0) + numVisited_);
    auto stride = input.size_from_dim(1);
    if (output->ndim() == 0 || output->dim(0) == 0 ||
        output->size_from_dim(1) != stride || numVisited_ < numToCollect_) {
      output->Resize(dims);
    }
    if (input.dim(0) > numToCollect_) {
      // just copy the last N rows
      context_.template Copy<DataType, CPUContext, Context>(
          output->size(),
          input.template data<DataType>() + input.size() - output->size(),
          output->template mutable_data<DataType>());
      numVisited_ = numToCollect_;
      return true;
    }
    // we have less elements than necessary
    auto numToCopy = input.dim(0);
    auto firstBlockSize =
        std::min<size_t>(numToCopy + numVisited_, numToCollect_) - numVisited_;
    context_.template Copy<DataType, CPUContext, Context>(
        firstBlockSize * stride,
        input.template data<DataType>(),
        output->template mutable_data<DataType>() + numVisited_ * stride);

    context_.template Copy<DataType, CPUContext, Context>(
        (numToCopy - firstBlockSize) * stride,
        input.template data<DataType>() + firstBlockSize * stride,
        output->template mutable_data<DataType>());

    numVisited_ += numToCopy;
    return true;
  }

 private:
  const int32_t numToCollect_;
  size_t numVisited_;
};

REGISTER_CPU_OPERATOR(
    LastNWindowCollector,
    LastNWindowCollectorOp<CPUContext, float>);

OPERATOR_SCHEMA(LastNWindowCollector)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
Collect the last N rows from input data. The purpose is to keep track of data
accross batches, so for example suppose the LastNWindowCollector is called
successively with the following input

[1,2,3,4]
[5,6,7]
[8,9,10,11]

And the number of items is set to 6, then the output after the 3rd call
will contain the following elements:
[6,7,8,9,10,11]

No guarantee is made on the ordering of elements in input. So a valid value for
output could have been
[11,10,9,8,7,6]

Also, this method works for any order tensor, treating the first dimension as
input rows and keeping the last N rows seen as input. So for instance:

[[1,2],[2,3],[3,4],[4,5]]
[[5,6],[6,7],[7,8]]
[[8,9],[9,10],[10,11],[11,12]]

A possible output would be
[[6,7],[7,8],[8,9],[9,10],[10,11],[11,12]]
)DOC")
    .Arg(
        "num_to_collect",
        "The number of random samples to append for each positive samples")
    .Input(
        0,
        "Output data",
        "Copy, just to say that the output depends on the previous iterations")
    .Output(0, "The last n", "Data stored in sessions");
SHOULD_NOT_DO_GRADIENT(LastNWindowCollector);
}
}
