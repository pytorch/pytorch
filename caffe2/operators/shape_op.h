
#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

// RecordShapeOp records the shape of the input tensor to a vector of int. You
// mostly don't need this operator explicitly, and it is mostly used in the
// autodiff process.
template <class Context>
class ShapeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  ShapeOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axes_(OperatorBase ::GetRepeatedArgument<int>("axes")) {}

  bool RunOnDevice() override {
    auto& data = Input(DATA);
    auto* output = Output(0);
    int numDims = data.dim();
    int numAxes = axes_.size();
    if (numAxes == 0) {
      output->Resize(numDims);
      int64_t* output_data = output->template mutable_data<int64_t>();
      context_.CopyBytesSameDevice(
          numDims * sizeof(int64_t), data.sizes().data(), output_data);
      return true;
    }

    output->Resize(numAxes);
    auto src = reinterpret_cast<const char*>(data.sizes().data());
    auto out = reinterpret_cast<char*>(output->template mutable_data<int64_t>());
    for (int i = 0; i < numAxes; i++) {
      auto axis = axes_[i];
      CAFFE_ENFORCE_LT(axis, numDims, "Axis out of range");
      CAFFE_ENFORCE_GE(axis, 0, "Each axis should be non-negative");
      context_.CopyBytesSameDevice(
          sizeof(int64_t), src + axis * sizeof(int64_t), out);
      out += sizeof(int64_t);
    }
    return true;
  }

  INPUT_TAGS(DATA);

 private:
  vector<int> axes_;
};

} // namespace caffe2
