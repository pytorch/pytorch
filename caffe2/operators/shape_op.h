
#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "c10/util/irange.h"

namespace caffe2 {

// RecordShapeOp records the shape of the input tensor to a vector of int. You
// mostly don't need this operator explicitly, and it is mostly used in the
// autodiff process.
template <class Context>
class ShapeOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  template <class... Args>
  explicit ShapeOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        axes_(OperatorBase ::GetRepeatedArgument<int>("axes")) {}

  bool RunOnDevice() override {
    auto& data = Input(DATA);

    int numDims = data.dim();
    int numAxes = axes_.size();
    if (numAxes == 0) {
      auto* output = Output(0, {numDims}, at::dtype<int64_t>());
      int64_t* output_data = output->template mutable_data<int64_t>();
      context_.CopyBytesSameDevice(
          numDims * sizeof(int64_t), data.sizes().data(), output_data);
      return true;
    }

    auto* output = Output(0, {numAxes}, at::dtype<int64_t>());
    auto src = reinterpret_cast<const char*>(data.sizes().data());
    auto out = reinterpret_cast<char*>(output->template mutable_data<int64_t>());
    for (const auto i : c10::irange(numAxes)) {
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
