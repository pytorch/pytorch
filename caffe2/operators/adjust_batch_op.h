#pragma once

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <class Context>
class AdjustBatchOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  AdjustBatchOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        max_batch_size_(
            this->template GetSingleArgument<int64_t>("max_batch_size", -1)) {}

  bool RunOnDevice() override {
    auto& input = Input(0);
    vector<int64_t> output_dims(input.sizes().vec());
    CAFFE_ENFORCE(!output_dims.empty());
    if (InputSize() > 1) {
      // TODO: if we have a second input and we have max_batch_size set, check
      // the batch size of the two inputs for consistency
      auto& batch_size = Input(1);
      int64_t real_batch_size = *batch_size.template data<int64_t>();
      int64_t max_batch_size = output_dims[0];
      CAFFE_ENFORCE_GE(max_batch_size, real_batch_size);
      output_dims[0] = real_batch_size;
      auto* output = Output(0, output_dims, input.dtype());
      this->context_.template CopyItems<Context, Context>(
          input.dtype(),
          input.numel() * real_batch_size / max_batch_size,
          input.raw_data(),
          output->raw_mutable_data(input.dtype()));
    } else {
      // Pad to max batch size
      CAFFE_ENFORCE_GT(
          max_batch_size_,
          0,
          "max_batch_size should be larger than 0. Got ",
          max_batch_size_);

      // TODO: ideally we can support the case when input batch is larger than
      // the max_batch_size, as we can just pad to the multiple of
      // max_batch_size.
      CAFFE_ENFORCE_GE(max_batch_size_, output_dims.front());

      int64_t real_batch_size = output_dims[0];
      output_dims[0] = max_batch_size_;
      auto* output = Output(0, output_dims, input.dtype());
      math::Set(
          output->nbytes(),
          static_cast<char>(0),
          static_cast<char*>(output->raw_data()),
          &context_);
      this->context_.template CopyItems<Context, Context>(
          input.dtype(),
          input.numel(),
          input.raw_data(),
          output->raw_mutable_data(input.dtype()));

      if (OutputSize() > 1) {
        auto* real_batch_tensor = Output(1, {1}, at::dtype<int64_t>());
        real_batch_tensor->template mutable_data<int64_t>()[0] =
            real_batch_size;
      }
    }

    return true;
  }

 private:
  int64_t max_batch_size_;
};
} // namespace caffe2
