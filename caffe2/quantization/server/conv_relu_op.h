#pragma once

#include "caffe2/operators/conv_op.h"
#include "caffe2/operators/conv_pool_op_base.h"

namespace caffe2 {

template <typename T, class Context>
class ConvReluOp final : public ConvPoolOpBase<Context> {
 public:
  ConvReluOp(const OperatorDef& operator_def, Workspace* ws)
      : ConvPoolOpBase<Context>(operator_def, ws) {
    for (auto name : operator_def.input()) {
      local_input_blobs_.push_back(local_ws_.CreateBlob(name));
      CHECK_NOTNULL(local_input_blobs_.back());
    }
    local_op_.reset(new ConvOp<T, Context>(operator_def, &local_ws_));
    for (auto name : operator_def.output()) {
      local_output_blobs_.push_back(local_ws_.GetBlob(name));
      CHECK_NOTNULL(local_output_blobs_.back());
    }
  }
  ~ConvReluOp() {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 private:
  Workspace local_ws_;
  std::vector<Blob*> local_input_blobs_;
  std::vector<Blob*> local_output_blobs_;
  std::unique_ptr<ConvOp<T, Context>> local_op_;
}; // class ConvReluOp

} // namespace caffe2
