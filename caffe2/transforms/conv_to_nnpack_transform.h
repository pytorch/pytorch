#pragma once

#include "caffe2/core/common.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/transforms/single_op_transform.h"
#include "caffe2/utils/proto_utils.h"

namespace caffe2 {

class CAFFE2_API ConvToNNPackTransform : public SingleOpTransform {
 protected:
  // Specify what the op needs to be to match the pattern.
  bool MatchOperator(const OperatorDef& op) override {
    return (
        op.type() == "Conv" && op.device_option().device_type() == PROTO_CPU &&
        op.engine() != "NNPACK");
  }

  // Specify how the operator should be replaced.
  void ReplaceOperator(OperatorDef* op) override {
    op->set_engine("NNPACK");
  }
};

} // namespace caffe2
