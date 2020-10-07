// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once
#include "caffe2/quantization/server/caffe2_dnnlowp_utils.h"
#include "caffe2/quantization/server/dnnlowp.h"

namespace caffe2 {

class ComputeEqualizationScaleOp final : public Operator<CPUContext> {
 public:
  ComputeEqualizationScaleOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<CPUContext>(operator_def, ws) {}

  bool RunOnDevice() override;

}; // class ComputeEqualizationScaleOp

} // namespace caffe2
