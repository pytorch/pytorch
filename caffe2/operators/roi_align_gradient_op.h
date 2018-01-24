// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef ROI_ALIGN_OP_H_
#define ROI_ALIGN_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T, class Context>
class RoIAlignGradientOp final : public Operator<Context> {
 public:
  RoIAlignGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        spatial_scale_(
            OperatorBase::GetSingleArgument<float>("spatial_scale", 1.)),
        pooled_height_(OperatorBase::GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(OperatorBase::GetSingleArgument<int>("pooled_w", 1)),
        sampling_ratio_(
            OperatorBase::GetSingleArgument<int>("sampling_ratio", -1)) {
    DCHECK_GT(spatial_scale_, 0);
    DCHECK_GT(pooled_height_, 0);
    DCHECK_GT(pooled_width_, 0);
    DCHECK_GE(sampling_ratio_, 0);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;
};

} // namespace caffe2

#endif // ROI_ALIGN_OP_H_
