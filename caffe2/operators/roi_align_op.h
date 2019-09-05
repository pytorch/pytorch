// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef ROI_ALIGN_OP_H_
#define ROI_ALIGN_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/export_caffe2_op_to_c10.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"

C10_DECLARE_EXPORT_CAFFE2_OP_TO_C10(RoIAlign)

namespace caffe2 {

template <typename T, class Context>
class RoIAlignOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit RoIAlignOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))),
        spatial_scale_(
            this->template GetSingleArgument<float>("spatial_scale", 1.)),
        pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
        sampling_ratio_(
            this->template GetSingleArgument<int>("sampling_ratio", -1)),
        aligned_(this->template GetSingleArgument<bool>("aligned", false)) {
    DCHECK_GT(spatial_scale_, 0);
    DCHECK_GT(pooled_height_, 0);
    DCHECK_GT(pooled_width_, 0);
    DCHECK_GE(sampling_ratio_, 0);
    DCHECK(order_ == StorageOrder::NCHW || order_ == StorageOrder::NHWC);
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  StorageOrder order_;
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  int sampling_ratio_;
  bool aligned_;
};

} // namespace caffe2

#endif // ROI_ALIGN_OP_H_
