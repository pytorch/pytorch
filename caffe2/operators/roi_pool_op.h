#ifndef ROI_POOL_OP_H_
#define ROI_POOL_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class RoIPoolOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit RoIPoolOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        is_test_(
            this->template GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))),
        pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
        spatial_scale_(
            this->template GetSingleArgument<float>("spatial_scale", 1.)) {
    CAFFE_ENFORCE(
        (is_test_ && OutputSize() == 1) || (!is_test_ && OutputSize() == 2),
        "Output size mismatch.");
    CAFFE_ENFORCE_GT(spatial_scale_, 0);
    CAFFE_ENFORCE_GT(pooled_height_, 0);
    CAFFE_ENFORCE_GT(pooled_width_, 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override;

 protected:
  bool is_test_;
  StorageOrder order_;
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

template <typename T, class Context>
class RoIPoolGradientOp final : public Operator<Context> {
 public:
  template <class... Args>
  explicit RoIPoolGradientOp(Args&&... args)
      : Operator<Context>(std::forward<Args>(args)...),
        spatial_scale_(
            this->template GetSingleArgument<float>("spatial_scale", 1.)),
        pooled_height_(this->template GetSingleArgument<int>("pooled_h", 1)),
        pooled_width_(this->template GetSingleArgument<int>("pooled_w", 1)),
        order_(StringToStorageOrder(
            this->template GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_ENFORCE_GT(spatial_scale_, 0);
    CAFFE_ENFORCE_GT(pooled_height_, 0);
    CAFFE_ENFORCE_GT(pooled_width_, 0);
    CAFFE_ENFORCE_EQ(
        order_, StorageOrder::NCHW, "Only NCHW order is supported right now.");
  }
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    CAFFE_NOT_IMPLEMENTED;
  }

 protected:
  float spatial_scale_;
  int pooled_height_;
  int pooled_width_;
  StorageOrder order_;
};

} // namespace caffe2

#endif // ROI_POOL_OP_H_
