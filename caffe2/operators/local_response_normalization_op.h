#ifndef CAFFE2_OPERATORS_LOCAL_RESPONSE_NORMALIZATION_OP_H_
#define CAFFE2_OPERATORS_LOCAL_RESPONSE_NORMALIZATION_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class LRNOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LRNOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        size_(OperatorBase::GetSingleArgument<int>("size", 0)),
        alpha_(OperatorBase::GetSingleArgument<float>("alpha", 0)),
        beta_(OperatorBase::GetSingleArgument<float>("beta", 0)),
        bias_(OperatorBase::GetSingleArgument<float>("bias", 1)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        pre_pad_((size_ - 1) / 2) {
    DCHECK_GT(size_, 0);
    DCHECK_EQ(size_ % 2, 1);
    DCHECK_GT(alpha_, 0);
    DCHECK_GT(beta_, 0);
  }

  bool RunOnDevice() override {
    switch (order_) {
      case StorageOrder::NHWC:
        return RunOnDeviceWithOrderNHWC();
      case StorageOrder::NCHW:
        return RunOnDeviceWithOrderNCHW();
      default:
        LOG(FATAL) << "Unknown storage order: " << order_;
    }
    // To suppress old compiler warnings
    return true;
  }

  virtual bool RunOnDeviceWithOrderNCHW() = 0;
  virtual bool RunOnDeviceWithOrderNHWC() = 0;

 protected:
  const int size_;
  const float alpha_;
  const float beta_;
  const float bias_;
  const StorageOrder order_;
  const int pre_pad_;
  // Input: X; Output: Y, scale.
};

template <typename T, class Context>
class LRNOp final : public LRNOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LRNOp(const OperatorDef& operator_def, Workspace* ws)
      : LRNOpBase<T, Context>(operator_def, ws) {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 protected:
  // Input: X; Output: Y, scale.
  OUTPUT_TAGS(OUTPUT, SCALE);
  Tensor<Context>* scale_ = nullptr;
  Tensor<Context> local_scale_tensor_;
};

template <typename T, class Context>
class LRNGradientOp final : public LRNOpBase<T, Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LRNGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : LRNOpBase<T, Context>(operator_def, ws) {}

  bool RunOnDeviceWithOrderNCHW() override;
  bool RunOnDeviceWithOrderNHWC() override;

 protected:
  // Input: X, Y, scale, dY; Output: dX
  INPUT_TAGS(INPUT, OUTPUT, SCALE, OUTPUT_GRAD);
  Tensor<Context>* scale_ = nullptr;
  Tensor<Context> local_scale_tensor_;
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_LOCAL_RESPONSE_NORMALIZATION_OP_H_
