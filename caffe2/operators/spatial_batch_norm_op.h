#ifndef CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_H_
#define CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class SpatialBNOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SpatialBNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        is_test_(OperatorBase::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)),
        momentum_(OperatorBase::GetSingleArgument<float>("momentum", 0.9f)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        num_batches_(OperatorBase::GetSingleArgument<int>("num_batches", 1)) {
    // TODO(jiayq): update the input and output size checks.
    CAFFE_ENFORCE(
        (is_test_ && OutputSize() == 1) || (!is_test_ && OutputSize() == 5));
    CAFFE_ENFORCE_GT(epsilon_, 0);
    CAFFE_ENFORCE_GE(momentum_, 0);
    CAFFE_ENFORCE_LE(momentum_, 1);
  }
  ~SpatialBNOp() {}

  bool RunOnDevice() override {
    return true;
  }

 protected:
  bool is_test_;
  double epsilon_;
  double momentum_;
  StorageOrder order_;
  int num_batches_;
  INPUT_TAGS(INPUT, SCALE, BIAS, EST_MEAN, EST_VAR, SUMS, SUMSQ);
  OUTPUT_TAGS(OUTPUT, RUNNING_MEAN, RUNNING_VAR, SAVED_MEAN, SAVED_INV_VAR);
};

template <class Context>
class SpatialBNGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SpatialBNGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        is_test_(OperatorBase::GetSingleArgument<int>(OpSchema::Arg_IsTest, 0)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))),
        num_batches_(OperatorBase::GetSingleArgument<int>("num_batches", 1)) {
    CAFFE_ENFORCE(InputSize() == 5 || InputSize() == 7);
    CAFFE_ENFORCE(OutputSize() == 3);
  }
  ~SpatialBNGradientOp() {}

  bool RunOnDevice() override {
    return true;
  }

 protected:
  bool is_test_;
  double epsilon_;
  StorageOrder order_;
  int num_batches_;

  INPUT_TAGS(
      INPUT,
      SCALE,
      OUTPUT_GRAD,
      SAVED_MEAN,
      SAVED_INV_VAR,
      AGGREGATE_SCALE_GRAD,
      AGGREGATE_BIAS_GRAD);
  OUTPUT_TAGS(INPUT_GRAD, SCALE_GRAD, BIAS_GRAD);
};

} // namespace caffe2

#endif // CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_H_
