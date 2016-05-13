#ifndef CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_H_
#define CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class SpatialBNOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SpatialBNOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        is_test_(OperatorBase::GetSingleArgument<int>("is_test", 0)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)),
        momentum_(OperatorBase::GetSingleArgument<float>("momentum", 0.9)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_CHECK((is_test_ && InputSize() == 5) ||
          (!is_test_ && InputSize() == 3));
    CAFFE_CHECK((is_test_ && OutputSize() == 1) ||
          (!is_test_ && (OutputSize() == 3 || OutputSize() == 5)));
    CAFFE_CHECK_GT(epsilon_, 0);
    CAFFE_CHECK_GE(momentum_, 0);
    CAFFE_CHECK_LE(momentum_, 1);
  }
  ~SpatialBNOpBase() {}

 protected:
  bool is_test_;
  double epsilon_;
  double momentum_;
  StorageOrder order_;
  INPUT_TAGS(INPUT, SCALE, BIAS, EST_MEAN, EST_INV_VAR);
  OUTPUT_TAGS(OUTPUT, RUNNING_MEAN, RUNNING_INV_VAR, SAVED_MEAN, SAVED_INV_VAR);
  DISABLE_COPY_AND_ASSIGN(SpatialBNOpBase);
};

template <class Context>
class SpatialBNGradientOpBase : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  SpatialBNGradientOpBase(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        is_test_(OperatorBase::GetSingleArgument<int>("is_test", 0)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5)),
        order_(StringToStorageOrder(
            OperatorBase::GetSingleArgument<string>("order", "NCHW"))) {
    CAFFE_CHECK((InputSize() == 5) || (!is_test_ && InputSize() == 3));
    CAFFE_CHECK_EQ(OutputSize(), 3);
  }
  ~SpatialBNGradientOpBase() {}

 protected:
  bool is_test_;
  double epsilon_;
  StorageOrder order_;

  INPUT_TAGS(INPUT, SCALE, OUTPUT_GRAD, SAVED_MEAN, SAVED_INV_VAR);
  OUTPUT_TAGS(INPUT_GRAD, SCALE_GRAD, BIAS_GRAD);
  DISABLE_COPY_AND_ASSIGN(SpatialBNGradientOpBase);
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_SPATIAL_BATCH_NORM_OP_H_
