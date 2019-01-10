#ifndef CAFFE2_OPERATORS_LAYER_NORM_OP_H
#define CAFFE2_OPERATORS_LAYER_NORM_OP_H

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <class Context>
class LayerNormOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LayerNormOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 1e-5f)) {}
  ~LayerNormOp() {}

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override {
    return DoRunWithType<float>();
  }

 protected:
  int axis_;
  float epsilon_;

  Tensor<Context> scratch_;
  Tensor<Context> seg_indices_;
};

template <class Context>
class LayerNormGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LayerNormGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(OperatorBase::GetSingleArgument<int>("axis", 1)),
        epsilon_(OperatorBase::GetSingleArgument<float>("epsilon", 0.001f)) {}
  ~LayerNormGradientOp() {}

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override {
    return DoRunWithType<float>();
  }

 protected:
  int axis_;
  float epsilon_;

  Tensor<Context> scratch_;
  Tensor<Context> gscratch_;
  Tensor<Context> seg_indices_;
  Tensor<Context> dstdev_;
  Tensor<Context> dmean_;
};

} // namespace caffe2

#endif /* CAFFE2_OPERATORS_LAYER_NORM_OP_H */
