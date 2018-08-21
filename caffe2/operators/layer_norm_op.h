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
        axis_(this->template GetSingleArgument<int>("axis", 1)),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 1e-5f)) {}
  ~LayerNormOp() {}

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override {
    return DoRunWithType<float>();
  }

 protected:
  int axis_;
  float epsilon_;

  Tensor scratch_{Context::GetDeviceType()};
  Tensor seg_indices_{Context::GetDeviceType()};
};

template <class Context>
class LayerNormGradientOp : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  LayerNormGradientOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        axis_(this->template GetSingleArgument<int>("axis", 1)),
        epsilon_(this->template GetSingleArgument<float>("epsilon", 0.001f)) {}
  ~LayerNormGradientOp() {}

  template <typename T>
  bool DoRunWithType();

  bool RunOnDevice() override {
    return DoRunWithType<float>();
  }

 protected:
  int axis_;
  float epsilon_;

  Tensor scratch_{Context::GetDeviceType()};
  Tensor gscratch_{Context::GetDeviceType()};
  Tensor seg_indices_{Context::GetDeviceType()};
  Tensor dstdev_{Context::GetDeviceType()};
  Tensor dmean_{Context::GetDeviceType()};
};

} // namespace caffe2

#endif /* CAFFE2_OPERATORS_LAYER_NORM_OP_H */
