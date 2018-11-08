#ifndef CAFFE2_OPENGL_OPERATORS_ACTIVATION_OPS_H_
#define CAFFE2_OPENGL_OPERATORS_ACTIVATION_OPS_H_

#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"

namespace caffe2 {

template <typename T>
class GLSigmoidOp final : public Operator<GLContext> {
public:
  GLSigmoidOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<GLContext>(operator_def, ws) {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  arm_compute::GCActivationLayer sigmoid_layer_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> X_;
};

template <typename T> class GLReluOp final : public Operator<GLContext> {
public:
  GLReluOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<GLContext>(operator_def, ws) {}
  virtual ~GLReluOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  arm_compute::GCActivationLayer relu_layer_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> X_;

};

} // namespace caffe2

#endif // CAFFE2_OPENGL_OPERATORS_ACTIVATION_OPS_H_
