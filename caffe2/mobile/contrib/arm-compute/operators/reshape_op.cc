#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"
#include "caffe2/operators/reshape_op.h"

namespace caffe2 {

template <typename T> class GLReshapeOp final : public Operator<GLContext> {
public:
  GLReshapeOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<GLContext>(operator_def, ws) {}
  virtual ~GLReshapeOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
};

template <typename T>
bool GLReshapeOp<T>::RunOnDevice() {
  auto *Xblob = OperatorBase::Inputs()[0];
  auto X = GLContext::getGLTensor<T>(Xblob);
  LOG(INFO) << "[C2DEBUG] X: " << X->dim32(0) << " " << X->dim32(1) << " " << X->dim32(2) << " " << X->dim32(3);
  auto arg = OperatorBase::GetRepeatedArgument<int>("shape");
  for (int i = 0; i < arg.size(); ++i) {
    LOG(INFO) << "[C2DEBUG] shape: " << arg[i];
  }
  return true;
}

REGISTER_GL_OPERATOR(Reshape, GLReshapeOp<DataType>);

} // namespace caffe2
