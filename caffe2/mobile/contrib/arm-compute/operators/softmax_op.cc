#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"

#include "caffe2/operators/softmax_op.h"

namespace caffe2 {

template <typename T> class GLSoftmaxOp final : public Operator<GLContext> {
public:
  GLSoftmaxOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<GLContext>(operator_def, ws) {}
  virtual ~GLSoftmaxOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  arm_compute::GCSoftmaxLayer softmax_layer_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> X_;
};

template <typename T>
bool GLSoftmaxOp<T>::RunOnDevice() {

  auto *Xblob = OperatorBase::Inputs()[0];
  if (first_run_) {
    X_ = GLContext::getGLTensor<T>(Xblob);
  }

  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();
  if (first_run_) {
    first_run_ = false;
    Y->ResizeLike(*X_);
    softmax_layer_.configure(X_->get_underlying(), Y->get_underlying());
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    if (second_run_) {
      second_run_ = false;
      Y->allocate();
    }
    softmax_layer_.run();
  }

  return true;
}

REGISTER_GL_OPERATOR(Softmax, GLSoftmaxOp<DataType>);

} // namespace caffe2
