#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"
#include "caffe2/operators/utility_ops.h"

namespace caffe2 {

template <typename T> class GLSumOp final : public Operator<GLContext> {
public:
  GLSumOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<GLContext>(operator_def, ws) {}
  virtual ~GLSumOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  arm_compute::GCArithmeticAddition add_layer_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> A_, B_;
};


template <typename T>
bool GLSumOp<T>::RunOnDevice() {

  auto *Ablob = OperatorBase::Inputs()[0];
  auto *Bblob = OperatorBase::Inputs()[1];

  if (first_run_) {
    A_ = GLContext::getGLTensor<T>(Ablob);
    B_ = GLContext::getGLTensor<T>(Bblob);
  }

  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();
  if (first_run_) {
    first_run_ = false;
    Y->ResizeLike(*A_);
    add_layer_.configure(A_->get_underlying(), B_->get_underlying(), Y->get_underlying(), arm_compute::ConvertPolicy::SATURATE);
  } else {
    A_->lazy_allocate(Ablob, second_run_, true);
    B_->lazy_allocate(Bblob, second_run_, true);
    if (second_run_) {
      Y->allocate();
      second_run_ = false;
    }
    add_layer_.run();
  }

  return true;
}

REGISTER_GL_OPERATOR(Sum, GLSumOp<DataType>);
REGISTER_GL_OPERATOR(Add, GLSumOp<DataType>);

} // namespace caffe2
