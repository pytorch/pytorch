#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"

#include "caffe2/operators/fully_connected_op.h"

namespace caffe2 {

template <typename T> class GLFullyConnectedOp final : public Operator<GLContext> {
public:
  GLFullyConnectedOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<GLContext>(operator_def, ws) {}
  virtual ~GLFullyConnectedOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  arm_compute::GCFullyConnectedLayer fc_layer_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> X_, W_, B_;
};

template <typename T>
bool GLFullyConnectedOp<T>::RunOnDevice() {

  auto Xblob = OperatorBase::Inputs()[0];
  auto *Wblob = OperatorBase::Inputs()[1];
  auto *Bblob = OperatorBase::Inputs()[2];

  if (first_run_) {
    X_ = GLContext::getGLTensor<T>(Xblob);
    W_ = GLContext::getGLTensor<T>(Wblob);
    B_ = GLContext::getGLTensor<T>(Bblob);
  }

  auto M = X_->dim32(0);
  auto CIn = X_->dim32(1);
  auto Height = X_->dim32(2);
  auto Width = X_->dim32(3);
  auto N = W_->dim32(0);

  CAFFE_ENFORCE_EQ(1, B_->ndim());
  CAFFE_ENFORCE_EQ(N, B_->dim32(0));

  vector<TIndex> output_dims = {M, N};
  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();
  if (first_run_) {
    first_run_ = false;
    Y->Resize(output_dims);

    fc_layer_.configure(X_->get_underlying(), W_->get_underlying(),
                     B_->get_underlying(), Y->get_underlying(), true, false);
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    W_->lazy_allocate(Wblob, second_run_, second_run_);
    B_->lazy_allocate(Bblob, second_run_, second_run_);
    if (second_run_) {
      second_run_ = false;
      Y->allocate();
    }
    fc_layer_.run();
  }

  return true;
}

REGISTER_GL_OPERATOR(FC, GLFullyConnectedOp<DataType>);

} // namespace caffe2
