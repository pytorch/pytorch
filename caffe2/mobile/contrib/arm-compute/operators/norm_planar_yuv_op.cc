#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"

namespace caffe2 {

template <typename T>
class GLNormalizePlanarYUVOp final : public Operator<GLContext> {
public:
  GLNormalizePlanarYUVOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<GLContext>(operator_def, ws) {}
  virtual ~GLNormalizePlanarYUVOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  arm_compute::GCNormalizePlanarYUVLayer norm_layer_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> X_, mean_, sd_;
};

template <typename T> bool GLNormalizePlanarYUVOp<T>::RunOnDevice() {

  auto Xblob = OperatorBase::Inputs()[0];
  auto *meanblob = OperatorBase::Inputs()[1];
  auto *sdblob = OperatorBase::Inputs()[2];

  if (first_run_) {
    X_ = GLContext::getGLTensor<T>(Xblob);
    mean_ = GLContext::getGLTensor<T>(meanblob);
    sd_ = GLContext::getGLTensor<T>(sdblob);
  }

  CAFFE_ENFORCE_EQ(X_->ndim(), 4);
  auto N = X_->dim32(0);
  auto C = X_->dim32(1);
  auto H = X_->dim32(2);
  auto W = X_->dim32(3);

  CAFFE_ENFORCE_EQ(C, mean_->dim32(1));
  CAFFE_ENFORCE_EQ(C, sd_->dim32(1));

  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();
  if (first_run_) {
    first_run_ = false;
    Y->ResizeLike(*X_);
    norm_layer_.configure(X_->get_underlying(), Y->get_underlying(), mean_->get_underlying(), sd_->get_underlying());
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    mean_->lazy_allocate(meanblob, second_run_, second_run_);
    sd_->lazy_allocate(sdblob, second_run_, second_run_);
    if (second_run_) {
      second_run_ = false;
      Y->allocate();
    }
    norm_layer_.run();
  }

  return true;
}

REGISTER_GL_OPERATOR(NormalizePlanarYUV, GLNormalizePlanarYUVOp<DataType>);

} // namespace caffe2
