#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"
#include "caffe2/operators/resize_op.h"

namespace caffe2 {

template<typename T>
class GLResizeNearestOp final : public Operator<GLContext> {
public:
  GLResizeNearestOp(const OperatorDef &operator_def, Workspace *ws)
    : Operator<GLContext>(operator_def, ws), width_scale_(1), height_scale_(1) {
    if (HasArgument("width_scale")) {
      width_scale_ = static_cast<float>(
          OperatorBase::GetSingleArgument<float>("width_scale", 1));
    }
    if (HasArgument("height_scale")) {
      height_scale_ = static_cast<float>(
          OperatorBase::GetSingleArgument<float>("height_scale", 1));
    }
    CAFFE_ENFORCE_GT(width_scale_, 0);
    CAFFE_ENFORCE_GT(height_scale_, 0);
  }
  virtual ~GLResizeNearestOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  float width_scale_;
  float height_scale_;
  arm_compute::GCScale resize_layer_;
  bool first_run_ = true, second_run_ = true;
  GLContext::deleted_unique_ptr<const GLTensor<T>> X_;
};

template <typename T>
bool GLResizeNearestOp<T>::RunOnDevice() {

  auto* Xblob = OperatorBase::Inputs()[0];

  X_ = GLContext::getGLTensor<T>(Xblob, X_.release());

  auto N = X_->dim32(0);
  auto C = X_->dim32(1);
  auto H = X_->dim32(2);
  auto W = X_->dim32(3);

  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();
  vector<TIndex> output_dims = {N, C, H * height_scale_, W * width_scale_};

  if (first_run_) {
    Y->Resize(output_dims);
    first_run_ = false;
    resize_layer_.configure(X_->get_underlying(), Y->get_underlying(), arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR, arm_compute::BorderMode::UNDEFINED);
  } else if (second_run_) {
    X_->lazy_allocate(Xblob, second_run_, true);
    second_run_ = false;
    Y->Resize(output_dims);
    Y->allocate();
    resize_layer_.run();
  } else {
    X_->lazy_allocate(Xblob, second_run_, true);
    bool need_allocation = Y->Resize(output_dims);
    resize_layer_.configure(X_->get_underlying(), Y->get_underlying(), arm_compute::InterpolationPolicy::NEAREST_NEIGHBOR, arm_compute::BorderMode::UNDEFINED);
    if (need_allocation) {
      Y->allocate();
    }
  }

  return true;
}

REGISTER_GL_OPERATOR(ResizeNearest, GLResizeNearestOp<DataType>);

} // namespace caffe2
