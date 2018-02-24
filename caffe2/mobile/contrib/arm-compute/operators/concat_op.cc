#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"
#include "caffe2/operators/concat_split_op.h"

namespace caffe2 {

template <typename T> class GLConcatOp final : public Operator<GLContext> {
public:
  GLConcatOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<GLContext>(operator_def, ws) {}
  virtual ~GLConcatOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  arm_compute::GCDepthConcatenateLayer concat_layer_;
  bool first_run_ = true, second_run_ = true;
  std::vector<GLContext::deleted_unique_ptr<const GLTensor<T>>> inputs_;
  int channelCount_ = 0;
};


template <typename T>
bool GLConcatOp<T>::RunOnDevice() {

  CAFFE_ENFORCE(InputSize() <= 4 && InputSize() >= 2, "Number \
  of input must be between 2 and 4.");

  auto *X0blob = OperatorBase::Inputs()[0];
  auto X0 = GLContext::getGLTensor<T>(X0blob);
  if (first_run_) {
    inputs_.push_back(std::move(X0));
  }

  int N = inputs_[0]->dim32(0);
  int channels = inputs_[0]->dim32(1);
  int height = inputs_[0]->dim32(2);
  int width = inputs_[0]->dim32(3);
  std::vector<const Blob*> inputsBlob;
  inputsBlob.push_back(X0blob);

  if (first_run_) {
    channelCount_ = channels;
    for (int i = 1; i < Inputs().size(); ++i) {
      auto *Xblob = OperatorBase::Inputs()[i];
      auto X = GLContext::getGLTensor<T>(Xblob);
      CAFFE_ENFORCE_EQ(N, X->dim32(0), X->dim32(0));
      CAFFE_ENFORCE_EQ(height, X->dim32(2), X->dim32(2));
      CAFFE_ENFORCE_EQ(width, X->dim32(3), X->dim32(3));
      channelCount_ += X->dim32(1);
      inputs_.push_back(std::move(X));
    }
  }

  for (int i = 1; i < Inputs().size(); ++i) {
    auto *Xblob = OperatorBase::Inputs()[i];
    inputsBlob.push_back(Xblob);
  }
  std::vector<int> output_dims = {N, channelCount_, height, width};
  GLTensor<T> *Y =
      OperatorBase::Outputs()[0]->template GetMutable<GLTensor<T>>();
  if (first_run_) {
    first_run_ = false;
    Y->Resize(output_dims);

    std::vector<arm_compute::IGCTensor*> inputsGC;
    for (int i = 0; i < inputs_.size(); ++i) {
      inputsGC.push_back(inputs_[i]->get_underlying());
    }
    concat_layer_.configure(inputsGC, Y->get_underlying());
  } else {
    for (int i = 0; i < inputs_.size(); ++i) {
      auto* X = inputs_[i].get();
      auto* Xblob = inputsBlob[i];
      X->lazy_allocate(Xblob, second_run_, true);
    }
    if (second_run_) {
      second_run_ = false;
      Y->allocate();
    }
    concat_layer_.run();
  }

  return true;
}

REGISTER_GL_OPERATOR(Concat, GLConcatOp<DataType>);

} // namespace caffe2
