#include "caffe2/mobile/contrib/arm-compute/core/context.h"
#include "caffe2/mobile/contrib/arm-compute/core/operator.h"
#include "caffe2/core/timer.h"

namespace caffe2 {

template <typename T> class CopyFromGLOp final : public Operator<GLContext> {
public:
  CopyFromGLOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<GLContext>(operator_def, ws) {}
  virtual ~CopyFromGLOp() noexcept {}
  USE_OPERATOR_FUNCTIONS(GLContext);
  bool RunOnDevice() override;
private:
  bool first_run_ = true, second_run_ = true;
  std::vector<GLContext::deleted_unique_ptr<const GLTensor<T>>> inputs_;
};

template <typename T>
bool CopyFromGLOp<T>::RunOnDevice() {

  std::vector<const Blob*> inputsBlob;

  for (int i = 0; i < Inputs().size(); ++i) {
    auto *Xblob = OperatorBase::Inputs()[i];
    inputsBlob.push_back(Xblob);
  }

  if (first_run_) {
    for (int i = 0; i < Inputs().size(); ++i) {
      auto *Xblob = inputsBlob[i];
      auto X = GLContext::getGLTensor<T>(Xblob);
      inputs_.push_back(std::move(X));
    }
  } else {
    for (int i = 0; i < Inputs().size(); ++i) {
      auto *Xblob = inputsBlob[i];
      auto X = GLContext::getGLTensor<T>(Xblob, inputs_[i].release());
      inputs_[i] = std::move(X);
    }
  }

  if (first_run_) {
    first_run_ = false;
    for (int i = 0; i < Inputs().size(); ++i) {
      auto* Y = OperatorBase::Outputs()[i]->template GetMutable<TensorCPU>();
      Y->Resize(inputs_[i]->dims());
      Y->template mutable_data<float>();
    }
  } else {
    for (auto i = 0; i < Inputs().size(); ++i) {
      // Blob
      auto* Xblob = inputsBlob[i];
      // GLTensor
      auto* X = inputs_[i].get();
      X->lazy_allocate(Xblob, second_run_, true);
      auto* Y = OperatorBase::Outputs()[i]->template GetMutable<TensorCPU>();
      Timer timer;
      timer.Start();
      getTensorCPU(*X, *Y);
      auto millis = timer.MilliSeconds();
      //LOG(ERROR) << "[C2DEBUG] copy_op " << X->dims() << " takes " << millis << " milliseconds";
    }
  }

  return true;
}

REGISTER_GL_OPERATOR(CopyFromGL, CopyFromGLOp<DataType>);

} // namespace caffe2
