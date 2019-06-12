#include "conv_relu_op.h"

namespace caffe2 {

template <typename T, class Context>
bool ConvReluOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  // Delegate to local conv operator
  for (int i = 0; i < OperatorBase::InputSize(); ++i) {
    local_input_blobs_[i]->ShareExternal(
      const_cast<void*>(OperatorBase::Inputs()[i]->GetRaw()),
      OperatorBase::Inputs()[i]->meta());
  }

  if (!local_op_->RunOnDeviceWithOrderNCHW()) return false;

  // Apply Relu
  Tensor *local_output =
    BlobGetMutableTensor(local_output_blobs_[0], Context::GetDeviceType());
  const T *output_local_data = local_output->template data<T>();

  Tensor *output = Operator<Context>::Output(0);
  output->ResizeLike(*local_output);
  T *output_data = output->template mutable_data<T>();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < output->numel(); ++i) {
    output_data[i] = std::max(static_cast<T>(0), output_local_data[i]);
  }

  return true;
}

template <typename T, class Context>
bool ConvReluOp<T, Context>::RunOnDeviceWithOrderNHWC() {
  // Delegate to local conv operator
  for (int i = 0; i < OperatorBase::InputSize(); ++i) {
    local_input_blobs_[i]->ShareExternal(
      const_cast<void*>(OperatorBase::Inputs()[i]->GetRaw()),
      OperatorBase::Inputs()[i]->meta());
  }

  if (!local_op_->RunOnDeviceWithOrderNHWC()) return false;

  // Apply Relu
  Tensor *local_output =
    BlobGetMutableTensor(local_output_blobs_[0], Context::GetDeviceType());
  const T *output_local_data = local_output->template data<T>();

  Tensor *output = Operator<Context>::Output(0);
  output->ResizeLike(*local_output);
  T *output_data = output->template mutable_data<T>();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < output->numel(); ++i) {
    output_data[i] = std::max(static_cast<T>(0), output_local_data[i]);
  }

  return true;
}

REGISTER_CPU_OPERATOR(ConvRelu, ConvReluOp<float, CPUContext>);

} // namespace caffe2
