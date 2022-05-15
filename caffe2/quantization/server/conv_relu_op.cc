#include "conv_relu_op.h"

namespace caffe2 {

template <typename T, class Context>
bool ConvReluOp<T, Context>::RunOnDeviceWithOrderNCHW() {
  // Delegate to local conv operator
  for (int i = 0; i < this->InputSize(); ++i) {
    local_input_blobs_[i]->ShareExternal(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<void*>(this->Inputs()[i]->GetRaw()),
        this->Inputs()[i]->meta());
  }

  if (!local_op_->RunOnDeviceWithOrderNCHW()) {
    return false;
  }

  // Apply Relu
  Tensor* local_output =
      BlobGetMutableTensor(local_output_blobs_[0], Context::GetDeviceType());
  const T* output_local_data = local_output->template data<T>();

  Tensor* output =
      Operator<Context>::Output(0, local_output->sizes(), at::dtype<T>());
  T* output_data = output->template mutable_data<T>();
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
  for (int i = 0; i < this->InputSize(); ++i) {
    local_input_blobs_[i]->ShareExternal(
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<void*>(this->Inputs()[i]->GetRaw()),
        this->Inputs()[i]->meta());
  }

  if (!local_op_->RunOnDeviceWithOrderNHWC()) {
    return false;
  }

  // Apply Relu
  Tensor* local_output =
      BlobGetMutableTensor(local_output_blobs_[0], Context::GetDeviceType());
  const T* output_local_data = local_output->template data<T>();

  Tensor* output =
      Operator<Context>::Output(0, local_output->sizes(), at::dtype<T>());
  T* output_data = output->template mutable_data<T>();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < output->numel(); ++i) {
    output_data[i] = std::max(static_cast<T>(0), output_local_data[i]);
  }

  return true;
}

OPERATOR_SCHEMA(ConvRelu)
    .NumInputs(2, 3)
    .NumOutputs(1)
    .TensorInferenceFunction(ConvPoolOpBase<CPUContext>::TensorInferenceForConv)
    .CostInferenceFunction(OpSchema::CostInferenceFunctionType(
        ConvPoolOpBase<CPUContext>::CostInferenceForConv));

REGISTER_CPU_OPERATOR(ConvRelu, ConvReluOp<float, CPUContext>);

} // namespace caffe2
