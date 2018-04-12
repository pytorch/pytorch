#include "caffe2/core/predictor.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace predictor_details {

void CopyInputTensor(
    Workspace* ws,
    const std::string& name,
    TensorCPU* input,
    CUDAContext *context) {
  enforceIsTensor<TensorCUDA>(ws, name);
  auto* blob = ws->GetBlob(name);
  CAFFE_ENFORCE(blob, "Blob: ", name, " does not exist");
  auto* tensor = blob->template GetMutable<TensorCUDA>();
  tensor->CopyFrom(*input, context);
}

std::shared_ptr<TensorCPU> CopyOutputTensor(
    Workspace* ws,
    const std::string& name,
    CUDAContext *context) {
  enforceIsTensor<TensorCUDA>(ws, name);
  auto* blob = ws->GetBlob(name);
  std::shared_ptr<TensorCPU> tensor = std::make_shared<TensorCPU>(
    *blob->template GetMutable<TensorCUDA>(),
    context
  );
  return tensor;
}

} // namespace predictor_details

template <>
bool Predictor<CUDAContext>::run_map(const TensorMap& inputs, OutputTensorVector* outputs) {
  if (!inputNames_.empty()) {
    CAFFE_ENFORCE_EQ(inputs.size(), inputNames_.size());
  }
  for (auto &input: inputs) {
    if (!inputNames_.empty()) {
      CAFFE_ENFORCE_GT(inputNames_.count(input.first), 0);
    }
    predictor_details::CopyInputTensor(&ws_, input.first, input.second, context_.get());
  }

  if (!ws_.RunNet(run_net_.name())) {
    return false;
  }

  outputs->resize(run_net_.external_output_size());
  for (auto i = 0; i < outputs->size(); ++i) {
    (*outputs)[i] = predictor_details::CopyOutputTensor(&ws_, run_net_.external_output(i), context_.get());
  }
  context_->FinishDeviceComputation();
  return true;
}

CAFFE_REGISTER_PREDICTOR(CUDA, Predictor<CUDAContext>);

} // namespace caffe2