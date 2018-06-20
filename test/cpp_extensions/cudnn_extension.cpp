#include <torch/torch.h>
#include <cuda.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Exceptions.h>
#include <ATen/cudnn/Handles.h>
#include <torch/csrc/cuda/cuda_check.h>
#include <torch/torch.h>
#include <iostream>

using namespace at;
using namespace std;
using namespace at::native;
const char *cudnn_relu_name = "cudnn_relu";

// Check arguments to cudnn_relu
int cudnn_relu_check(const Tensor &inputs, const Tensor &outputs) {
  // Create TensorArgs
  TensorArg arg_inputs(inputs, "inputs", 0);
  TensorArg arg_outputs(outputs, "outputs", 1);
  // Check arguments
  checkContiguous(cudnn_relu_name, arg_inputs);
  checkScalarType(cudnn_relu_name, arg_inputs, ScalarType::Float);                      \
  checkBackend(cudnn_relu_name, arg_inputs.tensor, Backend::CUDA);
  checkContiguous(cudnn_relu_name, arg_outputs);
  checkScalarType(cudnn_relu_name, arg_outputs, ScalarType::Float);                      \
  checkBackend(cudnn_relu_name, arg_outputs.tensor, Backend::CUDA);
  checkSameSize(cudnn_relu_name, arg_inputs, arg_outputs);
  return 0;
}

int cudnn_relu(const Tensor& inputs, const Tensor& outputs) {
  //Check inputs
  cudnn_relu_check(inputs, outputs);
  // Declarations
  cudnnHandle_t cuDnn = getCudnnHandle();
  TensorDescriptor* input_tensor_desc = new TensorDescriptor(inputs, 4);
  // 4 is minium size for TensorDescriptor
  cudnnActivationDescriptor_t activationDesc;
  cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU;
  cudnnNanPropagation_t reluNanOpt = CUDNN_PROPAGATE_NAN;
  double coef = 1.;
  float alpha = 1.;
  float beta = 0.;
  // Activation descriptor
  CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
  CUDNN_CHECK(cudnnSetActivationDescriptor(
      activationDesc,
      mode,
      reluNanOpt,
      coef));
  //Apply
  CUDNN_CHECK(cudnnActivationForward(
    cuDnn,
    activationDesc,
    &alpha,
    input_tensor_desc->desc(),
    inputs.data_ptr(),
    &beta,
    input_tensor_desc->desc(), // same size and type so we only need one descriptor
    outputs.data_ptr()));
  //Destroy
  CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
  delete input_tensor_desc;
  //Return
  return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cudnn_relu", &cudnn_relu, "CuDNN ReLU");
}
