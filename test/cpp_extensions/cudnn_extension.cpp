/*
 * CuDNN ReLU extension. Simple function but contains the general structure of
 * most CuDNN extensions:
 * 1) Check arguments. torch::check* functions provide a standard way to
 * validate input and provide pretty errors. 2) Create descriptors. Most CuDNN
 * functions require creating and setting a variety of descriptors. 3) Apply the
 * CuDNN function. 4) Destroy your descriptors. 5) Return something (optional).
 */

#include <torch/extension.h>

#include <ATen/cuda/Exceptions.h> // for CUDNN_CHECK
#include <ATen/cudnn/Descriptors.h> // for TensorDescriptor
#include <ATen/cudnn/Handle.h> // for getCudnnHandle

// Name of function in python module and name used for error messages by
// torch::check* functions.
const char* cudnn_relu_name = "cudnn_relu";

// Check arguments to cudnn_relu
void cudnn_relu_check(
    const torch::Tensor& inputs,
    const torch::Tensor& outputs) {
  // Create TensorArgs. These record the names and positions of each tensor as a
  // parameter.
  torch::TensorArg arg_inputs(inputs, "inputs", 0);
  torch::TensorArg arg_outputs(outputs, "outputs", 1);
  // Check arguments. No need to return anything. These functions with throw an
  // error if they fail. Messages are populated using information from
  // TensorArgs.
  torch::checkContiguous(cudnn_relu_name, arg_inputs);
  torch::checkScalarType(cudnn_relu_name, arg_inputs, torch::kFloat);
  torch::checkBackend(cudnn_relu_name, arg_inputs.tensor, torch::Backend::CUDA);
  torch::checkContiguous(cudnn_relu_name, arg_outputs);
  torch::checkScalarType(cudnn_relu_name, arg_outputs, torch::kFloat);
  torch::checkBackend(
      cudnn_relu_name, arg_outputs.tensor, torch::Backend::CUDA);
  torch::checkSameSize(cudnn_relu_name, arg_inputs, arg_outputs);
}

void cudnn_relu(const torch::Tensor& inputs, const torch::Tensor& outputs) {
  // Most CuDNN extensions will follow a similar pattern.
  // Step 1: Check inputs. This will throw an error if inputs are invalid, so no
  // need to check return codes here.
  cudnn_relu_check(inputs, outputs);
  // Step 2: Create descriptors
  cudnnHandle_t cuDnn = torch::native::getCudnnHandle();
  // Note: 4 is minimum dim for a TensorDescriptor. Input and output are same
  // size and type and contiguous, so one descriptor is sufficient.
  torch::native::TensorDescriptor input_tensor_desc(inputs, 4);
  cudnnActivationDescriptor_t activationDesc;
  // Note: Always check return value of cudnn functions using CUDNN_CHECK
  AT_CUDNN_CHECK(cudnnCreateActivationDescriptor(&activationDesc));
  AT_CUDNN_CHECK(cudnnSetActivationDescriptor(
      activationDesc,
      /*mode=*/CUDNN_ACTIVATION_RELU,
      /*reluNanOpt=*/CUDNN_PROPAGATE_NAN,
      /*coef=*/1.));
  // Step 3: Apply CuDNN function
  float alpha = 1.;
  float beta = 0.;
  AT_CUDNN_CHECK(cudnnActivationForward(
      cuDnn,
      activationDesc,
      &alpha,
      input_tensor_desc.desc(),
      inputs.data_ptr(),
      &beta,
      input_tensor_desc.desc(), // output descriptor same as input
      outputs.data_ptr()));
  // Step 4: Destroy descriptors
  AT_CUDNN_CHECK(cudnnDestroyActivationDescriptor(activationDesc));
  // Step 5: Return something (optional)
}

// Create the pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Use the same name as the check functions so error messages make sense
  m.def(cudnn_relu_name, &cudnn_relu, "CuDNN ReLU");
}
