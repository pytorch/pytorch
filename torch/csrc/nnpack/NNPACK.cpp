#include "NNPACK.h"

#include "TH/TH.h"
#include <stdlib.h>

namespace torch {
namespace nnpack {

// Stolen from Caffe2
static pthreadpool_t nnpack_threadpool_ = nullptr;

pthreadpool_t nnpack_threadpool() {
  if (nnpack_threadpool_ == nullptr) {
    enum nnp_status nnpack_status = nnp_initialize();
    if (nnpack_status != nnp_status_success) throw std::runtime_error("could not initialize NNPack");
    nnpack_threadpool_ = pthreadpool_create(16);
  }
  return nnpack_threadpool_;
}

// Assuming for now this is Thread-safe, but that seems like a stretch
static void *workspace = nullptr;
static size_t workspace_size = 0;

// NNPack has alignment requirements
const size_t nnpack_memory_alignment_boundary = 64;

static inline void deallocate_workspace() {
  if (workspace)
    std::free(workspace);
  workspace = nullptr;
}

static inline void allocate_workspace() {
  if (workspace)
    deallocate_workspace();
  posix_memalign(&workspace, nnpack_memory_alignment_boundary, workspace_size);
}

void SpatialConvolution_updateOutput(
    at::Tensor& input,
    at::Tensor& output,
    at::Tensor& weight,
    at::Tensor& bias,
    int kW,
    int kH,
    int padW,
    int padH) {
  // Setup parameters for the NNPack convolution output function call

  // For now, we use the default algorithm
  auto algorithm = nnp_convolution_algorithm_auto;

  // Our input Tensor must be in the form N,C,H,W
  if (input.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D input Tensor N,C,H,W");
  }
  // Our weight Tensor must be in the form oC,iC,kH,kW
  if (weight.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D weight Tensor oC,iC,kH,kW");
  }
  // Our weight Tensor must be in the form N,oC,oH,oW
  if (output.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D output Tensor N,oC,oH,oW");
  }

  // All Tensors must be float Tensors
  if (input.type().ID() != at::TypeID::CPUFloat ||
      weight.type().ID() != at::TypeID::CPUFloat ||
      output.type().ID() != at::TypeID::CPUFloat ||
      (bias.defined() && bias.type().ID() != at::TypeID::CPUFloat)) {
    throw std::runtime_error("Mismatched Tensor types in NNPack convolutionOutput");
  }

  const size_t batch_size = input.sizes()[0];
  const size_t input_channels = input.sizes()[1];
  const size_t output_channels = weight.sizes()[0];
  const struct nnp_size input_size = {
    .width = input.sizes()[3],
    .height = input.sizes()[2]
  };
  const struct nnp_padding input_padding = {
    .top = padH,
    .right = padW,
    .bottom = padH,
    .left = padW
  };
  const struct nnp_size kernel_size = {
    .width = kW,
    .height = kH
  };

  // If we don't have a defined bias Tensor, we need to create one filled with zeroes
  auto bias_ = bias.defined() ? bias : input.type().zeros({output_channels});

  // Note: we assume that the output is shaped correctly, probably should add an assert

  auto run = [&]() -> nnp_status {
    return nnp_convolution_output(
        algorithm,
        batch_size,
        input_channels,
        output_channels,
        input_size,
        input_padding,
        kernel_size,
        (float*)input.data_ptr(),
        (float*)weight.data_ptr(),
        (float*)bias_.data_ptr(),
        (float*)output.data_ptr(),
        workspace, // workspace_buffer
        &workspace_size, // workspace_size
        nnp_activation_identity,
        nullptr, // activation parameters
        nnpack_threadpool(),
        nullptr // profile
    );
  };

  auto size_and_allocate_ws = [&]() {
    // Run a single pass to get the size of memory workspace buffer
    auto status = run();
    if (status != nnp_status_success) {
      throw std::runtime_error("NNPACK SpatialConvolution_updateOutput failed");
    }
    allocate_workspace();
  };

  // If no workspace created yet, allocate it
  if (workspace == nullptr) {
    size_and_allocate_ws();
  }

  // Try to run with the newly created, or existing workspace
  auto status = run();

  if (status == nnp_status_insufficient_buffer) {
    // Need to reallocate the workspace
    deallocate_workspace();
    size_and_allocate_ws();

    // Try one more time
    status = run();
  }

  if (status != nnp_status_success) {
    throw std::runtime_error("NNPACK SpatialConvolution_updateOutput failed");
  }
}

void SpatialConvolution_updateGradInput(
    at::Tensor& input,
    at::Tensor& gradOutput,
    at::Tensor& gradInput,
    at::Tensor& weight,
    int kW,
    int kH,
    int padW,
    int padH) {
  // Setup parameters for the NNPACK convolution input gradient call

  // Use the default algorithm
  auto algorithm = nnp_convolution_algorithm_auto;

  // Our input and gradInput Tensors must be in the form N,C,H,W
  if (input.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D input Tensor N,C,H,W");
  }
  if (gradInput.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D gradInput Tensor N,C,H,W");
  }
  // Our weight Tensor must be in the form oC,iC,kH,kW
  if (weight.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D weight Tensor oC,iC,kH,kW");
  }
  // Our gradOutput Tensor must be in the form N,oC,oH,oW
  if (gradOutput.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D gradOutput Tensor N,oC,oH,oW");
  }

  const size_t batch_size = input.sizes()[0];
  const size_t input_channels = input.sizes()[1];
  const size_t output_channels = weight.sizes()[0];
  const struct nnp_size input_size = {
    .width = input.sizes()[3],
    .height = input.sizes()[2]
  };
  const struct nnp_padding input_padding = {
    .top = padH,
    .right = padW,
    .bottom = padH,
    .left = padW
  };
  const struct nnp_size kernel_size = {
    .width = kW,
    .height = kH
  };

  auto run = [&]() -> nnp_status {
    return nnp_convolution_input_gradient(
        algorithm,
        batch_size,
        input_channels,
        output_channels,
        input_size,
        input_padding,
        kernel_size,
        (float*)gradOutput.data_ptr(),
        (float*)weight.data_ptr(),
        (float*)gradInput.data_ptr(),
        workspace, // workspace_buffer
        &workspace_size, // workspace_size
        nnp_activation_identity,
        nullptr, // activation_parameters
        nnpack_threadpool(),
        nullptr  // profile
    );
  };

  auto size_and_allocate_ws = [&]() {
    // Run a single pass to get the size of memory workspace buffer
    auto status = run();
    if (status != nnp_status_success) {
      throw std::runtime_error("NNPACK SpatialConvolution_updateGradInput failed");
    }
    allocate_workspace();
  };

  // If no workspace created yet, allocate it
  if (workspace == nullptr) {
    size_and_allocate_ws();
  }

  // Try to run with the newly created, or existing workspace
  auto status = run();

  if (status == nnp_status_insufficient_buffer) {
    // Need to reallocate the workspace
    deallocate_workspace();
    size_and_allocate_ws();

    // Try one more time
    status = run();
  }

  if (status != nnp_status_success) {
    throw std::runtime_error("NNPACK SpatialConvolution_updateGradInput failed");
  }
}

void SpatialConvolution_accGradWeight(
    at::Tensor& input,
    at::Tensor& gradOutput,
    at::Tensor& gradWeight,
    int kW,
    int kH,
    int padW,
    int padH) {
  // Setup parameters for the NNPACK convolution kernel gradient call

  // Use the default algorithm
  auto algorithm = nnp_convolution_algorithm_auto;

  // Our input and gradInput Tensors must be in the form N,C,H,W
  if (input.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D input Tensor N,C,H,W");
  }
  // Our gradWeight Tensor must be in the form oC,iC,kH,kW
  if (gradWeight.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D gradWeight Tensor oC,iC,kH,kW");
  }
  // Our weight Tensor must be in the form N,oC,oH,oW
  if (gradOutput.ndimension() != 4) {
    throw std::runtime_error("NNPack convolutionOutput expects 4D gradOutput Tensor N,oC,oH,oW");
  }

  const size_t batch_size = input.sizes()[0];
  const size_t input_channels = input.sizes()[1];
  const size_t output_channels = gradWeight.sizes()[0];
  const struct nnp_size input_size = {
    .width = input.sizes()[3],
    .height = input.sizes()[2]
  };
  const struct nnp_padding input_padding = {
    .top = padH,
    .right = padW,
    .bottom = padH,
    .left = padW
  };
  const struct nnp_size kernel_size = {
    .width = kW,
    .height = kH
  };

  auto run= [&]() -> nnp_status {
    return nnp_convolution_kernel_gradient(
        algorithm,
        batch_size,
        input_channels,
        output_channels,
        input_size,
        input_padding,
        kernel_size,
        (float*)input.data_ptr(),
        (float*)gradOutput.data_ptr(),
        (float*)gradWeight.data_ptr(),
        workspace, // workspace_buffer
        &workspace_size, // workspace_size
        nnp_activation_identity,
        nullptr, // activation_parameters
        nnpack_threadpool(),
        nullptr  // profile
    );
  };

  auto size_and_allocate_ws = [&]() {
    // Run a single pass to get the size of memory workspace buffer
    auto status = run();
    if (status != nnp_status_success) {
      throw std::runtime_error("NNPACK SpatialConvolution_accGradWeight failed");
    }
    allocate_workspace();
  };

  // If no workspace created yet, allocate it
  if (workspace == nullptr) {
    size_and_allocate_ws();
  }

  // Try to run with the newly created, or existing workspace
  auto status = run();

  if (status == nnp_status_insufficient_buffer) {
    // Need to reallocate the workspace
    deallocate_workspace();
    size_and_allocate_ws();

    // Try one more time
    status = run();
  }

  if (status != nnp_status_success) {
    throw std::runtime_error("NNPACK SpatialConvolution_accGradWeight failed");
  }
}

} // torch::nnpack
} // torch
