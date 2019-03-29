#include <ATen/ATen.h>
#include <ATen/Config.h>

#if !AT_NNPACK_ENABLED()

namespace at {
namespace native {

at::Tensor _nnpack_spatial_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    IntArrayRef padding) {
  throw std::runtime_error(
      "nnpack_spatial_convolution: ATen not compiled with NNPACK support");
}

at::Tensor _nnpack_spatial_convolution_backward_input(
    const at::Tensor& input,
    const at::Tensor& gradOutput,
    const at::Tensor& weight,
    IntArrayRef padding) {
  throw std::runtime_error(
      "nnpack_spatial_convolution_backward_input: ATen not compiled with NNPACK support");
}

at::Tensor _nnpack_spatial_convolution_backward_weight(
    const at::Tensor& input,
    at::IntArrayRef weight_size,
    const at::Tensor& gradOutput,
    IntArrayRef padding) {
  throw std::runtime_error(
      "nnpack_spatial_convolution_backward_weight: ATen not compiled with NNPACK support");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
_nnpack_spatial_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& gradOutput,
    const at::Tensor& weight,
    IntArrayRef padding,
    std::array<bool, 3> output_mask) {
  throw std::runtime_error(
      "_nnpack_spatial_convolution_backward: ATen not compiled with NNPACK support");
}

bool _nnpack_available() {
  return false;
}

} // namespace native
} // namespace at

#else

#include "nnpack.h"

#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#else
#include <thread>
#endif

namespace at {
namespace native {

// Stolen from Caffe2
static pthreadpool_t nnpack_threadpool_ = nullptr;
static bool called_nnpack_threadpool_ = false;

pthreadpool_t nnpack_threadpool() {
  if (! called_nnpack_threadpool_) {
    called_nnpack_threadpool_ = true;
    enum nnp_status nnpack_status = nnp_initialize();
    if (nnpack_status != nnp_status_success) {
      if (nnpack_status == nnp_status_out_of_memory) {
	throw std::runtime_error("could not initialize NNPack (out of memory)");
      } else if (nnpack_status == nnp_status_unsupported_hardware) {
	throw std::runtime_error("could not initialize NNPack (unsupported hardware)");
      } else {
	throw std::runtime_error("could not initialize NNPack (unknown error)");
      }
    }
    unsigned int threads;
#ifdef _OPENMP
    threads = omp_get_num_threads();
#else
    threads = std::thread::hardware_concurrency();
#endif
    nnpack_threadpool_ = pthreadpool_create(threads);
    if (nnpack_threadpool_ == nullptr) {
      throw std::runtime_error("could not initialize NNPack's pthreadpool");
    }
  }
  return nnpack_threadpool_;
}

bool _nnpack_available() {
  if (! called_nnpack_threadpool_) {
    try {
      return nnpack_threadpool() != nullptr;
    } catch (std::runtime_error e) {
    }
  }
  return nnpack_threadpool() != nullptr;
}

// Make thread_local for safety in cases where we have multiple threads running
// Convs at once
static thread_local void* workspace = nullptr;
static thread_local size_t workspace_size = 0;

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
  // Won't work on Windows, but NNPACK doesn't support Windows either
  posix_memalign(&workspace, nnpack_memory_alignment_boundary, workspace_size);
}

constexpr int input_batch_size_dim = 0;
constexpr int input_channels_dim = 1;
constexpr int input_height_dim = 2;
constexpr int input_width_dim = 3;
constexpr int output_batch_size_dim = 0;
constexpr int output_channels_dim = 1;
constexpr int output_height_dim = 2;
constexpr int output_width_dim = 3;
constexpr int weight_output_channels_dim = 0;
constexpr int weight_input_channels_dim = 1;
constexpr int weight_height_dim = 2;
constexpr int weight_width_dim = 3;

// Often written as 2 + max_dim (extra dims for batch size and channels)
constexpr int max_dim = 3;

std::vector<int64_t> conv_output_size(
    IntArrayRef input_size,
    IntArrayRef weight_size,
    IntArrayRef padding) {
  auto dim = input_size.size();
  std::vector<int64_t> output_size(dim);
  output_size[output_batch_size_dim] = input_size[input_batch_size_dim];
  output_size[output_channels_dim] = weight_size[weight_output_channels_dim];
  output_size[output_height_dim] =
      input_size[input_height_dim] + 2 * padding[0] - (weight_size[2] - 1);
  output_size[output_width_dim] =
      input_size[input_width_dim] + 2 * padding[1] - (weight_size[3] - 1);
  return output_size;
}

Tensor _nnpack_spatial_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    IntArrayRef padding) {
  at::Tensor output = at::empty(
      conv_output_size(input.sizes(), weight.sizes(), padding),
      input.options());

  // Our input Tensor must be in the form N,C,H,W
  if (input.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolutionOutput expects 4D input Tensor N,C,H,W");
  }
  // Our weight Tensor must be in the form oC,iC,kH,kW
  if (weight.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolutionOutput expects 4D weight Tensor oC,iC,kH,kW");
  }
  // Our output Tensor must be in the form N,oC,oH,oW
  if (output.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolutionOutput expects 4D output Tensor N,oC,oH,oW");
  }

  // Some basic shape checking, not comprehensive
  if (input.size(1) != weight.size(1)) {
    std::stringstream err;
    err << "Mismatch between number of input channels in input Tensor ("
        << input.size(1) << ") and weight Tensor (" << weight.size(1)
        << ") in NNPack convolutionOutput";
    throw std::runtime_error(err.str());
  }
  if (weight.size(0) != output.size(1)) {
    std::stringstream err;
    err << "Mismatch between number of output channels in weight Tensor ("
        << weight.size(0) << ") and output Tensor (" << output.size(1)
        << ") in NNPack convolutionOutput";
    throw std::runtime_error(err.str());
  }
  if (input.size(0) != output.size(0)) {
    std::stringstream err;
    err << "Mismatch between batch size in input Tensor (" << input.size(0)
        << ") and output Tensor (" << output.size(0)
        << ") in NNPack convolutionOutput";
    throw std::runtime_error(err.str());
  }

  // Setup parameters for the NNPack convolution output function call

  // For now, we use the default algorithm
  auto algorithm = nnp_convolution_algorithm_auto;

  // All Tensors must be float Tensors
  if (input.type().ID() != at::TypeID::CPUFloat ||
      weight.type().ID() != at::TypeID::CPUFloat ||
      output.type().ID() != at::TypeID::CPUFloat ||
      (bias.defined() && bias.type().ID() != at::TypeID::CPUFloat)) {
    throw std::runtime_error(
        "Mismatched Tensor types in NNPack convolutionOutput");
  }

  const size_t batch_size = input.size(0);
  const size_t input_channels = input.size(1);
  const size_t output_channels = weight.size(0);
  const struct nnp_size input_size = {.width = (size_t)input.size(3),
                                      .height = (size_t)input.size(2)};
  const struct nnp_padding input_padding = {.top = (size_t)padding[0],
                                            .right = (size_t)padding[1],
                                            .bottom = (size_t)padding[0],
                                            .left = (size_t)padding[1]};
  const struct nnp_size kernel_size = {.width = (size_t)weight.size(3),
                                       .height = (size_t)weight.size(2)};

  // If we don't have a defined bias Tensor, we need to create one filled with
  // zeroes
  auto bias_ =
      bias.defined() ? bias : at::zeros({weight.size(0)}, input.options());

  // Note: we assume that the output is shaped correctly, probably should add an
  // assert
  auto input_ = input.contiguous();
  auto batched = [&]() -> nnp_status {
    return nnp_convolution_output(
        algorithm,
        batch_size,
        input_channels,
        output_channels,
        input_size,
        input_padding,
        kernel_size,
        (float*)input_.data_ptr(),
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

  auto single = [&]() -> nnp_status {
    const nnp_size output_subsample = {.width = 1, .height = 1};
    auto input_ = input.contiguous();
    return nnp_convolution_inference(
        algorithm,
        nnp_convolution_transform_strategy_compute,
        input_channels,
        output_channels,
        input_size,
        input_padding,
        kernel_size,
        output_subsample,
        (float*)input_.data_ptr(),
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
    auto status = batch_size == 1 ? single() : batched();
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
  auto status = batch_size == 1 ? single() : batched();

  if (status == nnp_status_insufficient_buffer) {
    // Need to reallocate the workspace
    deallocate_workspace();
    size_and_allocate_ws();

    // Try one more time
    status = batch_size == 1 ? single() : batched();
  }

  if (status != nnp_status_success) {
    throw std::runtime_error("NNPACK SpatialConvolution_updateOutput failed");
  }

  return output;
}

Tensor _nnpack_spatial_convolution_backward_input(
    const at::Tensor& input,
    const at::Tensor& gradOutput,
    const at::Tensor& weight,
    IntArrayRef padding) {
  at::Tensor gradInput = at::empty(input.sizes(), input.options());

  // Our input and gradInput Tensors must be in the form N,C,H,W
  if (input.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolution updateGradInput expects 4D input Tensor N,C,H,W");
  }
  if (gradInput.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolution updateGradInput expects 4D gradInput Tensor N,C,H,W");
  }
  // Our weight Tensor must be in the form oC,iC,kH,kW
  if (weight.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolution updateGradInput expects 4D weight Tensor oC,iC,kH,kW");
  }
  // Our gradOutput Tensor must be in the form N,oC,oH,oW
  if (gradOutput.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolution updateGradInput expects 4D gradOutput Tensor N,oC,oH,oW");
  }

  // Some basic shape checking, not comprehensive
  if (!input.sizes().equals(gradInput.sizes())) {
    std::stringstream err;
    err << "Mismatch between input size (" << input.sizes()
        << ") and gradInput size (" << gradInput.sizes()
        << ") in NNPack convolution updateGradInput";
    throw std::runtime_error(err.str());
  }
  if (input.size(1) != weight.size(1)) {
    std::stringstream err;
    err << "Mismatch between number of input channels in input Tensor ("
        << input.size(1) << ") and weight Tensor (" << weight.size(1)
        << ") in NNPack convolution updateGradInput";
    throw std::runtime_error(err.str());
  }
  if (weight.size(0) != gradOutput.size(1)) {
    std::stringstream err;
    err << "Mismatch between number of output channels in weight Tensor ("
        << weight.size(0) << ") and gradOutput Tensor (" << gradOutput.size(1)
        << ") in NNPack convolution updateGradInput";
    throw std::runtime_error(err.str());
  }
  if (input.size(0) != gradOutput.size(0)) {
    std::stringstream err;
    err << "Mismatch between batch size in input Tensor (" << input.size(0)
        << ") and gradOutput Tensor (" << gradOutput.size(0)
        << ") in NNPack convolution updateGradInput";
    throw std::runtime_error(err.str());
  }

  // Setup parameters for the NNPACK convolution input gradient call

  // Use the default algorithm
  auto algorithm = nnp_convolution_algorithm_auto;

  const size_t batch_size = input.size(0);
  const size_t input_channels = input.size(1);
  const size_t output_channels = weight.size(0);
  const struct nnp_size input_size = {.width = (size_t)input.size(3),
                                      .height = (size_t)input.size(2)};
  const struct nnp_padding input_padding = {.top = (size_t)padding[0],
                                            .right = (size_t)padding[1],
                                            .bottom = (size_t)padding[0],
                                            .left = (size_t)padding[1]};
  const struct nnp_size kernel_size = {.width = (size_t)weight.size(3),
                                       .height = (size_t)weight.size(2)};

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
        nullptr // profile
    );
  };

  auto size_and_allocate_ws = [&]() {
    // Run a single pass to get the size of memory workspace buffer
    auto status = run();
    if (status != nnp_status_success) {
      throw std::runtime_error(
          "NNPACK SpatialConvolution_updateGradInput failed");
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
    throw std::runtime_error(
        "NNPACK SpatialConvolution_updateGradInput failed");
  }

  return gradInput;
}

Tensor _nnpack_spatial_convolution_backward_weight(
    const at::Tensor& input,
    IntArrayRef weight_size,
    const at::Tensor& gradOutput,
    IntArrayRef padding) {
  at::Tensor gradWeight = at::empty(weight_size, input.options());

  // Our input and gradInput Tensors must be in the form N,C,H,W
  if (input.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolutionOutput expects 4D input Tensor N,C,H,W");
  }
  // Our gradWeight Tensor must be in the form oC,iC,kH,kW
  if (gradWeight.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolutionOutput expects 4D gradWeight Tensor oC,iC,kH,kW");
  }
  // Our weight Tensor must be in the form N,oC,oH,oW
  if (gradOutput.ndimension() != 4) {
    throw std::runtime_error(
        "NNPack convolutionOutput expects 4D gradOutput Tensor N,oC,oH,oW");
  }

  // Some basic shape checking, not comprehensive
  if (input.size(1) != gradWeight.size(1)) {
    std::stringstream err;
    err << "Mismatch between number of input channels in input Tensor ("
        << input.size(1) << ") and gradWeight Tensor (" << gradWeight.size(1)
        << ") in NNPack convolution accGradWeight";
    throw std::runtime_error(err.str());
  }
  if (gradWeight.size(0) != gradOutput.size(1)) {
    std::stringstream err;
    err << "Mismatch between number of output channels in gradWeight Tensor ("
        << gradWeight.size(0) << ") and gradOutput Tensor ("
        << gradOutput.size(1) << ") in NNPack convolution accGradWeight";
    throw std::runtime_error(err.str());
  }
  if (input.size(0) != gradOutput.size(0)) {
    std::stringstream err;
    err << "Mismatch between batch size in input Tensor (" << input.size(0)
        << ") and gradOutput Tensor (" << gradOutput.size(0)
        << ") in NNPack convolution accGradWeight";
    throw std::runtime_error(err.str());
  }

  // Setup parameters for the NNPACK convolution kernel gradient call

  // Use the default algorithm
  auto algorithm = nnp_convolution_algorithm_auto;

  const size_t batch_size = input.size(0);
  const size_t input_channels = input.size(1);
  const size_t output_channels = gradWeight.size(0);
  const struct nnp_size input_size = {.width = (size_t)input.size(3),
                                      .height = (size_t)input.size(2)};
  const struct nnp_padding input_padding = {.top = (size_t)padding[0],
                                            .right = (size_t)padding[1],
                                            .bottom = (size_t)padding[0],
                                            .left = (size_t)padding[1]};
  const struct nnp_size kernel_size = {.width = (size_t)weight_size[3],
                                       .height = (size_t)weight_size[2]};

  auto input_ = input.contiguous();
  auto run = [&]() -> nnp_status {
    return nnp_convolution_kernel_gradient(
        algorithm,
        batch_size,
        input_channels,
        output_channels,
        input_size,
        input_padding,
        kernel_size,
        (float*)input_.data_ptr(),
        (float*)gradOutput.data_ptr(),
        (float*)gradWeight.data_ptr(),
        workspace, // workspace_buffer
        &workspace_size, // workspace_size
        nnp_activation_identity,
        nullptr, // activation_parameters
        nnpack_threadpool(),
        nullptr // profile
    );
  };

  auto size_and_allocate_ws = [&]() {
    // Run a single pass to get the size of memory workspace buffer
    auto status = run();
    if (status != nnp_status_success) {
      throw std::runtime_error(
          "NNPACK SpatialConvolution_accGradWeight failed");
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

  return gradWeight;
}

std::tuple<Tensor, Tensor, Tensor> _nnpack_spatial_convolution_backward(
    const at::Tensor& input,
    const at::Tensor& grad_output,
    const at::Tensor& weight,
    IntArrayRef padding,
    std::array<bool, 3> output_mask) {
  Tensor grad_input, grad_weight, grad_bias;
  if (output_mask[0]) {
    grad_input = at::_nnpack_spatial_convolution_backward_input(
        input, grad_output, weight, padding);
  }
  if (output_mask[1]) {
    grad_weight = at::_nnpack_spatial_convolution_backward_weight(
        input, weight.sizes(), grad_output, padding);
  }
  if (output_mask[2]) {
    grad_bias = grad_output.contiguous()
                    .view({grad_output.size(0), grad_output.size(1), -1})
                    .sum(0)
                    .sum(1);
  }

  return std::tuple<Tensor, Tensor, Tensor>{grad_input, grad_weight, grad_bias};
}

} // namespace native
} // namespace at

#endif // AT_NNPACK_ENABLED
