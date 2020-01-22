#include <ATen/ATen.h>
#include <ATen/Config.h>

#if !AT_NNPACK_ENABLED()

namespace at {
namespace native {

at::Tensor _nnpack_spatial_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride) {
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

#include "caffe2/utils/threadpool/ThreadPoolMobile.h"
#include <ATen/native/ConvUtils.h>

namespace at {
namespace native {

static bool init_nnpack() {
  static std::once_flag once_;
  static bool nnpack_successfully_initialized_ = false;

  std::call_once(once_, []() {
    const nnp_status nnpack_status = nnp_initialize();
    nnpack_successfully_initialized_ = (nnp_status_success == nnpack_status);

    if (nnpack_status != nnp_status_success) {
      if (nnpack_status == nnp_status_out_of_memory) {
        LOG(WARNING) << "Could not initialize NNPACK! Reason: Out of memory.";
      } else if (nnpack_status == nnp_status_unsupported_hardware) {
        LOG(WARNING) << "Could not initialize NNPACK! Reason: Unsupported hardware.";
      } else {
        LOG(WARNING) << "Could not initialize NNPACK! Reason: Unknown error!";
      }
    }
  });

  return nnpack_successfully_initialized_;
}

static pthreadpool_t nnpack_threadpool() {
  // Try initializing a threadpool for NNPACK's use.  If we fail to
  // successfully initialize an implementation, return nullptr which will
  // instruct NNPACK to run single threaded.

#ifdef C10_MOBILE
  // If building for mobile, use Caffe 2's mobile-friendly threadpool.
  return caffe2::mobile_pthreadpool();
#else
  // Otherwise, try using pthreadpool if we manage to initialize it successfully.
  static pthreadpool_t nnpack_threadpool_ = nullptr;
  static bool called_nnpack_threadpool_ = false;

  if (!called_nnpack_threadpool_) {
    called_nnpack_threadpool_ = true;

#ifdef INTRA_OP_PARALLEL
    const uint32_t threads = at::get_num_threads();
#else
    const uint32_t threads = std::thread::hardware_concurrency();
#endif

    nnpack_threadpool_ = pthreadpool_create(threads);
    if (!nnpack_threadpool_) {
      LOG(WARNING) << "Failed to initialize pthreadpool! Running NNPACK in single-threaded mode.";
    }
  }

  return nnpack_threadpool_;
#endif
}

bool _nnpack_available() {
  return init_nnpack();
}

// Make thread_local for safety in cases where we have multiple threads running
// Convs at once
static thread_local void* workspace = nullptr;
static thread_local size_t workspace_size = 0;

static inline void deallocate_workspace() {
  if (workspace) {
    std::free(workspace);
    workspace = nullptr;
  }
}

static inline void allocate_workspace() {
  if (workspace) {
    deallocate_workspace();
  }

  // NNPack has alignment requirements
  constexpr size_t nnpack_memory_alignment_boundary = 64;

  // Won't work on Windows, but NNPACK doesn't support Windows either
  posix_memalign(&workspace, nnpack_memory_alignment_boundary, workspace_size);
}

Tensor _nnpack_spatial_convolution(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& bias,
    const IntArrayRef padding,
    const IntArrayRef stride) {
  at::Tensor output = at::empty(
      conv_output_size(input.sizes(), weight.sizes(), padding, stride),
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

  // All Tensors must be float Tensors
  if (input.device().type() != kCPU || input.scalar_type() != kFloat ||
      weight.device().type() != kCPU || weight.scalar_type() != kFloat ||
      output.device().type() != kCPU || output.scalar_type() != kFloat ||
      (bias.defined() && (bias.device().type() != kCPU || bias.scalar_type() != kFloat))) {
    throw std::runtime_error(
        "Mismatched Tensor types in NNPack convolutionOutput");
  }

  const auto algorithm = nnp_convolution_algorithm_auto;
  const size_t input_channels = input.size(1);
  const size_t output_channels = weight.size(0);
  const struct nnp_size input_size = {
      .width = (size_t)input.size(3),
      .height = (size_t)input.size(2),
  };
  const struct nnp_padding input_padding = {
      .top = (size_t)padding[0],
      .right = (size_t)padding[1],
      .bottom = (size_t)padding[0],
      .left = (size_t)padding[1],
  };
  const struct nnp_size kernel_size = {
      .width = (size_t)weight.size(3),
      .height = (size_t)weight.size(2),
  };
  const struct nnp_size output_size = {
      .width = (size_t)output.size(3),
      .height = (size_t)output.size(2),
  };
  const nnp_size output_subsample = {
      .width = stride[1],
      .height = stride[0],
  };

  const auto input_ = input.contiguous();
  // If we don't have a defined bias Tensor, we need to create one filled with zeroes
  const auto bias_ = bias.defined() ? bias : at::zeros({weight.size(0)}, input.options());

  const auto compute = [&](const size_t batch_size) -> nnp_status {
    if ((batch_size == 1) || (output_subsample.width != 1) || (output_subsample.height != 1)) {
      const size_t input_size_per_batch = input_channels * input_size.width * input_size.height;
      const size_t output_size_per_batch = output_channels * output_size.width * output_size.height;

      for (size_t batch = 0u; batch < batch_size; ++batch) {
        const nnp_status status = nnp_convolution_inference(
            algorithm,
            nnp_convolution_transform_strategy_compute,
            input_channels,
            output_channels,
            input_size,
            input_padding,
            kernel_size,
            output_subsample,
            input_.data_ptr<float>() + batch * input_size_per_batch,
            weight.data_ptr<float>(),
            bias_.data_ptr<float>(),
            output.data_ptr<float>() + batch * output_size_per_batch,
            workspace,
            &workspace_size,
            nnp_activation_identity,
            nullptr,
            nnpack_threadpool(),
            nullptr );

        if (nnp_status_success != status) {
          return status;
        }
      }

      return nnp_status_success;
    }
    else {
      return nnp_convolution_output(
        algorithm,
        batch_size,
        input_channels,
        output_channels,
        input_size,
        input_padding,
        kernel_size,
        input_.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_.data_ptr<float>(),
        output.data_ptr<float>(),
        workspace,
        &workspace_size,
        nnp_activation_identity,
        nullptr,
        nnpack_threadpool(),
        nullptr );
    }
  };

  const size_t batch_size = input.size(0);

  auto size_and_allocate_ws = [&]() {
    // Run a single pass to get the size of memory workspace buffer
    const auto status = compute(batch_size);
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
  auto status = compute(batch_size);

  if (status == nnp_status_insufficient_buffer) {
    // Need to reallocate the workspace
    deallocate_workspace();
    size_and_allocate_ws();

    // Try one more time
    status = compute(batch_size);
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
        gradOutput.data_ptr<float>(),
        weight.data_ptr<float>(),
        gradInput.data_ptr<float>(),
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
        input_.data_ptr<float>(),
        gradOutput.data_ptr<float>(),
        gradWeight.data_ptr<float>(),
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
