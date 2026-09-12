#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAConfig.h>
#include <ATen/native/GridSamplerUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cudnn_grid_sampler_backward_native.h>
#include <ATen/ops/cudnn_grid_sampler_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/grid_sampler_2d_backward.h>
#endif

#if !AT_CUDNN_ENABLED()

namespace at {
namespace native {

// See Note [ATen preprocessor philosophy]

Tensor cudnn_grid_sampler_forward(const Tensor& input_t, const Tensor& grid_t) {
  TORCH_CHECK(
      false,
      "cudnn_grid_sampler_forward: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor> cudnn_grid_sampler_backward(
    const Tensor& input_t,
    const Tensor& grid_t,
    const Tensor& grad_output_t) {
  TORCH_CHECK(
      false,
      "cudnn_grid_sampler_backward: ATen not compiled with cuDNN support");
}

} // namespace native
} // namespace at

#else // AT_CUDNN_ENABLED

#include <ATen/cuda/Exceptions.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <array>

#include <ATen/TensorUtils.h>
#include <c10/util/irange.h>

// TODO: descriptor checking

namespace at::native {

namespace {

void setSamplerDescriptor(
    SpatialTransformerDescriptor& desc,
    cudnnDataType_t dataType,
    const at::Tensor& tensor) {
  std::array<int, 4> inputSize{0};
  for (const auto i : c10::irange(tensor.dim())) {
    inputSize[i] = static_cast<int>(tensor.size(i));
  }
  desc.set(dataType, 4, inputSize.data());
}

void checkGridSize(CheckedFrom c, TensorArg grid, TensorArg input) {
  // assert size of grid is n*h*w*2
  // FYI: grid is between [-1, 1], where -1 left most pixel,
  // 1 represents right most pixel (and hence 0 is the center pixel)
  // if grid has values >1 or <-1, those values are ignored
  checkContiguous(c, grid);
  checkDim(c, grid, 4);
  // TODO: Maybe more user friendly to report where the expected size
  // came from
  checkSize(c, grid, 0, input->size(0));
  checkSize(c, grid, 3, 2);
}

} // namespace

Tensor cudnn_grid_sampler_forward(const Tensor& input_t, const Tensor& grid_t) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input_t, grid_t);
  TORCH_CHECK(
      cond_cudnn_grid_sampler(input_t, grid_t),
      "Invalid arguments to cudnn_grid_sampler_forward");

  auto input_contig = contiguousIfZeroInStrides(input_t);
  auto grid_contig = grid_t.contiguous();
  TensorArg input{input_contig, "input", 1}, grid{grid_contig, "grid", 2};
  CheckedFrom c = "cudnn_grid_sampler_forward";
  checkAllSameGPU(c, {input, grid});
  checkAllSameType(c, {input, grid});
  checkGridSize(c, grid, input);
  checkDim(c, input, 4);

  auto output_t = at::empty({0}, input->options());
  output_t.resize_(
      {input->size(0), input->size(1), grid->size(1), grid->size(2)});

  TensorDescriptor idesc{*input}; // input descriptor
  TensorDescriptor odesc{output_t}; // output descriptor
  SpatialTransformerDescriptor desc; // sampler descriptor

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*input);
  setSamplerDescriptor(desc, dataType, output_t);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  AT_CUDNN_CHECK(cudnnSpatialTfSamplerForward(
      handle,
      desc.desc(),
      &one,
      idesc.desc(),
      input->const_data_ptr(),
      grid->const_data_ptr(),
      &zero,
      odesc.desc(),
      output_t.data_ptr()));

  return output_t;
}

// This operator has no output mask and always returns both gradients.
std::tuple<Tensor, Tensor> cudnn_grid_sampler_backward(
    const Tensor& input_t,
    const Tensor& grid_t,
    const Tensor& grad_output_t) {
  // See NOTE [ grid_sampler Native Functions ].
  // Add checks here in case this is called instead of grid_sampler.
  check_grid_sampler_common(input_t, grid_t);
  TORCH_CHECK(
      cond_cudnn_grid_sampler(input_t, grid_t),
      "Invalid arguments to cudnn_grid_sampler_backward");
  TensorArg input{input_t, "input", 1}, grid{grid_t, "grid", 2},
      grad_output{grad_output_t, "grad_output", 3};
  CheckedFrom c = "cudnn_grid_sampler_backward";
  checkAllSameGPU(c, {input, grad_output, grid});
  checkDim(c, input, 4);
  checkDim(c, grad_output, 4);

  // Keep cuDNN forward semantics, but use native deterministic accumulation in
  // backward. The native implementation accepts the original tensor strides.
  return at::grid_sampler_2d_backward(
      grad_output_t, input_t, grid_t, 0, 0, true, {true, true});
}

} // namespace at::native

#endif
