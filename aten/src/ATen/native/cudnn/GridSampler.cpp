#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

Tensor cudnn_grid_sampler_forward(
    const Tensor& input_t, const Tensor& grid_t) {
  AT_ERROR("cudnn_grid_sampler_forward: ATen not compiled with cuDNN support");
}

std::tuple<Tensor, Tensor> cudnn_grid_sampler_backward(
    const Tensor& input_t, const Tensor& grid_t,
    const Tensor& grad_output_t) {
  AT_ERROR("cudnn_grid_sampler_backward: ATen not compiled with cuDNN support");
}

}}

#else // AT_CUDNN_ENABLED

#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cuda/Exceptions.h>

#include <ATen/TensorUtils.h>

// TODO: descriptor checking


namespace at { namespace native {

namespace {

void setSamplerDescriptor(SpatialTransformerDescriptor& desc, cudnnDataType_t dataType, const at::Tensor& tensor)
{
  int inputSize[4] = {0};
  for (int i = 0; i < tensor.dim(); ++i) {
    inputSize[i] = (int) tensor.size(i);
  }
  desc.set(dataType, 4, inputSize);
}

void checkGridSize(CheckedFrom c, TensorArg grid, TensorArg input)
{
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

}  // namespace

Tensor cudnn_grid_sampler_forward(
    const Tensor& input_t, const Tensor& grid_t)
{
  TensorArg input{ contiguousIfZeroInStrides(input_t), "input", 1 },
            grid{ grid_t.contiguous(), "grid", 2 };
  CheckedFrom c = "cudnn_grid_sampler_forward";
  checkAllSameGPU(c, {input, grid});
  checkAllSameType(c, {input, grid});
  checkGridSize(c, grid, input);
  checkDim(c, input, 4);

  auto output_t = at::empty({0}, input->options());
  output_t.resize_({input->size(0), input->size(1), grid->size(1), grid->size(2)});

  TensorDescriptor idesc{ *input };  // input descriptor
  TensorDescriptor odesc{ output_t };  // output descriptor
  SpatialTransformerDescriptor desc; // sampler descriptor

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*input);
  setSamplerDescriptor(desc, dataType, output_t);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  AT_CUDNN_CHECK(cudnnSpatialTfSamplerForward(
      handle, desc.desc(),
      &one, idesc.desc(), input->data_ptr(),
      grid->data_ptr(),
      &zero, odesc.desc(), output_t.data_ptr()
  ));

  return output_t;
}

// NB: CuDNN does not support output mask; you always get both
// gradients.
std::tuple<Tensor, Tensor> cudnn_grid_sampler_backward(
    const Tensor& input_t, const Tensor& grid_t,
    const Tensor& grad_output_t)
{
  TensorArg input{ contiguousIfZeroInStrides(input_t), "input", 1 },
            grid{ grid_t.contiguous(), "grid", 2 },
            grad_output{ contiguousIfZeroInStrides(grad_output_t), "grad_output", 3 };
  CheckedFrom c = "cudnn_grid_sampler_backward";
  checkAllSameGPU(c, {input, grad_output, grid});
  checkGridSize(c, grid, input);
  checkDim(c, input, 4);
  checkDim(c, grad_output, 4);

  auto grad_input_t = at::empty({0}, input->options());
  grad_input_t.resize_(input->sizes());
  auto grad_grid_t = at::empty({0}, grid->options());
  grad_grid_t.resize_(grid->sizes());

  TensorDescriptor idesc{ *input };  // input descriptor
  TensorDescriptor odesc{ *grad_output };  // grad_output descriptor
  TensorDescriptor gdesc{ grad_input_t };  // grad_input descriptor
  SpatialTransformerDescriptor desc; // sampler descriptor

  auto handle = getCudnnHandle();
  auto dataType = getCudnnDataType(*input);
  setSamplerDescriptor(desc, dataType, *grad_output);

  Constant one(dataType, 1);
  Constant zero(dataType, 0);
  AT_CUDNN_CHECK(cudnnSpatialTfSamplerBackward(
    handle, desc.desc(),
    &one, idesc.desc(), input->data_ptr(),
    &zero, gdesc.desc(), grad_input_t.data_ptr(),
    &one, odesc.desc(), grad_output->data_ptr(),
    // intruigingly, the outputs don't need descriptors
    grid->data_ptr(),
    &zero, grad_grid_t.data_ptr()
  ));

  return std::tuple<Tensor, Tensor>{ grad_input_t, grad_grid_t };
}

}}  // namespace at::cudnn

#endif
