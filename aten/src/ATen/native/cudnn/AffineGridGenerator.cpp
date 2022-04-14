#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <ATen/cuda/CUDAConfig.h>

#if !AT_CUDNN_ENABLED()

namespace at { namespace native {

// See Note [ATen preprocessor philosophy]

Tensor cudnn_affine_grid_generator_forward(
    const Tensor& theta,
    int64_t N, int64_t C, int64_t H, int64_t W) {
  AT_ERROR("cudnn_affine_grid_generator_forward: ATen not compiled with cuDNN support");
}

Tensor cudnn_affine_grid_generator_backward(
    const Tensor& grad_theta,
    int64_t N, int64_t C, int64_t H, int64_t W) {
  AT_ERROR("cudnn_affine_grid_generator_backward: ATen not compiled with cuDNN support");
}

}}

#else // AT_CUDNN_ENABLED()

#include <ATen/cudnn/cudnn-wrapper.h>
#include <ATen/cudnn/Handle.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/cudnn/Types.h>
#include <ATen/cudnn/Utils.h>
#include <ATen/cuda/Exceptions.h>

#include <ATen/TensorUtils.h>

namespace at { namespace native {

namespace {

void setSamplerDescriptor(SpatialTransformerDescriptor& desc,
                          cudnnDataType_t dataType,
                          int N, int C, int H, int W)
{
  int inputSize[4] = {N, C, H, W};
  desc.set(dataType, 4, inputSize);
}

}  // namespace

Tensor cudnn_affine_grid_generator_forward(
    const Tensor& theta_t,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
  auto theta_t_contig = theta_t.contiguous();
  TensorArg theta{ theta_t_contig, "theta", 1 };
  CheckedFrom c = "cudnn_affine_grid_generator_forward";
  checkContiguous(c, theta);
  checkSize(c, theta, {N, 2, 3});

  auto grid_t = at::empty({0}, theta->options());
  grid_t.resize_({N, H, W, 2});

  auto dataType = getCudnnDataType(*theta);
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W);
  AT_CUDNN_CHECK(cudnnSpatialTfGridGeneratorForward(getCudnnHandle(), desc.desc(),
                                                 theta->data_ptr(),
                                                 grid_t.data_ptr()));
  return grid_t;
}

Tensor cudnn_affine_grid_generator_backward(
    const Tensor& grad_grid_t,
    int64_t N, int64_t C, int64_t H, int64_t W)
{
  auto grad_grid_contig = grad_grid_t.contiguous();
  TensorArg grad_grid{ grad_grid_contig, "grad_grid", 1 };
  CheckedFrom c = "cudnn_affine_grid_generator_backward";
  checkContiguous(c, grad_grid);
  checkSize(c, grad_grid, {N, H, W, 2});

  auto grad_theta_t = at::empty({0}, grad_grid->options());
  grad_theta_t.resize_({N, 2, 3});

  auto dataType = getCudnnDataType(grad_theta_t);
  SpatialTransformerDescriptor desc;
  setSamplerDescriptor(desc, dataType, N, C, H, W);
  AT_CUDNN_CHECK(cudnnSpatialTfGridGeneratorBackward(getCudnnHandle(), desc.desc(),
                                                  grad_grid->data_ptr(),
                                                  grad_theta_t.data_ptr()));
  return grad_theta_t;
}

}}  // namespace at::native

#endif // AT_CUDNN_ENABLED()
