#include <ATen/ATen.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <c10/util/Exception.h>
#include <THC/THCDeviceUtils.cuh>
#include <ATen/native/SquareForm.h>


namespace at {
namespace native {
namespace {
void getSquareformCudaConfig(dim3& block, dim3& grid, int64_t totalElements)
{
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  block = cuda::getApplyBlock();
  AT_CHECK(cuda::getApplyGrid(totalElements, grid, curDevice),
      "Could not get squareform cuda config");
}
} // anonymous namespace

__host__ __device__ __forceinline__ int64_t getVectorIndex(int64_t n,
    int64_t i, int64_t j)
{
  // by definition of squareform, element of (i, j) square matrix with size n
  // is mapped to vector element at (n choose 2)-((n-i) choose 2)+(j-i- 1)
  return n * (n - 1) / 2 - (n - i) * (n - i - 1) / 2 + j - i - 1;
}

template <typename scalar_t>
__global__ void squareform_kernel(
    const scalar_t* input_data,
    scalar_t* output_data,
    int64_t square_size,
    int64_t input_dim,
    bool checks,
    scalar_t* check_data)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (input_dim == 1)
  {
    // from 1D to 2D
    for (; idx < square_size * square_size; idx += blockDim.x * gridDim.x)
    {
      int64_t i = idx / square_size, j = idx % square_size;
      scalar_t v(0);
      if (j > i)
      {
        int64_t vidx = getVectorIndex(square_size, i, j);
        v = input_data[vidx];
      }
      else if (j < i)
      {
        int64_t vidx = getVectorIndex(square_size, j, i);
        v = input_data[vidx];
      }
      *(output_data + square_size * i + j) = v;
    }
  }
  else
  {
    // from 2D to 1D
    for (; idx < square_size * square_size; idx += blockDim.x * gridDim.x)
    {
      int64_t i = idx / square_size, j = idx % square_size;
      if (j > i)
      {
        int64_t vidx = getVectorIndex(square_size, i, j);
        output_data[vidx] = *(input_data + square_size * i + j);
        if (checks)
        {
          if (*(input_data + square_size * i + j) != 
              *(input_data + square_size * j + i))
          {
            *check_data = 1;
          }
        }
      }
      else if (j == i)
      {
        if (checks)
        {
          if (*(input_data + square_size * i + j) != 0)
          {
            *check_data = 1;
          }
        }
      }
    }
  }
}

template <typename scalar_t>
__global__ void squareform_backward_kernel(
    const scalar_t* grad_input,
    scalar_t* grad_output,
    int64_t square_size,
    int64_t input_dim)
{
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (input_dim == 1)
  {
    /// backward for 2D to 1D
    for (; idx < square_size * square_size; idx += blockDim.x * gridDim.x)
    {
      int64_t i = idx / square_size, j = idx % square_size;
      if (j > i)
      {
        int64_t vidx = getVectorIndex(square_size, i, j);
        *(grad_output + square_size * i + j) = grad_input[vidx] / 2.0;
        *(grad_output + square_size * j + i) = grad_input[vidx] / 2.0;
      }
    }
  }
  else
  {
    /// backward for 1D to 2D
    for (; idx < square_size * square_size; idx += blockDim.x * gridDim.x)
    {
      int64_t i = idx / square_size, j = idx % square_size;
      if (j > i)
      {
        int64_t vidx = getVectorIndex(square_size, i, j);
        auto v = *(grad_input + square_size * i + j);
        atomicAdd(&grad_output[vidx], v);
      }
      else if (i > j)
      {
        int64_t vidx = getVectorIndex(square_size, j, i);
        auto v = *(grad_input + square_size * i + j);
        atomicAdd(&grad_output[vidx], v);
      }
    }
  }
}

Tensor squareform_cuda(const Tensor& self, bool checks)
{
  AT_CHECK((self.dim() == 1 || self.dim() == 2),
      "Expected vector-form distance or square-form distance as input",
      " but the dimension of input is ", self.dim());
  auto input = self.contiguous();
  Tensor out_tensor;
  int64_t vector_size = 0;
  int64_t d = 0;
  if (input.dim() == 1)
  {
    vector_size = input.size(0);
    d = getSquareSize(vector_size);
    out_tensor = at::empty({d, d}, input.options());
  }
  else
  {
    AT_CHECK(input.size(0) == input.size(1),
        "Expected square matrix as input, size(0) ",
        input.size(0), " != size(1) ", input.size(1));
    d = input.size(0);
    vector_size = d * (d - 1) / 2;
    out_tensor = at::empty({vector_size}, input.options());
  }
  dim3 gridSize, blockSize;
  getSquareformCudaConfig(blockSize, gridSize, d);
  Tensor check_tensor = at::zeros({1}, input.options());
  AT_DISPATCH_ALL_TYPES_AND_HALF(input.type(), "squareform", [&] {
    squareform_kernel<scalar_t>
    <<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data<scalar_t>(), out_tensor.data<scalar_t>(), d, input.dim(),
      checks, check_tensor.data<scalar_t>());
  });
  if (checks)
  {
    int64_t r = check_tensor[0].item<int64_t>();
    AT_CHECK(r == 0, " Input is not symmetric or has non-zero diagonal");
  }
  AT_CUDA_CHECK(cudaGetLastError());
  return out_tensor;
}

Tensor squareform_backward_cuda(const Tensor& grad, const Tensor& self)
{
  AT_CHECK((grad.dim() == 1 || grad.dim() == 2),
      "Expected the input dim is 1 or 2, but got ", grad.dim());
  auto input = grad.contiguous();
  Tensor out_tensor;
  int64_t vector_size = 0;
  int64_t d = 0;
  if (input.dim() == 1)
  {
    vector_size = input.size(0);
    d = getSquareSize(vector_size);
    out_tensor = at::zeros({d, d}, input.options());
  }
  else
  {
    AT_CHECK(input.size(0) == input.size(1),
        "expected square-form distance as input, size(0) ",
        input.size(0), " != size(1) ", input.size(1));
    d = input.size(0);
    vector_size = d * (d - 1) / 2;
    out_tensor = at::zeros({vector_size}, input.options());
  }
  dim3 gridSize, blockSize;
  getSquareformCudaConfig(blockSize, gridSize, d);
  AT_DISPATCH_ALL_TYPES_AND_HALF(input.type(), "squareform", [&] {
    squareform_backward_kernel<scalar_t>
    <<<gridSize, blockSize, 0, at::cuda::getCurrentCUDAStream()>>>(
      input.data<scalar_t>(), out_tensor.data<scalar_t>(), d, input.dim());
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return out_tensor;
}

} // at::native
} // at
