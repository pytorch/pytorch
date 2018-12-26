#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/TensorUtils.h"

#include "ATen/cuda/CUDAContext.h"
#include "c10/util/Exception.h"

namespace at {
namespace native {

template <typename T>
__host__ __device__ __forceinline__ T ceilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__global__ void MaxUnpooling2d_forward_kernel(
    const int64_t numInputElements,
    const T* input,
    const int64_t* indices,
    const int64_t numBatch,
    const int64_t numChannels,
    const int64_t inputHeight,
    const int64_t inputWidth,
    const int64_t outputHeight,
    const int64_t outputWidth,
    T* output) {
  for (int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < numInputElements;
       linearIndex += blockDim.x * gridDim.x) {
    int c = (linearIndex / inputWidth / inputHeight) % numChannels;
    int n = linearIndex / inputWidth / inputHeight / numChannels;
    output += (n * numChannels + c) * outputHeight * outputWidth;
    int maxind = indices[linearIndex];
    output[maxind] = input[linearIndex];
  }
}
template <typename T>
__global__ void MaxUnpooling3d_forward_kernel(
    const T* input,
    const int64_t* indices,
    const int64_t batchSize,
    const int64_t inputSlices,
    const int64_t inputTime,
    const int64_t inputHeight,
    const int64_t inputWidth,
    const int64_t oT,
    const int64_t oW,
    const int64_t oH,
    const int64_t dT,
    const int64_t dW,
    const int64_t dH,
    const int64_t pT,
    const int64_t pW,
    const int64_t pH,
    const int64_t offsetZ,
    T* output) {
  int64_t iColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t iRow = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t iFrame = (blockIdx.z + offsetZ) % inputTime; // intput frame/time
  int64_t slice = (blockIdx.z + offsetZ) / inputTime; // intput slice/feature
  if (iRow < inputHeight && iColumn < inputWidth) {
    int64_t newIndex = slice * (inputTime * inputHeight * inputWidth) +
        iFrame * (inputHeight * inputWidth) + iRow * inputWidth + iColumn;
    T val = input[newIndex];
    int64_t index = indices[newIndex];
    output[slice * oT * oH * oW + index] = val;
  }
}

at::Tensor MaxUnpooling2d_forward_cuda(
    const Tensor& self,
    const Tensor& indices,
    IntList output_size) {
  AT_CHECK(
      self.ndimension() == 4,
      "Input to MaxUnpooling2d should be a NCHW Tensor",
      self.sizes());
  AT_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");

  auto output = at::zeros(
      {self.size(0), self.size(1), output_size[0], output_size[1]},
      self.options());
  MaxUnpooling2d_forward_out_cuda(output, self, indices, output_size);
  return output;
}

at::Tensor& MaxUnpooling2d_forward_out_cuda(
    Tensor& output,
    const Tensor& self,
    const Tensor& indices,
    IntList output_size) {
  TensorArg output_arg{ output, "output", 1 },
          self_arg{ self, "self", 2 },
          indices_arg{ indices, "indices", 3 };
  checkAllSameGPU("MaxUnpooling2d_forward_cuda_out", {output_arg, self_arg, indices_arg});

  for (int64_t i = 0; i < self.ndimension(); i++) {
    AT_CHECK(
        self.size(i) > 0,
        "input must be nonempty, but input has sizes: ",
        self.sizes());
  }
  AT_CHECK(
      self.ndimension() == 4,
      "Input to MaxUnpooling2d should be a NCHW Tensor",
      self.sizes());
  AT_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");

  auto numBatch = self.size(0);
  auto numChannels = self.size(1);
  auto inputHeight = self.size(2);
  auto inputWidth = self.size(3);
  auto outputHeight = output_size[0];
  auto outputWidth = output_size[1];

  dim3 block(512);
  dim3 grid((output.numel() + 512 - 1) / 512);

  AT_DISPATCH_ALL_TYPES_AND_HALF(
      self.type(), "MaxUnpooling2d_forward_kernel", ([&] {
        MaxUnpooling2d_forward_kernel<<<
            grid,
            block,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            self.numel(),
            self.data<scalar_t>(),
            indices.data<int64_t>(),
            numBatch,
            numChannels,
            inputHeight,
            inputWidth,
            outputHeight,
            outputWidth,
            output.data<scalar_t>());
      }));
  AT_CHECK(
      cudaGetLastError() == cudaSuccess,
      "RoiPooling2d_forward_kernel failed with error code ",
      cudaGetLastError());
  return output;
}

void MaxUnpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding,
    bool check_grad) {
  // is_empty check
  for (int64_t i = 0; i < input.ndimension(); i++) {
    AT_CHECK(
        input.size(i) > 0,
        "input must be nonempty, but input has sizes: ",
        input.sizes());
  }
  AT_CHECK(input.sizes() == indices.sizes());
  AT_CHECK(
      stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
      "stride should be never greater than zero, but got stride: ",
      stride);

  int dimw = 4;
  int dimh = 3;
  int dimt = 2;
  int dimn = 1;

  int nslices = input.size(dimn);
  if (check_grad) {
    if (output_size[0] != gradOutput.size(dimt) ||
        output_size[1] != gradOutput.size(dimh) ||
        output_size[2] != gradOutput.size(dimw)) {
      AT_ERROR(
          "Inconsistent gradOutput size. output_size[1]= %d, output_size[1] = %d, output_size[2] = %d, gradOutput: %dx%dx%d",
          output_size[0],
          output_size[1],
          output_size[2],
          gradOutput[0],
          gradOutput[1],
          gradOutput[2]);
    }
    AT_CHECK(gradOutput.ndimension() == 5 && gradOutput.size(dimn) == nslices);
  }
}

at::Tensor MaxUnpooling3d_forward_cuda(
    const Tensor& self,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding) {
      AT_CHECK(
          self.ndimension() == 5,
          "Input to MaxUnpooling2d should be a NCDHW Tensor",
          self.sizes());
      AT_CHECK(
          output_size.size() == 3,
          "There should be exactly three elements (depth, height, width) in output_size");

  auto output = at::zeros(
      {self.size(1),
       self.size(2),
       output_size[0],
       output_size[1],
       output_size[2]},
      self.options());
  MaxUnpooling3d_forward_out_cuda(output, self, indices, output_size, stride, padding);
  return output;
}

at::Tensor& MaxUnpooling3d_forward_out_cuda(
    Tensor& output,
    const Tensor& self,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding) {
  AT_CHECK(
      self.ndimension() == 5,
      "Input to MaxUnpooling2d should be a NCDHW Tensor",
      self.sizes());
  AT_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size");

  auto batchSize = self.size(0);
  auto inputSlices = self.size(1);
  auto inputTime = self.size(2);
  auto inputHeight = self.size(3);
  auto inputWidth = self.size(4);
  auto outputTime = output_size[0];
  auto outputHeight = output_size[1];
  auto outputWidth = output_size[2];

  auto dT = stride[0];
  auto dH = stride[1];
  auto dW = stride[2];

  auto padT = padding[0];
  auto padH = padding[1];
  auto padW = padding[2];

  MaxUnpooling3d_shape_check(
      self, at::empty({}), indices, output_size, stride, padding, false);
  TensorArg output_arg{ output, "output", 1 },
          self_arg{ self, "self", 2 },
          indices_arg{ indices, "indices", 3 };
  checkAllSameGPU("MaxUnpooling3d_forward_out_cuda", {output_arg, self_arg, indices_arg});

  AT_CHECK(output.is_contiguous(), "output must be contiguous");

  int totalZ = inputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);

  while (totalZ > 0) {
    dim3 grid(
        ceilDiv(inputWidth, static_cast<int64_t>(block.x)),
        ceilDiv(inputHeight, static_cast<int64_t>(block.y)),
        totalZ > 65535 ? 65535 : totalZ);
    AT_DISPATCH_ALL_TYPES_AND_HALF(
        self.type(), "MaxUnpooling3d_forward_kernel", ([&] {
          MaxUnpooling3d_forward_kernel<<<
              grid,
              block,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              self.contiguous().data<scalar_t>(),
              indices.contiguous().data<int64_t>(),
              batchSize,
              inputSlices,
              inputTime,
              inputHeight,
              inputWidth,
              outputTime,
              outputHeight,
              outputWidth,
              dT,
              dH,
              dW,
              padT,
              padH,
              padW,
              offsetZ,
              output.data<scalar_t>());
        }));
    AT_CHECK(
        cudaGetLastError() == cudaSuccess,
        "RoiPooling3d_forward_kernel failed with error code ",
        cudaGetLastError());
    totalZ -= 65535;
    offsetZ += 65535;
  }
  return output;
}

} // namespace native
} // namespace at
