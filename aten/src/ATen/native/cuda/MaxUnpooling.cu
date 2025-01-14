#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <c10/util/Exception.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/max_unpool2d_native.h>
#include <ATen/ops/max_unpool3d_native.h>

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#endif

namespace at::native {

using namespace at::cuda::detail;

template <typename T>
__host__ __device__ __forceinline__ T ceilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename T>
__global__ void max_unpooling2d_forward_kernel(
    const int64_t numInputElements,
    const T* input,
    const int64_t* indices,
    const int64_t numChannels,
    const int64_t inputHeight,
    const int64_t inputWidth,
    const int64_t outputHeight,
    const int64_t outputWidth,
    T* output) {
  int64_t outputImageSize = outputHeight * outputWidth;
  CUDA_KERNEL_LOOP(linearIndex, numInputElements) {
    int c = (linearIndex / inputWidth / inputHeight) % numChannels;
    int n = linearIndex / inputWidth / inputHeight / numChannels;
    output += (n * numChannels + c) * outputHeight * outputWidth;
    int maxind = indices[linearIndex];
    CUDA_KERNEL_ASSERT(maxind >= 0 && maxind < outputImageSize);
    output[maxind] = input[linearIndex];
  }
}

template <typename T>
__global__ void max_unpooling3d_forward_kernel(
    PackedTensorAccessor64<const T, 4> input,
    PackedTensorAccessor64<const int64_t, 4> indices,
    T* output,
    const int64_t oT,
    const int64_t oH,
    const int64_t oW,
    const int64_t offsetZ) {
  int64_t iColumn = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t iRow = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t iFrame = (blockIdx.z + offsetZ) % input.size(1); // input frame/time
  int64_t slice = (blockIdx.z + offsetZ) / input.size(1); // input slice/feature
  int64_t outputImageSize = oT * oH * oW;
  if (iRow < input.size(2) && iColumn < input.size(3)) {
    const T val = input[slice][iFrame][iRow][iColumn];
    const int64_t index = indices[slice][iFrame][iRow][iColumn];
    CUDA_KERNEL_ASSERT(index >= 0 && index < outputImageSize);
    output[slice * oT * oH * oW + index] = val;
  }
}

template <typename T>
__global__ void max_unpooling2d_backward_kernel(
    const int64_t numInputElements,
    const T* input,
    const int64_t* indices,
    const int64_t numChannels,
    const int64_t inputHeight,
    const int64_t inputWidth,
    const int64_t outputHeight,
    const int64_t outputWidth,
    T* output) {
  CUDA_KERNEL_LOOP(linearIndex, numInputElements) {
    int c = (linearIndex / inputWidth / inputHeight) % numChannels;
    int n = linearIndex / inputWidth / inputHeight / numChannels;
    input += (n * numChannels + c) * outputHeight * outputWidth;
    int maxind = indices[linearIndex];
    output[linearIndex] = input[maxind];
  }
}

template <typename T>
__global__ void max_unpooling3d_backward_kernel(
    const T* gradOutputData,
    int64_t oT,
    int64_t oH,
    int64_t oW,
    PackedTensorAccessor64<int64_t, 4> indices,
    PackedTensorAccessor64<T, 4> gradInput,
    int offsetZ) {
  int iColumn = blockIdx.x * blockDim.x + threadIdx.x;
  int iRow = blockIdx.y * blockDim.y + threadIdx.y;
  int iFrame = (blockIdx.z + offsetZ) % gradInput.size(1); // output frame/time
  int slice =
      (blockIdx.z + offsetZ) / gradInput.size(1); // output slice/feature

  if (iRow < gradInput.size(2) && iColumn < gradInput.size(3)) {
    int64_t index = indices[slice][iFrame][iRow][iColumn];
    T grad_val = gradOutputData[slice * oT * oH * oW + index];
    gradInput[slice][iFrame][iRow][iColumn] = grad_val;
  }
}

Tensor& max_unpooling2d_forward_out_cuda(const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    Tensor& output) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic with duplicate indices
  at::globalContext().alertNotDeterministic("max_unpooling2d_forward_out");

  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64 but got: ", indices_.scalar_type());
  auto oheight = output_size[0];
  auto owidth = output_size[1];

  TensorArg output_arg{output, "output", 1}, self_arg{self_, "self_", 2},
      indices_arg{indices_, "indices_", 3};
  checkAllSameGPU(
      "max_unpooling2d_forward_out_cuda", {output_arg, self_arg, indices_arg});

  for (int64_t i = 1; i < self_.ndimension(); ++i) {
    TORCH_CHECK(self_.size(i) > 0, "max_unpooling2d_forward_out_cuda(): ",
                "Expected input to have non-zero size for non-batch dimensions, but got ",
                self_.sizes(), " with dimension ", i , " being empty.");
  }

  TORCH_CHECK(
      (self_.ndimension() == 3 || self_.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor, but got tensor with dimension: ", self_.ndimension());
  TORCH_CHECK(
      self_.sizes() == indices_.sizes(),
      "Expected shape of indices to be: ", self_.sizes(), " but got: ", indices_.sizes());
  TORCH_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size, but got ", output_size.size(), " elements.");

  int64_t dimw = 2;
  int64_t dimh = 1;
  int64_t numBatch = 1;

  int64_t numChannels;
  int64_t inputHeight;
  int64_t inputWidth;

  auto self = self_.contiguous();
  auto indices = indices_.contiguous();

  if (self.ndimension() == 4) {
    numBatch = self.size(0);
    dimw++;
    dimh++;
  }
  numChannels = self.size(dimh - 1);
  inputHeight = self.size(dimh);
  inputWidth = self.size(dimw);

  output.resize_({numBatch, numChannels, oheight, owidth});

  output.zero_();

  auto count = self.numel();
  if (count != 0) {
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16,
        self.scalar_type(), "max_unpooling2d_forward_kernel", ([&] {
          max_unpooling2d_forward_kernel<<<
              GET_BLOCKS(count),
              CUDA_NUM_THREADS,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              self.numel(),
              self.const_data_ptr<scalar_t>(),
              indices.const_data_ptr<int64_t>(),
              numChannels,
              inputHeight,
              inputWidth,
              oheight,
              owidth,
              output.mutable_data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }));
  }
  if (self.ndimension() == 3) {
    output.resize_({numChannels, oheight, owidth});
  }
  return output;
}

Tensor max_unpooling2d_forward_cuda(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  at::native::max_unpooling2d_forward_out_cuda(self, indices, output_size, output);
  return output;
}

static void max_unpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    const char *fn_name) {
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64 but got: ", indices.scalar_type());
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "Input to max_unpooling3d should be a 4d or 5d Tensor, but got a tensor with dim ", input.ndimension());
  TORCH_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size, but got ", output_size.size(), " elements.");
  TORCH_CHECK(
      stride.size() == 3,
      "There should be exactly three elements (depth, height, width) in stride, but got: ", stride.size(), " elements.");
  TORCH_CHECK(
      padding.size() == 3,
      "There should be exactly three elements (depth, height, width) in padding, but got: ", padding.size(), " elements.");
  TORCH_CHECK(
      input.sizes() == indices.sizes(),
      "Expected shape of indices to be: ", input.sizes(), " but got: ", indices.sizes());

  for (int64_t i = 1; i < input.ndimension(); ++i) {
    TORCH_CHECK(input.size(i) > 0, fn_name,
                ": Expected input to have non-zero size for non-batch dimensions, but got ",
                input.sizes(), " with dimension ", i , " being empty.");
  }

  TORCH_CHECK(
      stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
      "strides should be greater than zero, but got stride: ",
      stride);

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;

  if (input.ndimension() == 5) {
    dimw++;
    dimh++;
    dimt++;
    dimn++;
  }

  int nslices = input.size(dimn);

  if (gradOutput.defined()) {
    if (oT != gradOutput.size(dimt) || oH != gradOutput.size(dimh) ||
        oW != gradOutput.size(dimw)) {
      TORCH_CHECK(false,
          "Inconsistent gradOutput size. oT= ",
          oT,
          ", oH= ",
          oH,
          ", oW= ",
          oW,
          ". gradOutput: ",
          gradOutput.size(dimt),
          "x",
          gradOutput.size(dimh),
          "x",
          gradOutput.size(dimw));
    }
    TORCH_CHECK(
        gradOutput.ndimension() == input.ndimension() &&
            gradOutput.size(dimn) == nslices,
        "gradOutput and input Tensors should have same number of dimensions and also the same number of channels/slices");
  }
}

Tensor& max_unpooling3d_forward_out_cuda(const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic with duplicate indices
  at::globalContext().alertNotDeterministic("max_unpooling3d_forward_out");

  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  max_unpooling3d_shape_check(
    self_, Tensor(), indices_, output_size, stride, padding, "max_unpooling3d_forward_out_cuda()");

  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];

  TensorArg output_arg{output, "output", 1}, self_arg{self_, "self_", 2},
      indices_arg{indices_, "indices_", 3};
  checkAllSameGPU(
      "max_unpooling3d_forward_out_cuda", {output_arg, self_arg, indices_arg});

  auto self = self_.contiguous();
  auto indices = indices_.contiguous();

  int64_t batchSize;
  int64_t inputSlices;
  int64_t inputTime;
  int64_t inputHeight;
  int64_t inputWidth;

  if (self.ndimension() == 4) {
    batchSize = 1;
    inputSlices = self.size(0);
    inputTime = self.size(1);
    inputHeight = self.size(2);
    inputWidth = self.size(3);
    output.resize_({inputSlices, oT, oH, oW});
  } else {
    batchSize = self.size(0);
    inputSlices = self.size(1);
    inputTime = self.size(2);
    inputHeight = self.size(3);
    inputWidth = self.size(4);
    output.resize_({batchSize, inputSlices, oT, oH, oW});
  }

  output.zero_();

  // Collapse batch and feature dimensions if needed
  if (self.ndimension() == 5) {
    self = self.reshape({self.size(0) * self.size(1),
                         self.size(2),
                         self.size(3),
                         self.size(4)});
    indices = indices.reshape({indices.size(0) * indices.size(1),
                               indices.size(2),
                               indices.size(3),
                               indices.size(4)});
  }

  if (self.numel() == 0) {
    return output;
  }

  int totalZ = inputTime * inputSlices * batchSize;
  int offsetZ = 0;
  dim3 block(32, 8);

  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16,
      self.scalar_type(), "max_unpooling3d_forward_kernel", ([&] {
        while (totalZ > 0) {
          dim3 grid(
              ceilDiv(inputWidth, static_cast<int64_t>(block.x)),
              ceilDiv(inputHeight, static_cast<int64_t>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);
          max_unpooling3d_forward_kernel<<<
              grid,
              block,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              self.packed_accessor64<const scalar_t, 4>(),
              indices.packed_accessor64<const int64_t, 4>(),
              output.mutable_data_ptr<scalar_t>(),
              oT,
              oH,
              oW,
              offsetZ);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          totalZ -= 65535;
          offsetZ += 65535;
        }
      }));
  return output;
}

Tensor max_unpooling3d_forward_cuda(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto output = at::empty({0}, self.options());
  at::native::max_unpooling3d_forward_out_cuda(
      self, indices, output_size, stride, padding, output);
  return output;
}

at::Tensor& max_unpooling2d_backward_out_cuda(const Tensor& grad_output_,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    Tensor& grad_input) {
  int64_t oheight = output_size[0];
  int64_t owidth = output_size[1];
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64 but got type: ", indices_.scalar_type());
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2},
      self_arg{self_, "self_", 3}, indices_arg{indices_, "indices_", 4};
  checkAllSameGPU(
      "max_unpooling2d_backward_out_cuda",
      {grad_input_arg, grad_output_arg, self_arg, indices_arg});

  TORCH_CHECK(
      (self_.ndimension() == 3 || self_.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor, instead got: ",
      self_);

  TORCH_CHECK(
      self_.sizes() == indices_.sizes(),
      "Expected shape of indices to be: ", self_.sizes(), " but got: ", indices_.sizes());

  TORCH_CHECK(output_size.size() == 2, "output_size must have two elements, got size: ", output_size.size());

  int64_t nInputCols, nInputRows, nInputPlane;

  int dimw = 2;
  int dimh = 1;

  auto self = self_.contiguous();
  auto indices = indices_.contiguous();
  auto grad_output = grad_output_.contiguous();

  if (self.ndimension() == 3) {
    nInputPlane = self.size(0);
  } else {
    ++dimw;
    ++dimh;
    nInputPlane = self.size(1);
  }

  nInputCols = self.size(dimw);
  nInputRows = self.size(dimh);

  if (oheight != grad_output.size(dimh) || owidth != grad_output.size(dimw)) {
    TORCH_CHECK(false,
        "Inconsistent gradOutput size. output height: ",
        oheight,
        ", output width= ",
        owidth,
        ", gradOutput: ",
        grad_output.size(dimh),
        "x",
        grad_output.size(dimw));
  }

  grad_input.resize_as_(self);
  grad_input.zero_();

  int64_t count = self.numel();
  if (count == 0) {
    return grad_input;
  }

  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16,
      self.scalar_type(), "max_unpooling2d_backward_kernel", ([&] {
        max_unpooling2d_backward_kernel<<<
            GET_BLOCKS(count),
            CUDA_NUM_THREADS,
            0,
            at::cuda::getCurrentCUDAStream()>>>(
            count,
            grad_output.const_data_ptr<scalar_t>(),
            indices.const_data_ptr<int64_t>(),
            nInputPlane,
            nInputRows,
            nInputCols,
            oheight,
            owidth,
            grad_input.mutable_data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      }));
  return grad_input;
}
at::Tensor max_unpooling2d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto grad_input = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::native::max_unpooling2d_backward_out_cuda(
      grad_output, self, indices, output_size, grad_input);
  return grad_input;
}

at::Tensor& max_unpooling3d_backward_out_cuda(const Tensor& grad_output_,
    const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& grad_input) {
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];

  max_unpooling3d_shape_check(
    self_, grad_output_, indices_, output_size, stride, padding, "max_unpooling3d_backward_out_cuda()");

  int batchSize = 0;
  int inputSlices = 0;
  int inputTime = 0;
  int64_t inputHeight = 0;
  int64_t inputWidth = 0;

  TensorArg self_arg{self_, "self_", 1}, indices_arg{indices_, "indices_", 2},
      grad_output_arg{grad_output_, "grad_output_", 3},
      grad_input_arg{grad_input, "grad_input", 4};
  checkAllSameGPU(
      "max_unpooling3d_backward_out_cuda",
      {self_arg, indices_arg, grad_output_arg, grad_input_arg});

  auto self = self_.contiguous();
  auto indices = indices_.contiguous();
  auto grad_output = grad_output_.contiguous();

  if (self.ndimension() == 4) {
    batchSize = 1;
    inputSlices = self.size(0);
    inputTime = self.size(1);
    inputHeight = self.size(2);
    inputWidth = self.size(3);
  } else {
    batchSize = self.size(0);
    inputSlices = self.size(1);
    inputTime = self.size(2);
    inputHeight = self.size(3);
    inputWidth = self.size(4);
  }

  grad_input.resize_as_(self);
  grad_input.zero_();

  // Collapse batch and feature dimensions if needed
  auto grad_input_reshaped = grad_input;
  if (grad_input.ndimension() == 5) {
    grad_input_reshaped =
        grad_input.reshape({grad_input.size(0) * grad_input.size(1),
                            grad_input.size(2),
                            grad_input.size(3),
                            grad_input.size(4)});

    indices = indices.reshape({indices.size(0) * indices.size(1),
                               indices.size(2),
                               indices.size(3),
                               indices.size(4)});
  }
  if (grad_input.numel() == 0) {
    return grad_input;
  }

  int totalZ = inputTime * inputSlices * batchSize;
  int offsetZ = 0;

  dim3 block(32, 8);

  AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBFloat16,
      self.scalar_type(), "max_unpooling3d_backward_kernel", ([&] {
        while (totalZ > 0) {
          dim3 grid(
              ceilDiv(inputWidth, static_cast<int64_t>(block.x)),
              ceilDiv(inputHeight, static_cast<int64_t>(block.y)),
              totalZ > 65535 ? 65535 : totalZ);
          max_unpooling3d_backward_kernel<<<
              grid,
              block,
              0,
              at::cuda::getCurrentCUDAStream()>>>(
              grad_output.const_data_ptr<scalar_t>(),
              oT,
              oH,
              oW,
              indices.packed_accessor64<int64_t, 4>(),
              grad_input_reshaped.packed_accessor64<scalar_t, 4>(),
              offsetZ);
          C10_CUDA_KERNEL_LAUNCH_CHECK();
          totalZ -= 65535;
          offsetZ += 65535;
        }
      }));
  return grad_input;
}

at::Tensor max_unpooling3d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto grad_input = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::native::max_unpooling3d_backward_out_cuda(
      grad_output, self, indices, output_size, stride, padding, grad_input);
  return grad_input;
}

} // namespace at::native
