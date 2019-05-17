#include <tuple>
#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/pooling_shape.h"
#include "ATen/TensorUtils.h"

namespace at {
namespace native {
using namespace at;
namespace {
inline int start_index(int a, int b, int c) {
  return (int)std::floor((float)(a * c) / b);
}

inline int end_index(int a, int b, int c) {
  return (int)std::ceil((float)((a + 1) * c) / b);
}

static void avg_pool2d_shapecheck(
    Tensor& input,
    Tensor* output,
    IntList kernel_size,
    IntList stride_size,
    IntList pad_size,
    bool ceil_mode) {
  auto kH = kernel_size[0];
  auto kW = kernel_size[1];
  auto dH = stride_size[0];
  auto dW = stride_size[1];
  auto padH = pad_size[0];
  auto padW = pad_size[1];
  AT_CHECK(
      kW > 0 && kH > 0,
      5,
      "kernel size should be greater than zero, but got kH: %d kW: %d",
      kH,
      kW);
  AT_CHECK(
      dW > 0 && dH > 0,
      8,
      "stride should be greater than zero, but got dH: %d dW: %d",
      dH,
      dW);

  int ndim = input.dim();
  int dimf = 0;
  int dimh = 1;
  int dimw = 2;

  if (ndim == 4) {
    dimf++;
    dimh++;
    dimw++;
  }

  AT_CHECK(
      input.numel() > 0 && (ndim == 3 || ndim == 4),
      "non-empty 3D or 4D input tensor expected but got: %s",
      input);

  AT_CHECK(
      kW / 2 >= padW && kH / 2 >= padH,
      "pad should be smaller than half of kernel size, but got "
      "padW = %d, padH = %d, kW = %d, kH = %d",
      padW,
      padH,
      kW,
      kH);

  int64_t nInputPlane = input.size(dimh - 1);
  int64_t inputHeight = input.size(dimh);
  int64_t inputWidth = input.size(dimw);
  int64_t nOutputPlane = nInputPlane;

  int64_t outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
  int64_t outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

  

  if (outputWidth < 1 || outputHeight < 1)
    AT_ERROR(
        "Given input size: (%dx%dx%d). "
        "Calculated output size: (%dx%dx%d). Output size is too small",
        nInputPlane,
        inputHeight,
        inputWidth,
        nInputPlane,
        outputHeight,
        outputWidth);

  if (output != NULL) {
    auto output_co = TensorArg(*output, "out", -1);
    AT_CHECK(
      output->ndimension() == ndim,
      "output.ndimensions() must be %d, got %d",
      ndim, output->ndimension()
    )
    at::checkSize("avg_pool2d", output_co, dimf, nOutputPlane);
    at::checkSize("avg_pool2d", output_co, dimh, outputHeight);
    at::checkSize("avg_pool2d", output_co, dimw, outputWidth);
  }
}

template <typename scalar_t>
static void avg_pool2d_out_cpu_frame(
    Tensor& input,
    Tensor& output,
    int64_t nInputPlane,
    int64_t inputHeight,
    int64_t inputWidth,
    int64_t outputHeight,
    int64_t outputWidth,
    int64_t nbatch,
    IntList kernel_size,
    IntList stride_size,
    IntList pad_size,
    bool count_include_pad) {
  auto kH = kernel_size[0];
  auto kW = kernel_size[1];
  auto dH = stride_size[0];
  auto dW = stride_size[1];
  auto padH = pad_size[0];
  auto padW = pad_size[1];
  scalar_t* output_data = output.data<scalar_t>();
  scalar_t* input_data = input.data<scalar_t>();
  int64_t k;
#pragma omp parallel for private(k)
  for (k = 0; k < nInputPlane; k++) {
    int64_t p;
    for (p = 0; p < nbatch; p++) {
      int64_t xx, yy;
      /* For all output pixels... */
      scalar_t* ptr_output = output_data +
          p * nInputPlane * outputWidth * outputHeight +
          k * outputWidth * outputHeight;
      scalar_t* ptr_input = input_data +
          p * nInputPlane * inputWidth * inputHeight +
          k * inputWidth * inputHeight;
      int64_t i;
      for (i = 0; i < outputWidth * outputHeight; i++)
        ptr_output[i] = 0;

      for (yy = 0; yy < outputHeight; yy++) {
        for (xx = 0; xx < outputWidth; xx++) {
          /* Compute the mean of the input image... */
          int64_t hstart = yy * dH - padH;
          int64_t wstart = xx * dW - padW;
          int64_t hend = std::min(hstart + kH, inputHeight + padH);
          int64_t wend = std::min(wstart + kW, inputWidth + padW);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, (int64_t)0);
          wstart = std::max(wstart, (int64_t)0);
          hend = std::min(hend, inputHeight);
          wend = std::min(wend, inputWidth);

          scalar_t sum = 0;

          int divide_factor;
          if (count_include_pad)
            divide_factor = pool_size;
          else
            divide_factor = (hend - hstart) * (wend - wstart);

          int64_t kx, ky;

          for (ky = hstart; ky < hend; ky++) {
            for (kx = wstart; kx < wend; kx++)
              sum += ptr_input[ky * inputWidth + kx];
          }
          /* Update output */
          *ptr_output++ += sum / divide_factor;
        }
      }
    }
  }
}

static void avg_pool2d_out_cpu_template(
    Tensor& input,
    Tensor& output,
    IntList kernel_size,
    IntList stride_size,
    IntList pad_size,
    bool ceil_mode,
    bool count_include_pad) {
  auto kH = kernel_size[0];
  auto kW = kernel_size[1];
  auto dH = stride_size[0];
  auto dW = stride_size[1];
  auto padH = pad_size[0];
  auto padW = pad_size[1];

  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  int64_t nbatch = 1;

  int64_t inputWidth;
  int64_t inputHeight;
  int64_t outputWidth;
  int64_t outputHeight;
  int64_t nInputPlane; // number of channels (or colors)

  avg_pool2d_shapecheck(
      input, &output, kernel_size, stride_size, pad_size, ceil_mode);

  if (input.dim() == 4) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimc++;
  }

  inputWidth = input.size(dimw);
  inputHeight = input.size(dimh);
  nInputPlane = input.size(dimc);

  outputWidth =
      pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  outputHeight =
      pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  if (input.dim() == 3)
    output.resize_({nInputPlane, outputHeight, outputWidth});
  else
    output.resize_({input.size(0), nInputPlane, outputHeight, outputWidth});

  input = input.contiguous();

  if (!output.is_contiguous()) {
    AT_ERROR("output must be contiguous");
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    output.scalar_type(), "avg_pool2d", [&] {
      avg_pool2d_out_cpu_frame<scalar_t>(
          input,
          output,
          nInputPlane,
          inputHeight,
          inputWidth,
          outputHeight,
          outputWidth,
          nbatch,
          kernel_size,
          stride_size,
          pad_size,
          count_include_pad);
    });
}

template <typename scalar_t>
static void avg_pool2d_backward_cpu_frame(
  Tensor& gradInput,
  Tensor& gradOutput,
  int64_t nInputPlane,
  int64_t inputHeight,
  int64_t inputWidth,
  int64_t outputHeight,
  int64_t outputWidth,
  int64_t nbatch,
  bool count_include_pad,
  IntList kernel_size,
  IntList stride_size,
  IntList pad_size
) {
  scalar_t *gradInput_data = gradInput.data<scalar_t>();
  scalar_t *gradOutput_data = gradOutput.data<scalar_t>();
  int64_t k;
  auto kH = kernel_size[0];
  auto kW = kernel_size[1];
  auto dH = stride_size[0];
  auto dW = stride_size[1];
  auto padH = pad_size[0];
  auto padW = pad_size[1];

#pragma omp parallel for private(k)
  for(k = 0; k < nInputPlane; k++)
  {
    int64_t p;
    for(p = 0; p < nbatch; p++)
    {
      scalar_t *ptr_gradOutput = gradOutput_data + p*nInputPlane*outputHeight*outputWidth + k*outputWidth*outputHeight;
      int64_t xx, yy;

      scalar_t* ptr_gi = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;
      scalar_t *ptr_gradInput = gradInput_data + p*nInputPlane*inputWidth*inputHeight + k*inputWidth*inputHeight;

      int64_t i;
      for(i=0; i<inputWidth*inputHeight; i++)
        ptr_gi[i] = 0.0;

      for(yy = 0; yy < outputHeight; yy++)
      {
        for(xx = 0; xx < outputWidth; xx++)
        {
          int64_t hstart = yy * dH - padH;
          int64_t wstart = xx * dW - padW;
          int64_t hend = std::min(hstart + kH, inputHeight + padH);
          int64_t wend = std::min(wstart + kW, inputWidth + padW);
          int pool_size = (hend - hstart) * (wend - wstart);
          hstart = std::max(hstart, (int64_t) 0);
          wstart = std::max(wstart, (int64_t) 0);
          hend = std::min(hend, inputHeight);
          wend = std::min(wend, inputWidth);

          scalar_t z = *ptr_gradOutput++;

          int divide_factor;
          if(count_include_pad)
            divide_factor = pool_size;
          else
            divide_factor = (hend - hstart) * (wend - wstart);

          int64_t kx, ky;
          for(ky = hstart ; ky < hend; ky++)
          {
            for(kx = wstart; kx < wend; kx++)
              ptr_gradInput[ky*inputWidth + kx] += z/divide_factor;
          }
        }
      }
    }
  }
}

static void avg_pool2d_backward_cpu_template(
  Tensor& input,
  Tensor& gradOutput,
  Tensor& gradInput,
  IntList kernel_size,
  IntList stride_size,
  IntList pad_size,
  bool ceil_mode,
  bool count_include_pad)
{
  auto kH = kernel_size[0];
  auto kW = kernel_size[1];
  auto dH = stride_size[0];
  auto dW = stride_size[1];
  auto padH = pad_size[0];
  auto padW = pad_size[1];
  int dimw = 2;
  int dimh = 1;
  int dimc = 0;
  int64_t nbatch = 1;
  int64_t ndim = 3;

  int64_t inputWidth;
  int64_t inputHeight;
  int64_t outputWidth;
  int64_t outputHeight;
  int64_t nInputPlane; // number of channels (or colors)

  avg_pool2d_shapecheck(
      input, &gradOutput, kernel_size, stride_size, pad_size, ceil_mode);


  if (input.dim() == 4) {
    nbatch = input.size(0);
    dimw++;
    dimh++;
    dimc++;
    ndim = 4;
  }

  inputWidth = input.size(dimw);
  inputHeight = input.size(dimh);
  nInputPlane = input.size(dimc);

  outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);
  outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);

  gradInput.resize_as_(input);

  if (input.dim() == 3)
    gradOutput.resize_({nInputPlane, outputHeight, outputWidth});
  else
    gradOutput.resize_({input.size(0), nInputPlane, outputHeight, outputWidth});

  gradOutput = gradOutput.contiguous();
  if (!gradInput.is_contiguous())
    AT_ERROR("gradInput must be contiguous");

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    gradOutput.scalar_type(), "avg_pool2d", [&] {
      avg_pool2d_backward_cpu_frame<scalar_t>(
          gradInput,
          gradOutput,
          nInputPlane,
          inputHeight,
          inputWidth,
          outputHeight,
          outputWidth,
          nbatch,
          count_include_pad,
          kernel_size,
          stride_size,
          pad_size);
    });
}

} // namespace

Tensor& avg_pool2d_out_cpu(
  Tensor& input,
  IntList kernel_size,
  IntList stride,
  IntList padding,
  bool ceil_mode,
  bool count_include_pad,
  Tensor& output)
  {
    avg_pool2d_out_cpu_template(
      input, output, kernel_size, stride, padding, ceil_mode, count_include_pad);
    return output;
  }

Tensor avg_pool2d_cpu(
  Tensor& input,
  IntList kernel_size,
  IntList stride,
  IntList padding,
  bool ceil_mode,
  bool count_include_pad)
  {
    auto output = empty({0}, input.options());
    avg_pool2d_out_cpu_template(
      input, output, kernel_size, stride, padding, ceil_mode, count_include_pad);
    return output;
  }

  Tensor& avg_pool2d_backward_out_cpu(
    Tensor& gradOutput,
    Tensor& input,
    IntList kernel_size,
    IntList stride,
    IntList padding,
    bool ceil_mode,
    bool count_include_pad,
    Tensor& gradInput)
  {
    avg_pool2d_backward_cpu_template(
      input, gradOutput, gradInput, kernel_size, stride, padding, ceil_mode, count_include_pad);
    return gradOutput;
  }

  Tensor avg_pool2d_backward_cpu(
    Tensor& gradOutput,
    Tensor& input,
    IntList kernel_size,
    IntList stride,
    IntList padding,
    bool ceil_mode,
    bool count_include_pad)
  {
    auto gradInput = at::zeros_like(input);
    avg_pool2d_backward_cpu_template(
      input, gradOutput, gradInput, kernel_size, stride, padding, ceil_mode, count_include_pad);
    return gradOutput;
  }
} // namespace native
} // namespace at