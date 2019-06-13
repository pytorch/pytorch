#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/AvgPooling2d.h>
#include <tuple>


namespace at {
namespace native {

namespace {

  template <typename scalar_t>
  static void average_pool2d_single_out_frame(
          scalar_t *input_p,
          scalar_t *output_p,
          int64_t nslices,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t outputWidth,
          int64_t outputHeight,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool count_include_pad) 
  {
    at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
      for (auto k = start; k < end; k++) {
        int64_t xx, yy;
        /* For all output pixels... */
        scalar_t *ptr_output = output_p + k*outputWidth*outputHeight;
        scalar_t *ptr_input = input_p + k*inputWidth*inputHeight;
        int64_t i;
        for(i = 0; i < outputWidth*outputHeight; i++)
          ptr_output[i] = 0;

        for(yy = 0; yy < outputHeight; yy++)
        {
          for(xx = 0; xx < outputWidth; xx++)
          {
            /* Compute the mean of the input image... */
            int64_t hstart = yy * dH - padH;
            int64_t wstart = xx * dW - padW;
            int64_t hend = std::min(hstart + kH, inputHeight + padH);
            int64_t wend = std::min(wstart + kW, inputWidth + padW);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = std::max(hstart, (int64_t) 0);
            wstart = std::max(wstart, (int64_t) 0);
            hend = std::min(hend, inputHeight);
            wend = std::min(wend, inputWidth);

            scalar_t sum = 0;

            int divide_factor;
            if(count_include_pad)
              divide_factor = pool_size;
            else
              divide_factor = (hend - hstart) * (wend - wstart);

            int64_t kx, ky;

            for(ky = hstart; ky < hend; ky++)
            {
              for(kx = wstart; kx < wend; kx++)
                sum += ptr_input[ky*inputWidth + kx];
            }
            /* Update output */
            *ptr_output++ += sum/divide_factor;
          }
        }
      }
    });
  }

  template <typename scalar_t>
  static void avg_pool2d_out_frame(
          scalar_t *input_data,
          scalar_t *output_data,
          int64_t nbatch,
          int64_t nInputPlane,
          int64_t inputWidth,
          int64_t inputHeight,
          int64_t outputWidth,
          int64_t outputHeight,
          int kW,
          int kH,
          int dW,
          int dH,
          int padW,
          int padH,
          bool count_include_pad)
  {
    at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
      for (auto p = start; p < end; p++) {
        average_pool2d_single_out_frame(
          input_data+p*nInputPlane*inputWidth*inputHeight,
          output_data+p*nInputPlane*outputWidth*outputHeight,
          nInputPlane,
          inputWidth, inputHeight,
          outputWidth, outputHeight,
          kW, kH, dW, dH,
          padW, padH,
          count_include_pad);
      }
    });
  }

  void avg_pool2d_out_cpu_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad)
  {

    TORCH_CHECK((input_.ndimension() == 3 || input_.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

    const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
    const int kW = safe_downcast<int, int64_t>(kernel_size[1]);

    const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
    const int dW = stride.empty() ? kW : safe_downcast<int, int64_t>(stride[1]);

    const int padH = safe_downcast<int, int64_t>(padding[0]);
    const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

    /* sizes */
    const int64_t nbatch = input_.ndimension() == 4 ? input_.size(-4) : 1;
    const int64_t nInputPlane = input_.size(-3);
    const int64_t inputHeight = input_.size(-2);
    const int64_t inputWidth = input_.size(-1);

    const int64_t outputHeight = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
    const int64_t outputWidth = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

    avg_pool2d_shape_check(
      input_,
      kH, kW, dH, dW, padH, padW,
      ceil_mode,
      nInputPlane,
      inputHeight, inputWidth,
      outputHeight, outputWidth);

    Tensor input = input_.contiguous();
    AT_CHECK(input.is_contiguous(), "input must be contiguous");

    output.resize_({nbatch, nInputPlane, outputHeight, outputWidth});
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "avg_pool2d_cpu",
      [&] {
        /* get raw pointers */
        scalar_t *input_data = input.data<scalar_t>();
        scalar_t *output_data = output.data<scalar_t>();

        avg_pool2d_out_frame(
          input_data, output_data,
          nbatch,
          nInputPlane,
          inputWidth, inputHeight,
          outputWidth, outputHeight,
          kW, kH, dW, dH,
          padW, padH,
          count_include_pad);
      }
    );
  }

  template <typename scalar_t>
  static void avg_pool2d_backward_single_out_frame(
    scalar_t *gradInput_p,
    scalar_t *gradOutput_p,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t outputWidth,
    int64_t outputHeight,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool count_include_pad)
  {
    at::parallel_for(0, nInputPlane, 0, [&](int64_t start, int64_t end) {
      for (auto k = start; k < end; k++)
      {
        scalar_t *gradInput_pi_k = gradInput_p + k*inputHeight*inputWidth;
        scalar_t *gradInput_p_k = gradInput_p + k*inputHeight*inputWidth;
        scalar_t *gradOutput_p_k = gradOutput_p + k*outputWidth*outputHeight;

        int64_t i;
        for(i=0; i<inputWidth*inputHeight; i++)
          gradInput_pi_k[i] = 0.0;

        int64_t xx, yy;
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

            scalar_t z = *gradOutput_p_k++;

            int divide_factor;
            if(count_include_pad)
              divide_factor = pool_size;
            else
              divide_factor = (hend - hstart) * (wend - wstart);

            int64_t kx, ky;
            for(ky = hstart ; ky < hend; ky++)
            {
              for(kx = wstart; kx < wend; kx++)
                gradInput_p_k[ky*inputWidth + kx] += z/divide_factor;
            }
          }
        }
      }
    });
  }
  
  template <typename scalar_t>
  static void avg_pool2d_backward_out_frame(
    scalar_t *gradInput_data,
    scalar_t *gradOutput_data,
    int64_t nbatch,
    int64_t nInputPlane,
    int64_t inputWidth,
    int64_t inputHeight,
    int64_t outputWidth,
    int64_t outputHeight,
    int kW,
    int kH,
    int dW,
    int dH,
    int padW,
    int padH,
    bool count_include_pad)
  {
    at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
      for (auto p = start; p < end; p++) {
        avg_pool2d_backward_single_out_frame<scalar_t>(
          gradInput_data+p*nInputPlane*inputWidth*inputHeight,
          gradOutput_data+p*nInputPlane*outputWidth*outputHeight,
          nInputPlane,
          inputWidth, inputHeight,
          outputWidth, outputHeight,
          kW, kH,
          dW, dH,
          padW, padH,
          count_include_pad
        );
      };
    });
  }

  Tensor&  avg_pool2d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad
  )
  {
    // XXX JIT: Pooling.cpp allows stride.empty().
    TORCH_CHECK(kernel_size.size() == 2 &&
              (stride.empty() || stride.size() == 2) &&
              (padding.size() == 1 || padding.size() == 2),
      "max_pool2d_with_indices: internal error: all IntArrayRef sizes must be 2");

    TORCH_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

    const int kH = safe_downcast<int, int64_t>(kernel_size[0]);
    const int kW = safe_downcast<int, int64_t>(kernel_size[1]);

    const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[0]);
    const int dW = stride.empty() ? kW : safe_downcast<int, int64_t>(stride[1]);

    const int padH = safe_downcast<int, int64_t>(padding[0]);
    const int padW = padding.size() == 1 ? padH : safe_downcast<int, int64_t>(padding[1]);

    /* get contiguous gradOutput */
    const Tensor gradOutput = gradOutput_.contiguous();
    AT_CHECK(gradOutput.is_contiguous(), "gradOutput must be contiguous");

    /* resize */
    gradInput.resize_as_(input);
    gradInput.zero_();

    /* sizes */
    const int64_t nbatch = input.ndimension() == 4 ? input.size(-4) : 1;
    const int64_t nInputPlane = input.size(-3);
    const int64_t inputHeight = input.size(-2);
    const int64_t inputWidth = input.size(-1);
    const int64_t outputHeight = gradOutput.size(-2);
    const int64_t outputWidth = gradOutput.size(-1);

    /* XXX preserve the existing shape check behavior */
    const int64_t outputHeight_for_shape_check = pooling_output_shape<int64_t>(inputHeight, kH, padH, dH, 1, ceil_mode);
    const int64_t outputWidth_for_shape_check = pooling_output_shape<int64_t>(inputWidth, kW, padW, dW, 1, ceil_mode);

    avg_pool2d_shape_check(
      input, gradOutput_,
      kH, kW, dH, dW, padH, padW,
      ceil_mode,
      nInputPlane,
      inputHeight, inputWidth,
      outputHeight_for_shape_check, outputWidth_for_shape_check);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "avg_pool2d_backward_cpu",
      [&] {
        /* get raw pointers */
        scalar_t *gradInput_data = gradInput.data<scalar_t>();
        scalar_t *gradOutput_data = gradOutput.data<scalar_t>();
        
        avg_pool2d_backward_out_frame<scalar_t>(
          gradInput_data, gradOutput_data,
          nbatch,
          nInputPlane,
          inputWidth, inputHeight,
          outputWidth, outputHeight,
          kW, kH, dW, dH,
          padW, padH,
          count_include_pad);
      }
    );
    return gradInput;
  }
} // namespace

Tensor& avg_pool2d_out_cpu(
  Tensor& output,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad)
{
  avg_pool2d_out_cpu_template(
    output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad);
  return output;
}

Tensor avg_pool2d_cpu(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad)
{
  Tensor output = at::empty({0}, input.options());
  avg_pool2d_out_cpu_template(
    output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad);
  return output;
}

Tensor& avg_pool2d_backward_out_cpu(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad)
{
  avg_pool2d_backward_out_cpu_template(
    gradInput,
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad);
  return gradInput;
}

Tensor avg_pool2d_backward_cpu(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad)
{
  auto gradInput = at::zeros_like(input);
  avg_pool2d_backward_out_cpu_template(
    gradInput,
    gradOutput_,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad);
  return gradInput;
} 

} // at::native
} // at