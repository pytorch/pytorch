#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include <tuple>


namespace at {
namespace native {

namespace {

  inline int start_index(int a, int b, int c) {
    return (int)std::floor((float)(a * c) / b);
  }

  inline int end_index(int a, int b, int c) {
    return (int)std::ceil((float)((a + 1) * c) / b);
  }

  template <typename scalar_t>
  static void adaptive_avg_pool2d_out_frame(
            scalar_t *input_p,
            scalar_t *output_p,
            int64_t sizeD,
            int64_t isizeH,
            int64_t isizeW,
            int64_t osizeH,
            int64_t osizeW,
            int64_t istrideD,
            int64_t istrideH,
            int64_t istrideW)
  {
    int64_t d;
  #pragma omp parallel for private(d)
    for (d = 0; d < sizeD; d++)
    {
      /* loop over output */
      int64_t oh, ow;
      for(oh = 0; oh < osizeH; oh++)
      {
        int istartH = start_index(oh, osizeH, isizeH);
        int iendH   = end_index(oh, osizeH, isizeH);
        int kH = iendH - istartH;

        for(ow = 0; ow < osizeW; ow++)
        {
          int istartW = start_index(ow, osizeW, isizeW);
          int iendW   = end_index(ow, osizeW, isizeW);
          int kW = iendW - istartW;

          /* local pointers */
          scalar_t *ip = input_p   + d*istrideD + istartH*istrideH + istartW*istrideW;
          scalar_t *op = output_p  + d*osizeH*osizeW + oh*osizeW + ow;

          /* compute local average: */
          scalar_t sum = 0;
          int ih, iw;
          for(ih = 0; ih < kH; ih++)
          {
            for(iw = 0; iw < kW; iw++)
            {
              scalar_t val = *(ip + ih*istrideH + iw*istrideW);
              sum += val;
            }
          }

          /* set output to local average */
          *op = sum / kW / kH;
        }
      }
    }
  }

  void adaptive_avg_pool2d_out_cpu_template(
    at::Tensor& output,
    at::Tensor const& input,
    IntArrayRef output_size)
  {
    for (int64_t i = 0; i < input.ndimension(); i++) {
      AT_CHECK(input.size(i) > 0,
        "adaptive_avg_pooling2d(): expected input to have non-empty spatial dimensions, "
        "but input has sizes ", input.sizes(), " with dimension ", i, " being "
        "empty");
    }

    AT_CHECK((input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

    /* sizes */
    int64_t sizeD  = input.size(-3);
    int64_t isizeH = input.size(-2);
    int64_t isizeW = input.size(-1);
    /* strides */
    int64_t istrideD = input.stride(-3);
    int64_t istrideH = input.stride(-2);
    int64_t istrideW = input.stride(-1);

    auto osizeH = output_size[0];
    auto osizeW = output_size[1];

    /* resize output */
    if (input.ndimension() == 3)
    {
      output.resize_({sizeD, osizeH, osizeW});

      AT_DISPATCH_FLOATING_TYPES(input.type(), "adaptive_avg_pool2d", [&] {
          auto input_data = input.data<scalar_t>();
          auto output_data = output.data<scalar_t>();
          adaptive_avg_pool2d_out_frame<scalar_t>(input_data, output_data,
                                                  sizeD,
                                                  isizeH, isizeW,
                                                  osizeH, osizeW,
                                                  istrideD,
                                                  istrideH, istrideW);
        }
      );
    }
    else
    {
      output.resize_({input.size(-4), sizeD, osizeH, osizeW});
      int64_t b;
    #pragma omp parallel for private(b)
      for (b = 0; b < input.size(0); b++)
      {
        AT_DISPATCH_FLOATING_TYPES(input.type(), "adaptive_avg_pool2d", [&] {
            auto input_data = input.data<scalar_t>();
            auto output_data = output.data<scalar_t>();
            adaptive_avg_pool2d_out_frame<scalar_t>(input_data+b*input.stride(0), output_data+b*sizeD*osizeH*osizeW,
                                                    sizeD,
                                                    isizeH, isizeW,
                                                    osizeH, osizeW,
                                                    istrideD,
                                                    istrideH, istrideW);
          }
        );
      }
    }
  }

  template <typename scalar_t>
  static void adaptive_avg_pool2d_backward_out_frame(
    scalar_t *gradInput_p,
    scalar_t *gradOutput_p,
    int64_t sizeD,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeH,
    int64_t osizeW)
  {
    int64_t d;
  #pragma omp parallel for private(d)
    for (d = 0; d < sizeD; d++)
    {
      scalar_t *gradInput_p_d = gradInput_p + d*isizeW*isizeH;
      scalar_t *gradOutput_p_d = gradOutput_p + d*osizeW*osizeH;

      /* calculate average */
      int64_t oh, ow;
      for(oh = 0; oh < osizeH; oh++)
      {
        int istartH = start_index(oh, osizeH, isizeH);
        int iendH   = end_index(oh, osizeH, isizeH);
        int kH = iendH - istartH;

        for(ow = 0; ow < osizeW; ow++)
        {

          int istartW = start_index(ow, osizeW, isizeW);
          int iendW   = end_index(ow, osizeW, isizeW);
          int kW = iendW - istartW;

          scalar_t grad_delta = gradOutput_p_d[oh*osizeW +ow] / kH / kW;

          int ih, iw;
          for(ih = istartH; ih < iendH; ih++)
          {
            for(iw = istartW; iw < iendW; iw++)
            {
              /* update gradient */
              gradInput_p_d[ih*isizeW + iw] += grad_delta;
            }
          }
        }
      }
    }
  }

  Tensor& adaptive_avg_pool2d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input)
  {
    /* sizes */
    int sizeD  = input.size(-3);
    int isizeH = input.size(-2);
    int isizeW = input.size(-1);
    int osizeH = gradOutput_.size(-2);
    int osizeW = gradOutput_.size(-1);

    /* get contiguous gradOutput */
    auto gradOutput = gradOutput_.contiguous();

    /* backprop */
    if (input.ndimension() == 3)
    {
      AT_DISPATCH_FLOATING_TYPES(
        input.type(), "adaptive_avg_pool2d_backward", [&] {
          /* get raw pointers */
          scalar_t *gradInput_data = gradInput.data<scalar_t>();
          scalar_t *gradOutput_data = gradOutput.data<scalar_t>();

          adaptive_avg_pool2d_backward_out_frame<scalar_t>(
            gradInput_data, gradOutput_data,
            sizeD,
            isizeH, isizeW,
            osizeH, osizeW);
        }
      );
    }
    else
    {
      int64_t b;
    #pragma omp parallel for private(b)
      for (b = 0; b < input.size(0); b++)
      {
        AT_DISPATCH_FLOATING_TYPES(
          input.type(), "adaptive_avg_pool2d_backward", [&] {
            /* get raw pointers */
            scalar_t *gradInput_data = gradInput.data<scalar_t>();
            scalar_t *gradOutput_data = gradOutput.data<scalar_t>();

            adaptive_avg_pool2d_backward_out_frame<scalar_t>(
              gradInput_data+b*sizeD*isizeH*isizeW, gradOutput_data+b*sizeD*osizeH*osizeW,
              sizeD,
              isizeH, isizeW,
              osizeH, osizeW);
          }
        );
      }
    }
    return gradInput;
  }

} // namespace

  Tensor& adaptive_avg_pool2d_out_cpu(
    Tensor& output,
    const Tensor& input,
    IntArrayRef output_size)
  {
    adaptive_avg_pool2d_out_cpu_template(
      output, input, output_size);
    return output;
  }

  Tensor adaptive_avg_pool2d_cpu(
    at::Tensor const& input,
    IntArrayRef output_size)
  {
    auto output = at::empty({0}, input.options());
    adaptive_avg_pool2d_out_cpu_template(
      output, input, output_size);
    return output;
  }

  Tensor adaptive_avg_pool2d(  
    at::Tensor const& input,
    IntArrayRef output_size){
    if (output_size[0] == 1 && output_size[1] == 1) {
//in this case, adaptive pooling is just computing mean over hw dimensions, which can be done more efficiently
       int64_t mean_size = input.size(-1) * input.size(-2);
       Tensor out = input.contiguous().view({-1, mean_size}).mean(-1);
       return input.ndimension() == 3 ? out.view({input.size(0), 1, 1}) : out.view({input.size(0), input.size(1), 1, 1});
    } else {
       return _adaptive_avg_pool2d(input, output_size);
    }
  }

  Tensor& adaptive_avg_pool2d_backward_out_cpu(
    Tensor& gradInput,
    const Tensor& gradOutput,
    const Tensor& input)
  {
    gradInput.resize_as_(input);
    adaptive_avg_pool2d_backward_out_cpu_template(
      gradInput, gradOutput, input);
    return gradInput;
  }

  Tensor adaptive_avg_pool2d_backward_cpu(
    const Tensor& gradOutput,
    const Tensor& input)
  {
    auto gradInput = at::zeros_like(input);
    adaptive_avg_pool2d_backward_out_cpu_template(
      gradInput, gradOutput, input);
    return gradInput;
  }

} // at::native
} // at
