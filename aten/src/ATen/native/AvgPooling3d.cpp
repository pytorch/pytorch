#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/AvgPooling.h>
#include <tuple>

namespace at {
namespace native {

namespace {

  template <typename scalar_t>
  static void avg_pool3d_single_out_frame(
    scalar_t *input_p,
    scalar_t *output_p,
    int64_t nslices,
    int64_t itime, int64_t iwidth, int64_t iheight,
    int64_t otime, int64_t owidth, int64_t oheight,
    int kT, int kW, int kH,
    int dT, int dW, int dH,
    int pT, int pW, int pH,
    bool count_include_pad
  ) {
    at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
      for (auto k = start; k < end; k++)
      {
        int64_t i, j, ti;

        /* local pointers. */
        scalar_t *ip = input_p + k * itime * iwidth * iheight;
        scalar_t *op = output_p + k * otime * owidth * oheight;
        for (i = 0; i < otime * oheight * owidth; ++i)
          *(op + i) = 0;

        /* loop over output */
        for (ti = 0; ti < otime; ti++)
        {
          for (i = 0; i < oheight; i++)
          {
            for (j = 0; j < owidth; j++)
            {
              /* compute pool range. */
              int64_t tstart = ti * dT - pT;
              int64_t hstart = i  * dH - pH;
              int64_t wstart = j  * dW - pW;
              int64_t tend = std::min(tstart + kT, itime + pT);
              int64_t hend = std::min(hstart + kH, iheight + pH);
              int64_t wend = std::min(wstart + kW, iwidth + pW);
              int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
              tstart = std::max(tstart, (int64_t) 0);
              hstart = std::max(hstart, (int64_t) 0);
              wstart = std::max(wstart, (int64_t) 0);
              tend = std::min(tend, itime);
              hend = std::min(hend, iheight);
              wend = std::min(wend, iwidth);

              int divide_factor;
              if (count_include_pad)
                divide_factor = pool_size;
              else
                divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);

              /* compute local sum: */
              scalar_t sum = 0.0;
              int64_t x, y, z;

              for (z = tstart; z < tend; z++)
              {
                for (y = hstart; y < hend; y++)
                {
                  for (x = wstart; x < wend; x++)
                  {
                    sum +=  *(ip + z * iwidth * iheight + y * iwidth + x);
                  }
                }
              }

              /* set output to local max */
              *op++ += sum / divide_factor;
            }
          }
        }
      }
    });
  }

  template <typename scalar_t>
  static void avg_pool3d_out_frame(
    scalar_t *input_data,
    scalar_t *output_data,
    int64_t nbatch,
    int64_t nslices,
    int64_t istride, int64_t ostride,
    int64_t itime, int64_t iwidth, int64_t iheight,
    int64_t otime, int64_t owidth, int64_t oheight,
    int kT, int kW, int kH,
    int dT, int dW, int dH,
    int pT, int pW, int pH,
    bool count_include_pad)
  {
    at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
      for (auto p = start; p < end; p++)
      {
        avg_pool3d_single_out_frame(
          input_data + p * istride,
          output_data + p * ostride,
          nslices,
          itime, iwidth, iheight,
          otime, owidth, oheight,
          kT, kW, kH,
          dT, dW, dH,
          pT, pW, pH,
          count_include_pad);
      }
    });
  }

  void avg_pool3d_out_cpu_template(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad
  )
  {
    // XXX [JIT] Pooling.cpp allows stride.empty().
    // XXX [LIBTORCH] IntegrationTest.MNIST: padding.size() == 1 && dilation.size() == 1.
    TORCH_CHECK(kernel_size.size() == 3 &&
                (stride.empty() || stride.size() == 3) &&
                (padding.size() == 1 || padding.size() == 3),
      "avg_pool3d: internal error: all IntArrayRef sizes must be 3");

    TORCH_CHECK((input_.ndimension() == 4 || input_.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

    const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
    const int kH = safe_downcast<int, int64_t>(kernel_size[1]);
    const int kW = safe_downcast<int, int64_t>(kernel_size[2]);

    const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
    const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[1]);
    const int dW = stride.empty() ? kW : safe_downcast<int, int64_t>(stride[2]);

    const int pT = safe_downcast<int, int64_t>(padding[0]);
    const int pH = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
    const int pW = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

    const int64_t nslices = input_.size(-4);
    const int64_t itime = input_.size(-3);
    const int64_t iheight = input_.size(-2);
    const int64_t iwidth = input_.size(-1);

    const int64_t otime = pooling_output_shape<int64_t>(itime, kT, pT, dT, 1, ceil_mode);
    const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, pH, dH, 1, ceil_mode);
    const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, pW, dW, 1, ceil_mode);

    avg_pool3d_shape_check(
      input_, 
      nslices, 
      kT, kH, kW,
      dT, dH, dW,
      pT, pH, pW,
      itime, iheight, iwidth,
      otime, oheight, owidth);

    /* get contiguous input */
    Tensor input = input_.contiguous();

    if (input.dim() == 4) { /* non-batch mode */
      /* resize output */
      output.resize_({nslices, otime, oheight, owidth});

      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
        "avg_pool3d_cpu",
        [&] {
          scalar_t *input_data = input.data<scalar_t>();
          scalar_t *output_data = output.data<scalar_t>();

          avg_pool3d_single_out_frame(
            input_data, output_data,
            nslices,
            itime, iwidth, iheight,
            otime, owidth, oheight,
            kT, kW, kH,
            dT, dW, dH,
            pT, pW, pH,
            count_include_pad);
        });
    }
    else 
    {
      const int64_t nbatch = input.size(0);
      const int64_t istride = nslices * itime * iwidth * iheight;
      const int64_t ostride = nslices * otime * owidth * oheight;

      output.resize_({nbatch, nslices, otime, oheight, owidth});

      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
        "avg_pool3d_cpu",
        [&] {
          scalar_t *input_data = input.data<scalar_t>();
          scalar_t *output_data = output.data<scalar_t>();

          avg_pool3d_out_frame(
            input_data, output_data,
            nbatch,
            nslices,
            istride, ostride,
            itime, iwidth, iheight,
            otime, owidth, oheight,
            kT, kW, kH,
            dT, dW, dH,
            pT, pW, pH,
            count_include_pad);
        });
    }
  }

  template <typename scalar_t>
  static void avg_pool3d_backward_single_out_frame(
    scalar_t *gradInput_p,
    scalar_t *gradOutput_p,
    int64_t nslices,
    int64_t itime, int64_t iwidth, int64_t iheight,
    int64_t otime, int64_t owidth, int64_t oheight,
    int kT, int kW, int kH,
    int dT, int dW, int dH,
    int pT, int pW, int pH,
    bool count_include_pad)
  {
    at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
      for (auto k = start; k < end; k++)
      {
        int64_t i, j, ti;

        /* local pointers */
        scalar_t *ip = gradInput_p + k * itime * iwidth * iheight;
        scalar_t *op = gradOutput_p + k * otime * owidth * oheight;
        for (i = 0; i < itime*iwidth*iheight; i++)
          *(ip + i) = 0;

        /* loop over output */
        for (ti = 0; ti < otime; ti++)
        {
          for (i = 0; i < oheight; i++)
          {
            for (j = 0; j < owidth; j++)
            {
              int64_t tstart = ti * dT - pT;
              int64_t hstart = i  * dH - pH;
              int64_t wstart = j  * dW - pW;
              int64_t tend = std::min(tstart + kT, itime + pT);
              int64_t hend = std::min(hstart + kH, iheight + pH);
              int64_t wend = std::min(wstart + kW, iwidth + pW);
              int64_t pool_size = (tend -tstart) * (hend - hstart) * (wend - wstart);
              tstart = std::max(tstart, (int64_t) 0);
              hstart = std::max(hstart, (int64_t) 0);
              wstart = std::max(wstart, (int64_t) 0);
              tend = std::min(tend, itime);
              hend = std::min(hend, iheight);
              wend = std::min(wend, iwidth);

              int64_t divide_factor;
              if (count_include_pad)
                divide_factor = pool_size;
              else
                divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);

              /* scatter gradients out to footprint: */
              scalar_t val  = *op++;

              int64_t x,y,z;
              for (z = tstart; z < tend; z++)
              {
                for (y = hstart; y < hend; y++)
                {
                  for (x = wstart; x < wend; x++)
                  {
                    *(ip + z * iheight * iwidth + y * iwidth + x) += val / divide_factor;
                  }
                }
              }
            }
          }
        }
      }
    });
  }

  template <typename scalar_t>
  static void avg_pool3d_backward_out_frame(
    scalar_t *gradInput_data,
    scalar_t *gradOutput_data,
    int64_t nbatch,
    int64_t nslices,
    int64_t istride, int64_t ostride,
    int64_t itime, int64_t iwidth, int64_t iheight,
    int64_t otime, int64_t owidth, int64_t oheight,
    int kT, int kW, int kH,
    int dT, int dW, int dH,
    int pT, int pW, int pH,
    bool count_include_pad)
  {
    at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
      for (auto p = start; p < end; p++)
      {
        avg_pool3d_backward_single_out_frame(
          gradInput_data + p * istride,
          gradOutput_data + p * ostride,
          nslices,
          itime, iwidth, iheight,
          otime, owidth, oheight,
          kT, kW, kH,
          dT, dW, dH,
          pT, pW, pH,
          count_include_pad);
      }
    });
  }

  void avg_pool3d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    bool ceil_mode,
    bool count_include_pad)
  {
    // XXX [JIT] Pooling.cpp allows stride.empty().
    // XXX [LIBTORCH] IntegrationTest.MNIST: padding.size() == 1 && dilation.size() == 1.
    TORCH_CHECK(kernel_size.size() == 3 &&
                (stride.empty() || stride.size() == 3) &&
                (padding.size() == 1 || padding.size() == 3),
      "avg_pool3d: internal error: all IntArrayRef sizes must be 3");

    TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
      "non-empty 4D or 5D (batch mode) tensor expected for input");

    const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
    const int kH = safe_downcast<int, int64_t>(kernel_size[1]);
    const int kW = safe_downcast<int, int64_t>(kernel_size[2]);

    const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
    const int dH = stride.empty() ? kH : safe_downcast<int, int64_t>(stride[1]);
    const int dW = stride.empty() ? kW : safe_downcast<int, int64_t>(stride[2]);

    const int pT = safe_downcast<int, int64_t>(padding[0]);
    const int pH = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
    const int pW = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);


    const int64_t nslices = input.size(-4);
    const int64_t itime = input.size(-3);
    const int64_t iheight = input.size(-2);
    const int64_t iwidth = input.size(-1);

    /* get contiguous gradOutput */
    Tensor gradOutput = gradOutput_.contiguous();

    /* resize */
    gradInput.resize_as_(input);
    gradInput.zero_();

    const int64_t otime = gradOutput.size(-3);
    const int64_t oheight = gradOutput.size(-2);
    const int64_t owidth = gradOutput.size(-1);

    avg_pool3d_shape_check(
      input,
      gradOutput,
      nslices,
      kT, kH, kW,
      dT, dH, dW,
      pT, pH, pW,
      itime, iheight, iwidth,
      otime, oheight, owidth);

    if (input.ndimension() == 4)
    {
      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
        "avg_pool2d_backward",
        [&] {
          scalar_t *gradInput_data = gradInput.data<scalar_t>();
          scalar_t *gradOutput_data = gradOutput.data<scalar_t>();

          avg_pool3d_backward_single_out_frame(
            gradInput_data,
            gradOutput_data,
            nslices,
            itime, iwidth, iheight,
            otime, owidth, oheight,
            kT, kW, kH,
            dT, dW, dH,
            pT, pW, pH,
            count_include_pad);
        }
      );
    }
    else {
      const int64_t nbatch = input.size(0);
      const int64_t istride = nslices * itime * iheight * iwidth;
      const int64_t ostride = nslices * otime * oheight * owidth;

      AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
        "avg_pool2d_backward",
        [&] {
          scalar_t *gradInput_data = gradInput.data<scalar_t>();
          scalar_t *gradOutput_data = gradOutput.data<scalar_t>();

          avg_pool3d_backward_out_frame(
            gradInput_data,
            gradOutput_data,
            nbatch,
            nslices,
            istride, ostride,
            itime, iwidth, iheight,
            otime, owidth, oheight,
            kT, kW, kH,
            dT, dW, dH,
            pT, pW, pH,
            count_include_pad);
        }
      );
    }
  }

} // namespace

Tensor& avg_pool3d_out_cpu(
  Tensor& output,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad)
{
  avg_pool3d_out_cpu_template(
    output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad);
  return output;
}

Tensor avg_pool3d_cpu(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad)
{
  Tensor output = at::empty({0}, input.options());
  avg_pool3d_out_cpu_template(
    output,
    input,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad);
  return output;
} 

Tensor& avg_pool3d_backward_out_cpu(
  Tensor& gradInput,
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad)
{
  avg_pool3d_backward_out_cpu_template(
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

Tensor avg_pool3d_backward_cpu(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad)
{
  auto gradInput = at::zeros_like(input);
  avg_pool3d_backward_out_cpu_template(
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