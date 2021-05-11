#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <tuple>


namespace at {
namespace native {

namespace {

template <typename scalar_t>
static void max_pool3d_with_indices_single_out_frame(
          scalar_t *input_p,
          scalar_t *output_p,
          int64_t *indz_p,
          int64_t nslices,
          int64_t itime,
          int64_t iwidth,
          int64_t iheight,
          int64_t otime,
          int64_t owidth,
          int64_t oheight,
          int kT,
          int kW,
          int kH,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int dilationT,
          int dilationW,
          int dilationH)
{
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      /* loop over output */
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t i, j, ti;
      scalar_t *ip = input_p + k * itime * iwidth * iheight;
      for (ti = 0; ti < otime; ti++)
      {
        for (i = 0; i < oheight; i++)
        {
          for (j = 0; j < owidth; j++)
          {
            /* local pointers */

            int64_t start_t = ti * dT - pT;
            int64_t start_h = i * dH - pH;
            int64_t start_w = j * dW - pW;

            int64_t end_t = std::min(start_t + (kT - 1) * dilationT + 1, itime);
            int64_t end_h = std::min(start_h + (kH - 1) * dilationH + 1, iheight);
            int64_t end_w = std::min(start_w + (kW - 1) * dilationW + 1, iwidth);

            while(start_t < 0)
              start_t += dilationT;
            while(start_h < 0)
              start_h += dilationH;
            while(start_w < 0)
              start_w += dilationW;

            scalar_t *op = output_p + k * otime * owidth * oheight
              + ti * owidth * oheight + i * owidth + j;
            int64_t *indzp = indz_p + k * otime * owidth * oheight
              + ti * owidth * oheight + i * owidth + j;

            /* compute local max: */
            int64_t maxindex = start_t * iwidth * iheight + start_h * iwidth + start_w;
            scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();

            for (int64_t z = start_t; z < end_t; z += dilationT)
            {
              for (int64_t y = start_h; y < end_h; y += dilationH)
              {
                for (int64_t x = start_w; x < end_w; x += dilationW)
                {
                  int64_t index = z * iwidth * iheight + y * iwidth + x;
                  scalar_t val = ip[index];
                  if ((val > maxval) || std::isnan(val))
                  {
                    maxval = val;
                    maxindex = index;
                  }
                }
              }
            }

            // store location of max
            *indzp = maxindex;

            /* set output to local max */
            *op = maxval;
          }
        }
      }
    }
  });
}

template <typename scalar_t>
static void max_pool3d_with_indices_out_frame(
          scalar_t *input_data,
          scalar_t *output_data,
          int64_t *indices_data,
          int64_t nbatch,
          int64_t nslices,
          int64_t istride, int64_t ostride,
          int64_t itime, int64_t iwidth, int64_t iheight,
          int64_t otime, int64_t owidth, int64_t oheight,
          int kT, int kW, int kH,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          int dilationT, int dilationW, int dilationH)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; p++)
    {
      max_pool3d_with_indices_single_out_frame(
        input_data   + p * istride,
        output_data  + p * ostride,
        indices_data + p * ostride,
        nslices,
        itime, iwidth, iheight,
        otime, owidth, oheight,
        kT, kW, kH,
        dT, dW, dH,
        pT, pW, pH,
        dilationT, dilationW, dilationH
      );
    }
  });
}

void max_pool3d_with_indices_out_cpu_template(
          Tensor& output,
          Tensor& indices,
          const Tensor& input_,
          IntArrayRef kernel_size,
          IntArrayRef stride,
          IntArrayRef padding,
          IntArrayRef dilation,
          bool ceil_mode)
{
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
    "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "max_pool3d: padding must be either be a single int, or a tuple of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
    "max_pool3d: dilation must be either a single int, or a tuple of three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK((input_.ndimension() == 4 || input_.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");

  const int64_t nslices = input_.size(-4);
  const int64_t itime = input_.size(-3);
  const int64_t iheight = input_.size(-2);
  const int64_t iwidth = input_.size(-1);

  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, pT, dT, dilationT, ceil_mode);
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, pH, dH, dilationH, ceil_mode);
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, pW, dW, dilationW, ceil_mode);

  pool3d_shape_check(
    input_,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    pT, pH, pW,
    dilationT, dilationH, dilationW,
    itime, iheight, iwidth,
    otime, oheight, owidth);

  /* get contiguous input */
  Tensor input = input_.contiguous();

  if (input.dim() == 4) { /* non-batch mode */
    /* resize output */
    output.resize_({nslices, otime, oheight, owidth});
    /* indices will contain ti,i,j locations for each output point */
    indices.resize_({nslices, otime, oheight, owidth});

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "max_pool3d_with_indices_cpu",
      [&] {
        scalar_t *input_data = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        max_pool3d_with_indices_single_out_frame(
          input_data, output_data,
          indices_data,
          nslices,
          itime, iwidth, iheight,
          otime, owidth, oheight,
          kT, kW, kH,
          dT, dW, dH,
          pT, pW, pH,
          dilationT, dilationW, dilationH);
      }
    );
  }
  else { /* batch mode */
    const int64_t nbatch = input.size(0);
    const int64_t istride = nslices * itime * iwidth * iheight;
    const int64_t ostride = nslices * otime * owidth * oheight;

    /* resize output */
    output.resize_({nbatch, nslices, otime, oheight, owidth});
    /* indices will contain ti,i,j locations for each output point */
    indices.resize_({nbatch, nslices, otime, oheight, owidth});

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "max_pool3d_with_indices_cpu",
      [&] {
        scalar_t *input_data = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        max_pool3d_with_indices_out_frame(
          input_data,
          output_data,
          indices_data,
          nbatch,
          nslices,
          istride, ostride,
          itime, iwidth, iheight,
          otime, owidth, oheight,
          kT, kW, kH,
          dT, dW, dH,
          pT, pW, pH,
          dilationT, dilationW, dilationH);
     }
   );
  }
}

template <typename scalar_t>
static void max_pool3d_with_indices_backward_single_out_frame(
          scalar_t *gradInput_p,
          scalar_t *gradOutput_p,
          int64_t *indz_p,
          int64_t nslices,
          int64_t itime,
          int64_t iwidth,
          int64_t iheight,
          int64_t otime,
          int64_t owidth,
          int64_t oheight,
          int dT,
          int dW,
          int dH,
          int pT,
          int pW,
          int pH,
          int dilationT,
          int dilationW,
          int dilationH)
{
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (auto k = start; k < end; k++)
    {
      scalar_t *gradInput_p_k  = gradInput_p  + k * itime * iwidth * iheight;
      scalar_t *gradOutput_p_k = gradOutput_p + k * otime * owidth * oheight;
      int64_t *indz_p_k = indz_p + k * otime * owidth * oheight;

      /* calculate max points */
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t ti, i, j;
      for (ti = 0; ti < otime; ti++)
      {
        for (i = 0; i < oheight; i++)
        {
          for (j = 0; j < owidth; j++)
          {
            /* retrieve position of max */
            int64_t index = ti * oheight * owidth + i * owidth + j;
            int64_t maxp = indz_p_k[index];

            if (maxp != -1) {
              /* update gradient */
              gradInput_p_k[maxp] += gradOutput_p_k[index];
            }
          }
        }
      }
    }
  });
}

template <typename scalar_t>
static void max_pool3d_with_indices_backward_out_frame(
          scalar_t *gradInput_data,
          scalar_t *gradOutput_data,
          int64_t *indices_data,
          int64_t nbatch,
          int64_t nslices,
          int64_t istride, int64_t ostride,
          int64_t itime, int64_t iwidth, int64_t iheight,
          int64_t otime, int64_t owidth, int64_t oheight,
          int dT, int dW, int dH,
          int pT, int pW, int pH,
          int dilationT, int dilationW, int dilationH)
{
  at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
    for (auto p = start; p < end; p++)
    {
      max_pool3d_with_indices_backward_single_out_frame<scalar_t>(
        gradInput_data + p * istride,
        gradOutput_data + p * ostride,
        indices_data + p * ostride,
        nslices,
        itime, iwidth, iheight,
        otime, owidth, oheight,
        dT, dW, dH,
        pT, pW, pH,
        dilationT, dilationW, dilationH
      );
    }
  });
}

Tensor& max_pool3d_with_indices_backward_out_cpu_template(
          Tensor& gradInput,
          const Tensor& gradOutput_,
          const Tensor& input,
          const Tensor& indices,
          IntArrayRef kernel_size,
          IntArrayRef stride,
          IntArrayRef padding,
          IntArrayRef dilation,
          bool ceil_mode)
{
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "max_pool3d: kernel_size must either be a single int, or a tuple of three ints")
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.size() == 0 || stride.size() == 1 || stride.size() == 3,
    "max_pool3d: stride must either be omitted, a single int, or a tuple of three ints")
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "max_pool3d: padding must be either be a single int, or a tuple of three ints");
  const int pT = safe_downcast<int, int64_t>(padding[0]);
  const int pH = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[1]);
  const int pW = padding.size() == 1 ? pT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK(dilation.size() == 1 || dilation.size() == 3,
    "max_pool3d: dilation must be either a single int, or a tuple of three ints");
  const int dilationT = safe_downcast<int, int64_t>(dilation[0]);
  const int dilationH = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[1]);
  const int dilationW = dilation.size() == 1 ? dilationT : safe_downcast<int, int64_t>(dilation[2]);

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");

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

  max_pool3d_backward_shape_check(
    input,
    gradOutput,
    indices,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    pT, pH, pW,
    dilationT, dilationH, dilationW,
    itime, iheight, iwidth,
    otime, oheight, owidth);

  /* backprop */
  if (input.ndimension() == 4) /* non-batch mode*/
  {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "max_pool3d_with_indices_backward",
      [&] {
        /* get raw pointers */
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
        scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        max_pool3d_with_indices_backward_single_out_frame(
          gradInput_data, gradOutput_data,
          indices_data,
          nslices,
          itime, iwidth, iheight,
          otime, owidth, oheight,
          dT, dW, dH,
          pT, pW, pH,
          dilationT, dilationW, dilationH);
      }
    );
  }
  else /* batch mode */
  {
    const int64_t nbatch = input.size(0);
    const int64_t istride = nslices * itime * iwidth * iheight;
    const int64_t ostride = nslices * otime * owidth * oheight;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "max_pool3d_with_indices_backward",
      [&] {
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
        scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        max_pool3d_with_indices_backward_out_frame<scalar_t>(
          gradInput_data,
          gradOutput_data,
          indices_data,
          nbatch,
          nslices,
          istride, ostride,
          itime, iwidth, iheight,
          otime, owidth, oheight,
          dT, dW, dH,
          pT, pW, pH,
          dilationT, dilationW, dilationH);
      }
    );
  }

  return gradInput;
}

} // namespace

std::tuple<Tensor&, Tensor&> max_pool3d_with_indices_out_cpu(const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  Tensor& output,
  Tensor& indices)
{
  max_pool3d_with_indices_out_cpu_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> max_pool3d_with_indices_cpu(
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode)
{
  NoNamesGuard guard;

  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  max_pool3d_with_indices_out_cpu_template(
    output,
    indices,
    input,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);

  guard.reset();
  namedinference::propagate_names(output, input);
  namedinference::propagate_names(indices, input);

  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& max_pool3d_with_indices_backward_out_cpu(const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices,
  Tensor& gradInput)
{
  max_pool3d_with_indices_backward_out_cpu_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

Tensor max_pool3d_with_indices_backward_cpu(
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  IntArrayRef dilation,
  bool ceil_mode,
  const Tensor& indices)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  max_pool3d_with_indices_backward_out_cpu_template(
    gradInput,
    gradOutput_,
    input,
    indices,
    kernel_size,
    stride,
    padding,
    dilation,
    ceil_mode);
  return gradInput;
}

} // at::native
} // at
