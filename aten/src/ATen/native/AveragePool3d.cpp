#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ScalarOps.h>
#include <ATen/Parallel.h>
#include <ATen/native/Pool.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/avg_pool3d_backward_native.h>
#include <ATen/ops/avg_pool3d_native.h>
#endif

namespace at::meta {
using namespace ::at::native;

TORCH_META_FUNC(avg_pool3d) (
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  std::optional<int64_t> divisor_override
) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
    "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0,
    "divisor must be not zero");

  /* sizes */
  const int64_t nbatch = input.size(0);
  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  pool3d_shape_check(
    input,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    padT, padH, padW,
    1, 1, 1,
    itime, iheight, iwidth,
    otime, oheight, owidth,
    "avg_pool3d()",
    /*check_input_size=*/ true);

  /* resize output */
  if (input.ndimension() == 4) {
    set_output_raw_strided(0, {nslices, otime, oheight, owidth}, {}, input.options());
  }
  else {
    set_output_raw_strided(0, {nbatch, nslices, otime, oheight, owidth}, {}, input.options());
  }
}

TORCH_META_FUNC(avg_pool3d_backward) (
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  std::optional<int64_t> divisor_override
) {
  // #20866, #22032: Guarantee this for the official C++ API?
  TORCH_CHECK(kernel_size.size() == 1 || kernel_size.size() == 3,
    "avg_pool3d: kernel_size must be a single int, or a tuple of three ints");
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  TORCH_CHECK(stride.empty() || stride.size() == 1 || stride.size() == 3,
    "avg_pool3d: stride must be omitted, a single int, or a tuple of three ints");
  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  TORCH_CHECK(padding.size() == 1 || padding.size() == 3,
    "avg_pool3d: padding must be a single int, or a tuple of three ints");
  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(!divisor_override.has_value() || divisor_override.value() != 0, "divisor must be not zero");

  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  /* XXX shape check behavior from TH */
  const int64_t otime_for_shape_check = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight_for_shape_check = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth_for_shape_check = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  avg_pool3d_backward_shape_check(
    input,
    gradOutput_,
    nslices,
    kT, kH, kW,
    dT, dH, dW,
    padT, padH, padW,
    itime, iheight, iwidth,
    otime_for_shape_check, oheight_for_shape_check, owidth_for_shape_check,
    "avg_pool3d_backward()");

  /* resize output */
  set_output_raw_strided(0, input.sizes(), {}, input.options());
}

} // namespace at::meta

namespace at::native {

namespace {

template <typename scalar_t>
static void avg_pool3d_out_frame(
          const scalar_t *input_p,
          scalar_t *output_p,
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
          int padT,
          int padW,
          int padH,
          bool count_include_pad,
          std::optional<int64_t> divisor_override)
{
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (const auto k : c10::irange(start, end)) {
      /* local pointers. */
      const scalar_t *ip = input_p + k * itime * iwidth * iheight;
      scalar_t *op = output_p + k * otime * owidth * oheight;
      for (int64_t i = 0; i < otime * oheight * owidth; ++i)
        *(op + i) = 0;

      /* loop over output */
      for (int64_t ti = 0; ti < otime; ti++)
      {
        for (int64_t i = 0; i < oheight; i++)
        {
          for (int64_t j = 0; j < owidth; j++)
          {
            /* compute pool range. */
            int64_t tstart = ti * dT - padT;
            int64_t hstart = i  * dH - padH;
            int64_t wstart = j  * dW - padW;
            int64_t tend = std::min(tstart + kT, itime + padT);
            int64_t hend = std::min(hstart + kH, iheight + padH);
            int64_t wend = std::min(wstart + kW, iwidth + padW);
            int64_t pool_size = (tend - tstart) * (hend - hstart) * (wend - wstart);
            tstart = std::max(tstart, (int64_t) 0);
            hstart = std::max(hstart, (int64_t) 0);
            wstart = std::max(wstart, (int64_t) 0);
            tend = std::min(tend, itime);
            hend = std::min(hend, iheight);
            wend = std::min(wend, iwidth);

            if (tstart >= tend || hstart >= hend || wstart >= wend) {
              ++op;
              continue;
            }

            int64_t divide_factor = 0;
            if (divisor_override.has_value()) {
              divide_factor = divisor_override.value();
            } else {
              if(count_include_pad) {
                divide_factor = pool_size;
              } else {
                divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
              }
            }

            /* compute local sum: */
            scalar_t sum = 0.0;
            for (int64_t z = tstart; z < tend; z++)
            {
              for (int64_t y = hstart; y < hend; y++)
              {
                for (int64_t x = wstart; x < wend; x++)
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

} // anonymous namespace

TORCH_IMPL_FUNC(avg_pool3d_out_cpu) (
  const Tensor& input_,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  std::optional<int64_t> divisor_override,
  const Tensor& output
) {
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  const int64_t nslices = input_.size(-4);
  const int64_t itime = input_.size(-3);
  const int64_t iheight = input_.size(-2);
  const int64_t iwidth = input_.size(-1);

  const int64_t otime = pooling_output_shape<int64_t>(itime, kT, padT, dT, 1, ceil_mode);
  const int64_t oheight = pooling_output_shape<int64_t>(iheight, kH, padH, dH, 1, ceil_mode);
  const int64_t owidth = pooling_output_shape<int64_t>(iwidth, kW, padW, dW, 1, ceil_mode);

  /* get contiguous input */
  Tensor input = input_.contiguous();

  if (input.ndimension() == 4) /* non-batch mode */
  {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
      "avg_pool3d_out_frame",
      [&] {
        const scalar_t *input_data = input.const_data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();

        avg_pool3d_out_frame(
          input_data, output_data, nslices,
          itime, iwidth, iheight,
          otime, owidth, oheight,
          kT, kW, kH,
          dT, dW, dH,
          padT, padW, padH,
          count_include_pad,
          divisor_override);
    });
  }
  else  /* batch mode */
  {
    const int64_t nbatch = input.size(0);
    const int64_t istride = nslices * itime * iwidth * iheight;
    const int64_t ostride = nslices * otime * owidth * oheight;

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
      "avg_pool3d_out_frame",
      [&] {
        const scalar_t *input_data = input.const_data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();

        at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
          for (const auto p : c10::irange(start, end)) {
            avg_pool3d_out_frame(
              input_data + p * istride, output_data + p * ostride, nslices,
              itime, iwidth, iheight,
              otime, owidth, oheight,
              kT, kW, kH,
              dT, dW, dH,
              padT, padW, padH,
              count_include_pad,
              divisor_override
            );
          }
        });
    });
  }
}

namespace {

template <typename scalar_t>
static void avg_pool3d_backward_out_frame(
          scalar_t *gradInput_p,
          const scalar_t *gradOutput_p,
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
          int padT,
          int padW,
          int padH,
          bool count_include_pad,
          std::optional<int64_t> divisor_override)
{
  at::parallel_for(0, nslices, 0, [&](int64_t start, int64_t end) {
    for (const auto k : c10::irange(start, end)) {
      /* local pointers */
      scalar_t *ip = gradInput_p + k * itime * iwidth * iheight;
      const scalar_t *op = gradOutput_p + k * otime * owidth * oheight;
      for (int64_t i = 0; i < itime*iwidth*iheight; i++)
        *(ip + i) = 0;

      /* loop over output */
      for (int64_t ti = 0; ti < otime; ti++)
      {
        for (int64_t i = 0; i < oheight; i++)
        {
          for (int64_t j = 0; j < owidth; j++)
          {
            int64_t tstart = ti * dT - padT;
            int64_t hstart = i  * dH - padH;
            int64_t wstart = j  * dW - padW;
            int64_t tend = std::min(tstart + kT, itime + padT);
            int64_t hend = std::min(hstart + kH, iheight + padH);
            int64_t wend = std::min(wstart + kW, iwidth + padW);
            int64_t pool_size = (tend -tstart) * (hend - hstart) * (wend - wstart);
            tstart = std::max(tstart, (int64_t) 0);
            hstart = std::max(hstart, (int64_t) 0);
            wstart = std::max(wstart, (int64_t) 0);
            tend = std::min(tend, itime);
            hend = std::min(hend, iheight);
            wend = std::min(wend, iwidth);

            int64_t divide_factor = 0;
            if (divisor_override.has_value()) {
              divide_factor = divisor_override.value();
            } else {
              if(count_include_pad) {
                divide_factor = pool_size;
              } else {
                divide_factor = (tend - tstart) * (hend - hstart) * (wend - wstart);
              }
            }

            /* scatter gradients out to footprint: */
            scalar_t val  = *op++;

            for (auto z = tstart; z < tend; z++)
            {
              for (auto y = hstart; y < hend; y++)
              {
                for (auto x = wstart; x < wend; x++)
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

} // anonymous namespace

TORCH_IMPL_FUNC(avg_pool3d_backward_out_cpu) (
  const Tensor& gradOutput_,
  const Tensor& input,
  IntArrayRef kernel_size,
  IntArrayRef stride,
  IntArrayRef padding,
  bool ceil_mode,
  bool count_include_pad,
  std::optional<int64_t> divisor_override,
  const Tensor& gradInput
) {
  const int kT = safe_downcast<int, int64_t>(kernel_size[0]);
  const int kH = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[1]);
  const int kW = kernel_size.size() == 1 ? kT : safe_downcast<int, int64_t>(kernel_size[2]);

  const int dT = stride.empty() ? kT : safe_downcast<int, int64_t>(stride[0]);
  const int dH = stride.empty() ? kH :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[1]);
  const int dW = stride.empty() ? kW :
                 stride.size() == 1 ? dT : safe_downcast<int, int64_t>(stride[2]);

  const int padT = safe_downcast<int, int64_t>(padding[0]);
  const int padH = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[1]);
  const int padW = padding.size() == 1 ? padT : safe_downcast<int, int64_t>(padding[2]);

  const int64_t nslices = input.size(-4);
  const int64_t itime = input.size(-3);
  const int64_t iheight = input.size(-2);
  const int64_t iwidth = input.size(-1);

  /* get contiguous gradOutput */
  Tensor gradOutput = gradOutput_.contiguous();

  const int64_t otime = gradOutput.size(-3);
  const int64_t oheight = gradOutput.size(-2);
  const int64_t owidth = gradOutput.size(-1);

  gradInput.zero_();

  /* backprop */
  if (input.ndimension() == 4) /* non-batch mode*/
  {
    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
      "avg_pool3d_backward_out_frame",
      [&] {
       scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
       const scalar_t *gradOutput_data = gradOutput.const_data_ptr<scalar_t>();

       avg_pool3d_backward_out_frame(
         gradInput_data, gradOutput_data,
         nslices,
         itime, iwidth, iheight,
         otime, owidth, oheight,
         kT, kW, kH,
         dT, dW, dH,
         padT, padW, padH,
         count_include_pad,
         divisor_override);
    });
  }
  else /* batch mode */
  {
    const int64_t nbatch = input.size(0);
    const int64_t istride = nslices * itime * iwidth * iheight;
    const int64_t ostride = nslices * otime * owidth * oheight;

    AT_DISPATCH_FLOATING_TYPES_AND(at::ScalarType::Long, input.scalar_type(),
      "avg_pool3d_backward_out_frame",
      [&] {
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
        const scalar_t *gradOutput_data = gradOutput.const_data_ptr<scalar_t>();

        at::parallel_for(0, nbatch, 0, [&](int64_t start, int64_t end) {
          for (const auto p : c10::irange(start, end)) {
            avg_pool3d_backward_out_frame(
              gradInput_data  + p * istride, gradOutput_data + p * ostride, nslices,
              itime, iwidth, iheight,
              otime, owidth, oheight,
              kT, kW, kH,
              dT, dW, dH,
              padT, padW, padH,
              count_include_pad,
              divisor_override
            );
          }
        });
    });
  }
}

} // namespace at::native
