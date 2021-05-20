#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/NativeFunctions.h>
#include <tuple>


namespace at {
namespace meta {
TORCH_META_FUNC(adaptive_max_pool2d) (const Tensor& input, IntArrayRef output_size) {
  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_max_pool2d: expected input to have non-empty spatial dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (input.ndimension() == 3 || input.ndimension() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  TORCH_CHECK(
      output_size.size() == 2,
      "adaptive_max_pool2d: internal error: output_size.size() must be 2");

  int dimH = 1;
  int64_t sizeB = 1;
  int64_t sizeD = 0;

  if (input.ndimension() == 4) {
    sizeB = input.size(0);
    dimH++;
  }

  sizeD = input.size(dimH - 1);

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  /* resize output */
  if (input.ndimension() == 3) {
    set_output(0, {sizeD, osizeH, osizeW}, input.options());
    /* indices will contain i,j locations for each output point */
    set_output(1, {sizeD, osizeH, osizeW}, input.options().dtype(kLong));
  } else {
    set_output(0, {sizeB, sizeD, osizeH, osizeW}, input.options());
    /* indices will contain i,j locations for each output point */
    set_output(1, {sizeB, sizeD, osizeH, osizeW}, input.options().dtype(kLong));
  }
}

TORCH_META_FUNC(adaptive_max_pool2d_backward)
(const Tensor& gradOutput, const Tensor& input, const Tensor& indices) {
  set_output(0, input.sizes(), input.options());
}
} // namespace meta

namespace native {

namespace {

inline int start_index(int a, int b, int c) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return (int)std::floor((float)(a * c) / b);
}

inline int end_index(int a, int b, int c) {
  // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-narrowing-conversions)
  return (int)std::ceil((float)((a + 1) * c) / b);
}

// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

// 4d tensor B x D x H x W

template <typename scalar_t>
static void adaptive_max_pool2d_single_out_frame(
        scalar_t *input_p,
        scalar_t *output_p,
        int64_t *ind_p,
        int64_t sizeD,
        int64_t isizeH,
        int64_t isizeW,
        int64_t osizeH,
        int64_t osizeW,
        int64_t istrideD,
        int64_t istrideH,
        int64_t istrideW)
{
  at::parallel_for(0, sizeD, 0, [&](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++)
    {
      /* loop over output */
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
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
          int64_t *indp = ind_p   + d*osizeH*osizeW + oh*osizeW + ow;

          /* compute local max: */
          int ih=0, iw=0;
          int64_t maxindex = (ih+istartH)*isizeW + (iw+istartW);
          scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
          for(ih=0; ih < kH; ih++)
          {
            for(iw=0; iw < kW; iw++)
            {
              scalar_t val = *(ip + ih*istrideH + iw*istrideW);
              if ((val > maxval) || std::isnan(val))
              {
                maxval = val;
                maxindex = (ih+istartH)*isizeW + (iw+istartW);
              }
            }
          }

          /* set output to local max */
          *op = maxval;

          /* store location of max */
          *indp = maxindex;
        }
      }
    }
  });
}

template <typename scalar_t>
static void adaptive_max_pool2d_out_frame(
  scalar_t *input_data,
  scalar_t *output_data,
  int64_t *indices_data,
  int64_t sizeB,
  int64_t sizeD,
  int64_t isizeH,
  int64_t isizeW,
  int64_t osizeH,
  int64_t osizeW,
  int64_t istrideB,
  int64_t istrideD,
  int64_t istrideH,
  int64_t istrideW)
{
  at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
    for (auto b = start; b < end; b++) {
      adaptive_max_pool2d_single_out_frame<scalar_t>(input_data+b*istrideB, output_data+b*sizeD*osizeH*osizeW,
                                                     indices_data+b*sizeD*osizeH*osizeW,
                                                     sizeD,
                                                     isizeH, isizeW,
                                                     osizeH, osizeW,
                                                     istrideD,
                                                     istrideH, istrideW);
    }
  });
}

template <typename scalar_t>
static void adaptive_max_pool2d_backward_single_out_frame(
          scalar_t *gradInput_p,
          scalar_t *gradOutput_p,
          int64_t *indices,
          int64_t sizeD,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeH,
          int64_t osizeW)
{
  at::parallel_for(0, sizeD, 0, [&](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++)
    {
      scalar_t *gradInput_p_d = gradInput_p + d*isizeH*isizeW;
      scalar_t *gradOutput_p_d = gradOutput_p + d*osizeH*osizeW;
      int64_t *ind_p_d = indices + d*osizeH*osizeW;

      /* calculate max points */
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      int64_t oh, ow;
      for(oh = 0; oh < osizeH; oh++)
      {
        for(ow = 0; ow < osizeW; ow++)
        {
          /* retrieve position of max */
          int64_t maxp = ind_p_d[oh*osizeW + ow];

          /* update gradient */
          gradInput_p_d[maxp] += gradOutput_p_d[oh*osizeW + ow];
        }
      }
    }
  });
}

template <typename scalar_t>
static void adaptive_max_pool2d_backward_out_frame(
          scalar_t *gradInput_data,
          scalar_t *gradOutput_data,
          int64_t *indices_data,
          int64_t sizeB,
          int64_t sizeD,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeH,
          int64_t osizeW)
{
  at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
    for (auto b = start; b < end; b++) {
      adaptive_max_pool2d_backward_single_out_frame<scalar_t>(gradInput_data+b*sizeD*isizeH*isizeW,
                                                              gradOutput_data+b*sizeD*osizeH*osizeW,
                                                              indices_data+b*sizeD*osizeH*osizeW,
                                                              sizeD,
                                                              isizeH, isizeW,
                                                              osizeH, osizeW);
    }
  });
}
} // namespace

TORCH_IMPL_FUNC(adaptive_max_pool2d_out_cpu)
(const Tensor& input, IntArrayRef output_size, const Tensor& output, const Tensor& indices) {
  int dimW = 2;
  int dimH = 1;
  int64_t sizeB = 1;
  int64_t sizeD = 0;
  int64_t isizeH = 0;
  int64_t isizeW = 0;

  int64_t istrideD = 0;
  int64_t istrideH = 0;
  int64_t istrideW = 0;
  int64_t istrideB = 0;

  if (input.ndimension() == 4) {
    istrideB = input.stride(0);
    sizeB = input.size(0);
    dimW++;
    dimH++;
  }

  /* sizes */
  sizeD = input.size(dimH - 1);
  isizeH = input.size(dimH);
  isizeW = input.size(dimW);
  /* strides */
  istrideD = input.stride(dimH - 1);
  istrideH = input.stride(dimH);
  istrideW = input.stride(dimW);

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  /* resize output */
  if (input.ndimension() == 3) {
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "adaptive_max_pool2d_cpu", [&] {
          auto input_data = input.data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          auto indices_data = indices.data_ptr<int64_t>();

          adaptive_max_pool2d_single_out_frame<scalar_t>(
              input_data,
              output_data,
              indices_data,
              sizeD,
              isizeH,
              isizeW,
              osizeH,
              osizeW,
              istrideD,
              istrideH,
              istrideW);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "adaptive_max_pool2d_cpu", [&] {
          auto input_data = input.data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          auto indices_data = indices.data_ptr<int64_t>();

          adaptive_max_pool2d_out_frame<scalar_t>(
              input_data,
              output_data,
              indices_data,
              sizeB,
              sizeD,
              isizeH,
              isizeW,
              osizeH,
              osizeW,
              istrideB,
              istrideD,
              istrideH,
              istrideW);
        });
  }
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_cpu)
(const Tensor& gradOutput,
 const Tensor& input,
 const Tensor& indices,
 const Tensor& gradInput) {
  int dimW = 2;
  int dimH = 1;
  int64_t sizeB = 1;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int sizeD;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int isizeH;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int isizeW;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int osizeH;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int osizeW;

  /* get contiguous gradOutput */
  auto gradOutput_ = gradOutput.contiguous();

  /* zero */
  gradInput.zero_();

  if (input.ndimension() == 4) {
    sizeB = input.size(0);
    dimW++;
    dimH++;
  }

  sizeD = input.size(dimH - 1);
  isizeH = input.size(dimH);
  isizeW = input.size(dimW);
  osizeH = gradOutput_.size(dimH);
  osizeW = gradOutput_.size(dimW);

  /* backprop */
  if (input.ndimension() == 3) {
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "adaptive_max_pool2d_backward", [&] {
          /* get raw pointers */
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          scalar_t* gradOutput_data = gradOutput_.data_ptr<scalar_t>();
          int64_t* indices_data = indices.data_ptr<int64_t>();

          adaptive_max_pool2d_backward_single_out_frame<scalar_t>(
              gradInput_data,
              gradOutput_data,
              indices_data,
              sizeD,
              isizeH,
              isizeW,
              osizeH,
              osizeW);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "adaptive_max_pool2d_backward", [&] {
          /* get raw pointers */
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          scalar_t* gradOutput_data = gradOutput_.data_ptr<scalar_t>();
          int64_t* indices_data = indices.data_ptr<int64_t>();

          adaptive_max_pool2d_backward_out_frame<scalar_t>(
              gradInput_data,
              gradOutput_data,
              indices_data,
              sizeB,
              sizeD,
              isizeH,
              isizeW,
              osizeH,
              osizeW);
        });
  }
 }
} // at::native
} // at
