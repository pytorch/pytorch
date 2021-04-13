#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
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

// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

// 5d tensor B x D x T x H x W

template <typename scalar_t>
static void adaptive_max_pool3d_single_out_frame(
          scalar_t *input_p,
          scalar_t *output_p,
          int64_t *ind_p,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW,
          int64_t istrideD,
          int64_t istrideT,
          int64_t istrideH,
          int64_t istrideW)
{
  at::parallel_for(0, sizeD, 0, [&](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++)
    {
      /* loop over output */
      int64_t ot, oh, ow;
      for(ot = 0; ot < osizeT; ot++)
      {
        int64_t istartT = start_index(ot, osizeT, isizeT);
        int64_t iendT   = end_index(ot, osizeT, isizeT);
        int64_t kT = iendT - istartT;

        for(oh = 0; oh < osizeH; oh++)
        {
          int64_t istartH = start_index(oh, osizeH, isizeH);
          int64_t iendH   = end_index(oh, osizeH, isizeH);
          int64_t kH = iendH - istartH;

          for(ow = 0; ow < osizeW; ow++)
          {

            int64_t istartW = start_index(ow, osizeW, isizeW);
            int64_t iendW   = end_index(ow, osizeW, isizeW);
            int64_t kW = iendW - istartW;

            /* local pointers */
            scalar_t *ip = input_p   + d*istrideD + istartT *istrideT + istartH*istrideH + istartW*istrideW;
            scalar_t *op = output_p  + d*osizeT*osizeH*osizeW + ot*osizeH*osizeW + oh*osizeW + ow;
            int64_t *indp = ind_p   + d*osizeT*osizeH*osizeW + ot*osizeH*osizeW + oh*osizeW + ow;

            /* compute local max: */
            int64_t it = 0, ih = 0, iw = 0;
            int64_t maxindex = (it+istartT)*isizeH*isizeW + (ih+istartH)*isizeW + (iw+istartW);
            scalar_t maxval = -std::numeric_limits<scalar_t>::infinity();
            for(it = 0; it < kT; it++)
            {
              for(ih = 0; ih < kH; ih++)
              {
                for(iw = 0; iw < kW; iw++)
                {
                  scalar_t val = *(ip + it*istrideT + ih*istrideH + iw*istrideW);
                  if ((val > maxval) || std::isnan(val))
                  {
                    maxval = val;
                    maxindex = (it+istartT)*isizeH*isizeW + (ih+istartH)*isizeW + (iw+istartW);
                  }
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
    }
  });
}

template <typename scalar_t>
static void adaptive_max_pool3d_out_frame(
          scalar_t *input_data,
          scalar_t *output_data,
          int64_t *indices_data,
          int64_t sizeB,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW,
          int64_t istrideB,
          int64_t istrideD,
          int64_t istrideT,
          int64_t istrideH,
          int64_t istrideW)
{
  at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
    for (auto b = start; b < end; b++)
    {
      adaptive_max_pool3d_single_out_frame<scalar_t>(input_data+b*istrideB, output_data+b*sizeD*osizeT*osizeH*osizeW,
                                                     indices_data+b*sizeD*osizeT*osizeH*osizeW,
                                                     sizeD,
                                                     isizeT, isizeH, isizeW,
                                                     osizeT, osizeH, osizeW,
                                                     istrideD, istrideT,
                                                     istrideH, istrideW);
    }
  });
}

void adaptive_max_pool3d_out_cpu_template(
          Tensor& output,
          Tensor& indices,
          const Tensor& input,
          IntArrayRef output_size)
{
  int dimD = 0;
  int dimT = 1;
  int dimH = 2;
  int dimW = 3;
  int64_t sizeB = 1;
  int64_t sizeD = 0;
  int64_t isizeT = 0;
  int64_t isizeH = 0;
  int64_t isizeW = 0;

  int64_t istrideB = 0;
  int64_t istrideD = 0;
  int64_t istrideT = 0;
  int64_t istrideH = 0;
  int64_t istrideW = 0;

  for (int64_t i = 0; i < input.ndimension(); i++) {
    TORCH_CHECK(input.size(i) > 0,
      "adaptive_max_pool3d: expected input to have non-empty spatial dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i, " being "
      "empty");
  }

  TORCH_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "non-empty 4D or 5D (batch mode) tensor expected for input");

  TORCH_CHECK(output_size.size() == 3,
    "adaptive_max_pool3d: internal error: output_size.size() must be 3");

  if (input.ndimension() == 5)
  {
    istrideB = input.stride(0);
    sizeB = input.size(0);
    dimD++;
    dimT++;
    dimH++;
    dimW++;
  }

  /* sizes */
  sizeD  = input.size(dimD);
  isizeT = input.size(dimT);
  isizeH = input.size(dimH);
  isizeW = input.size(dimW);
  /* strides */
  istrideD = input.stride(dimD);
  istrideT = input.stride(dimT);
  istrideH = input.stride(dimH);
  istrideW = input.stride(dimW);

  int64_t osizeT = output_size[0];
  int64_t osizeH = output_size[1];
  int64_t osizeW = output_size[2];

  /* resize output */
  if (input.ndimension() == 4)
  {
    output.resize_({sizeD, osizeT, osizeH, osizeW});
    /* indices will contain max input locations for each output point */
    indices.resize_({sizeD, osizeT, osizeH, osizeW});

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cpu", [&] {
      auto input_data = input.data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();
      auto indices_data = indices.data_ptr<int64_t>();

      adaptive_max_pool3d_single_out_frame<scalar_t>(input_data, output_data,
                                                     indices_data,
                                                     sizeD,
                                                     isizeT, isizeH, isizeW,
                                                     osizeT, osizeH, osizeW,
                                                     istrideD, istrideT,
                                                     istrideH, istrideW);
      }
    );
  }
  else
  {
    output.resize_({sizeB, sizeD, osizeT, osizeH, osizeW});
    /* indices will contain max input locations for each output point */
    indices.resize_({sizeB, sizeD, osizeT, osizeH, osizeW});

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_max_pool3d_cpu", [&] {
      auto input_data = input.data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();
      auto indices_data = indices.data_ptr<int64_t>();

      adaptive_max_pool3d_out_frame<scalar_t>(input_data, output_data,
                                              indices_data,
                                              sizeB,
                                              sizeD,
                                              isizeT, isizeH, isizeW,
                                              osizeT, osizeH, osizeW,
                                              istrideB,
                                              istrideD, istrideT,
                                              istrideH, istrideW);
      }
    );
  }
}

template <typename scalar_t>
static void adaptive_max_pool3d_backward_single_out_frame(
          scalar_t *gradInput_p,
          scalar_t *gradOutput_p,
          int64_t *ind_p,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW)
{
  at::parallel_for(0, sizeD, 0, [&](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++)
    {
      scalar_t *gradInput_p_d = gradInput_p + d*isizeT*isizeH*isizeW;
      scalar_t *gradOutput_p_d = gradOutput_p + d*osizeT*osizeH*osizeW;
      int64_t *ind_p_d = ind_p + d*osizeT*osizeH*osizeW;

      /* calculate max points */
      int64_t ot, oh, ow;
      for(ot = 0; ot < osizeT; ot++)
      {
        for(oh = 0; oh < osizeH; oh++)
        {
          for(ow = 0; ow < osizeW; ow++)
          {
            /* retrieve position of max */
            int64_t maxp = ind_p_d[ot*osizeH*osizeW + oh*osizeW + ow];

            /* update gradient */
            gradInput_p_d[maxp] += gradOutput_p_d[ot*osizeH*osizeW + oh*osizeW + ow];
          }
        }
      }
    }
  });
}

template <typename scalar_t>
static void adaptive_max_pool3d_backward_out_frame(
          scalar_t *gradInput_data,
          scalar_t *gradOutput_data,
          int64_t *indices_data,
          int64_t sizeB,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW)
{
  at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
    for (auto b = start; b < end; b++)
    {
      adaptive_max_pool3d_backward_single_out_frame<scalar_t>(gradInput_data+b*sizeD*isizeT*isizeH*isizeW, gradOutput_data+b*sizeD*osizeT*osizeH*osizeW,
                                                              indices_data+b*sizeD*osizeT*osizeH*osizeW,
                                                              sizeD,
                                                              isizeT, isizeH, isizeW,
                                                              osizeT, osizeH, osizeW);
    }
  });
}

Tensor& adaptive_max_pool3d_backward_out_cpu_template(
          Tensor& gradInput,
          const Tensor& gradOutput_,
          const Tensor& input,
          const Tensor& indices)
{
  int dimD = 0;
  int dimT = 1;
  int dimH = 2;
  int dimW = 3;
  int64_t sizeB = 1;
  int64_t sizeD;
  int64_t isizeT;
  int64_t isizeH;
  int64_t isizeW;
  int64_t osizeT;
  int64_t osizeH;
  int64_t osizeW;

  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  /* resize */
  gradInput.resize_as_(input);
  gradInput.zero_();

  if (input.ndimension() == 5) {
    sizeB = input.size(0);
    dimD++;
    dimT++;
    dimH++;
    dimW++;
  }

  /* sizes */
  sizeD  = input.size(dimD);
  isizeT = input.size(dimT);
  isizeH = input.size(dimH);
  isizeW = input.size(dimW);
  osizeT = gradOutput.size(dimT);
  osizeH = gradOutput.size(dimH);
  osizeW = gradOutput.size(dimW);

  /* backprop */
  if (input.ndimension() == 4)
  {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "adaptive_max_pool3d_backward",
      [&] {
        /* get raw pointers */
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
        scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        adaptive_max_pool3d_backward_single_out_frame<scalar_t>(gradInput_data, gradOutput_data,
                                                                indices_data,
                                                                sizeD,
                                                                isizeT, isizeH, isizeW,
                                                                osizeT, osizeH, osizeW);
      }
    );
  }
  else
  {
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(),
      "adaptive_max_pool3d_backward",
      [&] {
        /* get raw pointers */
        scalar_t *gradInput_data = gradInput.data_ptr<scalar_t>();
        scalar_t *gradOutput_data = gradOutput.data_ptr<scalar_t>();
        int64_t *indices_data = indices.data_ptr<int64_t>();

        adaptive_max_pool3d_backward_out_frame<scalar_t>(gradInput_data, gradOutput_data,
                                                         indices_data,
                                                         sizeB,
                                                         sizeD,
                                                         isizeT, isizeH, isizeW,
                                                         osizeT, osizeH, osizeW);
      }
    );
  }

  return gradInput;
}

} // namespace

std::tuple<Tensor&, Tensor&> adaptive_max_pool3d_out_cpu(const Tensor& input,
  IntArrayRef output_size,
  Tensor& output,
  Tensor& indices)
{
  adaptive_max_pool3d_out_cpu_template(
    output,
    indices,
    input,
    output_size);
  return std::tuple<Tensor&, Tensor&>(output, indices);
}

std::tuple<Tensor, Tensor> adaptive_max_pool3d_cpu(
  const Tensor& input,
  IntArrayRef output_size)
{
  Tensor output = at::empty({0}, input.options());
  Tensor indices = at::empty({0}, input.options().dtype(kLong));
  adaptive_max_pool3d_out_cpu_template(
    output,
    indices,
    input,
    output_size);
  return std::tuple<Tensor, Tensor>(output, indices);
}

Tensor& adaptive_max_pool3d_backward_out_cpu(const Tensor& gradOutput_,
  const Tensor& input,
  const Tensor& indices,
  Tensor& gradInput)
{
  adaptive_max_pool3d_backward_out_cpu_template(
    gradInput,
    gradOutput_,
    input,
    indices);
  return gradInput;
}

Tensor adaptive_max_pool3d_backward_cpu(
  const Tensor& gradOutput_,
  const Tensor& input,
  const Tensor& indices)
{
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  adaptive_max_pool3d_backward_out_cpu_template(
    gradInput,
    gradOutput_,
    input,
    indices);
  return gradInput;
}

} // at::native
} // at
