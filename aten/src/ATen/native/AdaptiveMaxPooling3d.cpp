#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>
#include <tuple>

#include <ATen/native/AdaptivePooling.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_max_pool3d_backward_native.h>
#include <ATen/ops/adaptive_max_pool3d_native.h>
#endif

namespace at::meta {
TORCH_META_FUNC(adaptive_max_pool3d) (const Tensor& input, IntArrayRef output_size) {
  auto ndim = input.ndimension();
  TORCH_CHECK(
    ndim == 4 || ndim == 5,
    "adaptive_max_pool3d(): Expected 4D or 5D tensor, but got: ", input.sizes());
  for (const auto i : c10::irange(1, ndim)) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_max_pool3d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      output_size.size() == 3,
      "adaptive_max_pool3d(): internal error: output_size.size() must be 3");

  int dimD = 0;
  int64_t sizeB = 1;
  int64_t sizeD = 0;

  if (ndim == 5) {
    sizeB = input.size(0);
    dimD++;
  }

  /* sizes */
  sizeD = input.size(dimD);

  int64_t osizeT = output_size[0];
  int64_t osizeH = output_size[1];
  int64_t osizeW = output_size[2];

  /* resize output */
  if (ndim == 4) {
    set_output_raw_strided(0, {sizeD, osizeT, osizeH, osizeW}, {}, input.options());
    /* indices will contain max input locations for each output point */
    set_output_raw_strided(1, {sizeD, osizeT, osizeH, osizeW}, {}, input.options().dtype(kLong));
  } else {
    set_output_raw_strided(0, {sizeB, sizeD, osizeT, osizeH, osizeW}, {}, input.options());
    /* indices will contain max input locations for each output point */
    set_output_raw_strided(1, {sizeB, sizeD, osizeT, osizeH, osizeW}, {}, input.options().dtype(kLong));
  }
}

TORCH_META_FUNC(adaptive_max_pool3d_backward)
(const Tensor& gradOutput, const Tensor& input, const Tensor& indices) {
    at::native::adaptive_pool_empty_output_check(gradOutput, "adaptive_max_pool3d_backward");
    set_output_raw_strided(0, input.sizes(), {}, input.options());
}
} // namespace meta

namespace at::native {

namespace {

// #define START_IND(a,b,c) a * c / b
// #define END_IND(a,b,c)  (a + 1) * c / b + ((a + 1) * c % b > 0)?1:0

// 5d tensor B x D x T x H x W

template <typename scalar_t>
static void adaptive_max_pool3d_single_out_frame(
          const scalar_t *input_p,
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
    for (const auto d : c10::irange(start, end)) {
      /* loop over output */
      int64_t ot = 0, oh = 0, ow = 0;
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
            const scalar_t *ip = input_p   + d*istrideD + istartT *istrideT + istartH*istrideH + istartW*istrideW;
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
          const scalar_t *input_data,
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
    for (const auto b : c10::irange(start, end)) {
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

template <typename scalar_t>
static void adaptive_max_pool3d_backward_single_out_frame(
          scalar_t *gradInput_p,
          const scalar_t *gradOutput_p,
          const int64_t *ind_p,
          int64_t sizeD,
          int64_t isizeT,
          int64_t isizeH,
          int64_t isizeW,
          int64_t osizeT,
          int64_t osizeH,
          int64_t osizeW)
{
  at::parallel_for(0, sizeD, 0, [&](int64_t start, int64_t end) {
    for (const auto d : c10::irange(start, end)) {
      scalar_t *gradInput_p_d = gradInput_p + d*isizeT*isizeH*isizeW;
      const scalar_t *gradOutput_p_d = gradOutput_p + d*osizeT*osizeH*osizeW;
      const int64_t *ind_p_d = ind_p + d*osizeT*osizeH*osizeW;

      /* calculate max points */
      int64_t ot = 0, oh = 0, ow = 0;
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
          const scalar_t *gradOutput_data,
          const int64_t *indices_data,
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
    for (const auto b : c10::irange(start, end)) {
      adaptive_max_pool3d_backward_single_out_frame<scalar_t>(gradInput_data+b*sizeD*isizeT*isizeH*isizeW, gradOutput_data+b*sizeD*osizeT*osizeH*osizeW,
                                                              indices_data+b*sizeD*osizeT*osizeH*osizeW,
                                                              sizeD,
                                                              isizeT, isizeH, isizeW,
                                                              osizeT, osizeH, osizeW);
    }
  });
}
} // namespace

TORCH_IMPL_FUNC(adaptive_max_pool3d_out_cpu)
(const Tensor& input, IntArrayRef output_size, const Tensor& output, const Tensor& indices) {
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

  if (input.ndimension() == 5) {
    istrideB = input.stride(0);
    sizeB = input.size(0);
    dimD++;
    dimT++;
    dimH++;
    dimW++;
  }

  /* sizes */
  sizeD = input.size(dimD);
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

  if (input.ndimension() == 4) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf,
        input.scalar_type(), "adaptive_max_pool3d_cpu", [&] {
          auto input_data = input.const_data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          auto indices_data = indices.data_ptr<int64_t>();

          adaptive_max_pool3d_single_out_frame<scalar_t>(
              input_data,
              output_data,
              indices_data,
              sizeD,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW,
              istrideD,
              istrideT,
              istrideH,
              istrideW);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf,
        input.scalar_type(), "adaptive_max_pool3d_cpu", [&] {
          auto input_data = input.const_data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          auto indices_data = indices.data_ptr<int64_t>();

          adaptive_max_pool3d_out_frame<scalar_t>(
              input_data,
              output_data,
              indices_data,
              sizeB,
              sizeD,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW,
              istrideB,
              istrideD,
              istrideT,
              istrideH,
              istrideW);
        });
  }
}

TORCH_IMPL_FUNC(adaptive_max_pool3d_backward_out_cpu)
(const Tensor& gradOutput,
 const Tensor& input,
 const Tensor& indices,
 const Tensor& gradInput) {
  int dimD = 0;
  int dimT = 1;
  int dimH = 2;
  int dimW = 3;
  int64_t sizeB = 1;
  int64_t sizeD = 0;
  int64_t isizeT = 0;
  int64_t isizeH = 0;
  int64_t isizeW = 0;
  int64_t osizeT = 0;
  int64_t osizeH = 0;
  int64_t osizeW = 0;

  /* get contiguous gradOutput */
  auto gradOutput_ = gradOutput.contiguous();

  /* resize */
  gradInput.zero_();

  if (input.ndimension() == 5) {
    sizeB = input.size(0);
    dimD++;
    dimT++;
    dimH++;
    dimW++;
  }

  /* sizes */
  sizeD = input.size(dimD);
  isizeT = input.size(dimT);
  isizeH = input.size(dimH);
  isizeW = input.size(dimW);
  osizeT = gradOutput_.size(dimT);
  osizeH = gradOutput_.size(dimH);
  osizeW = gradOutput_.size(dimW);

  /* backprop */
  if (input.ndimension() == 4) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf,
        input.scalar_type(), "adaptive_max_pool3d_backward", [&] {
          /* get raw pointers */
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          const scalar_t* gradOutput_data = gradOutput_.const_data_ptr<scalar_t>();
          const int64_t* indices_data = indices.const_data_ptr<int64_t>();

          adaptive_max_pool3d_backward_single_out_frame<scalar_t>(
              gradInput_data,
              gradOutput_data,
              indices_data,
              sizeD,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW);
        });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf,
        input.scalar_type(), "adaptive_max_pool3d_backward", [&] {
          /* get raw pointers */
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          const scalar_t* gradOutput_data = gradOutput_.const_data_ptr<scalar_t>();
          const int64_t* indices_data = indices.const_data_ptr<int64_t>();

          adaptive_max_pool3d_backward_out_frame<scalar_t>(
              gradInput_data,
              gradOutput_data,
              indices_data,
              sizeB,
              sizeD,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW);
        });
  }
}
} // namespace at::native
