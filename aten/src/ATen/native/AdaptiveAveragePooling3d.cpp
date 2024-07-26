#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <c10/util/irange.h>

#include <ATen/native/AdaptivePooling.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_adaptive_avg_pool3d.h>
#include <ATen/ops/_adaptive_avg_pool3d_backward_native.h>
#include <ATen/ops/_adaptive_avg_pool3d_native.h>
#include <ATen/ops/adaptive_avg_pool3d_backward_native.h>
#include <ATen/ops/adaptive_avg_pool3d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#endif

namespace at::native {

namespace {

template <typename scalar_t>
static void adaptive_avg_pool3d_out_frame(
    const scalar_t* input_p,
    scalar_t* output_p,
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
    int64_t istrideW) {
  at::parallel_for(0, sizeD, 1, [&](int64_t start, int64_t end) {
    for (const auto d : c10::irange(start, end)) {
      /* loop over output */
      for (const auto ot : c10::irange(osizeT)) {
        int istartT = start_index(ot, osizeT, isizeT);
        int iendT = end_index(ot, osizeT, isizeT);
        int kT = iendT - istartT;

        for (const auto oh : c10::irange(osizeH)) {
          int istartH = start_index(oh, osizeH, isizeH);
          int iendH = end_index(oh, osizeH, isizeH);
          int kH = iendH - istartH;

          for (const auto ow : c10::irange(osizeW)) {
            int istartW = start_index(ow, osizeW, isizeW);
            int iendW = end_index(ow, osizeW, isizeW);
            int kW = iendW - istartW;

            /* local pointers */
            const scalar_t* ip = input_p + d * istrideD + istartT * istrideT +
                istartH * istrideH + istartW * istrideW;
            scalar_t* op = output_p + d * osizeT * osizeH * osizeW +
                ot * osizeH * osizeW + oh * osizeW + ow;

            /* compute local average: */
            scalar_t sum = 0;
            for (const auto it : c10::irange(kT)) {
              for (const auto ih : c10::irange(kH)) {
                for (const auto iw : c10::irange(kW)) {
                  scalar_t val =
                      *(ip + it * istrideT + ih * istrideH + iw * istrideW);
                  sum += val;
                }
              }
            }

            /* set output to local average */
            *op = sum / kT / kH / kW;
          }
        }
      }
    }
  });
}

void adaptive_avg_pool3d_out_cpu_template(
    Tensor& output,
    Tensor const& input,
    IntArrayRef output_size) {
  TORCH_CHECK(output_size.size() == 3, "adaptive_avg_pool3d: output_size must be 3");

  for (const auto i : c10::irange(1, input.ndimension())) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pool3d(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being "
        "empty");
  }

  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "adaptive_avg_pool3d(): Expected 4D or 5D tensor, but got ",
      input.sizes());
  TORCH_CHECK(input.dtype() == output.dtype(),
      "expected dtype ", input.dtype(), " for `output` but got dtype ", output.dtype());

  /* sizes */
  int64_t sizeD = input.size(-4);
  int64_t isizeT = input.size(-3);
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  /* strides */
  int64_t istrideD = input.stride(-4);
  int64_t istrideT = input.stride(-3);
  int64_t istrideH = input.stride(-2);
  int64_t istrideW = input.stride(-1);
  /* output sizes */
  auto osizeT = output_size[0];
  auto osizeH = output_size[1];
  auto osizeW = output_size[2];

  if (input.ndimension() == 4) {
    output.resize_({sizeD, osizeT, osizeH, osizeW});

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
        input.scalar_type(), "adaptive_avg_pool3d_cpu", [&] {
          auto input_data = input.const_data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          adaptive_avg_pool3d_out_frame<scalar_t>(
              input_data,
              output_data,
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
    output.resize_({input.size(-5), sizeD, osizeT, osizeH, osizeW});
    int64_t n = input.size(0);

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
        input.scalar_type(), "adaptive_avg_pool3d_cpu", [&] {
          auto input_data = input.const_data_ptr<scalar_t>();
          auto output_data = output.data_ptr<scalar_t>();
          at::parallel_for(0, n, 1, [&](int64_t start, int64_t end) {
            for (const auto b : c10::irange(start, end)) {
              adaptive_avg_pool3d_out_frame<scalar_t>(
                  input_data + b * input.stride(0),
                  output_data + b * sizeD * osizeT * osizeH * osizeW,
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
            }
          });
    });
  }
}

template <typename scalar_t>
static void adaptive_avg_pool3d_backward_out_frame(
    scalar_t* gradInput_p,
    const scalar_t* gradOutput_p,
    int64_t sizeD,
    int64_t isizeT,
    int64_t isizeH,
    int64_t isizeW,
    int64_t osizeT,
    int64_t osizeH,
    int64_t osizeW) {
  at::parallel_for(0, sizeD, 1, [&](int64_t start, int64_t end) {
    for (const auto d : c10::irange(start, end)) {
      scalar_t* gradInput_p_d = gradInput_p + d * isizeT * isizeW * isizeH;
      const scalar_t* gradOutput_p_d = gradOutput_p + d * osizeT * osizeW * osizeH;

      /* calculate average */
      for (const auto ot : c10::irange(osizeT)) {
        int istartT = start_index(ot, osizeT, isizeT);
        int iendT = end_index(ot, osizeT, isizeT);
        int kT = iendT - istartT;

        for (const auto oh : c10::irange(osizeH)) {
          int istartH = start_index(oh, osizeH, isizeH);
          int iendH = end_index(oh, osizeH, isizeH);
          int kH = iendH - istartH;

          for (const auto ow : c10::irange(osizeW)) {
            int istartW = start_index(ow, osizeW, isizeW);
            int iendW = end_index(ow, osizeW, isizeW);
            int kW = iendW - istartW;

            scalar_t grad_delta =
                gradOutput_p_d[ot * osizeH * osizeW + oh * osizeW + ow] / kT /
                kH / kW;

            for (const auto it : c10::irange(istartT, iendT)) {
              for (const auto ih : c10::irange(istartH, iendH)) {
                for (const auto iw : c10::irange(istartW, iendW)) {
                  /* update gradient */
                  gradInput_p_d[it * isizeH * isizeW + ih * isizeW + iw] +=
                      grad_delta;
                }
              }
            }
          }
        }
      }
    }
  });
}

Tensor& adaptive_avg_pool3d_backward_out_cpu_template(
    Tensor& gradInput,
    const Tensor& gradOutput_,
    const Tensor& input) {
  /* get contiguous gradOutput */
  auto gradOutput = gradOutput_.contiguous();

  adaptive_pool_empty_output_check(gradOutput_, "adaptive_avg_pool3d_backward");

  /* sizes */
  int64_t sizeD = input.size(-4);
  int64_t isizeT = input.size(-3);
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  int64_t osizeT = gradOutput.size(-3);
  int64_t osizeH = gradOutput.size(-2);
  int64_t osizeW = gradOutput.size(-1);

  /* backprop */
  if (input.ndimension() == 4) {
    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
        input.scalar_type(), "adaptive_avg_pool3d_backward_cpu", [&] {
          /* get raw pointers */
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          const scalar_t* gradOutput_data = gradOutput.const_data_ptr<scalar_t>();

          adaptive_avg_pool3d_backward_out_frame<scalar_t>(
              gradInput_data,
              gradOutput_data,
              sizeD,
              isizeT,
              isizeH,
              isizeW,
              osizeT,
              osizeH,
              osizeW);
        });
  } else {
    int64_t n = input.size(0);

    AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
        input.scalar_type(), "adaptive_avg_pool3d_backward_cpu", [&] {
          /* get raw pointers */
          scalar_t* gradInput_data = gradInput.data_ptr<scalar_t>();
          const scalar_t* gradOutput_data = gradOutput.const_data_ptr<scalar_t>();
          at::parallel_for(0, n, 1, [&](int64_t start, int64_t end) {
            for (const auto b : c10::irange(start, end)) {
              adaptive_avg_pool3d_backward_out_frame<scalar_t>(
                  gradInput_data + b * sizeD * isizeT * isizeH * isizeW,
                  gradOutput_data + b * sizeD * osizeT * osizeH * osizeW,
                  sizeD,
                  isizeT,
                  isizeH,
                  isizeW,
                  osizeT,
                  osizeH,
                  osizeW);
            }
          });
    });
  }
  return gradInput;
}

} // namespace

Tensor& adaptive_avg_pool3d_out_cpu(const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  adaptive_avg_pool3d_out_cpu_template(output, input, output_size);
  return output;
}

Tensor adaptive_avg_pool3d_cpu(Tensor const& input, IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  adaptive_avg_pool3d_out_cpu_template(output, input, output_size);
  return output;
}

Tensor adaptive_avg_pool3d_symint(Tensor const& input, SymIntArrayRef output_size) {
  TORCH_CHECK(output_size.size() == 3, "adaptive_avg_pool3d: output_size must be 3");
  TORCH_CHECK(
        (output_size[0] >= 0 && output_size[1] >= 0 && output_size[2] >= 0),
        "adaptive_avg_pool3d: elements of output_size must be greater than or equal to 0 ",
        "but received {", output_size[0], ", ", output_size[1], ",", output_size[2], "}");

  if (output_size[0] == 1 && output_size[1] == 1 && output_size[2] == 1 && !input.is_xpu()) {
    // in this case, adaptive pooling is just computing mean over hw
    // dimensions, which can be done more efficiently
    Tensor out = input.mean({-1, -2, -3}, /* keepdim = */ true);
    if (input.suggest_memory_format() == at::MemoryFormat::ChannelsLast3d) {
      // assert ndim == 5, since ndim = 4 doesn't give channels_last
      const auto n = input.sym_size(0);
      const auto c = input.sym_size(1);
      out.as_strided__symint({n, c, 1, 1, 1}, {c, 1, c, c, c});
    }
    return out;
  } else {
    return _adaptive_avg_pool3d_symint(input, output_size);
  }
}

Tensor& adaptive_avg_pool3d_backward_out_cpu(const Tensor& gradOutput_,
    const Tensor& input,
    Tensor& gradInput) {
  gradInput.resize_as_(input).zero_();
  adaptive_avg_pool3d_backward_out_cpu_template(gradInput, gradOutput_, input);
  return gradInput;
}

Tensor adaptive_avg_pool3d_backward_cpu(const Tensor& gradOutput_,
    const Tensor& input) {
  auto gradInput = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  adaptive_avg_pool3d_backward_out_cpu_template(gradInput, gradOutput_, input);
  return gradInput;
}

} // namespace at::native
