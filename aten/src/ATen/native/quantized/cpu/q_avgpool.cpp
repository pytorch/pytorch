#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/core/op_registration/op_registration.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace at { namespace native {
namespace {

inline int start_index(int a, int b, int c) {
  return (int)std::floor((float)(a * c) / b);
}

inline int end_index(int a, int b, int c) {
  return (int)std::ceil((float)((a + 1) * c) / b);
}


template <typename scalar_t, typename underlying_t>
static void adaptive_avg_pool2d_single_out_frame(scalar_t *input_p,
                                                 scalar_t *output_p,
                                                 int64_t sizeD,
                                                 int64_t isizeH,
                                                 int64_t isizeW,
                                                 int64_t osizeH,
                                                 int64_t osizeW,
                                                 int64_t istrideD,
                                                 int64_t istrideH,
                                                 int64_t istrideW) {
  at::parallel_for(0, sizeD, 0, [&](int64_t start, int64_t end) {
    for (auto d = start; d < end; d++) {
      /* loop over output */
      int64_t oh, ow;
      for(oh = 0; oh < osizeH; oh++) {
        int istartH = start_index(oh, osizeH, isizeH);
        int iendH   = end_index(oh, osizeH, isizeH);
        int kH = iendH - istartH;
        float kHr = 1.0 / kH;

        for(ow = 0; ow < osizeW; ow++) {
          int istartW = start_index(ow, osizeW, isizeW);
          int iendW   = end_index(ow, osizeW, isizeW);
          int kW = iendW - istartW;
          float kHWr = kHr / kW;

          /* local pointers */
          scalar_t *ip = input_p   + d*istrideD + istartH*istrideH + istartW*istrideW;
          scalar_t *op = output_p  + d*osizeH*osizeW + oh*osizeW + ow;

          /* compute local average: */
          int64_t sum = 0;
          int ih, iw;
          for(ih = 0; ih < kH; ih++) {
            for(iw = 0; iw < kW; iw++) {
              int64_t val = (ip + ih*istrideH + iw*istrideW)->val_;
              sum += val;
            }
          }

          /* set output to local average */
          op->val_ = static_cast<underlying_t>(std::nearbyint(sum * kHWr));
        }
      }
    }
  });
}

template <typename scalar_t, typename underlying_t>
void adaptive_avg_pool2d_out_frame(scalar_t *input_p,
                                   scalar_t *output_p,
                                   int64_t sizeB,
                                   int64_t sizeD,
                                   int64_t isizeH,
                                   int64_t isizeW,
                                   int64_t osizeH,
                                   int64_t osizeW,
                                   int64_t istrideB,
                                   int64_t istrideD,
                                   int64_t istrideH,
                                   int64_t istrideW) {
  at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
    for (auto b = start; b < end; b++) {
      adaptive_avg_pool2d_single_out_frame<scalar_t, underlying_t>(
        input_p + b * istrideB,
        output_p + b * sizeD * osizeH * osizeW,
        sizeD,
        isizeH, isizeW,
        osizeH, osizeW,
        istrideD,
        istrideH, istrideW);
    }
  });
}

void adaptive_avg_pool2d_out_template(Tensor& output, Tensor input,
                                      IntArrayRef output_size) {
  for (int64_t i = 0; i < input.dim(); i++) {
    TORCH_CHECK(input.size(i) > 0,
      "adaptive_avg_pooling2d(): expected input to have non-empty spatial "
      "dimensions, but input has sizes ", input.sizes(), " with dimension ", i,
      " being empty");
  }

  TORCH_CHECK((input.dim() == 3 || input.dim() == 4),
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
  std::vector<int64_t> output_sizes;
  int64_t sizeB = 0;
  if (input.dim() == 3) {
    output_sizes = {sizeD, osizeH, osizeW};
  } else {
    sizeB = input.size(-4);
    output_sizes = {sizeB, sizeD, osizeH, osizeW};
  }
  if (!output.is_quantized() || output.sizes() != output_sizes) {
    output = at::_empty_affine_quantized(output_sizes, input.options(),
                                         input.q_scale(), input.q_zero_point());
  }

  if (input.dim() == 3 || input.size(0) == 1) {
    AT_DISPATCH_QINT_TYPES(input.scalar_type(),
      "quantized_adaptive_avg_pool2d", [&] {
        auto input_data = input.data<scalar_t>();
        auto output_data = output.data<scalar_t>();
        adaptive_avg_pool2d_single_out_frame<scalar_t, underlying_t>(
          input_data,
          output_data,
          sizeD,
          isizeH, isizeW,
          osizeH, osizeW,
          istrideD,
          istrideH, istrideW);
      }
    );
  } else {
    int64_t istrideB = input.stride(-4);

    AT_DISPATCH_QINT_TYPES(input.scalar_type(),
      "quantized_adaptive_avg_pool2d", [&] {
        auto input_data = input.data<scalar_t>();
        auto output_data = output.data<scalar_t>();
        adaptive_avg_pool2d_out_frame<scalar_t, underlying_t>(
          input_data,
          output_data,
          sizeB,
          sizeD,
          isizeH, isizeW,
          osizeH, osizeW,
          istrideB,
          istrideD,
          istrideH, istrideW);
      }
    );
  }
}

class QAdaptiveAvgPool2D final : public torch::OperatorKernel {
 public:
  Tensor operator()(Tensor qx, std::vector<int64_t> output_size) {
    return at::adaptive_avg_pool2d(qx, output_size);
  }
};

static auto registry = torch::RegisterOperators().op(
  "quantized::adaptive_avg_pool2d(Tensor qx, int[] output_size) -> Tensor",
  torch::RegisterOperators::options()
    .kernel<QAdaptiveAvgPool2D>(QuantizedCPUTensorId()));

}  // namespace


Tensor& quantized_adaptive_avg_pool2d_out(Tensor& output, const Tensor& input,
                                          IntArrayRef output_size) {
  adaptive_avg_pool2d_out_template(output, input, output_size);
  return output;
}

Tensor quantized_adaptive_avg_pool2d(const at::Tensor& input,
                                     IntArrayRef output_size) {
  Tensor output;
  adaptive_avg_pool2d_out_template(output, input, output_size);
  return output;
}

}}  // namespace at::native
