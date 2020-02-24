#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <ATen/native/quantized/cpu/quantized_ops.h>
#include <c10/util/math_compat.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace at {
namespace native {

DEFINE_DISPATCH(qadaptive_avg_pool2d_nhwc_stub);

namespace {

inline int start_index(int out_idx, int out_len, int in_len) {
  /*
   * out_idx: the current index of output matrix
   * out_len: the dimension_size of output matrix
   * in_len: the dimension_size of input matrix
   * Basically, in_len / out_len gives the number of
   * elements in each average computation.
   * This function computes the start index on input matrix.
   */
  return (int)std::floor((float)(out_idx * in_len) / out_len);
}

inline int end_index(int out_idx, int out_len, int in_len) {
  /*
   * Parameter definition is the same as start_index.
   * This function computes the end index on input matrix.
   */
  return (int)std::ceil((float)((out_idx + 1) * in_len) / out_len);
}

template <typename scalar_t>
static void adaptive_avg_pool2d_single_out_frame(
    scalar_t* input_p,
    scalar_t* output_p,
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
      for (oh = 0; oh < osizeH; oh++) {
        int istartH = start_index(oh, osizeH, isizeH);
        int iendH = end_index(oh, osizeH, isizeH);
        int kH = iendH - istartH;
        float kHr = 1.0 / kH;

        for (ow = 0; ow < osizeW; ow++) {
          int istartW = start_index(ow, osizeW, isizeW);
          int iendW = end_index(ow, osizeW, isizeW);
          int kW = iendW - istartW;
          float kHWr = kHr / kW;

          /* local pointers */
          scalar_t* ip =
              input_p + d * istrideD + istartH * istrideH + istartW * istrideW;
          scalar_t* op = output_p + d * osizeH * osizeW + oh * osizeW + ow;

          /* compute local average: */
          int64_t sum = 0;
          int ih, iw;
          for (ih = 0; ih < kH; ih++) {
            for (iw = 0; iw < kW; iw++) {
              int64_t val = (ip + ih * istrideH + iw * istrideW)->val_;
              sum += val;
            }
          }

          /* set output to local average */
          // TODO: add the max/min clip
          op->val_ = static_cast<typename scalar_t::underlying>(
              std::nearbyint(sum * kHWr));
        }
      }
    }
  });
}

std::vector<int64_t> get_output_shape(
    const Tensor& input,
    IntArrayRef output_size) {
  for (int64_t i = 0; i < input.dim(); i++) {
    TORCH_CHECK(
        input.size(i) > 0,
        "adaptive_avg_pooling2d(): expected input to have non-empty spatial "
        "dimensions, but input has sizes ",
        input.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  TORCH_CHECK(
      (input.dim() == 3 || input.dim() == 4),
      "non-empty 3D or 4D (batch mode) tensor expected for input");

  /* sizes */
  int64_t sizeD = input.size(-3);
  const auto osizeH = output_size[0];
  const auto osizeW = output_size[1];

  /* resize output */
  std::vector<int64_t> output_shape;
  int64_t sizeB = 0;
  if (input.dim() == 3) {
    output_shape = {sizeD, osizeH, osizeW};
  } else {
    sizeB = input.size(-4);
    output_shape = {sizeB, sizeD, osizeH, osizeW};
  }

  return output_shape;
}

template <typename scalar_t>
Tensor q_adaptive_avg_pool2d(const Tensor& input, IntArrayRef output_size) {
  Tensor output;
  const auto output_shape = get_output_shape(input, output_size);
  /* sizes */
  int64_t sizeD = input.size(-3);
  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  /* strides */
  int64_t istrideD = input.stride(-3);
  int64_t istrideH = input.stride(-2);
  int64_t istrideW = input.stride(-1);

  auto osizeH = output_shape[output_shape.size() - 2];
  auto osizeW = output_shape[output_shape.size() - 1];
  int64_t sizeB = output_shape.size() == 3 ? 0 : output_shape[0];

  if (input.is_contiguous(c10::MemoryFormat::ChannelsLast)) {
    // Fast path for NHWC
    Tensor output = at::_empty_affine_quantized(
        output_shape,
        input.options().memory_format(input.suggest_memory_format()),
        input.q_scale(),
        input.q_zero_point());
    if (input.dim() == 3 || input.size(0) == 1) {
      qadaptive_avg_pool2d_nhwc_stub(
          input.device().type(),
          input,
          output,
          0,
          sizeD,
          isizeH,
          isizeW,
          osizeH,
          osizeW,
          0,
          istrideD,
          istrideH,
          istrideW);
    } else {
      int64_t istrideB = input.stride(-4);
      at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
        for (auto b = start; b < end; b++) {
          qadaptive_avg_pool2d_nhwc_stub(
              input.device().type(),
              input,
              output,
              b,
              sizeD,
              isizeH,
              isizeW,
              osizeH,
              osizeW,
              istrideB,
              istrideD,
              istrideH,
              istrideW);
        }
      });
    }
    return output;
  } else {
    Tensor output = at::_empty_affine_quantized(
        output_shape, input.options(), input.q_scale(), input.q_zero_point());
    auto input_contig = input.contiguous();
    auto input_data = input_contig.data_ptr<scalar_t>();
    auto output_data = output.data_ptr<scalar_t>();

    if (input.dim() == 3 || input.size(0) == 1) {
      adaptive_avg_pool2d_single_out_frame<scalar_t>(
          input_data,
          output_data,
          sizeD,
          isizeH,
          isizeW,
          osizeH,
          osizeW,
          istrideD,
          istrideH,
          istrideW);
    } else {
      int64_t istrideB = input.stride(-4);
      at::parallel_for(0, sizeB, 0, [&](int64_t start, int64_t end) {
        for (auto b = start; b < end; b++) {
          adaptive_avg_pool2d_single_out_frame<scalar_t>(
              input_data + b * istrideB,
              output_data + b * sizeD * osizeH * osizeW,
              sizeD,
              isizeH,
              isizeW,
              osizeH,
              osizeW,
              istrideD,
              istrideH,
              istrideW);
        }
      });
    }
    return output;
  }
}
} // namespace

Tensor quantized_adaptive_avg_pool2d(
    const at::Tensor& input,
    IntArrayRef output_size) {
  Tensor output;
  AT_DISPATCH_QINT_TYPES(
      input.scalar_type(), "quantized_adaptive_avg_pool2d", [&]() {
        output = q_adaptive_avg_pool2d<scalar_t>(input, output_size);
      });
  return output;
}

} // namespace native
} // namespace at
